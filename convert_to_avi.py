"""
convert_to_avi.py — Convert microscopy files to uncompressed AVI.

Strategy (tried in order):
  1. ImageJ/Fiji (recommended) — handles any Bio-Formats-supported format,
     writes truly uncompressed AVI with correct FPS.  Looks for Fiji on both
     Windows (for running from WSL) and Linux.
  2. Python fallback — uses cv2 to read frames and a pure-Python RIFF AVI
     writer to produce uncompressed output without ffmpeg.

Inputs supported by Fiji: .nd2, .tif, .tiff, .czi, .lif, .isxd, .bmp, .png ...
Inputs supported by cv2 fallback: .tif/.tiff stacks, .bmp/.png sequences.

Usage:
    python convert_to_avi.py \\
        --data_dir /mnt/z/ephacoffice/DColameo/Ca_Anand_AllData \\
        --fps 30 \\
        --workers 4

    # Convert only specific extensions:
    python convert_to_avi.py --data_dir /mnt/z/... --fps 30 --ext .nd2

    # Convert a single file:
    python convert_to_avi.py --file /path/to/recording.nd2 --fps 30

    # Dry run — show what would be converted:
    python convert_to_avi.py --data_dir /mnt/z/... --fps 30 --dry_run

Notes:
    • Output AVI files are written alongside the source files (same directory,
      same base name with .avi extension).
    • Already-present .avi files are skipped unless --overwrite is passed.
    • FPS is the most important parameter — check your recording software to
      confirm the correct value before converting.
"""

import os
import re
import sys
import shutil
import struct
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONVERTIBLE_EXTENSIONS = {".nd2", ".tif", ".tiff", ".czi", ".lif", ".isxd",
                           ".bmp", ".png", ".mp4", ".mkv", ".mov"}

MACRO_PATH = Path(__file__).parent / "imagej_macros" / "convert_to_avi.ijm"

# Common Fiji/ImageJ install paths (Windows paths accessible from WSL, and Linux)
FIJI_CANDIDATES = [
    # WSL → Windows Fiji
    "/mnt/c/Fiji.app/ImageJ-win64.exe",
    "/mnt/c/Program Files/Fiji.app/ImageJ-win64.exe",
    "/mnt/c/Users/DColameo/AppData/Local/Fiji.app/ImageJ-win64.exe",
    "/mnt/c/Users/DColameo/Desktop/Fiji.app/ImageJ-win64.exe",
    "/mnt/c/Users/DColameo/Downloads/Fiji.app/ImageJ-win64.exe",
    # Linux/WSL Fiji
    "/opt/fiji/ImageJ-linux64",
    "/usr/local/fiji/ImageJ-linux64",
    str(Path.home() / "Fiji.app" / "ImageJ-linux64"),
    str(Path.home() / "fiji" / "ImageJ-linux64"),
]


# ── Fiji discovery ────────────────────────────────────────────────────────────

def find_fiji() -> Optional[str]:
    """Return path to Fiji/ImageJ executable, or None if not found."""
    # Check PATH first
    for name in ("fiji", "ImageJ-linux64", "ImageJ-win64.exe"):
        exe = shutil.which(name)
        if exe:
            return exe

    for candidate in FIJI_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate

    return None


# ── Fiji-based conversion ─────────────────────────────────────────────────────

def convert_with_fiji(
    fiji_exe: str,
    input_path: str,
    output_path: str,
    fps: int,
) -> bool:
    """
    Run the ImageJ macro to convert one file.  Returns True on success.
    """
    if not MACRO_PATH.exists():
        logger.error(f"ImageJ macro not found: {MACRO_PATH}")
        return False

    # Use | as delimiter to avoid issues with Windows paths containing spaces/commas
    macro_args = f"{input_path}|{output_path}|{fps}"

    cmd = [fiji_exe, "--headless", "--console", "-macro", str(MACRO_PATH), macro_args]
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per file max
        )
        if result.returncode != 0:
            logger.error(f"Fiji failed (code {result.returncode}):\n{result.stderr}")
            return False
        if "ERROR" in result.stdout:
            logger.error(f"Fiji macro error:\n{result.stdout}")
            return False
        logger.debug(result.stdout.strip())
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Fiji timed out on: {input_path}")
        return False
    except Exception as e:
        logger.error(f"Fiji subprocess error: {e}")
        return False


# ── Pure-Python uncompressed AVI writer ───────────────────────────────────────
# Implements just enough of the RIFF/AVI spec to write uncompressed 8-bit
# grayscale AVIs that MPS / ffmpeg-probe can read correctly.

def _pack(fmt: str, *args) -> bytes:
    return struct.pack(fmt, *args)

def _fourcc(s: str) -> bytes:
    return s.encode("ascii")[:4]


def write_uncompressed_avi(
    output_path: str,
    frames,          # iterable of 2-D numpy uint8 arrays (H, W)
    fps: int,
    height: int,
    width: int,
):
    """
    Write an uncompressed 8-bit grayscale AVI file without ffmpeg.

    Each frame is a (height, width) numpy uint8 array.
    """
    import numpy as np

    frame_list = list(frames)
    n_frames = len(frame_list)
    if n_frames == 0:
        raise ValueError("No frames to write.")

    # BITMAPINFOHEADER for 8-bit grayscale (BI_RGB, biBitCount=8)
    bmp_info = (
        _pack("<I", 40)             # biSize
        + _pack("<i", width)        # biWidth
        + _pack("<i", height)       # biHeight (positive = bottom-up; MPS reads either)
        + _pack("<H", 1)            # biPlanes
        + _pack("<H", 8)            # biBitCount (8 = grayscale palette)
        + _pack("<I", 0)            # biCompression (0 = BI_RGB)
        + _pack("<I", width * height)  # biSizeImage
        + _pack("<i", 0)            # biXPelsPerMeter
        + _pack("<i", 0)            # biYPelsPerMeter
        + _pack("<I", 256)          # biClrUsed (greyscale palette has 256 entries)
        + _pack("<I", 0)            # biClrImportant
    )
    # Greyscale palette (256 entries × 4 bytes: B G R Reserved)
    palette = b"".join(_pack("BBBB", i, i, i, 0) for i in range(256))
    strf_data = bmp_info + palette

    # AVI main header (avih)
    us_per_frame = int(1_000_000 / fps)
    max_bytes_per_sec = width * height * fps
    avih = (
        _pack("<I", us_per_frame)   # dwMicroSecPerFrame
        + _pack("<I", max_bytes_per_sec)  # dwMaxBytesPerSec
        + _pack("<I", 0)            # dwPaddingGranularity
        + _pack("<I", 0x910)        # dwFlags (AVIF_HASINDEX | AVIF_ISINTERLEAVED | AVIF_TRUSTCKTYPE)
        + _pack("<I", n_frames)     # dwTotalFrames
        + _pack("<I", 0)            # dwInitialFrames
        + _pack("<I", 1)            # dwStreams
        + _pack("<I", width * height)  # dwSuggestedBufferSize
        + _pack("<I", width)        # dwWidth
        + _pack("<I", height)       # dwHeight
        + _pack("<I", 0) * 4        # dwReserved[4]
    )

    # Stream header (strh) for video
    strh = (
        _fourcc("vids")             # fccType
        + _fourcc("\x00\x00\x00\x00")  # fccHandler (uncompressed)
        + _pack("<I", 0)            # dwFlags
        + _pack("<H", 0)            # wPriority
        + _pack("<H", 0)            # wLanguage
        + _pack("<I", 0)            # dwInitialFrames
        + _pack("<I", 1)            # dwScale
        + _pack("<I", fps)          # dwRate  (fps = dwRate / dwScale)
        + _pack("<I", 0)            # dwStart
        + _pack("<I", n_frames)     # dwLength
        + _pack("<I", width * height)  # dwSuggestedBufferSize
        + _pack("<I", 0xFFFFFFFF)   # dwQuality (-1 = default)
        + _pack("<I", 0)            # dwSampleSize
        + _pack("<hhhh", 0, 0, width, height)  # rcFrame
    )

    def _chunk(tag: str, data: bytes) -> bytes:
        padded = data + (b"\x00" if len(data) % 2 else b"")
        return _fourcc(tag) + _pack("<I", len(data)) + padded

    def _list(tag: str, data: bytes) -> bytes:
        return _fourcc("LIST") + _pack("<I", 4 + len(data)) + _fourcc(tag) + data

    # Build hdrl LIST
    hdrl = _chunk("avih", avih) + _list("strl", _chunk("strh", strh) + _chunk("strf", strf_data))

    # Build movi LIST and index
    frame_data_list = []
    for f in frame_list:
        arr = np.asarray(f, dtype=np.uint8)
        if arr.shape != (height, width):
            import cv2
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
        # AVI stores rows bottom-up for BI_RGB positive height; flip for correct display
        arr = np.flipud(arr)
        frame_data_list.append(arr.tobytes())

    movi_content = b""
    idx_entries = []
    offset = 4  # initial offset inside movi (after 'movi' fourcc)
    for raw in frame_data_list:
        tag = b"00dc"
        chunk_size = len(raw)
        padded = raw + (b"\x00" if chunk_size % 2 else b"")
        idx_entries.append((offset, chunk_size))
        movi_content += tag + _pack("<I", chunk_size) + padded
        offset += 8 + len(padded)

    movi = _fourcc("LIST") + _pack("<I", 4 + len(movi_content)) + _fourcc("movi") + movi_content

    # Build idx1 (legacy index — needed for seekability)
    movi_offset = (
        12          # RIFF header
        + 8 + 4 + len(_list("hdrl", hdrl))  # hdrl
        + 8         # movi LIST header
    )
    idx1_content = b""
    for chunk_offset, chunk_size in idx_entries:
        abs_offset = movi_offset + chunk_offset
        idx1_content += (
            _fourcc("00dc")
            + _pack("<I", 0x10)       # AVIIF_KEYFRAME
            + _pack("<I", abs_offset)
            + _pack("<I", chunk_size)
        )

    hdrl_block = _list("hdrl", hdrl)
    riff_content = hdrl_block + movi + _chunk("idx1", idx1_content)
    riff = _fourcc("RIFF") + _pack("<I", 4 + len(riff_content)) + _fourcc("AVI ") + riff_content

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(riff)

    logger.info(f"Wrote {n_frames} frames → {output_path}  ({os.path.getsize(output_path)/(1024**2):.1f} MB)")


def convert_with_python(input_path: str, output_path: str, fps: int) -> bool:
    """
    Python fallback: read frames with cv2 (no ffmpeg needed for reading TIFFs),
    write with pure-Python AVI writer.
    Works best for .tif/.tiff stacks and image sequences.
    """
    try:
        import cv2
        import numpy as np

        ext = Path(input_path).suffix.lower()

        if ext in (".tif", ".tiff"):
            # Try reading as a multi-page TIFF
            try:
                import tifffile
                frames_arr = tifffile.imread(input_path)
                if frames_arr.ndim == 2:
                    frames_arr = frames_arr[np.newaxis]  # single frame
                elif frames_arr.ndim == 3 and frames_arr.shape[2] <= 4:
                    # Might be (H, W, C) — single colour frame
                    frames_arr = frames_arr[np.newaxis]
                # Convert to 8-bit if needed
                if frames_arr.dtype != np.uint8:
                    f_min, f_max = frames_arr.min(), frames_arr.max()
                    if f_max > f_min:
                        frames_arr = ((frames_arr - f_min) / (f_max - f_min) * 255).astype(np.uint8)
                    else:
                        frames_arr = np.zeros_like(frames_arr, dtype=np.uint8)
                # Ensure 2-D frames (grayscale)
                if frames_arr.ndim == 4:
                    # (N, H, W, C) → convert to grayscale
                    frames_arr = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_arr])
                H, W = frames_arr.shape[1], frames_arr.shape[2]
                write_uncompressed_avi(output_path, frames_arr, fps, H, W)
                return True
            except ImportError:
                pass

            # Fall back to cv2 multi-frame read
            cap = cv2.VideoCapture(input_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            cap.release()
            if not frames:
                logger.error(f"cv2 could not read any frames from {input_path}")
                return False
            H, W = frames[0].shape
            write_uncompressed_avi(output_path, frames, fps, H, W)
            return True

        else:
            # For MP4/MKV/MOV: cv2 may or may not need ffmpeg
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"cv2 cannot open: {input_path}")
                return False
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            cap.release()
            H, W = frames[0].shape
            write_uncompressed_avi(output_path, frames, fps, H, W)
            return True

    except Exception as e:
        logger.error(f"Python fallback failed for {input_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


# ── File discovery ────────────────────────────────────────────────────────────

_EXCLUDE_DIR_RE = re.compile(r'^(40[Xx]|\.)', re.IGNORECASE)


def find_files_to_convert(
    root: str,
    extensions: set,
    overwrite: bool = False,
    output_root: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Return (input_path, output_avi_path) pairs for files that need conversion.

    Directories matching _EXCLUDE_DIR_RE (40X, 40X BF, hidden dirs) are skipped
    because they contain single reference images, not video recordings.

    Args:
        output_root: If given, AVIs are written under output_root, mirroring the
                     directory structure from root.  If None, AVIs are written
                     alongside the source files (same directory).
    """
    pairs = []
    for dirpath, subdirs, filenames in os.walk(root, topdown=True):
        subdirs[:] = sorted(d for d in subdirs if not _EXCLUDE_DIR_RE.match(d))
        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if ext not in extensions:
                continue
            input_path = os.path.join(dirpath, fname)
            stem = Path(fname).stem

            if output_root:
                rel_dir = os.path.relpath(dirpath, root)
                out_dir = os.path.join(output_root, rel_dir)
            else:
                out_dir = dirpath

            output_path = os.path.join(out_dir, stem + ".avi")

            if not overwrite and os.path.exists(output_path):
                logger.info(f"Skip (exists): {output_path}")
                continue
            pairs.append((input_path, output_path))
    return pairs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _relpath_safe(path: str, start: str = None) -> str:
    """os.path.relpath that never raises on Windows cross-drive paths."""
    try:
        return os.path.relpath(path, start) if start else os.path.relpath(path)
    except ValueError:
        return path


# ── Per-file conversion dispatch ──────────────────────────────────────────────

def convert_one(
    input_path: str,
    output_path: str,
    fps: int,
    fiji_exe: Optional[str],
    dry_run: bool = False,
) -> Tuple[str, bool]:
    """Convert one file. Returns (output_path, success)."""
    rel = _relpath_safe(input_path)
    if dry_run:
        print(f"  [DRY RUN] {rel}")
        print(f"         → {output_path}")
        return output_path, True

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Converting: {rel}")

    # Try Fiji first
    if fiji_exe:
        ok = convert_with_fiji(fiji_exe, input_path, output_path, fps)
        if ok and os.path.exists(output_path):
            logger.info(f"  Fiji → OK  ({os.path.getsize(output_path)/(1024**2):.1f} MB)")
            return output_path, True
        else:
            logger.warning(f"  Fiji failed — trying Python fallback...")

    # Python fallback
    ok = convert_with_python(input_path, output_path, fps)
    if ok and os.path.exists(output_path):
        logger.info(f"  Python → OK  ({os.path.getsize(output_path)/(1024**2):.1f} MB)")
        return output_path, True

    logger.error(f"  FAILED: {rel}")
    return output_path, False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert microscopy TIF stacks to uncompressed AVI (no ffmpeg required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_dir",    help="Root directory to scan recursively.")
    src.add_argument("--file",        help="Convert a single file.")

    parser.add_argument("--fiji",      required=True,
                        help="Path to the Fiji/ImageJ executable.  "
                             "Example (WSL → Windows Fiji): "
                             "/mnt/c/Users/YourName/Desktop/Fiji.app/ImageJ-win64.exe")
    parser.add_argument("--fps",       type=int, required=True,
                        help="Recording frame rate in Hz.  For 250ms exposure: 4.  "
                             "For 500ms: 2.  For 100ms: 10.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for converted AVIs.  The folder structure "
                             "from --data_dir is mirrored under output_dir.  "
                             "If omitted, AVIs are written alongside the source TIFs.")
    parser.add_argument("--ext",       default=None,
                        help="Only convert this extension, e.g. .tif (default: all supported).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert even if .avi already exists.")
    parser.add_argument("--workers",   type=int, default=1,
                        help="Parallel Fiji instances (default: 1).  "
                             "Safe to use 4–8 on a workstation.")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Show what would be converted without doing it.")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── Validate Fiji ─────────────────────────────────────────────────────────
    fiji_exe = args.fiji
    if not os.path.isfile(fiji_exe):
        logger.error(
            f"Fiji executable not found: {fiji_exe}\n"
            "  Typical WSL path: /mnt/c/Users/<name>/Desktop/Fiji.app/ImageJ-win64.exe\n"
            "  Download Fiji from https://fiji.sc if not installed."
        )
        sys.exit(1)
    logger.info(f"Fiji: {fiji_exe}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output dir: {args.output_dir}")

    extensions = CONVERTIBLE_EXTENSIONS
    if args.ext:
        ext = args.ext if args.ext.startswith(".") else "." + args.ext
        extensions = {ext.lower()}

    # ── Build file list ───────────────────────────────────────────────────────
    if args.file:
        input_path = args.file
        if args.output_dir:
            output_path = os.path.join(args.output_dir, Path(input_path).stem + ".avi")
        else:
            output_path = str(Path(input_path).with_suffix(".avi"))
        pairs = [(input_path, output_path)]
    else:
        pairs = find_files_to_convert(
            args.data_dir, extensions,
            overwrite=args.overwrite,
            output_root=args.output_dir,
        )

    if not pairs:
        logger.info("Nothing to convert (use --overwrite to force re-conversion).")
        return

    logger.info(f"Files to convert: {len(pairs)}")
    if args.dry_run:
        data_root = args.data_dir or os.path.dirname(pairs[0][0])
        for inp, out in pairs:
            rel_in  = _relpath_safe(inp, data_root)
            rel_out = _relpath_safe(out, args.output_dir) if args.output_dir else os.path.basename(out)
            dest = f"[output_dir]/{rel_out}" if args.output_dir else f"(alongside) {rel_out}"
            print(f"  {rel_in}")
            print(f"    → {dest}")
        logger.info("--dry_run: no files written.")
        return

    # ── Convert ───────────────────────────────────────────────────────────────
    results = {}
    if args.workers == 1:
        for inp, out in pairs:
            _, ok = convert_one(inp, out, args.fps, fiji_exe)
            results[inp] = ok
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert_one, inp, out, args.fps, fiji_exe): inp
                for inp, out in pairs
            }
            for future in as_completed(futures):
                inp = futures[future]
                try:
                    _, ok = future.result()
                    results[inp] = ok
                except Exception as e:
                    logger.error(f"{inp}: {e}")
                    results[inp] = False

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok  = sum(1 for ok in results.values() if ok)
    n_err = len(results) - n_ok
    logger.info(f"\nConversion complete: {n_ok} succeeded, {n_err} failed.")
    if n_err:
        for path, ok in results.items():
            if not ok:
                logger.error(f"  FAILED: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
