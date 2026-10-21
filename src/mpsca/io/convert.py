"""Convert raw recordings of one experiment into an experiment store.

All files are binned lazily and streamed into ``movie`` chunk by chunk, so memory use
is bounded by a few chunks regardless of the recording length.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import dask.array as da
import numpy as np
import pandas as pd

from .._progress import compute_with_progress
from ..preprocess import bin_movie, detect_low_signal
from .metadata import Recording
from .mm_tiff import open_tiff_lazy
from .scan import ScanReport, scan_recording
from .store import ExperimentStore

log = logging.getLogger(__name__)


@dataclass
class ConvertParams:
    source_fps: float = 4.0
    target_fps: float = 1.0
    # blocks recorded with less camera binning are binned up to this (0 = coarsest block)
    target_camera_binning: int = 2
    # extra spatial binning applied after all blocks share one pixel grid
    spatial_binning: int = 1
    # explicit spatial factor per block (overrides the automatic choice above)
    block_spatial_factors: dict[int, int] | None = None
    low_signal_thresh: float = 0.5
    chunk_frames: int = 16
    pixel_size_um: float | None = None  # output pixel size, if known and not in the metadata
    read_timestamps: bool = True        # use Micro-Manager per-frame times when present

    @property
    def temporal_factor(self) -> int:
        ratio = self.source_fps / self.target_fps
        # tolerate a detected rate like 3.98 fps (millisecond timestamps)
        if ratio < 0.99 or abs(ratio - round(ratio)) > 0.02 * ratio:
            raise ValueError(f"source_fps/target_fps must be a whole number, got {ratio}")
        return int(round(ratio))

    def factors_for(self, report: ScanReport) -> dict[int, int]:
        factors = report.proposed_factors(self.spatial_binning, self.target_camera_binning)
        overrides = self.block_spatial_factors or report.recording.block_factors
        if overrides:
            factors.update({int(k): int(v) for k, v in overrides.items() if int(k) in factors})
        return factors


def spatial_factors(shapes, binning: int) -> list[int]:
    """Per-file spatial bin factors that bring every file to the same output size.

    The lowest-resolution file is binned by ``binning`` and larger ones by
    ``binning`` × their integer size ratio.
    """
    ref_h, ref_w = min(h for h, _ in shapes), min(w for _, w in shapes)
    out = []
    for h, w in shapes:
        rh, rw = h / ref_h, w / ref_w
        if abs(rh - rw) > 1e-9 or abs(rh - round(rh)) > 1e-9:
            raise ValueError(f"frame size {h}x{w} is not an integer multiple of {ref_h}x{ref_w}")
        out.append(binning * int(round(rh)))
    return out


def _fill_missing(ts: np.ndarray, fps: float) -> np.ndarray:
    """Replace NaN timestamps by the nearest known one shifted by the frame interval."""
    ok = np.isfinite(ts)
    if ok.all():
        return ts
    idx = np.arange(len(ts))
    anchors = np.flatnonzero(ok)
    pos = np.searchsorted(anchors, idx)
    left = anchors[np.clip(pos - 1, 0, len(anchors) - 1)]
    right = anchors[np.clip(pos, 0, len(anchors) - 1)]
    nearest = np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)
    return np.where(ok, ts, ts[nearest] + (idx - nearest) / fps)


def frame_times(report: ScanReport, fps: float,
                use_timestamps: bool = True) -> tuple[list[np.ndarray], list[str]]:
    """Seconds since the first frame for every readable frame of every file.

    Acquisition timestamps are used when every file has some, so pauses between
    condition blocks are kept; otherwise frames are assumed back to back at ``fps``.
    """
    warnings = []
    files = report.files
    usable = use_timestamps and all(np.isfinite(f.info.times[:f.n_frames]).any() for f in files)
    # a truncated last page has a header but no readable frame: its timestamp is dropped
    if usable and any(len(f.info.times) < f.n_frames for f in files):
        warnings.append("fewer timestamps than frames; using frame index for time")
        usable = False
    if not usable:
        out, offset = [], 0
        for f in files:
            out.append((offset + np.arange(f.n_frames)) / fps)
            offset += f.n_frames
        return out, warnings
    out = [_fill_missing(f.info.times[:f.n_frames], fps) for f in files]
    t0 = out[0][0]
    return [ts - t0 for ts in out], warnings


def convert_recording(
    rec: Recording,
    out_path: str | Path,
    params: ConvertParams | None = None,
    overwrite: bool = False,
    progress: Callable[[float], None] | None = None,
    report: ScanReport | None = None,
) -> ExperimentStore:
    """Bin every included file of ``rec`` and write the working movie plus frame table.

    ``report`` is the result of :func:`scan_recording`; it is computed if not given.
    Spatial factors: ``params.block_spatial_factors``, else ``rec.block_factors``, else
    the proposal of the scan.
    """
    params = params or ConvertParams()
    t = params.temporal_factor
    if report is None:
        report = scan_recording(rec, params.source_fps)
    if not report.files:
        raise ValueError(f"{rec.name}: no input files found on disk")
    factors = params.factors_for(report)
    out_shapes = report.output_shape(factors)
    if len(out_shapes) > 1:
        raise ValueError(f"spatial factors {factors} give different frame sizes {sorted(out_shapes)}")
    times, warnings = frame_times(report, params.source_fps, params.read_timestamps)
    warnings = list(report.issues) + warnings

    parts, tables, sources = [], [], []
    for fs, ts in zip(report.files, times):
        fe, f = fs.entry, factors[fs.entry.block]
        raw = open_tiff_lazy(fe.path, sources)[: fs.n_frames]
        binned = bin_movie(raw, t, f)
        src_idx = np.arange(binned.shape[0], dtype=np.int64) * t
        tables.append(pd.DataFrame({
            "source_file": fe.filename,
            "source_frame": src_idx.astype(np.int32),
            "block": np.int16(fe.block),
            "condition": fe.condition or f"block {fe.block}",
            "use_for_seg": fe.use_for_seg,
            "time_s": ts[src_idx],
            "camera_binning": np.int8(fs.info.camera_binning or 0),
            "spatial_factor": np.int8(f),
        }))
        parts.append(binned)
        log.info("%s: %s -> %s (spatial /%d)", fe.filename, raw.shape, binned.shape, f)

    movie = da.concatenate(parts, axis=0)
    chunks = (params.chunk_frames, *movie.shape[1:])
    movie = movie.rechunk(chunks)

    store = ExperimentStore.create(out_path, overwrite=overwrite)
    target = store.create_movie(movie.shape, chunks)
    write = da.store(movie, target, lock=False, compute=False)
    try:
        _, frame_means = compute_with_progress(write, movie.mean(axis=(1, 2)), progress=progress)
    finally:
        for src in sources:
            src.close()

    frames = pd.concat(tables, ignore_index=True)
    is_bad, cutoff = detect_low_signal(frame_means, params.low_signal_thresh)
    frames["frame_mean"] = frame_means.astype(np.float32)
    frames["is_bad"] = is_bad
    store.write_frames(frames)

    pixel = params.pixel_size_um
    if pixel is None:
        for fs in report.files:
            if fs.info.pixel_size_um:
                pixel = fs.info.pixel_size_um * factors[fs.entry.block]
                break
    blocks = {}
    for fs in report.files:
        b = blocks.setdefault(fs.entry.block, {"block": fs.entry.block, "condition": fs.entry.condition,
                                               "info": fs.entry.info, "files": []})
        b["files"].append(fs.entry.relpath)
    store.set_meta(name=rec.name, description=rec.description, blocks=list(blocks.values()),
                   source_files=[str(fs.entry.path) for fs in report.files],
                   raw_shapes=[[fs.n_frames, fs.height, fs.width] for fs in report.files],
                   camera_binning=sorted({fs.info.camera_binning for fs in report.files
                                          if fs.info.camera_binning}),
                   pixel_size_um=pixel, fps=params.target_fps,
                   missing_files=rec.missing,
                   frame_warnings=warnings)
    p = asdict(params)
    p["block_spatial_factors"] = {str(k): v for k, v in factors.items()}
    store.set_params("convert", {**p, "bad_frame_cutoff": cutoff})
    store.log(f"converted {len(report.files)} files -> movie {movie.shape}, {int(is_bad.sum())} bad frames")
    store.mark_complete()
    return store
