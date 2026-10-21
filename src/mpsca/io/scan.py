"""Header-only inspection of an experiment's files before conversion.

The scan reads TIFF headers and Micro-Manager page metadata (no pixel data) and
reports anything the user should decide on: frame sizes that differ between blocks,
frame counts that differ from the metadata, frame-rate deviations, truncated files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import tifffile

from .metadata import ExperimentMeta, FileEntry
from .mm_tiff import FrameInfo, _page_metadata, _received_time

log = logging.getLogger(__name__)


@dataclass
class FileScan:
    entry: FileEntry
    n_frames: int
    height: int
    width: int
    n_pages: int
    info: FrameInfo


@dataclass
class ScanReport:
    meta: ExperimentMeta
    files: list[FileScan] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def shapes(self) -> set[tuple[int, int]]:
        return {(f.height, f.width) for f in self.files}

    @property
    def resolution_mismatch(self) -> bool:
        return len(self.shapes) > 1

    def proposed_factors(self, extra_binning: int = 1, target_camera_binning: int = 2) -> dict[int, int]:
        """Spatial bin factor per block for a common pixel grid.

        When every file records its camera binning, each block is binned up to
        ``target_camera_binning`` (2×2: unbinned 2200×3200 -> 1100×1600, camera-binned
        blocks untouched), so all experiments share one pixel size. Otherwise blocks are
        brought to the coarsest frame size of this experiment. ``extra_binning`` is
        applied on top.
        """
        if target_camera_binning > 0 and all(f.info.camera_binning for f in self.files):
            return {f.entry.block: extra_binning * max(1, target_camera_binning // f.info.camera_binning)
                    for f in self.files}
        ref_h = min(h for h, _ in self.shapes)
        return {f.entry.block: extra_binning * (f.height // ref_h) for f in self.files}

    def needs_decision(self, extra_binning: int = 1, target_camera_binning: int = 2) -> bool:
        """True when blocks differ in size or the proposal bins some block in software."""
        factors = self.proposed_factors(extra_binning, target_camera_binning)
        return self.resolution_mismatch or any(v != extra_binning for v in factors.values())

    def output_shape(self, factors: dict[int, int]) -> set[tuple[int, int]]:
        return {(f.height // factors[f.entry.block], f.width // factors[f.entry.block])
                for f in self.files}

    def block_summary(self) -> pd.DataFrame:
        rows = []
        for block in sorted({f.entry.block for f in self.files}):
            fs = [f for f in self.files if f.entry.block == block]
            t = np.concatenate([f.info.times for f in fs]) if fs else np.array([])
            t = t[np.isfinite(t)]
            rows.append({
                "block": block,
                "application": fs[0].entry.application,
                "phase": fs[0].entry.phase,
                "files": len(fs),
                "frames": sum(f.n_frames for f in fs),
                "expected": fs[0].entry.expected_frames,
                "size": " / ".join(sorted({f"{f.height}x{f.width}" for f in fs})),
                "camera_binning": " / ".join(sorted({str(f.info.camera_binning) for f in fs})),
                "duration_s": round(float(t.max() - t.min()), 1) if len(t) > 1 else None,
                "seg_files": sum(f.entry.use_for_seg for f in fs),
            })
        return pd.DataFrame(rows)


def scan_file(entry: FileEntry) -> FileScan:
    times, binning, pixel = [], None, None
    with tifffile.TiffFile(str(entry.path), is_ome=False, is_mmstack=False) as tf:
        series = tf.series[0]
        shape = series.shape
        n_pages = len(tf.pages)
        for page in tf.pages:
            md = _page_metadata(page)
            times.append(_received_time(md))
            if binning is None and md.get("Binning") is not None:
                binning = int(md["Binning"])
            if pixel is None and md.get("PixelSizeUm"):
                pixel = float(md["PixelSizeUm"])
    n_frames = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    return FileScan(entry, n_frames, int(shape[-2]), int(shape[-1]), n_pages,
                    FrameInfo(np.asarray(times, dtype=np.float64), binning, pixel))


def scan_experiment(meta: ExperimentMeta, source_fps: float = 4.0,
                    progress: Callable[[float], None] | None = None) -> ScanReport:
    report = ScanReport(meta)
    entries = [fe for fe in meta.files if fe.path is not None]
    for fe in meta.files:
        if fe.path is None:
            report.issues.append(f"missing file: {fe.filename}")
    for i, fe in enumerate(entries):
        fs = scan_file(fe)
        report.files.append(fs)
        if fs.n_pages != fs.n_frames:
            report.issues.append(f"{fe.filename}: {fs.n_pages} pages but {fs.n_frames} readable "
                                 "frames (truncated or damaged file)")
        if fs.n_frames == 1:
            report.issues.append(f"{fe.filename}: single image, not a recording; remove it")
        t = fs.info.times[np.isfinite(fs.info.times)]
        if len(t) > 2:
            dt = float(np.median(np.diff(t)))
            if abs(dt * source_fps - 1) > 0.1:
                report.issues.append(f"{fe.filename}: frame interval {dt:.3f} s, "
                                     f"expected {1 / source_fps:.3f} s")
        if progress:
            progress((i + 1) / len(entries))

    blocks = report.block_summary() if report.files else pd.DataFrame()
    for _, b in blocks.iterrows():
        if pd.notna(b["expected"]) and b["frames"] != b["expected"]:
            report.issues.append(f"block {b['block']} ({b['application']}, {b['phase']}): "
                                 f"{b['frames']} frames on disk, {int(b['expected'])} listed")
    if report.resolution_mismatch:
        sizes = ", ".join(f"block {b['block']}: {b['size']}" for _, b in blocks.iterrows())
        report.issues.append(f"frame size differs between blocks ({sizes}); "
                             "choose a spatial factor per block")
    return report
