"""ΔF/F with a rolling-percentile baseline.

Differences from ``ca_pipeline_v2.ipynb``:
- vectorised (``scipy.ndimage.percentile_filter``) instead of a Python triple loop;
- the baseline never spans a pause in the recording (e.g. between condition blocks);
- bad frames are bridged by interpolation for the baseline and returned as NaN-free values;
- an optional camera offset is subtracted first. These recordings sit on an offset of
  about 100 counts with only a few counts of signal, so without it ΔF/F is strongly
  compressed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import percentile_filter


@dataclass
class DffParams:
    baseline_window_s: float = 60.0
    percentile: float = 8.0
    offset: float | None = None  # camera offset subtracted from F; None = estimate from the mean image
    gap_factor: float = 3.0   # a pause longer than gap_factor × frame interval splits segments


def segments(time_s: np.ndarray, fps: float, gap_factor: float = 3.0) -> list[slice]:
    """Contiguous runs of frames without a pause longer than ``gap_factor`` frame intervals."""
    if len(time_s) == 0:
        return []
    breaks = np.flatnonzero(np.diff(time_s) > gap_factor / fps) + 1
    edges = [0, *breaks.tolist(), len(time_s)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _bridge(F: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any() or bad.all():
        return F
    t = np.arange(F.shape[1])
    good = ~bad
    out = F.copy()
    for i in range(F.shape[0]):
        out[i, bad] = np.interp(t[bad], t[good], F[i, good])
    return out


def estimate_offset(image: np.ndarray, percentile: float = 0.5) -> float:
    """Rough camera offset: a low percentile of a (mean) image."""
    return float(np.percentile(image, percentile))


def compute_dff(F: np.ndarray, time_s: np.ndarray, fps: float, bad: np.ndarray | None = None,
                params: DffParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dff, F0)``, both shaped like ``F`` (n_rois, T)."""
    p = params or DffParams()
    F = np.asarray(F, dtype=np.float32) - (p.offset or 0.0)
    bad = np.zeros(F.shape[1], bool) if bad is None else np.asarray(bad, bool)
    F = _bridge(F, bad)
    win = max(1, int(p.baseline_window_s * fps))
    size = 2 * (win // 2) + 1  # same centred window as the notebook: t ± win//2
    F0 = np.empty_like(F)
    for seg in segments(time_s, fps, p.gap_factor):
        F0[:, seg] = percentile_filter(F[:, seg], p.percentile, size=(1, size), mode="nearest")
    dff = (F - F0) / (np.abs(F0) + 1e-6)
    return dff.astype(np.float32), F0.astype(np.float32)
