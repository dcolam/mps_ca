"""Summary images of the movie used for segmentation.

``corr`` is the local correlation image: for each pixel, the mean temporal
correlation with its 8 neighbours. Pixels inside an active cell fluctuate together,
so cells stand out even when the mean image is dominated by noise or background.
"""
from __future__ import annotations

from typing import Callable

import dask.array as da
import numpy as np
from scipy.ndimage import uniform_filter

from .._progress import compute_with_progress


def _neighbour_product(block: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (block - mean) / std
    # mean of the 8 neighbours = (3x3 box sum - centre) / 8
    nb = (uniform_filter(z, size=(1, 3, 3), mode="nearest") * 9 - z) / 8
    return (z * nb).sum(axis=0, keepdims=True)


def compute_projections(movie: da.Array, frame_idx: np.ndarray | None = None,
                        progress: Callable[[float], None] | None = None) -> dict[str, np.ndarray]:
    """Mean, max, std and local-correlation images over ``frame_idx`` (all frames if None)."""
    sel = movie if frame_idx is None else movie[np.sort(np.asarray(frame_idx))]
    if sel.shape[0] == 0:
        raise ValueError("no frames selected for the projections")
    half = (lambda f: progress(f / 2)) if progress else None
    mean, mx, sq = compute_with_progress(sel.mean(axis=0), sel.max(axis=0),
                                         (sel.astype(np.float64) ** 2).mean(axis=0), progress=half)
    std = np.sqrt(np.maximum(sq - mean.astype(np.float64) ** 2, 0)).astype(np.float32)
    safe_std = np.where(std > 0, std, 1).astype(np.float32)
    prod = sel.map_blocks(_neighbour_product, mean, safe_std,
                          chunks=((1,) * sel.numblocks[0], *sel.chunks[1:]), dtype=np.float32)
    rest = (lambda f: progress(0.5 + f / 2)) if progress else None
    (corr_sum,) = compute_with_progress(prod.sum(axis=0), progress=rest)
    corr = (corr_sum / sel.shape[0]).astype(np.float32)
    corr[std == 0] = 0
    return {"mean": mean.astype(np.float32), "max": mx.astype(np.float32),
            "std": std, "corr": corr}


def parse_frames(text: str, n: int) -> np.ndarray | None:
    """Frame indices from text such as ``"0-299, 1200-"`` (0-based, ends included).

    ``a-`` runs to the last frame; ends beyond the movie are clipped. Empty text gives None.
    """
    if not text or not text.strip():
        return None
    idx = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        try:
            lo = int(a)
            hi = (int(b) if b.strip() else n - 1) if sep else lo
        except ValueError:
            raise ValueError(f"cannot read frame range '{part}' (use e.g. 0-299, 1200-)") from None
        if lo >= n:
            raise ValueError(f"frame range '{part}' is outside the movie (frames 0-{n - 1})")
        if hi < lo:
            raise ValueError(f"frame range '{part}' ends before it starts")
        idx.extend(range(lo, min(hi, n - 1) + 1))
    return np.unique(np.asarray(idx, dtype=int))


def segmentation_frames(frames, selection: str = "") -> np.ndarray:
    """Frames used for projections, never the bad ones.

    With a ``selection`` (see :func:`parse_frames`) those frames are used, otherwise the
    frames flagged for segmentation (fallback: all good frames).
    """
    good = ~frames["is_bad"].to_numpy()
    chosen = parse_frames(selection, len(frames))
    if chosen is not None:
        keep = chosen[good[chosen]]
        if not len(keep):
            raise ValueError(f"frames '{selection}' are all bad frames")
        return keep
    seg = frames["use_for_seg"].to_numpy() & good
    return np.flatnonzero(seg if seg.any() else good)
