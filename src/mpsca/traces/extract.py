"""Fluorescence traces: mean intensity of every ROI in every frame, in one streaming pass."""
from __future__ import annotations

from typing import Callable

import dask.array as da
import numpy as np
from scipy import sparse

from .._progress import compute_with_progress


def roi_matrix(labels: np.ndarray) -> sparse.csr_matrix:
    """(n_rois, n_pixels) matrix whose rows average the pixels of each ROI."""
    flat = labels.ravel()
    pix = np.flatnonzero(flat)
    rows = flat[pix] - 1
    n = int(labels.max())
    area = np.bincount(rows, minlength=n).astype(np.float32)
    weights = 1.0 / area[rows]
    return sparse.csr_matrix((weights, (rows, pix)), shape=(n, flat.size), dtype=np.float32)


def extract_traces(movie: da.Array, labels: np.ndarray,
                   progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Return F with shape (n_rois, T)."""
    if labels.shape != movie.shape[1:]:
        raise ValueError(f"labels {labels.shape} do not match the movie frames {movie.shape[1:]}")
    A = roi_matrix(labels)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, movie.shape[0]), dtype=np.float32)

    def _block(b):
        return np.asarray((A @ b.reshape(b.shape[0], -1).T).T, dtype=np.float32)

    traces = movie.map_blocks(_block, drop_axis=2, chunks=(movie.chunks[0], (n,)), dtype=np.float32)
    (F,) = compute_with_progress(traces, progress=progress)
    return F.T.copy()
