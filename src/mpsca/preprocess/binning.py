"""Temporal and spatial binning of (T, Y, X) movies, lazily via dask."""
from __future__ import annotations

import dask.array as da
import numpy as np

# raw frames per dask chunk is a multiple of the temporal factor near this budget
_TARGET_CHUNK_BYTES = 64 * 2**20


def _bin_block(block: np.ndarray, t: int, s: int) -> np.ndarray:
    a = block.astype(np.float32)
    T, H, W = a.shape
    if t > 1:
        a = a.reshape(T // t, t, H, W).mean(axis=1)
    if s > 1:
        a = a.reshape(a.shape[0], H // s, s, W // s, s).mean(axis=(2, 4))
    return a


def bin_movie(arr: da.Array, t: int = 1, s: int = 1) -> da.Array:
    """Average non-overlapping ``t`` frames and ``s``×``s`` pixel blocks.

    Trailing frames/rows/columns that do not fill a whole bin are dropped,
    as in the original pipeline. Returns float32.
    """
    if t < 1 or s < 1:
        raise ValueError("binning factors must be >= 1")
    T, H, W = arr.shape
    Tt, Hb, Wb = (T // t) * t, (H // s) * s, (W // s) * s
    if Tt == 0:
        return da.zeros((0, H // s, W // s), dtype=np.float32)
    a = arr[:Tt, :Hb, :Wb]
    frame_bytes = Hb * Wb * a.dtype.itemsize
    per_chunk = max(t, (_TARGET_CHUNK_BYTES // frame_bytes) // t * t)
    a = a.rechunk((min(per_chunk, Tt), Hb, Wb))
    out_chunks = (tuple(c // t for c in a.chunks[0]), (Hb // s,), (Wb // s,))
    return a.map_blocks(_bin_block, t, s, chunks=out_chunks, dtype=np.float32)
