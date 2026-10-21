"""Lazy access to Micro-Manager (MMStack) OME-TIFF recordings."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
import zarr

log = logging.getLogger(__name__)


def open_tiff_lazy(path: str | Path, stores: list | None = None) -> da.Array:
    """Return the frames stored in *this* file as a lazy (T, Y, X) dask array.

    Micro-Manager splits long recordings into ``*_MMStack_Default.ome.tif``,
    ``*_1.ome.tif``, ... and the OME/MMStack metadata of the first file describes the
    whole series. Opening that file normally can therefore return every file's frames,
    which would double-count frames when each file is also listed separately. The OME
    and MMStack interpretations are disabled so each file contributes only its own pages.

    The file stays open while the array is used; pass a list as ``stores`` to collect the
    underlying stores and ``close()`` them when done.
    """
    flags = dict(is_ome=False, is_mmstack=False)
    with tifffile.TiffFile(str(path), **flags) as tf:
        n_pages = len(tf.pages)
    store = tifffile.imread(str(path), aszarr=True, **flags)
    if stores is not None:
        stores.append(store)
    arr = da.from_zarr(zarr.open(store, mode="r"))
    n_planes = math.prod(arr.shape[:-2])
    if n_planes != n_pages:
        log.warning("%s: series has %d planes but the file has %d pages; "
                    "frames may be missing or read from other files", path, n_planes, n_pages)
    # drop singleton channel/z axes; keep the last three as (T, Y, X)
    while arr.ndim > 3 and 1 in arr.shape[:-2]:
        arr = arr.squeeze(axis=arr.shape[:-2].index(1))
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a (T, Y, X) stack, got shape {arr.shape}")
    return arr


_MM_TAG = 51123  # MicroManagerMetadata: per-page JSON


@dataclass
class FrameInfo:
    """Per-frame acquisition metadata read from Micro-Manager page tags."""
    times: np.ndarray                  # wall-clock seconds since the epoch, NaN where unknown
    camera_binning: int | None = None
    pixel_size_um: float | None = None


def _page_metadata(page) -> dict:
    tag = page.tags.get(_MM_TAG)
    if tag is None:
        return {}
    value = tag.value
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _received_time(md: dict) -> float:
    value = md.get("ReceivedTime")
    if not value:
        return np.nan
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f %z").timestamp()
    except ValueError:
        return np.nan
