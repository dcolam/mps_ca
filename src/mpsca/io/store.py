"""The experiment store: one Zarr group per experiment holding every processing product.

Layout::

    <exp>.mpsca.zarr/
      movie             (T, Y, X) float32, binned working movie, small time chunks
      frames/<column>   per-frame table, one 1-D array per column
      projections/...   (later steps)
      labels/...        (later steps)
      traces/...        (later steps)
    attrs: format_version, experiment (name, description, blocks, ...), params (per step), log
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec

from .. import __version__

FORMAT_VERSION = 1
_COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")


class ExperimentStore:
    def __init__(self, group: zarr.Group):
        self.group = group

    # ---- open / create ----------------------------------------------------
    @classmethod
    def create(cls, path: str | Path, overwrite: bool = False) -> "ExperimentStore":
        group = zarr.open_group(str(path), mode="w" if overwrite else "w-")
        group.attrs.update({"format_version": FORMAT_VERSION, "mpsca_version": __version__,
                            "complete": False, "experiment": {}, "params": {}, "log": []})
        return cls(group)

    @classmethod
    def open(cls, path: str | Path, mode: str = "r+") -> "ExperimentStore":
        group = zarr.open_group(str(path), mode=mode)
        if "format_version" not in group.attrs:
            raise ValueError(f"{path} is not an mpsca experiment store")
        return cls(group)

    @property
    def path(self) -> Path:
        return Path(str(self.group.store.root))

    @property
    def complete(self) -> bool:
        """False while a conversion is running or if it was interrupted."""
        return bool(self.group.attrs.get("complete", False))

    def mark_complete(self) -> None:
        self.group.attrs["complete"] = True

    # ---- metadata ---------------------------------------------------------
    @property
    def meta(self) -> dict:
        meta = dict(self.group.attrs.get("experiment", {}))
        if "description" not in meta and meta.get("genotype"):  # written before version 0.2
            meta["description"] = {"sheet": meta.get("sheet"), "genotype": meta["genotype"]}
        return meta

    def set_meta(self, **values) -> None:
        self.group.attrs["experiment"] = {**self.meta, **values}

    def params(self, step: str) -> dict:
        return dict(self.group.attrs.get("params", {}).get(step, {}))

    def set_params(self, step: str, values: dict) -> None:
        params = dict(self.group.attrs.get("params", {}))
        params[step] = values
        self.group.attrs["params"] = params

    def log(self, message: str) -> None:
        entry = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
        self.group.attrs["log"] = [*self.group.attrs.get("log", []), entry]

    # ---- movie ------------------------------------------------------------
    def create_movie(self, shape, chunks, dtype="float32") -> zarr.Array:
        return self.group.create_array("movie", shape=shape, chunks=chunks, dtype=dtype,
                                       compressors=_COMPRESSOR, overwrite=True)

    @property
    def movie(self) -> da.Array:
        return da.from_zarr(self.group["movie"])

    # ---- derived arrays (projections, labels, traces, ...) -----------------
    def save_array(self, name: str, data, **attrs) -> zarr.Array:
        """Write ``data`` at ``name`` (e.g. ``"projections/mean"``), replacing any old one."""
        data = np.asarray(data)
        arr = self.group.create_array(name, shape=data.shape, dtype=data.dtype,
                                      compressors=_COMPRESSOR, overwrite=True)
        arr[...] = data
        arr.attrs.update(attrs)
        return arr

    def load_array(self, name: str) -> np.ndarray:
        return self.group[name][...]

    def array_attrs(self, name: str) -> dict:
        return dict(self.group[name].attrs)

    def has(self, name: str) -> bool:
        try:
            self.group[name]
        except KeyError:
            return False
        return True

    def list_arrays(self, group: str) -> list[str]:
        if not self.has(group):
            return []
        return sorted(k for k, _ in self.group[group].arrays())

    # ---- frame table ------------------------------------------------------
    def write_frames(self, df: pd.DataFrame) -> None:
        """Store the per-frame table; text columns are stored as category codes."""
        grp = self.group.require_group("frames")
        categories = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == bool or pd.api.types.is_numeric_dtype(s):
                data = s.to_numpy()
            else:
                cat = pd.Categorical(s.astype("string").fillna(""))
                categories[col] = [str(c) for c in cat.categories]
                data = cat.codes.astype(np.int32)
            grp.create_array(col, data=data, overwrite=True)
        grp.attrs.update({"columns": list(df.columns), "categories": categories})

    def read_frames(self) -> pd.DataFrame:
        grp = self.group["frames"]
        categories = grp.attrs.get("categories", {})
        cols = {}
        for col in grp.attrs["columns"]:
            data = grp[col][:]
            if col in categories:
                data = pd.Categorical.from_codes(data, categories=categories[col])
            cols[col] = data
        df = pd.DataFrame(cols)
        if "condition" not in df and "application" in df:  # stores written before version 0.2
            app, phase = df["application"].astype(str), df["phase"].astype(str)
            df["condition"] = pd.Categorical(np.where(phase != "", app + " · " + phase, app))
        return df

    def update_frames_column(self, col: str, values) -> None:
        df = self.read_frames()
        df[col] = values
        self.write_frames(df)

    def __repr__(self) -> str:
        shape = self.group["movie"].shape if "movie" in self.group else None
        return f"ExperimentStore({self.meta.get('name', '?')!r}, movie={shape})"
