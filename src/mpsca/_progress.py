"""Progress reporting for dask computations (used by the CLI and the GUI)."""
from __future__ import annotations

from typing import Callable

import dask
from dask.callbacks import Callback


class ProgressCallback(Callback):
    """Report the fraction of finished dask tasks to ``fn``."""

    def __init__(self, fn: Callable[[float], None]):
        super().__init__()
        self._fn = fn
        self._total = 1

    def _start_state(self, dsk, state):
        self._total = max(1, sum(len(state[k]) for k in ("ready", "waiting", "running", "finished")))

    def _posttask(self, key, result, dsk, state, worker_id):
        self._fn(len(state["finished"]) / self._total)


def compute_with_progress(*args, progress: Callable[[float], None] | None = None):
    if progress is None:
        return dask.compute(*args)
    with ProgressCallback(progress):
        return dask.compute(*args)
