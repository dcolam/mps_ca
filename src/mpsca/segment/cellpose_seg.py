"""Cellpose segmentation of projection images (optional dependency).

Cellpose runs on one 2-D image or a stack of up to 3 channels (e.g. mean + corr): the
channels are combined into ONE segmentation, like the colours of an RGB image. Usually
that image is a projection, so it is usable on CPU-only machines; segmenting the movie
window by window (``pipeline.run_cellpose_windows``) is slow without a GPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .postprocess import filter_area

log = logging.getLogger(__name__)


@dataclass
class CellposeParams:
    model: str = "cpsam_v2"        # Cellpose 4 default; a path to a fine-tuned model also works
    diameter: float | None = None  # pixels; None lets the model decide
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_area: int = 10
    max_area: int | None = None
    gpu: bool | None = None        # None = use a GPU when available


def available() -> bool:
    try:
        import cellpose  # noqa: F401
    except Exception:
        return False
    return True


def gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available() or torch.backends.mps.is_available())


def _normalise(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.8])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def build_input(projections: dict[str, np.ndarray], channels: list[str]) -> np.ndarray:
    """Stack the chosen projections, each normalised to 0..1, as (C, Y, X)."""
    if not channels:
        raise ValueError("choose at least one input image")
    if len(channels) > 3:
        raise ValueError("Cellpose takes at most 3 input images (channels)")
    return np.stack([_normalise(projections[c]) for c in channels])


_MODELS: dict = {}


def free_gpu() -> None:
    """Return cached GPU memory to the driver (the GPU may be shared with other users)."""
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _segment(image: np.ndarray, p: CellposeParams, gpu: bool) -> np.ndarray:
    from cellpose import models

    key = (p.model, gpu)
    if key not in _MODELS:  # loading the weights takes seconds; keep them between runs
        _MODELS[key] = models.CellposeModel(gpu=gpu, pretrained_model=p.model)
    kwargs = dict(diameter=p.diameter or None, flow_threshold=p.flow_threshold,
                  cellprob_threshold=p.cellprob_threshold, normalize=True)
    if image.ndim == 3:
        kwargs["channel_axis"] = 0
    return _MODELS[key].eval(image, **kwargs)[0]


def run_cellpose(image: np.ndarray, params: CellposeParams | None = None) -> np.ndarray:
    """Segment a (Y, X) or (C, Y, X) image; returns int32 labels.

    When the GPU runs out of memory (e.g. another user's job), Cellpose is rerun on the CPU.
    """
    p = params or CellposeParams()
    gpu = gpu_available() if p.gpu is None else p.gpu
    try:
        masks = _segment(image, p, gpu)
    except (RuntimeError, MemoryError) as e:
        if not (gpu and _is_oom(e)):
            raise
        log.warning("GPU out of memory, running Cellpose on the CPU instead")
        _MODELS.pop((p.model, True), None)
        free_gpu()
        masks = _segment(image, p, False)
    finally:
        if gpu:
            free_gpu()
    return filter_area(masks, p.min_area, p.max_area)
