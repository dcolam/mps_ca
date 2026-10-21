"""Threshold + watershed segmentation sweep, ported from ``ca_pipeline_v2.ipynb``.

Every combination of base image (built from mean/std/max projections) and global
threshold method is segmented; the combination with the best composite score of
ROI count, total ROI area and mean activity (std) per ROI wins.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage import feature, filters, measure, morphology, segmentation

from .postprocess import filter_area

THRESHOLD_METHODS = {
    "otsu": filters.threshold_otsu, "yen": filters.threshold_yen, "li": filters.threshold_li,
    "triangle": filters.threshold_triangle, "isodata": filters.threshold_isodata,
}
BASE_IMAGES = ("mean", "std", "combined", "max", "max_std")


@dataclass
class ClassicParams:
    gauss_sigma: float = 2.0
    min_peak_distance: int = 8
    min_cell_area: int = 10
    max_cell_area: int = 1000
    score_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)  # n_rois, area, activity


@dataclass
class SweepResult:
    labels: np.ndarray
    best: tuple[str, str]
    threshold: float
    bases: list[str]
    methods: list[str]
    n_rois: np.ndarray
    area: np.ndarray
    activity: np.ndarray
    score: np.ndarray
    grid: dict = field(repr=False, default_factory=dict)  # (base, method) -> labels


def _norm(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-9)


def _n01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def _remove_small(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drop objects smaller than ``min_area`` (skimage renamed min_size to max_size in 0.26)."""
    if "max_size" in inspect.signature(morphology.remove_small_objects).parameters:
        return morphology.remove_small_objects(binary, max_size=min_area - 1)
    return morphology.remove_small_objects(binary, min_size=min_area)


def segment_watershed(seg: np.ndarray, threshold: float, p: ClassicParams) -> np.ndarray:
    binary = ndimage.binary_fill_holes(_remove_small(seg > threshold, p.min_cell_area))
    peaks = feature.peak_local_max(seg * binary, min_distance=p.min_peak_distance, labels=binary)
    markers = np.zeros(binary.shape, dtype=int)
    if len(peaks):
        markers[tuple(peaks.T)] = 1
    labels = segmentation.watershed(-seg, measure.label(markers), mask=binary)
    return filter_area(labels, p.min_cell_area, p.max_cell_area)


def classic_sweep(mean: np.ndarray, std: np.ndarray, mx: np.ndarray,
                  params: ClassicParams | None = None) -> SweepResult:
    p = params or ClassicParams()
    mn, sn, xn = _norm(mean), _norm(std), _norm(mx)
    bases = {"mean": mn, "std": sn, "combined": mn * sn, "max": xn, "max_std": xn * sn}
    segs = {k: _norm(filters.gaussian(v, sigma=p.gauss_sigma)) for k, v in bases.items()}
    names, methods = list(BASE_IMAGES), list(THRESHOLD_METHODS)
    shape = (len(names), len(methods))
    n_rois, area, activity = np.zeros(shape, int), np.zeros(shape), np.zeros(shape)
    grid, thresholds = {}, {}
    for bi, b in enumerate(names):
        for mi, m in enumerate(methods):
            th = float(THRESHOLD_METHODS[m](segs[b]))
            labels = segment_watershed(segs[b], th, p)
            n = int(labels.max())
            n_rois[bi, mi] = n
            if n:
                area[bi, mi] = float((labels > 0).sum())
                activity[bi, mi] = float(ndimage.mean(std, labels, np.arange(1, n + 1)).mean())
            grid[(b, m)], thresholds[(b, m)] = labels, th
    w_n, w_a, w_act = p.score_weights
    score = w_n * _n01(n_rois.astype(float)) + w_a * _n01(area) + w_act * _n01(activity)
    bi, mi = np.unravel_index(score.argmax(), shape)
    best = (names[bi], methods[mi])
    return SweepResult(grid[best], best, thresholds[best], names, methods,
                       n_rois, area, activity, score, grid)
