"""Clean-up of label images."""
from __future__ import annotations

import numpy as np
from skimage.segmentation import relabel_sequential


def filter_area(labels: np.ndarray, min_area: int = 0, max_area: int | None = None) -> np.ndarray:
    """Remove ROIs outside ``[min_area, max_area]`` pixels and renumber them 1..n."""
    labels = np.asarray(labels)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    if max_area:
        keep &= areas <= max_area
    keep[0] = False
    out = np.where(keep[labels], labels, 0)
    return relabel_sequential(out)[0].astype(np.int32)
