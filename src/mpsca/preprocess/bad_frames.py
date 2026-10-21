"""Bad-frame detection. Bad frames are masked in the frame table, never deleted."""
from __future__ import annotations

import numpy as np


def detect_low_signal(frame_means: np.ndarray, thresh: float = 0.5) -> tuple[np.ndarray, float]:
    """Flag frames whose mean intensity is below ``thresh`` × the median frame mean.

    Returns ``(is_bad, threshold_value)``.
    """
    frame_means = np.asarray(frame_means, dtype=np.float64)
    if frame_means.size == 0:
        return np.zeros(0, dtype=bool), 0.0
    cutoff = float(np.median(frame_means) * thresh)
    return frame_means < cutoff, cutoff
