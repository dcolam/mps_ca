"""Population analysis of ΔF/F traces, ported from ``ca_analysis_batch.ipynb``.

Fixed relative to the notebook: epochs come from the per-frame table (real block
lengths and times), bad frames never count as active, and bursts cannot span a pause
between blocks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import label as nd_label
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform

from ..traces.dff import segments

EPOCH_PALETTE = ["#AED6F1", "#A9DFBF", "#FAE6B0", "#F9C3C0", "#D7BDE2", "#FDEBD0", "#D5DBDB"]


@dataclass
class AnalysisParams:
    corr_threshold: float = 0.4
    dist_threshold_px: float = 60.0
    spike_rms_factor: float = 5.0
    burst_rms_factor: float = 1.2
    rms_window_s: float = 60.0
    burst_min_duration_s: float = 0.5
    n_pcs: int = 10
    # False: noise = rolling RMS of ΔF/F (as in the notebook; active cells raise their own
    # threshold). True: noise = 1.4826 × median absolute deviation, unaffected by transients.
    robust_noise: bool = False


@dataclass
class AnalysisResult:
    params: AnalysisParams
    epochs: list[dict]
    corr: np.ndarray
    order: list[int]
    mean_corr: float
    pair_dist: np.ndarray
    pair_corr: np.ndarray
    active: np.ndarray            # (n_cells, T) bool
    pop_activity: np.ndarray      # (T,)
    pop_threshold: np.ndarray     # (T,)
    bursts: list[dict]
    epoch_bursts: dict[str, int]
    pca_time: np.ndarray | None = None
    pca_cells: np.ndarray | None = None
    pca_cells_var: np.ndarray | None = None
    burst_participation: np.ndarray | None = None
    summary: dict = field(default_factory=dict)


def epochs_from_frames(frames: pd.DataFrame, fps: float) -> list[dict]:
    """Contiguous runs of the same (application, phase), with start/end times and colours."""
    t = frames["time_s"].to_numpy()
    keys = list(zip(frames["application"].astype(str), frames["phase"].astype(str)))
    out, colours = [], {}
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            app, phase = keys[start]
            colours.setdefault(app, EPOCH_PALETTE[len(colours) % len(EPOCH_PALETTE)])
            out.append({"application": app, "phase": phase,
                        "label": f"{app} ({phase})" if phase else app,
                        "start_s": float(t[start]), "end_s": float(t[i - 1] + 1 / fps),
                        "first_frame": start, "last_frame": i - 1, "color": colours[app]})
            start = i
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)


def analyse(dff: np.ndarray, time_s: np.ndarray, centroids: np.ndarray, frames: pd.DataFrame,
            fps: float, params: AnalysisParams | None = None) -> AnalysisResult:
    p = params or AnalysisParams()
    n, T = dff.shape
    bad = frames["is_bad"].to_numpy() if "is_bad" in frames else np.zeros(T, bool)
    epochs = epochs_from_frames(frames, fps)

    # ---- correlation & clustering order ----------------------------------
    good = ~bad
    corr = np.corrcoef(dff[:, good]) if n > 1 else np.ones((n, n))
    corr = np.nan_to_num(corr)
    if n > 2:
        d = np.clip(1 - corr, 0, None)
        np.fill_diagonal(d, 0)
        order = dendrogram(linkage(squareform((d + d.T) / 2, checks=False), "ward"),
                           no_plot=True)["leaves"]
    else:
        order = list(range(n))
    iu = np.triu_indices(n, k=1)
    mean_corr = float(corr[iu].mean()) if n > 1 else float("nan")
    pair_dist = squareform(pdist(centroids))[iu] if n > 1 else np.zeros(0)
    pair_corr = corr[iu]

    # ---- spikes (rolling RMS) & bursts -----------------------------------
    win = max(3, int(p.rms_window_s * fps))
    if p.robust_noise:
        g = dff[:, good]
        med = np.median(g, axis=1, keepdims=True)
        noise = 1.4826 * np.median(np.abs(g - med), axis=1, keepdims=True)
        active = (dff - med > p.spike_rms_factor * noise) & good[None, :]
    else:
        rms = np.sqrt(uniform_filter1d(dff ** 2, size=win, axis=1, mode="nearest"))
        active = (dff > p.spike_rms_factor * rms) & good[None, :]
    pop = active.mean(axis=0) if n else np.zeros(T)
    pop_thr = p.burst_rms_factor * np.sqrt(uniform_filter1d(pop ** 2, size=win, mode="nearest"))
    min_frames = max(1, int(p.burst_min_duration_s * fps))
    bursts = []
    for seg in segments(time_s, fps):
        lab, nb = nd_label((pop > pop_thr)[seg])
        for b in range(1, nb + 1):
            fr = np.flatnonzero(lab == b) + seg.start
            if len(fr) < min_frames:
                continue
            bursts.append({
                "frame_start": int(fr[0]), "frame_end": int(fr[-1]),
                "t_start": float(time_s[fr[0]]), "t_end": float(time_s[fr[-1]]),
                "duration_s": float(time_s[fr[-1]] - time_s[fr[0]] + 1 / fps),
                "peak_frac": float(pop[fr].max()),
                "n_recruits": int(active[:, fr].any(axis=1).sum()),
            })
    epoch_bursts: dict[str, int] = {}
    for ep in epochs:
        cnt = sum(1 for b in bursts if ep["first_frame"] <= b["frame_start"] <= ep["last_frame"])
        epoch_bursts[ep["label"]] = epoch_bursts.get(ep["label"], 0) + cnt

    res = AnalysisResult(p, epochs, corr, order, mean_corr, pair_dist, pair_corr, active,
                         pop, pop_thr, bursts, epoch_bursts)

    # ---- PCA (time- and cell-resolved) -----------------------------------
    k = min(p.n_pcs, n - 1, T - 1)
    if k >= 2:
        from sklearn.decomposition import PCA
        Z = _zscore(dff)
        res.pca_time = PCA(n_components=k).fit_transform(Z.T)
        pca = PCA(n_components=k)
        res.pca_cells = pca.fit_transform(Z)
        res.pca_cells_var = pca.explained_variance_ratio_ * 100
        part = np.zeros(n)
        for b in bursts:
            part += active[:, b["frame_start"]:b["frame_end"] + 1].mean(axis=1)
        res.burst_participation = part

    duration = float(time_s[-1] - time_s[0] + 1 / fps) if T else 0.0
    durs = [b["duration_s"] for b in bursts]
    ibi = [b2["t_start"] - b1["t_end"] for b1, b2 in zip(bursts, bursts[1:])
           if b2["frame_start"] > b1["frame_end"]]
    res.summary = {
        "n_cells": n, "mean_corr": round(mean_corr, 4), "n_bursts": len(bursts),
        "burst_freq_hz": round(len(bursts) / duration, 5) if duration else 0.0,
        "mean_dur_s": round(float(np.mean(durs)), 3) if durs else float("nan"),
        "mean_ibi_s": round(float(np.mean(ibi)), 3) if ibi else float("nan"),
        "mean_recruit": round(float(np.mean([b["n_recruits"] for b in bursts])), 1) if bursts else 0.0,
        "epoch_bursts": epoch_bursts, "params": asdict(p),
    }
    return res
