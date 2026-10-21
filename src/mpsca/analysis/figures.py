"""Figures as ``matplotlib.figure.Figure`` objects: embeddable in Qt or saved to PNG.

No pyplot state is used, so these are safe to build from worker threads.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from .core import AnalysisResult


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def shade_epochs(axes, epochs, alpha=0.15, labels=True):
    for ep in epochs:
        for ax in axes:
            ax.axvspan(ep["start_s"], ep["end_s"], color=ep["color"], alpha=alpha, lw=0, zorder=0)
    if labels and axes:
        top, seen = axes[0], set()
        for ep in epochs:
            if ep["label"] in seen and ep is not epochs[0]:
                continue
            seen.add(ep["label"])
            top.text((ep["start_s"] + ep["end_s"]) / 2, 1.01, ep["label"], ha="center", va="bottom",
                     fontsize=6.5, transform=top.get_xaxis_transform(), clip_on=False)


def gapped(t, *ys, factor=3.0):
    """Insert NaNs at pauses so lines are not drawn across them."""
    t = np.asarray(t, float)
    if len(t) < 3:
        return (t, *ys)
    dt = np.median(np.diff(t))
    cut = np.flatnonzero(np.diff(t) > factor * dt) + 1
    if not len(cut):
        return (t, *ys)
    out = [np.insert(t, cut, np.nan)]
    for y in ys:
        out.append(np.insert(np.asarray(y, float), cut, np.nan, axis=-1))
    return tuple(out)


def _pct(img, lo=1, hi=99):
    return np.percentile(img, [lo, hi])


def frame_means(time_s, means, is_bad, cutoff=None, epochs=()):
    fig = Figure(figsize=(9, 2.6), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(*gapped(time_s, means), lw=0.8, color="steelblue", label="frame mean")
    if cutoff is not None:
        ax.axhline(cutoff, color="crimson", ls="--", lw=1, label="bad-frame cutoff")
    if np.any(is_bad):
        ax.scatter(time_s[is_bad], means[is_bad], color="crimson", s=12, zorder=5,
                   label=f"bad frames ({int(np.sum(is_bad))})")
    shade_epochs([ax], list(epochs))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean intensity"); ax.legend(fontsize=7)
    _clean(ax)
    return fig


def overview(mean_img, labels, title=""):
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    lo, hi = _pct(mean_img)
    a1.imshow(mean_img, cmap="gray", vmin=lo, vmax=hi); a1.set_title("Mean image"); a1.axis("off")
    norm = np.clip((mean_img - lo) / (hi - lo + 1e-9), 0, 1)
    a2.imshow(np.clip(label2rgb(labels, image=norm, bg_label=0, alpha=0.35), 0, 1))
    edges = find_boundaries(labels)
    a2.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", interpolation="nearest")
    a2.set_title(f"ROIs (n={int(labels.max())})"); a2.axis("off")
    fig.suptitle(title, fontsize=9)
    return fig


def dff_heatmap(dff, time_s, epochs=(), order=None):
    """Heatmap against frame number (pauses between blocks are not drawn)."""
    fig = Figure(figsize=(10, 4), layout="constrained")
    ax = fig.add_subplot()
    data = dff if order is None else dff[order]
    v = np.percentile(np.abs(data), 99) if data.size else 1
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                   extent=[-0.5, data.shape[1] - 0.5, data.shape[0] + 0.5, 0.5])
    fig.colorbar(im, ax=ax, label="ΔF/F", shrink=0.8)
    for ep in epochs:
        ax.axvline(ep["first_frame"] - 0.5, color="k", lw=0.6, alpha=0.6)
        ax.text((ep["first_frame"] + ep["last_frame"]) / 2, 1.01, ep["label"], ha="center",
                va="bottom", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Frame (pauses between blocks not shown)"); ax.set_ylabel("Cell #")
    return fig


def cell_trace(time_s, F, dff, cell_id, epochs=(), F0=None):
    fig = Figure(figsize=(9, 3.2), layout="constrained")
    a1, a2 = fig.subplots(2, 1, sharex=True)
    a1.plot(*gapped(time_s, F), lw=0.7, color="0.3", label="F")
    if F0 is not None:
        a1.plot(*gapped(time_s, F0), lw=0.8, color="crimson", label="F0")
    a1.legend(fontsize=7, loc="upper right"); a1.set_ylabel("F")
    a2.plot(*gapped(time_s, dff), lw=0.7, color="steelblue"); a2.set_ylabel("ΔF/F")
    a2.set_xlabel("Time (s)")
    shade_epochs([a1, a2], list(epochs))
    fig.suptitle(f"Cell {cell_id}", fontsize=9, x=0.02, ha="left")
    _clean(a1); _clean(a2)
    return fig


def correlation(res: AnalysisResult):
    fig = Figure(figsize=(10, 4.2), layout="constrained")
    for ax, mat, title in zip(fig.subplots(1, 2), [res.corr, res.corr[np.ix_(res.order, res.order)]],
                              ["Correlation", "Correlation (clustered)"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("Cell #"); ax.set_ylabel("Cell #")
    fig.colorbar(im, ax=fig.axes, label="Pearson r", shrink=0.8)
    return fig


def proximity(res: AnalysisResult):
    p = res.params
    near, co = res.pair_dist < p.dist_threshold_px, res.pair_corr > p.corr_threshold
    colours = np.where(near & co, "crimson", np.where(near, "orange", np.where(co, "steelblue", "lightgray")))
    fig = Figure(figsize=(7, 4.2), layout="constrained")
    ax = fig.add_subplot()
    ax.scatter(res.pair_dist, res.pair_corr, c=colours, s=8, alpha=0.6, linewidths=0)
    ax.axhline(p.corr_threshold, color="steelblue", ls="--", lw=1)
    ax.axvline(p.dist_threshold_px, color="orange", ls="--", lw=1)
    ax.set_xlabel("Distance (px)"); ax.set_ylabel("Pearson r"); ax.set_title("Proximity vs co-activity")
    _clean(ax)
    return fig


def bursts(res: AnalysisResult, time_s):
    fig = Figure(figsize=(11, 6), layout="constrained")
    ax_r, ax_p = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    n = res.active.shape[0]
    cells, frames = np.nonzero(res.active)
    ax_r.scatter(time_s[frames], cells, s=1.5, c="k", linewidths=0)
    for b in res.bursts:
        for ax in (ax_r, ax_p):
            ax.axvspan(b["t_start"], b["t_end"], color="salmon", alpha=0.3, lw=0)
    ax_r.set_ylim(n - 0.5, -0.5); ax_r.set_ylabel("Cell #")
    noise = "MAD noise" if res.params.robust_noise else "RMS"
    ax_r.set_title(f"Spikes (> {res.params.spike_rms_factor}× {noise}), {len(res.bursts)} bursts")
    t, pop, thr = gapped(time_s, res.pop_activity, res.pop_threshold)
    ax_p.fill_between(t, np.nan_to_num(pop), where=np.isfinite(pop), alpha=0.5, color="steelblue",
                      label="fraction active")
    ax_p.plot(t, thr, color="crimson", ls="--", lw=1, label=f"{res.params.burst_rms_factor}× RMS")
    ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("Frac. active"); ax_p.legend(fontsize=7)
    shade_epochs([ax_r, ax_p], res.epochs)
    _clean(ax_r); _clean(ax_p)
    return fig


def pca(res: AnalysisResult, mean_img, centroids):
    fig = Figure(figsize=(13, 4.2), layout="constrained")
    a0, a1, a2 = fig.subplots(1, 3)
    if res.pca_time is None:
        a1.text(0.5, 0.5, "too few cells for PCA", ha="center", transform=a1.transAxes)
        return fig
    for ep in res.epochs:
        sl = slice(ep["first_frame"], ep["last_frame"] + 1)
        a0.scatter(res.pca_time[sl, 0], res.pca_time[sl, 1], color=ep["color"], s=6, label=ep["label"])
    burst_t = np.zeros(len(res.pop_activity), bool)
    for b in res.bursts:
        burst_t[b["frame_start"]:b["frame_end"] + 1] = True
    if burst_t.any():
        a0.scatter(res.pca_time[burst_t, 0], res.pca_time[burst_t, 1], c="crimson", s=12, label="burst")
    a0.legend(fontsize=6, markerscale=2); a0.set_xlabel("PC1"); a0.set_ylabel("PC2")
    a0.set_title("Time-resolved PCA")
    sc = a1.scatter(res.pca_cells[:, 0], res.pca_cells[:, 1], c=res.burst_participation, cmap="YlOrRd",
                    s=40, edgecolors="k", linewidths=0.4)
    fig.colorbar(sc, ax=a1, label="Burst participation", shrink=0.8)
    a1.set_xlabel(f"PC1 ({res.pca_cells_var[0]:.1f}%)"); a1.set_ylabel(f"PC2 ({res.pca_cells_var[1]:.1f}%)")
    a1.set_title("Cell-resolved PCA")
    lo, hi = _pct(mean_img)
    a2.imshow(mean_img, cmap="gray", vmin=lo, vmax=hi)
    v = np.abs(res.pca_cells[:, 0]).max() or 1
    sc2 = a2.scatter(centroids[:, 1], centroids[:, 0], c=res.pca_cells[:, 0], cmap="coolwarm",
                     vmin=-v, vmax=v, s=30, edgecolors="k", linewidths=0.4)
    fig.colorbar(sc2, ax=a2, label="PC1 loading", shrink=0.8)
    a2.set_title("Spatial map, PC1"); a2.axis("off")
    return fig


def sweep_grid(sweep, background):
    nb, nm = len(sweep.bases), len(sweep.methods)
    fig = Figure(figsize=(nm * 2.6, nb * 2.6), layout="constrained")
    axes = fig.subplots(nb, nm)
    lo, hi = _pct(background)
    norm = np.clip((background - lo) / (hi - lo + 1e-9), 0, 1)
    for bi, b in enumerate(sweep.bases):
        for mi, m in enumerate(sweep.methods):
            ax = axes[bi, mi]
            ax.imshow(np.clip(label2rgb(sweep.grid[(b, m)], image=norm, bg_label=0, alpha=0.4), 0, 1))
            best = (b, m) == sweep.best
            ax.set_title(f"{m} n={sweep.n_rois[bi, mi]} sc={sweep.score[bi, mi]:.2f}", fontsize=7,
                         color="darkgoldenrod" if best else "k", fontweight="bold" if best else "normal")
            if mi == 0:
                ax.set_ylabel(b, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("gold" if best else "0.7"); sp.set_linewidth(3 if best else 0.5)
    return fig
