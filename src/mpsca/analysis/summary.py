"""Cross-experiment summary from the analysis results stored in each experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..io.store import ExperimentStore

GENOTYPE_COLOURS = {"WT": "steelblue", "KO": "coral"}
METRICS = [("n_cells", "Cells"), ("mean_corr", "Mean pairwise r"),
           ("burst_freq_hz", "Burst frequency (Hz)"), ("mean_dur_s", "Burst duration (s)"),
           ("mean_ibi_s", "Inter-burst interval (s)"), ("mean_recruit", "Cells per burst")]


def collect(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        store = ExperimentStore.open(p, mode="r")
        s = dict(store.group.attrs.get("analysis_summary") or {})
        if not s:
            continue
        meta = store.meta
        row = {"experiment": meta.get("name", Path(p).name), "genotype": meta.get("genotype", "unknown"),
               "path": str(p)}
        row.update({k: s.get(k) for k, _ in METRICS})
        for label, n in (s.get("epoch_bursts") or {}).items():
            row[f"bursts: {label}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 6.5), layout="constrained")
    axes = fig.subplots(2, 3).ravel()
    groups = sorted(df["genotype"].unique())
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, METRICS):
        for gi, g in enumerate(groups):
            vals = pd.to_numeric(df.loc[df["genotype"] == g, key], errors="coerce").dropna()
            ax.scatter(gi + rng.uniform(-0.12, 0.12, len(vals)), vals, s=45, edgecolors="k",
                       linewidths=0.5, color=GENOTYPE_COLOURS.get(g, "gray"), zorder=3)
            if len(vals):
                ax.plot([gi - 0.25, gi + 0.25], [vals.mean()] * 2, color="k", lw=2)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_title(title, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Experiments by genotype (dots = experiments, bar = mean)", fontsize=10)
    return fig


def epoch_figure(df: pd.DataFrame) -> Figure:
    cols = [c for c in df.columns if c.startswith("bursts: ")]
    fig = Figure(figsize=(max(6, 2.2 * len(cols)), max(3, 0.5 * len(df) + 1.5)), layout="constrained")
    ax = fig.add_subplot()
    if not cols:
        return fig
    hm = df[cols].fillna(0).to_numpy(float)
    im = ax.imshow(hm, aspect="auto", cmap="YlOrRd", vmin=0)
    fig.colorbar(im, ax=ax, label="Bursts")
    ax.set_xticks(range(len(cols)), [c[8:] for c in cols], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)), df["experiment"], fontsize=7)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, int(hm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Bursts per condition", fontsize=10)
    return fig
