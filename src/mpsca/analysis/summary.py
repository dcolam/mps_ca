"""Cross-experiment summary from the analysis results stored in each experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..io.store import ExperimentStore

PALETTE = ["steelblue", "coral", "seagreen", "goldenrod", "orchid", "slategray", "sienna", "teal"]
METRICS = [("n_cells", "Cells"), ("mean_corr", "Mean pairwise r"),
           ("burst_freq_hz", "Burst frequency (Hz)"), ("mean_dur_s", "Burst duration (s)"),
           ("mean_ibi_s", "Inter-burst interval (s)"), ("mean_recruit", "Cells per burst")]


def collect(paths, descriptions: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per analysed recording: its description fields, then the metrics.

    ``descriptions`` (path -> fields) overrides the description saved in the store,
    e.g. with fields edited in the project after conversion.
    """
    rows = []
    for p in paths:
        store = ExperimentStore.open(p, mode="r")
        s = dict(store.group.attrs.get("analysis_summary") or {})
        if not s:
            continue
        meta = store.meta
        row = {"recording": meta.get("name", Path(p).name)}
        desc = (descriptions or {}).get(str(p))
        row.update({str(k): v for k, v in (desc if desc is not None else meta.get("description") or {}).items()})
        row["path"] = str(p)
        row.update({k: s.get(k) for k, _ in METRICS})
        for label, n in (s.get("epoch_bursts") or {}).items():
            row[f"bursts: {label}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def description_columns(df: pd.DataFrame) -> list[str]:
    metric = {k for k, _ in METRICS} | {"recording", "path"}
    return [c for c in df.columns if c not in metric and not c.startswith("bursts: ")]


def figure(df: pd.DataFrame, group_by: str | None = None) -> Figure:
    fig = Figure(figsize=(12, 6.5), layout="constrained")
    axes = fig.subplots(2, 3).ravel()
    key = df[group_by].fillna("–").astype(str) if group_by in df else pd.Series(["all"] * len(df), index=df.index)
    groups = sorted(key.unique())
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, METRICS):
        for gi, g in enumerate(groups):
            vals = pd.to_numeric(df.loc[key == g, metric], errors="coerce").dropna()
            ax.scatter(gi + rng.uniform(-0.12, 0.12, len(vals)), vals, s=45, edgecolors="k",
                       linewidths=0.5, color=PALETTE[gi % len(PALETTE)], zorder=3)
            if len(vals):
                ax.plot([gi - 0.25, gi + 0.25], [vals.mean()] * 2, color="k", lw=2)
        ax.set_xticks(range(len(groups)), groups, rotation=20 if len(groups) > 3 else 0, fontsize=8)
        ax.set_title(title, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    by = f"by {group_by}" if group_by in df else "all recordings"
    fig.suptitle(f"Recordings {by} (dots = recordings, bar = mean)", fontsize=10)
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
    ax.set_yticks(range(len(df)), df["recording"], fontsize=7)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, int(hm[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Bursts per condition", fontsize=10)
    return fig
