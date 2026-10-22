"""Processing steps that read from and write to an experiment store.

Each step stores its result plus the parameters used, so the GUI and the CLI share
one implementation and every result can be traced back to its settings.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from skimage import measure

from .analysis import AnalysisParams, AnalysisResult, analyse
from .io.store import ExperimentStore
from .preprocess import compute_projections, detect_low_signal, parse_frames, segmentation_frames
from .segment import ClassicParams, SweepResult, classic_sweep
from .segment.cellpose_seg import CellposeParams
from .traces import DffParams, compute_dff, estimate_offset, extract_traces

log = logging.getLogger(__name__)
Progress = Callable[[float], None] | None
PROJECTIONS = ("mean", "max", "std", "corr")


def fps_of(store: ExperimentStore) -> float:
    return float(store.meta.get("fps") or store.params("convert").get("target_fps", 1.0))


# ---- bad frames & projections ----------------------------------------------
def set_bad_frames(store: ExperimentStore, thresh: float) -> tuple[np.ndarray, float]:
    frames = store.read_frames()
    is_bad, cutoff = detect_low_signal(frames["frame_mean"].to_numpy(), thresh)
    store.update_frames_column("is_bad", is_bad)
    store.set_params("bad_frames", {"low_signal_thresh": thresh, "cutoff": cutoff})
    return is_bad, cutoff


def run_projections(store: ExperimentStore, frames: str = "",
                    progress: Progress = None) -> dict[str, np.ndarray]:
    """Projections over the frames in ``frames`` (e.g. ``"0-299"``), else the 'Segment.' frames."""
    idx = segmentation_frames(store.read_frames(), frames)
    proj = compute_projections(store.movie, idx, progress=progress)
    for name, img in proj.items():
        store.save_array(f"projections/{name}", img, n_frames=int(len(idx)))
    store.set_params("projections", {"frames": frames, "n_frames": int(len(idx))})
    store.log(f"projections over {len(idx)} frames" + (f" ({frames})" if frames else ""))
    return proj


def load_projections(store: ExperimentStore) -> dict[str, np.ndarray]:
    return {n: store.load_array(f"projections/{n}") for n in PROJECTIONS
            if store.has(f"projections/{n}")}


# ---- labels -----------------------------------------------------------------
def save_labels(store: ExperimentStore, labels: np.ndarray, name: str, **info) -> None:
    """Store a label image under ``labels/<name>`` and make it the active one."""
    store.save_array(f"labels/{name}", np.asarray(labels, dtype=np.int32), **info)
    store.group.attrs["active_labels"] = name
    store.log(f"labels '{name}': {int(np.max(labels))} ROIs")


def active_labels(store: ExperimentStore) -> tuple[str, np.ndarray] | None:
    name = store.group.attrs.get("active_labels")
    if name and store.has(f"labels/{name}"):
        return name, store.load_array(f"labels/{name}")
    return None


def run_classic(store: ExperimentStore, params: ClassicParams | None = None) -> SweepResult:
    p = params or ClassicParams()
    proj = load_projections(store)
    sweep = classic_sweep(proj["mean"], proj["std"], proj["max"], p)
    save_labels(store, sweep.labels, "classic", method="classic", best=list(sweep.best),
                threshold=sweep.threshold, params=asdict(p))
    store.set_params("segment", {"method": "classic", **asdict(p)})
    return sweep


def run_cellpose(store: ExperimentStore, channels: list[str], params=None) -> np.ndarray:
    from .segment.cellpose_seg import build_input, run_cellpose as _run

    p = params or CellposeParams()
    labels = _run(build_input(load_projections(store), channels), p)
    save_labels(store, labels, "cellpose", method="cellpose", channels=channels, params=asdict(p))
    store.set_params("segment", {"method": "cellpose", "channels": channels, **asdict(p)})
    return labels


def run_cellpose_image(store: ExperimentStore, image: np.ndarray, source: str,
                       params: CellposeParams | None = None) -> np.ndarray:
    """Cellpose on any image of the movie's frame size (e.g. one frame, or a layer in the viewer)."""
    from .segment.cellpose_seg import run_cellpose as _run

    if image.shape != store.movie.shape[1:]:
        raise ValueError(f"the image is {image.shape}, but ROIs must match the movie frames "
                         f"{store.movie.shape[1:]}")
    p = params or CellposeParams()
    labels = _run(np.asarray(image, dtype=np.float32), p)
    save_labels(store, labels, "cellpose", method="cellpose", source=source, params=asdict(p))
    store.set_params("segment", {"method": "cellpose", "source": source, **asdict(p)})
    return labels


def frame_windows(n_frames: int, size: int, frames: str = "") -> list[tuple[int, int]]:
    """Consecutive windows of ``size`` frames, (first, last) inclusive, within ``frames``."""
    idx = parse_frames(frames, n_frames)
    idx = np.arange(n_frames) if idx is None else idx
    out = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        out += [(int(run[i]), int(run[min(i + size, len(run)) - 1])) for i in range(0, len(run), size)]
    return out


def run_cellpose_windows(store: ExperimentStore, size: int, params: CellposeParams | None = None,
                         frames: str = "", progress: Progress = None,
                         should_stop: Callable[[], bool] | None = None) -> int:
    """Cellpose on the movie: every ``size`` frames are averaged (1 = every single frame).

    Results go to ``windows/labels`` (one label image per window, ROI numbers are NOT
    matched between windows) with the frame range of each window in ``windows/first`` and
    ``windows/last``. Bad frames are left out of the averages. Returns the number of
    windows segmented (fewer than planned when stopped).
    """
    from .segment.cellpose_seg import run_cellpose as _run

    p = params or CellposeParams()
    movie = store.movie
    wins = frame_windows(movie.shape[0], max(1, int(size)), frames)
    if not wins:
        raise ValueError("no frames selected")
    bad = store.read_frames()["is_bad"].to_numpy()
    first, last = np.array([w[0] for w in wins]), np.array([w[1] for w in wins])
    store.save_array("windows/first", first)
    store.save_array("windows/last", last)
    arr = store.create_array("windows/labels", (len(wins), *movie.shape[1:]), (1, *movie.shape[1:]),
                             np.int32, size=int(size), frames=frames, params=asdict(p), done=0)
    n_rois = np.zeros(len(wins), dtype=np.int32)
    done = 0
    for k, (a, b) in enumerate(wins):
        if should_stop is not None and should_stop():
            break
        idx = np.arange(a, b + 1)
        idx = idx[~bad[idx]]
        if len(idx):
            img = np.asarray(movie[idx].mean(axis=0), dtype=np.float32)
            labels = _run(img, p)
            arr[k] = labels
            n_rois[k] = labels.max()
        done = k + 1
        if progress:
            progress(done / len(wins))
    arr.attrs["done"] = done
    store.save_array("windows/n_rois", n_rois)
    store.log(f"cellpose on {done}/{len(wins)} windows of {size} frame(s)")
    return done


def load_windows(store: ExperimentStore):
    """The window segmentation as a lazy (T, Y, X) stack aligned with the movie, or None.

    Frame t shows the labels of the window that contains it (empty outside the windows).
    """
    import dask.array as da

    if not store.has("windows/labels"):
        return None
    lab = store.lazy_array("windows/labels")
    first, last = store.load_array("windows/first"), store.load_array("windows/last")
    n = store.movie.shape[0]
    which = np.full(n, len(first))  # index of the empty image appended below
    for k, (a, b) in enumerate(zip(first, last)):
        which[a:b + 1] = k
    lab = da.concatenate([lab, da.zeros((1, *lab.shape[1:]), dtype=lab.dtype, chunks=lab.chunksize)])
    return lab[which]


# ---- traces -----------------------------------------------------------------
def run_traces(store: ExperimentStore, params: DffParams | None = None,
               progress: Progress = None) -> dict[str, np.ndarray]:
    p = DffParams(**asdict(params or DffParams()))
    act = active_labels(store)
    if act is None:
        raise ValueError("no ROIs yet: run a segmentation or save edited labels first")
    if p.offset is None:
        proj = load_projections(store)
        p.offset = round(estimate_offset(proj["mean"]), 1) if "mean" in proj else 0.0
    name, labels = act
    frames = store.read_frames()
    F = extract_traces(store.movie, labels, progress=progress)
    time_s = frames["time_s"].to_numpy()
    dff, F0 = compute_dff(F, time_s, fps_of(store), frames["is_bad"].to_numpy(), p)
    props = measure.regionprops(labels)
    centroids = np.array([rp.centroid for rp in props]).reshape(-1, 2)
    areas = np.array([rp.area for rp in props])
    for key, arr in {"F": F, "F0": F0, "dff": dff, "centroids": centroids, "areas": areas,
                     "time_s": time_s}.items():
        store.save_array(f"traces/{key}", arr, labels=name)
    store.set_params("traces", {"labels": name, **asdict(p)})
    store.log(f"traces for {F.shape[0]} ROIs from labels '{name}'")
    return {"F": F, "F0": F0, "dff": dff, "centroids": centroids, "time_s": time_s}


def load_traces(store: ExperimentStore) -> dict[str, np.ndarray] | None:
    if not store.has("traces/dff"):
        return None
    return {k: store.load_array(f"traces/{k}") for k in ("F", "F0", "dff", "centroids", "areas", "time_s")}


# ---- analysis & export ---------------------------------------------------------
def run_analysis(store: ExperimentStore, params: AnalysisParams | None = None) -> AnalysisResult:
    tr = load_traces(store)
    if tr is None:
        raise ValueError("no traces yet: extract traces first")
    res = analyse(tr["dff"], tr["time_s"], tr["centroids"], store.read_frames(), fps_of(store), params)
    store.group.attrs["analysis_summary"] = json.loads(json.dumps(res.summary, default=float))
    store.set_params("analysis", asdict(res.params))
    store.log(f"analysis: {res.summary['n_bursts']} bursts, mean r={res.summary['mean_corr']}")
    return res


def default_output_dir(store: ExperimentStore) -> Path:
    p = store.path
    return p.with_name(p.name.replace(".mpsca.zarr", "") + "_results")


def export_results(store: ExperimentStore, out_dir: str | Path | None = None,
                   res: AnalysisResult | None = None, figures: bool = True) -> Path:
    """Write CSV tables (and PNG figures) next to the store."""
    from .analysis import figures as figs

    out = Path(out_dir) if out_dir else default_output_dir(store)
    out.mkdir(parents=True, exist_ok=True)
    frames = store.read_frames()
    frames.to_csv(out / "frames.csv", index_label="frame")
    tr = load_traces(store)
    if tr is not None:
        cols = [f"cell_{i + 1}" for i in range(tr["dff"].shape[0])]
        for key in ("F", "dff"):
            df = pd.DataFrame(tr[key].T, columns=cols)
            df.insert(0, "time_s", tr["time_s"])
            df.to_csv(out / f"{key}.csv", index=False)
        pd.DataFrame({"cell_id": np.arange(1, len(tr["areas"]) + 1),
                      "y_px": tr["centroids"][:, 0].round(2), "x_px": tr["centroids"][:, 1].round(2),
                      "area_px": tr["areas"]}).to_csv(out / "cells.csv", index=False)
    if res is not None:
        pd.DataFrame(res.bursts).to_csv(out / "bursts.csv", index=False)
        (out / "summary.json").write_text(json.dumps(
            {"experiment": store.meta, **res.summary}, indent=2, default=float))
    if figures:
        proj = load_projections(store)
        act = active_labels(store)
        if res is not None and tr is not None and "mean" in proj and act is not None:
            for name, fig in {
                "01_overview": figs.overview(proj["mean"], act[1], store.meta.get("name", "")),
                "02_dff": figs.dff_heatmap(tr["dff"], tr["time_s"], res.epochs, res.order),
                "03_correlation": figs.correlation(res),
                "04_proximity": figs.proximity(res),
                "05_bursts": figs.bursts(res, tr["time_s"]),
                "06_pca": figs.pca(res, proj["mean"], tr["centroids"]),
            }.items():
                fig.savefig(out / f"{name}.png", dpi=120)
    store.log(f"exported results to {out}")
    return out


# ---- settings shared by the GUI, the batch run and the CLI -----------------------
STEPS = ("bad_frames", "projections", "segment", "traces", "analysis", "export")
STEP_LABELS = {"bad_frames": "Bad frames", "projections": "Projections", "segment": "Segmentation",
               "traces": "Traces", "analysis": "Analysis", "export": "Export"}


def _build(cls, values: dict | None):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (values or {}).items() if k in names})


@dataclass
class PipelineSettings:
    """Everything the processing steps need; tested on a few recordings, then run on all."""
    bad_frame_thresh: float = 0.5
    proj_frames: str = ""                 # e.g. "0-299, 1200-"; empty = frames ticked 'Segment.'
    seg_method: str = "cellpose"          # "cellpose" or "classic"
    seg_channels: list[str] = field(default_factory=lambda: ["mean", "corr"])
    keep_edited_rois: bool = True         # never replace ROIs that were corrected by hand
    cellpose: CellposeParams = field(default_factory=CellposeParams)
    classic: ClassicParams = field(default_factory=ClassicParams)
    traces: DffParams = field(default_factory=DffParams)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PipelineSettings":
        d = dict(d or {})
        sub = {"cellpose": CellposeParams, "classic": ClassicParams, "traces": DffParams,
               "analysis": AnalysisParams}
        out = _build(cls, {k: v for k, v in d.items() if k not in sub})
        for k, c in sub.items():
            setattr(out, k, _build(c, d.get(k)))
        return out


def run_steps(store: ExperimentStore, settings: PipelineSettings, steps=STEPS,
              out_dir: str | Path | None = None, progress: Progress = None,
              say: Callable[[str], None] | None = None) -> dict:
    """Run the chosen steps in order on one recording; returns a short summary."""
    say = say or log.info
    todo = [s for s in STEPS if s in steps]
    res, out = None, {}
    for i, step in enumerate(todo):
        sub = (lambda f, i=i: progress((i + f) / len(todo))) if progress else None
        if step == "bad_frames":
            is_bad, _ = set_bad_frames(store, settings.bad_frame_thresh)
            out["bad_frames"] = int(is_bad.sum())
        elif step == "projections":
            run_projections(store, settings.proj_frames, progress=sub)
        elif step == "segment":
            act = active_labels(store)
            if settings.keep_edited_rois and act is not None and act[0] == "edited":
                say(f"keeping {int(act[1].max())} hand-edited ROIs")
                out["rois"] = int(act[1].max())
            elif settings.seg_method == "cellpose":
                out["rois"] = int(run_cellpose(store, settings.seg_channels, settings.cellpose).max())
            else:
                out["rois"] = int(run_classic(store, settings.classic).labels.max())
        elif step == "traces":
            run_traces(store, settings.traces, progress=sub)
        elif step == "analysis":
            res = run_analysis(store, settings.analysis)
            out["bursts"] = res.summary["n_bursts"]
        elif step == "export":
            if res is None and store.has("traces/dff"):
                res = run_analysis(store, settings.analysis)
            out["export"] = str(export_results(store, out_dir, res))
        say(f"{STEP_LABELS[step]} done")
        if progress:
            progress((i + 1) / len(todo))
    return out
