import numpy as np
import pytest

from mpsca import pipeline
from mpsca.analysis import epochs_from_frames
from mpsca.segment import filter_area
from mpsca.traces import DffParams, compute_dff, segments


def _matched(labels, truth, min_iou=0.3):
    """Number of true cells overlapped by some ROI with IoU above ``min_iou``."""
    hits = 0
    for t in range(1, truth.max() + 1):
        cell = truth == t
        ids = np.unique(labels[cell])
        ious = [(cell & (labels == i)).sum() / (cell | (labels == i)).sum() for i in ids if i]
        hits += max(ious, default=0) > min_iou
    return hits


def test_bad_frames_projections_classic_traces(synthetic_store):
    store, truth, gt = synthetic_store["store"], synthetic_store["truth"], synthetic_store["traces"]
    is_bad, _ = pipeline.set_bad_frames(store, 0.5)
    assert np.flatnonzero(is_bad).tolist() == [100]

    proj = pipeline.run_projections(store)
    assert store.load_array("projections/corr").shape == truth.shape
    # cells are far more correlated with their neighbours than background is
    assert proj["corr"][truth > 0].mean() > 3 * abs(proj["corr"][truth == 0].mean())

    sweep = pipeline.run_classic(store)
    assert _matched(sweep.labels, truth) >= 10
    assert pipeline.active_labels(store)[0] == "classic"

    tr = pipeline.run_traces(store, DffParams(offset=100.0))
    assert tr["F"].shape == (int(sweep.labels.max()), 240)
    # every ROI that overlaps a true cell follows that cell's trace
    for i in range(1, sweep.labels.max() + 1):
        cell = np.bincount(truth[sweep.labels == i]).argmax()
        if cell:
            good = ~is_bad
            r = np.corrcoef(tr["F"][i - 1][good], gt[cell - 1][good])[0, 1]
            assert r > 0.9


def test_traces_need_labels(synthetic_store):
    with pytest.raises(ValueError, match="no ROIs"):
        pipeline.run_traces(synthetic_store["store"])


def test_traces_use_saved_edits(synthetic_store):
    store, truth = synthetic_store["store"], synthetic_store["truth"]
    pipeline.save_labels(store, truth, "edited")
    tr = pipeline.run_traces(store)
    assert tr["F"].shape[0] == truth.max()
    assert store.params("traces")["labels"] == "edited"


def test_dff_respects_pauses_and_offset():
    t = np.r_[np.arange(100.0), 1000 + np.arange(100.0)]
    F = np.r_[np.full(100, 110.0), np.full(100, 150.0)][None, :]
    assert [(s.start, s.stop) for s in segments(t, 1.0)] == [(0, 100), (100, 200)]
    dff, F0 = compute_dff(F, t, 1.0, params=DffParams(offset=100.0, baseline_window_s=30))
    # each block gets its own baseline: no artefact at the block boundary
    np.testing.assert_allclose(dff, 0, atol=1e-6)
    np.testing.assert_allclose(F0[0, :100], 10)


def test_filter_area_relabels():
    lab = np.zeros((10, 10), int)
    lab[0, 0] = 5                 # area 1 -> removed
    lab[2:5, 2:5] = 7             # area 9 -> kept
    out = filter_area(lab, min_area=2)
    assert out.max() == 1 and (out == 1).sum() == 9


def test_analysis_and_export(synthetic_store, tmp_path):
    store = synthetic_store["store"]
    pipeline.save_labels(store, synthetic_store["truth"], "edited")
    pipeline.run_projections(store)
    pipeline.run_traces(store, DffParams(offset=100.0))
    res = pipeline.run_analysis(store)
    assert res.summary["n_cells"] == 12
    assert [e["label"] for e in res.epochs] == ["2mM Glucose", "20mM Glucose"]
    assert res.epochs[1]["start_s"] == 720.0  # real time, pause kept
    assert all(b["frame_end"] < 120 or b["frame_start"] >= 120 for b in res.bursts)
    out = pipeline.export_results(store, tmp_path / "res", res)
    for name in ("dff.csv", "cells.csv", "bursts.csv", "summary.json", "05_bursts.png"):
        assert (out / name).exists(), name


def test_epochs_from_frames(synthetic_store):
    frames = synthetic_store["store"].read_frames()
    eps = epochs_from_frames(frames, 1.0)
    assert [(e["first_frame"], e["last_frame"]) for e in eps] == [(0, 119), (120, 239)]


def test_robust_noise_finds_spikes_in_active_cells(synthetic_store):
    from mpsca.analysis import AnalysisParams

    store, gt = synthetic_store["store"], synthetic_store["traces"]
    pipeline.save_labels(store, synthetic_store["truth"], "edited")
    pipeline.run_traces(store, DffParams(offset=100.0))
    res = pipeline.run_analysis(store, AnalysisParams(robust_noise=True))
    # detected spikes sit on the true transients
    hits = res.active & (gt > 1.0)
    assert res.active.sum() > 50 and hits.sum() / res.active.sum() > 0.9


def test_cli_process_classic(synthetic_store, tmp_path, capsys):
    from mpsca.cli import main

    out = tmp_path / "results"
    assert main(["process", str(synthetic_store["store"].path), "--method", "classic",
                 "--out", str(out)]) == 0
    assert (out / "summary.json").exists() and (out / "01_overview.png").exists()
    assert "ROIs" in capsys.readouterr().out


def test_offset_auto_is_estimated_and_recorded(synthetic_store):
    store = synthetic_store["store"]
    pipeline.save_labels(store, synthetic_store["truth"], "edited")
    pipeline.run_projections(store)
    pipeline.run_traces(store)  # offset=None -> estimated
    assert 95 < store.params("traces")["offset"] < 105
    pipeline.run_traces(store, DffParams(offset=0.0))  # explicit 0 is kept (notebook behaviour)
    assert store.params("traces")["offset"] == 0.0
