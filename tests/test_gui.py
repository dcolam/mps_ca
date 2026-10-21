"""End-to-end GUI test on synthetic data (runs offscreen; needs napari + pytest-qt)."""
import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from mpsca.gui._main import MainWidget  # noqa: E402

WAIT = 120_000


def _wait(qtbot, tab):
    qtbot.waitUntil(lambda: not tab.runner.running, timeout=WAIT)
    assert not tab.status.text().startswith("Failed"), tab.status.text()


@pytest.fixture
def widget(make_napari_viewer, qtbot, tmp_path):
    from qtpy.QtCore import QSettings

    # keep the remembered folders of the tests out of the user's real settings
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    viewer = make_napari_viewer()
    w = MainWidget(viewer)
    qtbot.addWidget(w)
    return w


def test_import_scan_mismatch_and_convert(widget, qtbot, mixed_resolution_experiment, tmp_path, monkeypatch):
    tab = widget.import_tab
    asked = []
    monkeypatch.setattr(tab, "_ask_resolution", lambda: asked.append(True))
    tab.load_folder(mixed_resolution_experiment["exp"])
    assert tab.files.rowCount() == 2
    tab.scan()
    _wait(qtbot, tab)
    assert asked, "the user must be asked about differing frame sizes"
    assert tab.block_factors() == {0: 2, 1: 1}
    assert tab.form.widget("source_fps").value() == pytest.approx(4.0)
    # user decides: 4x4 and 2x2 instead
    tab._factor_boxes[0].setValue(4)
    tab._factor_boxes[1].setValue(2)
    tab.ed_out.setText(str(tmp_path))
    tab.convert(overwrite=True)
    _wait(qtbot, tab)
    store = widget.session.store
    assert store is not None and store.movie.shape[1:] == (5, 6)
    assert "movie" in widget.viewer.layers


def test_full_processing_chain(widget, qtbot, synthetic_store, tmp_path):
    s = widget.session
    s.open_store(synthetic_store["store"].path)
    assert widget.preprocess_tab.plot.canvas is not None

    pre = widget.preprocess_tab
    pre.sp_thresh.setValue(0.5)
    pre.apply_threshold()
    pre.compute_projections()
    _wait(qtbot, pre)
    assert "corr" in widget.viewer.layers

    seg = widget.segment_tab
    seg.cb_method.setCurrentIndex(1)  # classic: no model download in this test
    seg.run()
    _wait(qtbot, seg)
    layer = s.roi_layer()
    assert layer is not None and layer.data.max() >= 8
    # manual edit: erase one ROI, save
    n_before = int(layer.data.max())
    data = layer.data.copy()
    data[data == 1] = 0
    layer.data = data
    seg.save()
    assert int(s.roi_layer().data.max()) == n_before - 1

    tr = widget.traces_tab
    tr.estimate()
    assert 90 < tr.form.widget("offset").value() < 110
    tr.run()
    _wait(qtbot, tr)
    assert tr.traces["dff"].shape[0] == n_before - 1
    tr.show_cell(2)
    assert tr.trace.canvas is not None

    an = widget.analysis_tab
    an.run()
    _wait(qtbot, an)
    assert an.result is not None and "cells" in an.summary.text()
    out = tmp_path / "results"
    an.export(str(out))
    _wait(qtbot, an)
    assert (out / "dff.csv").exists() and (out / "05_bursts.png").exists()

    sm = widget.summary_tab
    sm.add(s.store.path)
    sm.compare()
    assert sm.df is not None and len(sm.df) == 1
