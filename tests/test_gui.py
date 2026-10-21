"""End-to-end GUI tests on synthetic data (run offscreen; need napari + pytest-qt)."""
import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")

from qtpy.QtCore import Qt  # noqa: E402

from mpsca.gui._main import MainWidget  # noqa: E402
from mpsca.project import Project, accept_proposals  # noqa: E402

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


def test_project_convert_test_and_run_all(widget, qtbot, two_recordings, monkeypatch):
    d = two_recordings
    tab = widget.project_tab
    asked = []

    def confirm(reports):  # the dialog, answered with "Use proposals"
        from mpsca.project import needs_decision
        pending = {n: r for n, r in reports.items() if needs_decision(tab.project, tab.project.get(n), r)}
        asked.extend(pending)
        accept_proposals(tab.project, pending)
        return True

    monkeypatch.setattr(tab, "_confirm", confirm)
    tab.ed_input.setText(str(d["root"]))
    tab.ed_output.setText(str(d["out"]))
    tab.find()
    proj = widget.session.project
    assert [r.name for r in proj.recordings] == ["R1", "R2"]
    assert tab.recs.rowCount() == 2

    # describe the recordings: a new column, edited in the table
    monkeypatch.setattr("mpsca.gui._project.QInputDialog.getText", lambda *a, **k: ("group", True))
    tab._add_column()
    tab.recs.item(0, 4).setText("control")
    tab.recs.item(1, 4).setText("treated")
    assert proj.get("R2").description == {"group": "treated"}

    # convert all: scans first, asks once, converts every recording
    tab.convert()
    qtbot.waitUntil(lambda: not tab.runner.running and "converted" in tab.status.text(), timeout=WAIT)
    assert asked == ["R2"]
    assert {proj.status(r) for r in proj.recordings} == {"converted"}
    tab.recs.selectRow(1)
    assert tab.blocks.rowCount() == 2 and tab.block_factors() == {0: 2, 1: 2}

    # test the settings on R1
    tab.recs.selectRow(0)
    tab.open_selected()
    assert widget.session.store.meta["name"] == "R1"
    pre = widget.preprocess_tab
    pre.sp_thresh.setValue(0.4)
    pre.compute_projections()
    _wait(qtbot, pre)
    seg = widget.segment_tab
    seg.cb_method.setCurrentIndex(1)  # classic: no model download in this test
    seg.cl_form.widget("min_cell_area").setValue(15)
    seg.run()
    _wait(qtbot, seg)
    an = widget.analysis_tab
    an.form.widget("robust_noise").setChecked(True)
    widget.traces_tab.run()
    _wait(qtbot, widget.traces_tab)
    an.run()
    _wait(qtbot, an)
    # the tested values are the project settings, saved with the project
    saved = Project.load(d["out"]).settings
    assert saved.bad_frame_thresh == pytest.approx(0.4)
    assert saved.seg_method == "classic" and saved.classic.min_cell_area == 15

    # run all with exactly these settings
    batch = widget.batch_tab
    widget.tabs.setCurrentIndex(5)
    assert batch.list.count() == 2 and batch.ticked() == ["R1", "R2"]
    assert "robust_noise: yes" in batch.settings_view.toPlainText()
    batch.run()
    _wait(qtbot, batch)
    assert {proj.status(r) for r in proj.recordings} == {"analysed"}
    assert any("R2" in batch.log.item(i).text() for i in range(batch.log.count()))

    sm = widget.summary_tab
    sm.add_project()
    sm.compare()
    assert len(sm.df) == 2 and set(sm.df["group"]) == {"control", "treated"}
    sm.cb_group.setCurrentText("group")
    assert sm.p_main.canvas is not None

    # reopening the project restores recordings, descriptions and settings
    widget.session.project = None
    tab.open_project(d["out"])
    assert widget.session.project.get("R1").description == {"group": "control"}
    assert widget.segment_tab.cb_method.currentIndex() == 1
    assert widget.preprocess_tab.sp_thresh.value() == pytest.approx(0.4)


def test_file_edits_and_scan_notes(widget, qtbot, two_recordings, monkeypatch):
    d = two_recordings
    tab = widget.project_tab
    monkeypatch.setattr(tab, "_confirm", lambda reports: False)
    tab.ed_input.setText(str(d["root"]))
    tab.ed_output.setText(str(d["out"]))
    tab.find()
    tab.recs.selectRow(0)
    assert tab.files.rowCount() == 2
    tab.files.item(1, 2).setText("high glucose")
    assert widget.session.project.get("R1").conditions() == {0: "1_1", 1: "high glucose"}
    tab.recs.item(1, 0).setCheckState(Qt.Unchecked)
    assert not widget.session.project.get("R2").include
    tab.scan()
    _wait(qtbot, tab)
    assert set(tab.reports) == {"R1"}
    tab.recs.selectRow(0)
    assert tab.issues.count() >= 1


def test_single_recording_without_project(widget, qtbot, synthetic_store, tmp_path):
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
    seg.cb_method.setCurrentIndex(1)
    seg.run()
    _wait(qtbot, seg)
    layer = s.roi_layer()
    assert layer is not None and layer.data.max() >= 8
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
