"""Step 5: population analysis and export. Plus the cross-experiment summary."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QMessageBox,
                            QProgressBar, QPushButton, QTabWidget, QVBoxLayout, QWidget)

from .. import pipeline
from ..analysis import AnalysisParams
from ..analysis import figures as figs
from ..analysis import summary
from ._common import FigurePanel, ParamForm, Runner, hint
from ._session import Session, settings

ANALYSIS_SPECS = {
    "corr_threshold": {"label": "Co-activity r threshold", "max": 1, "step": 0.05},
    "dist_threshold_px": {"label": "Neighbour distance (px)", "max": 10000, "step": 5},
    "spike_rms_factor": {"label": "Spike threshold (× RMS)", "max": 100, "step": 0.5},
    "burst_rms_factor": {"label": "Burst threshold (× RMS)", "max": 100, "step": 0.1},
    "rms_window_s": {"label": "RMS window (s)", "min": 1, "max": 3600, "step": 5},
    "burst_min_duration_s": {"label": "Min burst duration (s)", "max": 600, "step": 0.5},
    "n_pcs": {"label": "PCA components", "min": 2, "max": 100},
    "robust_noise": {"label": "Robust noise (MAD) for spikes",
                     "tooltip": "Off: threshold = factor × rolling RMS of ΔF/F, as in the original "
                                "notebook (very active cells raise their own threshold). On: "
                                "factor × noise from the median absolute deviation."},
}


def _fmt(v, unit=""):
    return "–" if v is None or v != v else f"{v}{unit}"


class AnalysisTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = None
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Correlation between cells, proximity vs co-activity, spikes and "
                           "population bursts per condition, and PCA. 'Export' writes CSV tables "
                           "and PNG figures next to the converted data."))
        self.form = ParamForm(AnalysisParams, ANALYSIS_SPECS)
        lay.addWidget(self.form)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run analysis")
        self.btn_export = QPushButton("Export results…")
        self.btn_export.setEnabled(False)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_export)
        lay.addLayout(row)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.panels = {}
        for name in ("Bursts", "Correlation", "Proximity", "PCA", "Overview"):
            self.panels[name] = FigurePanel()
            self.tabs.addTab(self.panels[name], name)
        lay.addWidget(self.tabs)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        self.runner = Runner(self, self.bar, self.status)
        self.btn_run.clicked.connect(self.run)
        self.btn_export.clicked.connect(self.export)
        session.store_changed.connect(self._restore_params)

    def read_settings(self, s):
        self.form.set_value(s.analysis)

    def write_settings(self, s):
        s.analysis = self.form.value()

    def _restore_params(self):
        self.result = None
        self.btn_export.setEnabled(False)
        p = self.session.store.params("analysis")
        if p and self.session.project is None:
            self.form.set_value(p)

    def run(self):
        if self.session.store is None:
            return
        self.session.pull_settings()
        self.runner.run(pipeline.run_analysis, self.session.store, self.form.value(), with_progress=False,
                        done=self._show, message="Analysing…", lock=[self.btn_run])

    def _show(self, res):
        self.result = res
        store = self.session.store
        tr = pipeline.load_traces(store)
        proj = pipeline.load_projections(store)
        labels = pipeline.active_labels(store)[1]
        self.panels["Bursts"].set_figure(figs.bursts(res, tr["time_s"]))
        self.panels["Correlation"].set_figure(figs.correlation(res))
        self.panels["Proximity"].set_figure(figs.proximity(res))
        self.panels["PCA"].set_figure(figs.pca(res, proj["mean"], tr["centroids"]))
        self.panels["Overview"].set_figure(figs.overview(proj["mean"], labels, self.session.name))
        s = res.summary
        per_epoch = ", ".join(f"{k}: {v}" for k, v in s["epoch_bursts"].items())
        self.summary.setText(f"{s['n_cells']} cells · mean r {_fmt(s['mean_corr'])} · {s['n_bursts']} bursts "
                             f"({s['burst_freq_hz']} Hz) · mean duration {_fmt(s['mean_dur_s'], ' s')} · "
                             f"mean IBI {_fmt(s['mean_ibi_s'], ' s')}\nBursts per condition: {per_epoch}")
        self.btn_export.setEnabled(True)

    def export(self, out_dir: str | None = None):
        store = self.session.store
        if store is None or self.result is None:
            return
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(self, "Export to",
                                                       str(pipeline.default_output_dir(store).parent))
            if not out_dir:
                return
            out_dir = Path(out_dir) / pipeline.default_output_dir(store).name
        self.runner.run(pipeline.export_results, store, out_dir, self.result, with_progress=False,
                        done=lambda p: self.status.setText(f"Exported to {p}"),
                        message="Exporting…", lock=[self.btn_export])


class SummaryTab(QWidget):
    """Compare analysed recordings, grouped by any description field."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.df = None
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Compare analysed recordings. 'Project recordings' adds every analysed "
                           "recording of the open project; recordings can also be added one by one "
                           "(.mpsca.zarr folders). Group them by any description field."))
        self.list = QListWidget()
        self.list.setMaximumHeight(130)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        b_proj = QPushButton("Project recordings")
        b_add = QPushButton("Add…")
        b_rm = QPushButton("Remove")
        for b in (b_proj, b_add, b_rm):
            row.addWidget(b)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Group by"))
        self.cb_group = QComboBox()
        self.cb_group.setMinimumWidth(140)
        row2.addWidget(self.cb_group)
        self.btn_make = QPushButton("Compare")
        self.btn_csv = QPushButton("Save table…")
        row2.addWidget(self.btn_make)
        row2.addWidget(self.btn_csv)
        row2.addStretch()
        lay.addLayout(row2)
        self.tabs = QTabWidget()
        self.p_main, self.p_epoch = FigurePanel(), FigurePanel()
        self.tabs.addTab(self.p_main, "By group")
        self.tabs.addTab(self.p_epoch, "Bursts per condition")
        lay.addWidget(self.tabs)
        self.status = QLabel("")
        lay.addWidget(self.status)
        b_proj.clicked.connect(self.add_project)
        b_add.clicked.connect(self._add)
        b_rm.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        self.btn_make.clicked.connect(self.compare)
        self.btn_csv.clicked.connect(self._save_csv)
        self.cb_group.currentIndexChanged.connect(self._redraw)

    def add(self, path):
        path = str(path)
        if path not in [self.list.item(i).text() for i in range(self.list.count())]:
            self.list.addItem(path)

    def add_project(self):
        proj = self.session.project
        if proj is None:
            self.status.setText("Open or create a project first.")
            return
        for rec in proj.select():
            if proj.status(rec) == "analysed":
                self.add(proj.store_path(rec))
        self.status.setText(f"{self.list.count()} analysed recordings.")

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Add recording (.mpsca.zarr)", settings().value("out_dir", ""))
        if d:
            self.add(d)

    def compare(self):
        paths = [self.list.item(i).text() for i in range(self.list.count())]
        proj = self.session.project
        desc = {str(proj.store_path(r)): r.description for r in proj.recordings} if proj else None
        self.df = summary.collect(paths, desc)
        if self.df.empty:
            self.status.setText("None of these recordings has been analysed yet.")
            return
        current = self.cb_group.currentText()
        self.cb_group.blockSignals(True)
        self.cb_group.clear()
        self.cb_group.addItems(["(none)", *summary.description_columns(self.df)])
        if current:
            self.cb_group.setCurrentText(current)
        self.cb_group.blockSignals(False)
        self._redraw()
        self.status.setText(f"{len(self.df)} recordings")

    def _redraw(self):
        if self.df is None or self.df.empty:
            return
        group = self.cb_group.currentText()
        self.p_main.set_figure(summary.figure(self.df, None if group in ("", "(none)") else group))
        self.p_epoch.set_figure(summary.epoch_figure(self.df))

    def _save_csv(self):
        if self.df is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save summary", "summary.csv", "CSV (*.csv)")
        if f:
            self.df.to_csv(f, index=False)
