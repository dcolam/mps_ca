"""Step 4: fluorescence traces and ΔF/F; click an ROI to see its trace."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                            QSpinBox, QVBoxLayout, QWidget)

from .. import pipeline
from ..analysis import epochs_from_frames
from ..analysis import figures as figs
from ..traces import DffParams, estimate_offset
from ._common import FigurePanel, ParamForm, Runner, hint
from ._session import Session

DFF_SPECS = {
    "baseline_window_s": {"label": "Baseline window (s)", "min": 1, "max": 3600, "step": 5},
    "percentile": {"label": "Baseline percentile", "max": 100, "step": 1},
    "offset": {"label": "Camera offset (counts)", "min": -1, "max": 65535, "step": 1, "decimals": 1,
               "optional": True,
               "tooltip": "Dark level subtracted before ΔF/F (auto = low percentile of the mean "
                          "image). These cameras sit near 100 counts; without subtracting it "
                          "ΔF/F is strongly compressed. 0 = no subtraction, as in the notebook"},
    "gap_factor": {"label": "Pause = gap > n frames", "min": 1, "max": 1000},
}


class TracesTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.traces = None
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Extract the mean intensity of every saved ROI in every frame and compute "
                           "ΔF/F against a rolling-percentile baseline (separately for each "
                           "recording block). Click an ROI in the viewer to plot its trace."))
        self.form = ParamForm(DffParams, DFF_SPECS)
        lay.addWidget(self.form)
        row = QHBoxLayout()
        self.btn_offset = QPushButton("Estimate offset")
        self.btn_run = QPushButton("Extract traces")
        row.addWidget(self.btn_offset)
        row.addWidget(self.btn_run)
        lay.addLayout(row)
        self.heat = FigurePanel(placeholder="No traces yet.")
        lay.addWidget(self.heat)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Cell"))
        self.sp_cell = QSpinBox()
        self.sp_cell.setRange(1, 1)
        crow.addWidget(self.sp_cell)
        crow.addStretch()
        lay.addLayout(crow)
        self.trace = FigurePanel()
        lay.addWidget(self.trace)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addStretch()
        self.runner = Runner(self, self.bar, self.status)

        self.btn_offset.clicked.connect(self.estimate)
        self.btn_run.clicked.connect(self.run)
        self.sp_cell.valueChanged.connect(self.show_cell)
        session.traces_changed.connect(self.load)
        session.store_changed.connect(self._restore_params)
        session.viewer.mouse_drag_callbacks.append(self._on_click)

    def _restore_params(self):
        p = self.session.store.params("traces")
        if p:
            self.form.set_value(p)  # shows the offset actually used last time

    def estimate(self):
        proj = pipeline.load_projections(self.session.store) if self.session.store else {}
        if "mean" not in proj:
            self.status.setText("Compute the projections first.")
            return
        self.form.widget("offset").setValue(round(estimate_offset(proj["mean"]), 1))

    def _labels_unsaved(self) -> bool:
        layer, act = self.session.roi_layer(), pipeline.active_labels(self.session.store)
        if layer is None:
            return False
        return act is None or layer.data.shape != act[1].shape or not np.array_equal(layer.data, act[1])

    def run(self):
        store = self.session.store
        if store is None:
            return
        if self._labels_unsaved():
            ans = QMessageBox.question(self, "Unsaved ROIs",
                                       "The ROIs layer has changes that are not saved. Save them first?")
            if ans == QMessageBox.Yes:
                from ..segment import filter_area
                labels = filter_area(self.session.roi_layer().data, 1)
                pipeline.save_labels(store, labels, "edited", method="edited")
                self.session.show_labels(labels)
        self.runner.run(pipeline.run_traces, store, self.form.value(), done=lambda _: self.load(),
                        message="Extracting traces…", lock=[self.btn_run])

    def load(self):
        store = self.session.store
        self.traces = pipeline.load_traces(store)
        if self.traces is None:
            return
        frames = store.read_frames()
        self.epochs = epochs_from_frames(frames, pipeline.fps_of(store))
        dff = self.traces["dff"]
        self.heat.set_figure(figs.dff_heatmap(dff, self.traces["time_s"], self.epochs))
        self.sp_cell.setRange(1, max(1, dff.shape[0]))
        self.status.setText(f"{dff.shape[0]} cells × {dff.shape[1]} frames. Next: Analysis.")
        self.show_cell(self.sp_cell.value())

    def show_cell(self, cell: int):
        if self.traces is None or not (1 <= cell <= self.traces["dff"].shape[0]):
            return
        i = cell - 1
        tr = self.traces
        offset = self.session.store.params("traces").get("offset") or 0.0
        self.trace.set_figure(figs.cell_trace(tr["time_s"], tr["F"][i], tr["dff"][i], cell,
                                              self.epochs, tr["F0"][i] + offset))

    def _on_click(self, viewer, event):
        layer = self.session.roi_layer()
        if layer is None or self.traces is None or event.type != "mouse_press":
            return
        value = layer.get_value(event.position, world=True)
        if value:
            self.sp_cell.setValue(int(value))
