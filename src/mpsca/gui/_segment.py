"""Step 3: segmentation (Cellpose or the classic sweep) and manual correction."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QProgressBar,
                            QPushButton, QSizePolicy, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from .. import pipeline
from ..analysis import figures as figs
from ..segment import ClassicParams, filter_area
from ..segment import cellpose_seg
from ._common import FigurePanel, ParamForm, Runner, hint
from ._session import Session

CELLPOSE_SPECS = {
    "model": {"label": "Model", "tooltip": "cpsam_v2 (built in) or the path of a fine-tuned model"},
    "diameter": {"label": "Cell diameter (px)", "optional": True, "decimals": 1, "max": 500,
                 "tooltip": "auto = let the model decide. Use 'Measure' to take it from a line you draw"},
    "flow_threshold": {"label": "Flow threshold", "max": 3, "step": 0.05,
                       "tooltip": "Higher keeps more (less regular) shapes"},
    "cellprob_threshold": {"label": "Cell probability threshold", "min": -6, "max": 6, "step": 0.25,
                           "tooltip": "Lower finds more / larger cells"},
    "min_area": {"label": "Min area (px)", "max": 100000},
    "max_area": {"label": "Max area (px)", "optional": True, "decimals": 0, "max": 1e6, "float": True},
    "gpu": {"skip": True},
}
CLASSIC_SPECS = {
    "gauss_sigma": {"label": "Smoothing σ (px)", "max": 20},
    "min_peak_distance": {"label": "Min peak distance (px)", "min": 1, "max": 200},
    "min_cell_area": {"label": "Min area (px)", "max": 100000},
    "max_cell_area": {"label": "Max area (px)", "max": 1000000},
}
CHANNELS = ("mean", "corr", "std", "max")


class SegmentTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._sweep = None
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Find the cells on the projection images. Cellpose usually copes far "
                           "better with noisy recordings; the classic threshold sweep is the "
                           "method of the original notebook. Afterwards, fix mistakes directly "
                           "in the 'ROIs' layer (paint, erase, fill, pick a new label with M) "
                           "and press 'Save ROIs'."))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["Cellpose", "Classic threshold sweep"])
        lay.addWidget(self.cb_method)

        self.stack = QStackedWidget()
        # -- cellpose page
        cp = QWidget()
        cl = QVBoxLayout(cp)
        cl.setContentsMargins(0, 0, 0, 0)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Input:"))
        self.ch_boxes = {}
        for name in CHANNELS:
            box = QCheckBox(name)
            box.setChecked(name in ("mean", "corr"))
            self.ch_boxes[name] = box
            crow.addWidget(box)
        cl.addLayout(crow)
        self.cp_form = ParamForm(cellpose_seg.CellposeParams, CELLPOSE_SPECS)
        cl.addWidget(self.cp_form)
        mrow = QHBoxLayout()
        self.btn_measure = QPushButton("Measure diameter from a drawn line")
        self.cb_gpu = QCheckBox("Use GPU")
        mrow.addWidget(self.btn_measure)
        mrow.addWidget(self.cb_gpu)
        cl.addLayout(mrow)
        self.stack.addWidget(cp)
        # -- classic page
        cs = QWidget()
        sl = QVBoxLayout(cs)
        sl.setContentsMargins(0, 0, 0, 0)
        self.cl_form = ParamForm(ClassicParams, CLASSIC_SPECS)
        sl.addWidget(self.cl_form)
        self.btn_grid = QPushButton("Show sweep grid")
        self.btn_grid.setEnabled(False)
        sl.addWidget(self.btn_grid)
        self.stack.addWidget(cs)
        self.stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay.addWidget(self.stack)

        self.btn_run = QPushButton("Run segmentation")
        lay.addWidget(self.btn_run)

        edit = QGroupBox("Correct and save")
        el = QVBoxLayout(edit)
        frow = QHBoxLayout()
        self.sp_min = QSpinBox()
        self.sp_min.setRange(0, 100000)
        self.sp_min.setValue(10)
        self.btn_clean = QPushButton("Remove ROIs smaller than")
        frow.addWidget(self.btn_clean)
        frow.addWidget(self.sp_min)
        frow.addWidget(QLabel("px"))
        el.addLayout(frow)
        self.btn_save = QPushButton("Save ROIs")
        el.addWidget(self.btn_save)
        lay.addWidget(edit)

        self.count = QLabel("")
        lay.addWidget(self.count)
        self.grid = FigurePanel()
        self.grid.setVisible(False)
        lay.addWidget(self.grid)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addStretch()
        self.runner = Runner(self, self.bar, self.status)

        if not cellpose_seg.available():
            self.cb_method.setCurrentIndex(1)
            self.cb_method.model().item(0).setEnabled(False)
            self.cb_method.setToolTip("Cellpose is not installed: pip install mpsca[cellpose]")
        self.stack.setCurrentIndex(self.cb_method.currentIndex())
        self.cb_gpu.setChecked(cellpose_seg.available() and cellpose_seg.gpu_available())
        self.cb_gpu.setEnabled(self.cb_gpu.isChecked())

        self.cb_method.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.btn_run.clicked.connect(self.run)
        self.btn_measure.clicked.connect(self.measure_diameter)
        self.btn_clean.clicked.connect(self.clean)
        self.btn_save.clicked.connect(self.save)
        self.btn_grid.clicked.connect(self._toggle_grid)
        session.labels_changed.connect(self._count)

    def read_settings(self, s):
        self.cb_method.setCurrentIndex(1 if s.seg_method == "classic" or not cellpose_seg.available() else 0)
        for name, box in self.ch_boxes.items():
            box.setChecked(name in s.seg_channels)
        self.cp_form.set_value(s.cellpose)
        if self.cb_gpu.isEnabled() and s.cellpose.gpu is not None:
            self.cb_gpu.setChecked(bool(s.cellpose.gpu))
        self.cl_form.set_value(s.classic)

    def write_settings(self, s):
        s.seg_method = "cellpose" if self.cb_method.currentIndex() == 0 else "classic"
        s.seg_channels = self.channels()
        s.cellpose = self.cp_form.value()
        s.cellpose.gpu = self.cb_gpu.isChecked()
        s.classic = self.cl_form.value()

    def _count(self):
        layer = self.session.roi_layer()
        self.count.setText(f"{int(layer.data.max())} ROIs" if layer is not None else "")

    def channels(self) -> list[str]:
        return [n for n, b in self.ch_boxes.items() if b.isChecked()]

    def run(self):
        store = self.session.store
        if store is None:
            return
        if not pipeline.load_projections(store):
            self.status.setText("Compute the projections first (Preprocess).")
            return
        lock = [self.btn_run, self.btn_save, self.btn_clean]
        self.session.pull_settings()
        if self.stack.currentIndex() == 0:
            p = self.cp_form.value()
            p.gpu = self.cb_gpu.isChecked()
            ch = self.channels()
            if not ch:
                self.status.setText("Tick at least one input image.")
                return
            self.runner.run(pipeline.run_cellpose, store, ch, p, with_progress=False,
                            done=self._got_labels, lock=lock,
                            message="Running Cellpose (first run downloads the model)…")
        else:
            self.runner.run(pipeline.run_classic, store, self.cl_form.value(), with_progress=False,
                            done=self._got_sweep, lock=lock, message="Running the threshold sweep…")

    def _got_labels(self, labels):
        self.session.show_labels(labels)

    def _got_sweep(self, sweep):
        self._sweep = sweep
        self.btn_grid.setEnabled(True)
        self.session.show_labels(sweep.labels)
        self.status.setText(f"Best: {sweep.best[0]} + {sweep.best[1]} ({int(sweep.labels.max())} ROIs)")

    def _toggle_grid(self):
        if self._sweep is None:
            return
        if not self.grid.isVisible():
            bg = pipeline.load_projections(self.session.store)["mean"]
            self.grid.set_figure(figs.sweep_grid(self._sweep, bg))
        self.grid.setVisible(not self.grid.isVisible())

    def measure_diameter(self):
        shapes = [l for l in self.session.viewer.layers if l._type_string == "shapes"]
        lines = [d for l in shapes for d, t in zip(l.data, l.shape_type) if t == "line"]
        if not lines:
            layer = self.session.viewer.add_shapes(name="diameter", edge_color="cyan", edge_width=1)
            layer.mode = "add_line"
            self.status.setText("Draw a line across a typical cell in the 'diameter' layer, "
                                "then press Measure again.")
            return
        d = float(np.mean([np.linalg.norm(np.asarray(l)[-1, -2:] - np.asarray(l)[0, -2:]) for l in lines]))
        self.cp_form.widget("diameter").setValue(d)
        self.status.setText(f"Diameter set to {d:.1f} px (mean of {len(lines)} line(s)).")

    def clean(self):
        layer = self.session.roi_layer()
        if layer is not None:
            self.session.show_labels(filter_area(layer.data, self.sp_min.value()))

    def save(self):
        layer, store = self.session.roi_layer(), self.session.store
        if layer is None or store is None:
            return
        labels = filter_area(layer.data, 1)  # renumber 1..n after manual edits
        pipeline.save_labels(store, labels, "edited", method="edited")
        self.session.show_labels(labels)
        self.status.setText(f"Saved {int(labels.max())} ROIs. Next: Traces.")
