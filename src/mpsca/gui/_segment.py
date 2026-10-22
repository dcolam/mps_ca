"""Step 3: segmentation (Cellpose or the classic sweep) and manual correction."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                            QProgressBar, QPushButton, QSizePolicy, QSpinBox, QStackedWidget,
                            QVBoxLayout, QWidget)

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
SOURCES = ("Projections", "Selected image layer (movie: the current frame)",
           "Movie, window by window")


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
                           "and press 'Save ROIs'. Cellpose can also run on any image layer, on one movie "
                           "frame, or on the movie window by window (averages of N frames) to see "
                           "how the cells change; Run all always uses the projections."))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["Cellpose", "Classic threshold sweep"])
        lay.addWidget(self.cb_method)

        self.stack = QStackedWidget()
        # -- cellpose page
        cp = QWidget()
        cl = QVBoxLayout(cp)
        cl.setContentsMargins(0, 0, 0, 0)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Input:"))
        self.cb_source = QComboBox()
        self.cb_source.addItems(SOURCES)
        self.cb_source.setToolTip("Run all always segments the projections")
        srow.addWidget(self.cb_source, 1)
        cl.addLayout(srow)
        # projections: ticked images are combined into one multi-channel input
        self.ch_row = QWidget()
        crow = QHBoxLayout(self.ch_row)
        crow.setContentsMargins(0, 0, 0, 0)
        self.ch_boxes = {}
        for name in CHANNELS:
            box = QCheckBox(name)
            box.setChecked(name in ("mean", "corr"))
            box.setToolTip("Ticked images are combined into ONE input with up to 3 channels "
                           "(like the colours of an RGB image); there is one result")
            box.toggled.connect(lambda on, b=box: self._limit_channels(b, on))
            self.ch_boxes[name] = box
            crow.addWidget(box)
        cl.addWidget(self.ch_row)
        # selected layer
        self.layer_info = QLabel("")
        self.layer_info.setWordWrap(True)
        cl.addWidget(self.layer_info)
        # movie, window by window
        self.win_row = QWidget()
        wrow = QHBoxLayout(self.win_row)
        wrow.setContentsMargins(0, 0, 0, 0)
        wrow.addWidget(QLabel("Average every"))
        self.sp_window = QSpinBox()
        self.sp_window.setRange(1, 100000)
        self.sp_window.setValue(20)
        self.sp_window.setToolTip("1 = every single frame (slow without a GPU)")
        wrow.addWidget(self.sp_window)
        wrow.addWidget(QLabel("frames;  frames"))
        self.ed_win_frames = QLineEdit()
        self.ed_win_frames.setPlaceholderText("all   e.g. 0-599")
        wrow.addWidget(self.ed_win_frames, 1)
        cl.addWidget(self.win_row)
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

        rrow = QHBoxLayout()
        self.btn_run = QPushButton("Run segmentation")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        rrow.addWidget(self.btn_run, 1)
        rrow.addWidget(self.btn_stop)
        lay.addLayout(rrow)
        self.btn_take = QPushButton("Use the ROIs of the current window")
        self.btn_take.setToolTip("Copy the window segmentation at the time slider's frame "
                                 "to the ROIs layer (then correct and save it)")
        lay.addWidget(self.btn_take)

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

        self._last_image = None
        self.cb_method.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.cb_method.currentIndexChanged.connect(self._update_source)
        self.cb_source.currentIndexChanged.connect(self._update_source)
        self.btn_run.clicked.connect(self.run)
        self.btn_stop.clicked.connect(self.runner.stop)
        self.btn_take.clicked.connect(self.take_window)
        self.runner.busy.connect(lambda on: self.btn_stop.setEnabled(on and self._source() == 2))
        session.viewer.layers.selection.events.active.connect(self._layer_selected)
        session.store_changed.connect(self._update_source)
        self._update_source()
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

    # ---- input source -------------------------------------------------------
    def _source(self) -> int:
        return self.cb_source.currentIndex() if self.cb_method.currentIndex() == 0 else 0

    def _update_source(self, *_):
        src = self._source()
        self.ch_row.setVisible(src == 0)
        self.layer_info.setVisible(src == 1)
        self.win_row.setVisible(src == 2)
        self.btn_stop.setVisible(src == 2)
        self.btn_take.setVisible(self.session.windows_layer() is not None)
        self._show_layer_info()

    def _limit_channels(self, box, on):
        if on and len(self.channels()) > 3:
            box.setChecked(False)
            self.status.setText("Cellpose takes at most 3 input images.")

    def _layer_selected(self, event=None):
        layer = self.session.viewer.layers.selection.active
        if layer is not None and layer._type_string == "image" and not layer.rgb:
            self._last_image = layer
        self._show_layer_info()

    def _show_layer_info(self):
        layer = self._last_image
        if layer is None or layer not in self.session.viewer.layers:
            self.layer_info.setText("Select an image layer in napari's layer list.")
        elif layer.ndim > 2:
            self.layer_info.setText(f"Image: '{layer.name}', the frame shown by the time slider.")
        else:
            self.layer_info.setText(f"Image: '{layer.name}'.")

    def current_image(self) -> tuple[np.ndarray, str]:
        """The last selected image layer; for a movie, the frame at the time slider."""
        layer = self._last_image
        if layer is None or layer not in self.session.viewer.layers:
            raise ValueError("Select an image layer in napari's layer list first.")
        data = layer.data[0] if layer.multiscale else layer.data
        lead = layer.ndim - 2
        step = tuple(self.session.viewer.dims.current_step[-layer.ndim:][:lead])
        step = tuple(min(max(int(i), 0), n - 1) for i, n in zip(step, data.shape))
        source = f"layer '{layer.name}'" + (f", frame {', '.join(map(str, step))}" if lead else "")
        return np.asarray(data[step], dtype=np.float32), source

    def take_window(self):
        layer = self.session.windows_layer()
        if layer is None:
            return
        t = int(self.session.viewer.dims.current_step[0])
        labels = np.asarray(layer.data[t])
        if not labels.any():
            self.status.setText(f"No window segmentation at frame {t}.")
            return
        self.session.show_labels(filter_area(labels, 1))
        self.status.setText(f"ROIs of frame {t} copied to 'ROIs': correct them, then Save ROIs.")

    def _count(self):
        layer = self.session.roi_layer()
        self.count.setText(f"{int(layer.data.max())} ROIs" if layer is not None else "")

    def channels(self) -> list[str]:
        return [n for n, b in self.ch_boxes.items() if b.isChecked()]

    def run(self):
        store = self.session.store
        if store is None:
            return
        lock = [self.btn_run, self.btn_save, self.btn_clean, self.btn_take]
        self.session.pull_settings()
        src = self._source()
        if src == 1:
            try:
                image, source = self.current_image()
            except ValueError as e:
                self.status.setText(str(e))
                return
            p = self.cp_form.value()
            p.gpu = self.cb_gpu.isChecked()
            self.runner.run(pipeline.run_cellpose_image, store, image, source, p, with_progress=False,
                            done=self._got_labels, lock=lock, message=f"Running Cellpose on {source}…")
            return
        if src == 2:
            p = self.cp_form.value()
            p.gpu = self.cb_gpu.isChecked()
            frames = self.ed_win_frames.text().strip()
            try:
                n = len(pipeline.frame_windows(store.movie.shape[0], self.sp_window.value(), frames))
            except ValueError as e:
                self.status.setText(str(e))
                return
            self.runner.run(pipeline.run_cellpose_windows, store, self.sp_window.value(), p, frames,
                            with_stop=True, done=self._got_windows, lock=lock,
                            message=f"Running Cellpose on {n} windows…")
            return
        if not pipeline.load_projections(store):
            self.status.setText("Compute the projections first (Preprocess).")
            return
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

    def _got_windows(self, done):
        self.session.show_windows()
        self._update_source()
        n = self.session.store.load_array("windows/n_rois")[:done]
        total = len(self.session.store.load_array("windows/first"))
        if not done:
            self.status.setText("Stopped before the first window was done.")
            return
        self.status.setText(f"{done} of {total} windows segmented, {n.min()}–{n.max()} ROIs each "
                            "(numbers are not matched between windows). Scroll in time, then "
                            "'Use the ROIs of the current window' to keep one.")

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
