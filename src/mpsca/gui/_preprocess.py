"""Step 2: bad frames and projection images."""
from __future__ import annotations

from qtpy.QtWidgets import (QDoubleSpinBox, QFormLayout, QLabel, QProgressBar, QPushButton,
                            QVBoxLayout, QWidget)

from .. import pipeline
from ..analysis import epochs_from_frames
from ..analysis import figures as figs
from ._common import FigurePanel, Runner, hint
from ._session import Session


class PreprocessTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Frames much darker than the rest (shutter, focus loss) are flagged as bad. "
                           "They stay in the data but are ignored by the projections, the ΔF/F "
                           "baseline and the analysis. Then compute the projection images used "
                           "for segmentation (frames ticked 'Segment.' only)."))
        self.plot = FigurePanel(placeholder="Open or convert an experiment first.")
        lay.addWidget(self.plot)
        form = QFormLayout()
        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(0, 1)
        self.sp_thresh.setSingleStep(0.05)
        self.sp_thresh.setValue(0.5)
        form.addRow("Bad-frame threshold (× median)", self.sp_thresh)
        lay.addLayout(form)
        self.btn_bad = QPushButton("Apply threshold")
        self.btn_proj = QPushButton("Compute projections (mean, max, std, correlation)")
        lay.addWidget(self.btn_bad)
        lay.addWidget(self.btn_proj)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addStretch()
        self.runner = Runner(self, self.bar, self.status)

        self.btn_bad.clicked.connect(self.apply_threshold)
        self.btn_proj.clicked.connect(self.compute_projections)
        session.store_changed.connect(self.refresh)

    def read_settings(self, s):
        self.sp_thresh.setValue(s.bad_frame_thresh)

    def write_settings(self, s):
        s.bad_frame_thresh = self.sp_thresh.value()

    def refresh(self):
        store = self.session.store
        if store is None:
            return
        p = store.params("bad_frames") or store.params("convert")
        if self.session.project is None:  # in a project the settings stay as being tested
            self.sp_thresh.setValue(float(p.get("low_signal_thresh", 0.5)))
        self._draw(p.get("cutoff", p.get("bad_frame_cutoff")))

    def _draw(self, cutoff=None):
        store = self.session.store
        frames = store.read_frames()
        t = frames["time_s"].to_numpy()
        bad = frames["is_bad"].to_numpy()
        eps = epochs_from_frames(frames, pipeline.fps_of(store))
        self.plot.set_figure(figs.frame_means(t, frames["frame_mean"].to_numpy(), bad, cutoff, eps))
        seg = int((frames["use_for_seg"] & ~frames["is_bad"]).sum())
        warn = store.meta.get("frame_warnings", [])
        self.info.setText(f"{len(frames)} frames, {int(bad.sum())} bad, {seg} used for projections."
                          + ("\nNotes from conversion:\n• " + "\n• ".join(warn) if warn else ""))

    def apply_threshold(self):
        if self.session.store is None:
            return
        self.session.pull_settings()
        _, cutoff = pipeline.set_bad_frames(self.session.store, self.sp_thresh.value())
        self._draw(cutoff)

    def compute_projections(self):
        if self.session.store is None:
            return
        self.runner.run(pipeline.run_projections, self.session.store, done=self._show,
                        message="Computing projections…", lock=[self.btn_proj, self.btn_bad])

    def _show(self, proj):
        for name in ("mean", "max", "std", "corr"):
            self.session.show_image(name, proj[name], visible=name == "corr")
        self.info.setText(self.info.text() + "\nProjections ready: next, Segment.")
