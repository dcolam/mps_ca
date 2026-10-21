"""The dock widget: one tab per processing step."""
from __future__ import annotations

from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from ._analysis import AnalysisTab, SummaryTab
from ._import import ImportTab
from ._preprocess import PreprocessTab
from ._segment import SegmentTab
from ._session import Session
from ._traces import TracesTab


def _scrolled(w: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(w)
    return area


class MainWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.session = Session(napari_viewer)
        self.import_tab = ImportTab(self.session)
        self.preprocess_tab = PreprocessTab(self.session)
        self.segment_tab = SegmentTab(self.session)
        self.traces_tab = TracesTab(self.session)
        self.analysis_tab = AnalysisTab(self.session)
        self.summary_tab = SummaryTab(self.session)
        self.tabs = QTabWidget()
        for title, tab in [("1 Import", self.import_tab), ("2 Preprocess", self.preprocess_tab),
                           ("3 Segment", self.segment_tab), ("4 Traces", self.traces_tab),
                           ("5 Analysis", self.analysis_tab), ("Summary", self.summary_tab)]:
            self.tabs.addTab(_scrolled(tab), title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        self.setMinimumWidth(520)
        self.session.store_changed.connect(self._opened)

    def _opened(self):
        self.viewer.title = f"mpsca: {self.session.name}"
        if self.tabs.currentIndex() == 0:
            self.tabs.setCurrentIndex(1)
