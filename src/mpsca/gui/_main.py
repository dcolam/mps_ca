"""The dock widget: one tab per processing step."""
from __future__ import annotations

from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from ._analysis import AnalysisTab, SummaryTab
from ._batch import BatchTab
from ._preprocess import PreprocessTab
from ._project import ProjectTab
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
        self.project_tab = ProjectTab(self.session)
        self.preprocess_tab = PreprocessTab(self.session)
        self.segment_tab = SegmentTab(self.session)
        self.traces_tab = TracesTab(self.session)
        self.analysis_tab = AnalysisTab(self.session)
        self.batch_tab = BatchTab(self.session)
        self.summary_tab = SummaryTab(self.session)
        # the processing settings live in the project; these tabs edit them
        self.session.settings_tabs = [self.preprocess_tab, self.segment_tab, self.traces_tab,
                                      self.analysis_tab, self.batch_tab]
        self.tabs = QTabWidget()
        for title, tab in [("1 Project", self.project_tab), ("2 Preprocess", self.preprocess_tab),
                           ("3 Segment", self.segment_tab), ("4 Traces", self.traces_tab),
                           ("5 Analysis", self.analysis_tab), ("6 Run all", self.batch_tab),
                           ("Summary", self.summary_tab)]:
            self.tabs.addTab(_scrolled(tab), title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        self.setMinimumWidth(560)
        self.session.store_changed.connect(self._opened)
        self.tabs.currentChanged.connect(self._tab_changed)

    def _opened(self):
        self.viewer.title = f"mpsca: {self.session.name}"
        self.project_tab.refresh_status()
        if self.tabs.currentIndex() == 0:
            self.tabs.setCurrentIndex(1)

    def _tab_changed(self, index: int):
        if self.tabs.widget(index).widget() is self.batch_tab:
            self.batch_tab.refresh()
        elif self.tabs.widget(index).widget() is self.project_tab:
            self.project_tab.refresh_status()
