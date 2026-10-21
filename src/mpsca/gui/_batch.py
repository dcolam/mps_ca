"""Step 6: run the tested settings on all converted recordings."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                            QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from ..pipeline import STEP_LABELS, STEPS
from ..project import run_all
from ._common import Runner, hint
from ._session import Session

NOT_READY = ("not converted", "incomplete", "unreadable")


def _value(v) -> str:
    if v is None:
        return "auto"
    if isinstance(v, (list, tuple)):
        return ", ".join(_value(x) for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(round(v, 4) if isinstance(v, float) else v)


def _format(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"    {kk}: {_value(vv)}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {_value(v)}")
    return "\n".join(lines)


class BatchTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._offered: set[str] = set()  # recordings that could already be ticked
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Runs the processing on many recordings with exactly the settings now "
                           "in tabs 2–5 (shown below). Try them on one or two recordings first. "
                           "ROIs you corrected and saved by hand are kept unless you untick that."))

        sbox = QGroupBox("Settings that will be used")
        sl = QVBoxLayout(sbox)
        self.settings_view = QPlainTextEdit()
        self.settings_view.setReadOnly(True)
        self.settings_view.setMaximumHeight(160)
        sl.addWidget(self.settings_view)
        self.btn_refresh = QPushButton("Refresh")
        sl.addWidget(self.btn_refresh)
        lay.addWidget(sbox)

        stbox = QGroupBox("Steps")
        stl = QHBoxLayout(stbox)
        self.step_boxes = {}
        for step in STEPS:
            box = QCheckBox(STEP_LABELS[step])
            box.setChecked(True)
            self.step_boxes[step] = box
            stl.addWidget(box)
        lay.addWidget(stbox)
        self.cb_keep = QCheckBox("Keep ROIs that were corrected and saved by hand")
        self.cb_keep.setChecked(True)
        lay.addWidget(self.cb_keep)

        lay.addWidget(QLabel("Recordings (converted ones can be ticked)"))
        self.list = QListWidget()
        self.list.setMinimumHeight(140)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        b_all = QPushButton("All")
        b_none = QPushButton("None")
        b_todo = QPushButton("Not analysed yet")
        for b in (b_all, b_none, b_todo):
            row.addWidget(b)
        lay.addLayout(row)
        self.btn_run = QPushButton("Run")
        lay.addWidget(self.btn_run)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.log = QListWidget()
        self.log.setMinimumHeight(120)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.log)
        lay.addStretch()
        self.runner = Runner(self, self.bar, self.status, self.log)

        self.btn_refresh.clicked.connect(self.refresh)
        b_all.clicked.connect(lambda: self._tick(lambda st: True))
        b_none.clicked.connect(lambda: self._tick(lambda st: False))
        b_todo.clicked.connect(lambda: self._tick(lambda st: st != "analysed"))
        self.btn_run.clicked.connect(self.run)
        session.project_changed.connect(self._project_changed)

    def _project_changed(self):
        self._offered.clear()
        self.refresh()

    def read_settings(self, s):
        self.cb_keep.setChecked(s.keep_edited_rois)

    def write_settings(self, s):
        s.keep_edited_rois = self.cb_keep.isChecked()

    def refresh(self):
        proj = self.session.project
        s = self.session.pull_settings()
        d = s.to_dict()
        if d["seg_method"] == "cellpose":
            d.pop("classic")
        else:
            d.pop("cellpose")
            d.pop("seg_channels")
        self.settings_view.setPlainText(_format(d))
        ticked = self.ticked()
        self.list.clear()
        if proj is None:
            return
        for rec in proj.select():
            status = proj.status(rec)
            it = QListWidgetItem(f"{rec.name}  [{status}]")
            it.setData(Qt.UserRole, rec.name)
            it.setData(Qt.UserRole + 1, status)
            if status in NOT_READY:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
                it.setCheckState(Qt.Unchecked)
            else:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                # newly converted recordings start ticked; earlier choices are kept
                it.setCheckState(Qt.Checked if rec.name in ticked or rec.name not in self._offered
                                 else Qt.Unchecked)
                self._offered.add(rec.name)
            self.list.addItem(it)

    def _tick(self, rule):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.flags() & Qt.ItemIsEnabled:
                it.setCheckState(Qt.Checked if rule(it.data(Qt.UserRole + 1)) else Qt.Unchecked)

    def ticked(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def steps(self) -> list[str]:
        return [s for s, b in self.step_boxes.items() if b.isChecked()]

    def run(self):
        proj = self.session.project
        if proj is None:
            QMessageBox.information(self, "Run", "Open or create a project first (tab 1).")
            return
        names, steps = self.ticked(), self.steps()
        if not names or not steps:
            QMessageBox.information(self, "Run", "Tick at least one recording and one step.")
            return
        self.refresh()  # collect and show the settings exactly as they will be used
        names = [n for n in names if n in self.ticked()]
        self.log.clear()
        self.runner.run(run_all, proj, proj.select(names), steps, with_say=True,
                        lock=[self.btn_run], done=self._finished,
                        message=f"Processing {len(names)} recordings…")

    def _finished(self, results: dict[str, str]):
        failed = {n: r for n, r in results.items() if r.startswith("failed")}
        self.status.setText(f"{len(results) - len(failed)} of {len(results)} recordings done."
                            + (" Failed: " + "; ".join(f"{n}: {r}" for n, r in failed.items()) if failed else
                               " Compare them in Summary."))
        self.refresh()
        store = self.session.store
        if store is not None and store.path.name in {f"{n}.mpsca.zarr" for n in results}:
            self.session.open_store(store.path)  # show the new results of the open recording
