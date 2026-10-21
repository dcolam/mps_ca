"""Step 1: find the recordings, describe them, scan all, convert all."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
                            QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget,
                            QTableWidgetItem, QVBoxLayout, QWidget)

from ..io import ConvertParams
from ..io.metadata import GROUPINGS, Recording
from ..io.mm_tiff import open_tiff_lazy
from ..io.scan import ScanReport
from ..project import PROJECT_FILE, Project, accept_proposals, convert_all, create_project, scan_all
from ._common import ParamForm, Runner, hint
from ._session import Session, settings

REC_FIXED = ["Recording", "Status", "Files", "Blocks"]  # then the description columns
FILE_COLS = ["File", "Block", "Condition", "Seg.", "Use", "Note"]
BLOCK_COLS = ["Block", "Condition", "Frames", "Listed", "Size", "Cam bin", "Factor", "Output"]

CONVERT_SPECS = {
    "source_fps": {"label": "Recorded fps", "min": 0.1, "max": 1000,
                   "tooltip": "Checked against the acquisition timestamps by 'Scan all'"},
    "target_fps": {"label": "Working fps", "min": 0.01, "max": 1000,
                   "tooltip": "Frames are averaged in time: recorded/working must be a whole number"},
    "target_camera_binning": {"label": "Common grid (camera binning)", "min": 0, "max": 8,
                              "tooltip": "Blocks recorded with less camera binning are binned to this "
                                         "grid (2 = 2×2, so 2200×3200 becomes 1100×1600). "
                                         "0 = bring blocks to the coarsest block of each recording"},
    "spatial_binning": {"label": "Extra spatial binning", "min": 1, "max": 16,
                        "tooltip": "Applied after all blocks share one pixel grid"},
    "low_signal_thresh": {"label": "Bad-frame threshold", "min": 0, "max": 1, "step": 0.05,
                          "tooltip": "Frames darker than this fraction of the median are flagged (adjustable later)"},
    "chunk_frames": {"label": "Frames per chunk", "min": 1, "max": 512},
    "pixel_size_um": {"label": "Pixel size (µm)", "optional": True, "decimals": 3,
                      "tooltip": "Output pixel size; auto = from Micro-Manager metadata"},
    "read_timestamps": {"label": "Use acquisition timestamps"},
    "block_spatial_factors": {"skip": True},
}


def _item(text, editable=False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


def _check(on: bool) -> QTableWidgetItem:
    it = QTableWidgetItem()
    it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
    it.setCheckState(Qt.Checked if on else Qt.Unchecked)
    return it


def _path_row(edit: QLineEdit, browse) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(edit)
    b = QPushButton("…")
    b.setMaximumWidth(30)
    b.clicked.connect(browse)
    row.addWidget(b)
    return row


class ProjectTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.reports: dict[str, ScanReport] = {}
        self._factor_boxes: dict[int, QSpinBox] = {}
        self._filling = False

        lay = QVBoxLayout(self)
        lay.addWidget(hint("1) Choose the folder with the videos and, optionally, an Excel file that "
                           "describes them, then 'Find recordings'. 2) Check the recordings and add "
                           "description columns (e.g. genotype). 3) 'Scan all' checks every file, "
                           "4) 'Convert all' makes the working copies. Then test the processing on a "
                           "few recordings (tabs 2–5) and run it on all of them (Run all)."))

        src = QGroupBox("Input")
        form = QFormLayout(src)
        last = settings()
        self.ed_input = QLineEdit(last.value("data_dir", ""))
        self.ed_pattern = QLineEdit("*.ome.tif")
        self.ed_pattern.setToolTip("File name pattern, searched in all subfolders; several separated by ';'")
        self.ed_excel = QLineEdit()
        self.ed_excel.setPlaceholderText("optional")
        self.cb_grouping = QComboBox()
        for key, text in GROUPINGS.items():
            self.cb_grouping.addItem(text, key)
        self.ed_conditions = QLineEdit()
        self.ed_conditions.setPlaceholderText("auto: the columns that change between blocks")
        self.ed_conditions.setToolTip("Excel columns that name a block (condition), comma-separated")
        self.ed_output = QLineEdit(last.value("out_dir", str(Path.home() / "mpsca_data")))
        form.addRow("Video folder", _path_row(self.ed_input, self._browse_input))
        form.addRow("File pattern", self.ed_pattern)
        form.addRow("Excel (optional)", _path_row(self.ed_excel, self._browse_excel))
        form.addRow("Block names from", self.ed_conditions)
        form.addRow("Without Excel", self.cb_grouping)
        form.addRow("Output folder", _path_row(self.ed_output, self._browse_output))
        row = QHBoxLayout()
        self.btn_find = QPushButton("Find recordings")
        self.btn_open = QPushButton("Open project…")
        row.addWidget(self.btn_find)
        row.addWidget(self.btn_open)
        form.addRow(row)
        lay.addWidget(src)

        self.recs = QTableWidget(0, len(REC_FIXED) + 1)
        self.recs.verticalHeader().setVisible(False)
        self.recs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.recs.setMinimumHeight(170)
        lay.addWidget(QLabel("Recordings (tick the ones to use; description cells are editable)"))
        lay.addWidget(self.recs)
        rrow = QHBoxLayout()
        self.btn_col = QPushButton("Add description column…")
        self.btn_test = QPushButton("Open selected recording")
        self.btn_test.setToolTip("Open a converted recording to try the processing settings on it")
        rrow.addWidget(self.btn_col)
        rrow.addWidget(self.btn_test)
        lay.addLayout(rrow)

        det = QGroupBox("Selected recording")
        dl = QVBoxLayout(det)
        self.files = QTableWidget(0, len(FILE_COLS))
        self.files.setHorizontalHeaderLabels(FILE_COLS)
        self.files.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.files.verticalHeader().setVisible(False)
        self.files.setMinimumHeight(150)
        dl.addWidget(QLabel("Files: Block, Condition, Seg. (use for segmentation) and Use are editable"))
        dl.addWidget(self.files)
        self.btn_preview = QPushButton("Preview selected file")
        dl.addWidget(self.btn_preview)
        self.blocks = QTableWidget(0, len(BLOCK_COLS))
        self.blocks.setHorizontalHeaderLabels(BLOCK_COLS)
        self.blocks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.blocks.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.blocks.verticalHeader().setVisible(False)
        self.blocks.setMinimumHeight(110)
        dl.addWidget(QLabel("After the scan (Factor = spatial binning of each block)"))
        dl.addWidget(self.blocks)
        self.issues = QListWidget()
        self.issues.setWordWrap(True)
        self.issues.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.issues.setMaximumHeight(110)
        dl.addWidget(self.issues)
        lay.addWidget(det)

        cbox = QGroupBox("Conversion (same for all recordings)")
        cl = QVBoxLayout(cbox)
        self.form = ParamForm(ConvertParams, CONVERT_SPECS)
        cl.addWidget(self.form)
        brow = QHBoxLayout()
        self.btn_scan = QPushButton("Scan all")
        self.btn_convert = QPushButton("Convert all")
        self.btn_convert.setToolTip("Converts the ticked recordings that are not converted yet")
        brow.addWidget(self.btn_scan)
        brow.addWidget(self.btn_convert)
        cl.addLayout(brow)
        lay.addWidget(cbox)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.log = QListWidget()
        self.log.setMaximumHeight(120)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addWidget(self.log)
        lay.addStretch()
        self.runner = Runner(self, self.bar, self.status, self.log)
        self._buttons = [self.btn_find, self.btn_open, self.btn_scan, self.btn_convert, self.btn_test]

        self.btn_find.clicked.connect(self.find)
        self.btn_open.clicked.connect(self._open_project)
        self.btn_col.clicked.connect(self._add_column)
        self.btn_test.clicked.connect(self.open_selected)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_scan.clicked.connect(self.scan)
        self.btn_convert.clicked.connect(self.convert)
        self.recs.itemChanged.connect(self._rec_edited)
        self.recs.itemSelectionChanged.connect(self._show_selected)
        self.files.itemChanged.connect(self._file_edited)
        for name in ("spatial_binning", "target_camera_binning"):
            self.form.widget(name).valueChanged.connect(self._grid_changed)

    # ---- project ---------------------------------------------------------------
    @property
    def project(self) -> Project | None:
        return self.session.project

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with the videos", self.ed_input.text())
        if d:
            self.ed_input.setText(d)

    def _browse_excel(self):
        f, _ = QFileDialog.getOpenFileName(self, "Excel file describing the videos",
                                           self.ed_excel.text() or self.ed_input.text(), "Excel (*.xlsx)")
        if f:
            self.ed_excel.setText(f)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder (use a fast local disk)", self.ed_output.text())
        if d:
            self.ed_output.setText(d)

    def find(self):
        inp, out = self.ed_input.text().strip(), self.ed_output.text().strip()
        if not inp or not Path(inp).is_dir():
            QMessageBox.warning(self, "Find recordings", "Choose an existing video folder first.")
            return
        if not out:
            QMessageBox.warning(self, "Find recordings", "Choose an output folder first.")
            return
        existing = Path(out) / PROJECT_FILE
        previous = self.project
        if previous is None and existing.exists():
            ans = QMessageBox.question(self, "Project exists",
                                       f"{existing} already exists. Keep its descriptions and settings?")
            if ans == QMessageBox.Yes:
                previous = Project.load(existing, input_dir=inp)
        cols = [c.strip() for c in self.ed_conditions.text().split(",") if c.strip()] or None
        try:
            proj = create_project(inp, out, self.ed_pattern.text().strip() or "*.ome.tif",
                                  self.ed_excel.text().strip() or None, self.cb_grouping.currentData(),
                                  cols, previous=previous)
        except Exception as e:
            QMessageBox.warning(self, "Find recordings", str(e))
            return
        if previous is None:
            proj.convert = self.form.value()
        settings().setValue("data_dir", inp)
        settings().setValue("out_dir", out)
        proj.save()
        self.reports = {}
        self.session.set_project(proj)
        self.show_project()
        n = sum(len(r.active_files) for r in proj.recordings)
        self.status.setText(f"{len(proj.recordings)} recordings, {n} files. Next: Scan all.")

    def _open_project(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder of the project", self.ed_output.text())
        if d:
            self.open_project(d)

    def open_project(self, path):
        path = Path(path)
        file = path / PROJECT_FILE if path.is_dir() else path
        if not file.exists():
            QMessageBox.warning(self, "Open project", f"No {PROJECT_FILE} in {path}")
            return
        proj = Project.load(file)
        if not proj.input_found:
            d = QFileDialog.getExistingDirectory(self, f"Video folder moved? Locate {Path(proj.input_dir).name}",
                                                 str(Path(proj.input_dir).parent))
            if d:
                proj = Project.load(file, input_dir=d)
        self.reports = {}
        self.session.set_project(proj)
        self.show_project()

    def show_project(self):
        proj = self.project
        self.ed_input.setText(proj.input_dir)
        self.ed_output.setText(proj.output_dir)
        self.ed_pattern.setText(proj.pattern)
        self.ed_excel.setText(proj.excel or "")
        self.ed_conditions.setText(", ".join(proj.condition_columns or []))
        self.cb_grouping.setCurrentIndex(max(0, self.cb_grouping.findData(proj.grouping)))
        self._filling = True
        self.form.set_value(proj.convert)
        self._filling = False
        self.fill_recordings()
        self.issues.clear()
        for msg in proj.issues:
            self.issues.addItem(msg)

    # ---- recordings table ---------------------------------------------------------
    def fill_recordings(self):
        proj = self.project
        keys = proj.description_keys()
        self._filling = True
        self.recs.setColumnCount(len(REC_FIXED) + len(keys))
        self.recs.setHorizontalHeaderLabels(REC_FIXED + keys)
        self.recs.setRowCount(len(proj.recordings))
        for r, rec in enumerate(proj.recordings):
            name = _item(rec.name)
            name.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            name.setCheckState(Qt.Checked if rec.include else Qt.Unchecked)
            self.recs.setItem(r, 0, name)
            self.recs.setItem(r, 1, _item(self._status_text(rec)))
            missing = f" ({len(rec.missing)} missing)" if rec.missing else ""
            self.recs.setItem(r, 2, _item(f"{len(rec.active_files)}{missing}"))
            self.recs.setItem(r, 3, _item(rec.n_blocks))
            for c, k in enumerate(keys):
                self.recs.setItem(r, 4 + c, _item(rec.description.get(k, ""), editable=True))
        self.recs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._filling = False

    def _status_text(self, rec: Recording) -> str:
        text = self.project.status(rec)
        rep = self.reports.get(rec.name)
        if rep is not None and text == "not converted":
            text = "scanned" + (f", {len(rep.issues)} note(s)" if rep.issues else ", ok")
        return text

    def refresh_status(self):
        if self.project is None:
            return
        self._filling = True
        for r, rec in enumerate(self.project.recordings):
            self.recs.setItem(r, 1, _item(self._status_text(rec)))
        self._filling = False

    def _rec_edited(self, item):
        if self._filling or self.project is None:
            return
        rec = self.project.recordings[item.row()]
        keys = self.project.description_keys()
        if item.column() == 0:
            rec.include = item.checkState() == Qt.Checked
        elif item.column() >= len(REC_FIXED):
            rec.description[keys[item.column() - len(REC_FIXED)]] = item.text().strip()
        self.project.save()

    def _add_column(self):
        if self.project is None:
            return
        name, ok = QInputDialog.getText(self, "Description column", "Name (e.g. Genotype):")
        name = name.strip()
        if ok and name:
            for rec in self.project.recordings:
                rec.description.setdefault(name, "")
            self.project.save()
            self.fill_recordings()

    def selected(self) -> Recording | None:
        rows = {i.row() for i in self.recs.selectedIndexes()}
        if self.project is None or not rows:
            return None
        return self.project.recordings[min(rows)]

    def select(self, name: str):
        for r, rec in enumerate(self.project.recordings):
            if rec.name == name:
                self.recs.selectRow(r)

    def open_selected(self):
        rec = self.selected()
        if rec is None:
            QMessageBox.information(self, "Open", "Select a recording first.")
            return
        path = self.project.store_path(rec)
        if self.project.status(rec) in ("not converted", "incomplete", "unreadable"):
            QMessageBox.information(self, "Open", f"{rec.name} is not converted yet.")
            return
        try:
            self.session.open_store(path)
        except Exception as e:
            QMessageBox.warning(self, "Cannot open", str(e))

    # ---- selected recording ------------------------------------------------------
    def _show_selected(self):
        rec = self.selected()
        self._filling = True
        self.files.setRowCount(0 if rec is None else len(rec.files))
        if rec is not None:
            for r, fe in enumerate(rec.files):
                rel = fe.relpath[len(rec.folder):].lstrip("/") if rec.folder else fe.relpath
                name = _item(rel + ("" if fe.path is not None else "  (MISSING)"))
                if fe.path is None and fe.include:
                    name.setForeground(Qt.red)
                self.files.setItem(r, 0, name)
                self.files.setItem(r, 1, _item(fe.block, editable=True))
                self.files.setItem(r, 2, _item(fe.condition, editable=True))
                self.files.setItem(r, 3, _check(fe.use_for_seg))
                self.files.setItem(r, 4, _check(fe.include))
                self.files.setItem(r, 5, _item(fe.note))
        self._filling = False
        self._fill_blocks()

    def _file_edited(self, item):
        rec = self.selected()
        if self._filling or rec is None:
            return
        fe = rec.files[item.row()]
        col = item.column()
        if col == 1:
            try:
                fe.block = int(item.text())
            except ValueError:
                pass
        elif col == 2:
            for other in rec.files:  # a condition names the whole block
                if other.block == fe.block:
                    other.condition = item.text().strip()
        elif col == 3:
            fe.use_for_seg = item.checkState() == Qt.Checked
        elif col == 4:
            fe.include = item.checkState() == Qt.Checked
        self.reports.pop(rec.name, None)  # the scan no longer matches the files
        self.project.save()
        self._show_selected()
        self.fill_recordings()
        self.select(rec.name)

    def _fill_blocks(self):
        rec = self.selected()
        rep = self.reports.get(rec.name) if rec is not None else None
        self.blocks.setRowCount(0)
        self._factor_boxes = {}
        self.issues.clear()
        if rep is None:
            for msg in (self.project.issues if self.project else []):
                self.issues.addItem(msg)
            return
        for msg in rep.issues or ["No problems found."]:
            self.issues.addItem(msg)
        if not rep.files:
            return
        factors = self.form.value().factors_for(rep)
        summary = rep.block_summary()
        self.blocks.setRowCount(len(summary))
        for row, (_, b) in enumerate(summary.iterrows()):
            listed = "" if b["expected"] is None or b["expected"] != b["expected"] else int(b["expected"])
            vals = [b["block"], b["condition"], b["frames"], listed, b["size"], b["camera_binning"]]
            for c, v in enumerate(vals):
                it = _item(v)
                if c == 2 and listed != "" and int(listed) != int(b["frames"]):
                    it.setForeground(Qt.red)
                self.blocks.setItem(row, c, it)
            spin = QSpinBox()
            spin.setRange(1, 32)
            spin.setValue(factors[int(b["block"])])
            spin.valueChanged.connect(self._factors_edited)
            self._factor_boxes[int(b["block"])] = spin
            self.blocks.setCellWidget(row, 6, spin)
        self._update_output_sizes()

    def block_factors(self) -> dict[int, int]:
        return {b: s.value() for b, s in self._factor_boxes.items()}

    def _factors_edited(self):
        rec = self.selected()
        if rec is not None:
            rec.block_factors = self.block_factors()  # an explicit choice for this recording
            self.project.save()
        self._update_output_sizes()

    def _update_output_sizes(self):
        rec = self.selected()
        rep = self.reports.get(rec.name) if rec is not None else None
        if rep is None:
            return
        f = self.block_factors()
        sizes = {}
        for fs in rep.files:
            sizes.setdefault(fs.entry.block, (fs.height // f[fs.entry.block], fs.width // f[fs.entry.block]))
        consistent = len(set(sizes.values())) <= 1
        for row in range(self.blocks.rowCount()):
            h, w = sizes.get(int(self.blocks.item(row, 0).text()), (0, 0))
            it = _item(f"{h}x{w}")
            if not consistent:
                it.setForeground(Qt.red)
            self.blocks.setItem(row, 7, it)

    def _grid_changed(self):
        if self.project is None or self._filling:
            return
        self.project.convert = self.form.value()
        for rec in self.project.recordings:  # proposals change with the grid: decide again
            rec.block_factors = None
        self.project.save()
        self._fill_blocks()

    def _preview(self):
        rec = self.selected()
        row = self.files.currentRow()
        if rec is None or row < 0 or rec.files[row].path is None:
            QMessageBox.information(self, "Preview", "Select a file in the table first.")
            return
        fe = rec.files[row]
        data = open_tiff_lazy(fe.path)
        lo, hi = np.percentile(np.asarray(data[0]), [1, 99.8])
        self.session.viewer.add_image(data, name=f"raw {fe.filename}", contrast_limits=(lo, hi + 1e-3))

    # ---- scan and convert ---------------------------------------------------------
    def _targets(self) -> list[Recording]:
        return self.project.select() if self.project is not None else []

    def scan(self, then_convert: bool = False):
        if self.project is None:
            QMessageBox.information(self, "Scan", "Find recordings or open a project first.")
            return
        self.project.convert = self.form.value()
        recs = self._targets()
        if not recs:
            QMessageBox.information(self, "Scan", "No recording is ticked.")
            return
        self.runner.run(scan_all, self.project, recs, with_say=True, lock=self._buttons,
                        done=lambda reps: self._scanned(reps, then_convert),
                        message=f"Reading the headers of {len(recs)} recordings…")

    def _scanned(self, reports: dict[str, ScanReport], then_convert: bool = False):
        self.reports.update(reports)
        self._snap_fps(reports)
        self.fill_recordings()
        rec = self.selected()
        if rec is None and self.project.recordings:
            self.recs.selectRow(0)
        else:
            self._show_selected()
        n_notes = sum(len(r.issues) for r in reports.values())
        self.status.setText(f"Scanned {len(reports)} recordings, {n_notes} note(s); select a recording "
                            "to see its blocks and notes.")
        if not self._confirm(reports):
            return
        if then_convert:
            self._start_convert()

    def _snap_fps(self, reports):
        dts = [np.diff(t) for rep in reports.values() for f in rep.files
               for t in [f.info.times[np.isfinite(f.info.times)]] if len(t) > 2]
        if dts:
            fps = 1 / float(np.median(np.concatenate(dts)))
            # timestamps have millisecond resolution: snap 3.98 -> 4
            fps = round(fps) if abs(fps - round(fps)) < 0.02 * fps else round(fps, 2)
            if abs(fps - self.form.value().source_fps) > 1e-6:
                self.form.widget("source_fps").setValue(fps)
                self.project.convert = self.form.value()
                self.project.save()

    def _confirm(self, reports) -> bool:
        """One dialog for every recording whose spatial binning needs a decision."""
        from ..project import needs_decision

        pending = {n: r for n, r in reports.items() if needs_decision(self.project, self.project.get(n), r)}
        if not pending:
            return True
        p = self.project.convert
        lines = []
        for name, rep in pending.items():
            parts = []
            factors = rep.proposed_factors(p.spatial_binning, p.target_camera_binning)
            for _, b in rep.block_summary().iterrows():
                parts.append(f"{b['size']} (camera bin {b['camera_binning']}) ÷{factors[int(b['block'])]}")
            out = next(iter(rep.output_shape(factors)))
            lines.append(f"• {name}: " + ", ".join(parts) + f" → {out[0]}×{out[1]}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Spatial binning needed")
        box.setText(f"{len(pending)} recording(s) have blocks recorded with less camera binning than the "
                    "common grid, or blocks of different sizes. Proposal (÷ = spatial binning per block):\n\n"
                    + "\n".join(lines))
        box.setInformativeText("'Use proposals' applies them to all of these recordings. To choose "
                               "yourself, select a recording and change Factor in its block table.")
        use = box.addButton("Use proposals", QMessageBox.AcceptRole)
        box.addButton("Review", QMessageBox.RejectRole)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        if box.clickedButton() is use:
            accept_proposals(self.project, pending)
            self._fill_blocks()
            return True
        return False

    def convert(self, overwrite: bool = False):
        if self.project is None:
            QMessageBox.information(self, "Convert", "Find recordings or open a project first.")
            return
        recs = self._targets()
        if any(r.name not in self.reports for r in recs):
            self.scan(then_convert=True)  # always check every file before the long run
            return
        if not self._confirm({r.name: self.reports[r.name] for r in recs}):
            return
        self._start_convert(overwrite)

    def _start_convert(self, overwrite: bool = False):
        from ..project import needs_decision

        recs = self._targets()
        missing = [r.name for r in recs if r.missing]
        undecided = [r.name for r in recs if needs_decision(self.project, r, self.reports[r.name])]
        problems = []
        if missing:
            problems.append(f"files missing in: {', '.join(missing)}")
        if undecided:
            problems.append(f"spatial binning not chosen for: {', '.join(undecided)}")
        if problems:
            ans = QMessageBox.question(self, "Convert", "These recordings will be skipped:\n"
                                       + "\n".join(problems) + "\n\nConvert the others?")
            if ans != QMessageBox.Yes:
                return
        out = Path(self.project.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.project.convert = self.form.value()
        self.project.save()
        self.runner.run(convert_all, self.project, recs, self.reports, overwrite=overwrite,
                        with_say=True, lock=self._buttons, done=self._converted,
                        message=f"Converting {len(recs)} recordings into {out}…")

    def _converted(self, results: dict[str, str]):
        self.refresh_status()
        failed = {n: r for n, r in results.items() if r.startswith("failed")}
        ok = len(results) - len(failed)
        text = f"{ok} of {len(results)} recordings converted or already done."
        if failed:
            text += " Failed: " + "; ".join(f"{n}: {r}" for n, r in failed.items())
        else:
            text += " Next: select one, 'Open selected recording', and try the processing (tabs 2–5)."
        self.status.setText(text)
