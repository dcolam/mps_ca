"""Step 1: choose files, describe conditions, scan headers, resolve mismatches, convert."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                            QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
                            QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget,
                            QTableWidgetItem, QVBoxLayout, QWidget)

from ..io import ConvertParams, convert_experiment, parse_excel, resolve_paths, scan_experiment
from ..io.metadata import ExperimentMeta, entries_from_folder, infer_genotype
from ..io.mm_tiff import open_tiff_lazy
from ..io.scan import ScanReport
from ._common import ParamForm, Runner, hint
from ._session import Session, settings

FILE_COLS = ["File", "Folder", "Block", "Application", "Phase", "Seg."]
BLOCK_COLS = ["Block", "Condition", "Frames", "Listed", "Size", "Cam bin", "Factor", "Output"]

CONVERT_SPECS = {
    "source_fps": {"label": "Recorded fps", "min": 0.1, "max": 1000, "tooltip": "Detected from timestamps when available"},
    "target_fps": {"label": "Working fps", "min": 0.01, "max": 1000,
                   "tooltip": "Frames are averaged in time: recorded/working must be a whole number"},
    "target_camera_binning": {"label": "Common grid (camera binning)", "min": 0, "max": 8,
                              "tooltip": "Blocks recorded with less camera binning are binned to this "
                                         "grid (2 = 2×2, so 2200×3200 becomes 1100×1600). "
                                         "0 = bring blocks to the coarsest block of this experiment"},
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


class ImportTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.meta: ExperimentMeta | None = None
        self.report: ScanReport | None = None
        self._factor_boxes: dict[int, QSpinBox] = {}

        lay = QVBoxLayout(self)
        lay.addWidget(hint("Pick the recordings of one experiment, check the conditions, scan the "
                           "files (headers only), then convert them into a working copy that "
                           "the other steps use. An existing working copy can be opened directly."))
        row = QHBoxLayout()
        self.btn_folder = QPushButton("Add folder…")
        self.btn_excel = QPushButton("Load from Excel…")
        self.btn_open = QPushButton("Open converted…")
        for b in (self.btn_folder, self.btn_excel, self.btn_open):
            row.addWidget(b)
        lay.addLayout(row)

        info = QFormLayout()
        self.ed_name = QLineEdit()
        self.cb_genotype = QComboBox()
        self.cb_genotype.setEditable(True)
        self.cb_genotype.addItems(["WT", "KO", "unknown"])
        info.addRow("Experiment", self.ed_name)
        info.addRow("Genotype", self.cb_genotype)
        lay.addLayout(info)

        self.files = QTableWidget(0, len(FILE_COLS))
        self.files.setHorizontalHeaderLabels(FILE_COLS)
        self.files.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.files.verticalHeader().setVisible(False)
        self.files.setMinimumHeight(160)
        lay.addWidget(QLabel("Files: Block, Application, Phase and Seg. (use for segmentation) are editable"))
        lay.addWidget(self.files)
        prow = QHBoxLayout()
        self.btn_preview = QPushButton("Preview selected file")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_scan = QPushButton("Scan files")
        prow.addWidget(self.btn_preview)
        prow.addWidget(self.btn_remove)
        prow.addWidget(self.btn_scan)
        lay.addLayout(prow)

        box = QGroupBox("Scan result (Factor = spatial binning per block)")
        bl = QVBoxLayout(box)
        self.blocks = QTableWidget(0, len(BLOCK_COLS))
        self.blocks.setHorizontalHeaderLabels(BLOCK_COLS)
        self.blocks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.blocks.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.blocks.verticalHeader().setVisible(False)
        self.blocks.setMinimumHeight(120)
        bl.addWidget(self.blocks)
        self.issues = QListWidget()
        self.issues.setWordWrap(True)
        self.issues.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.issues.setMaximumHeight(130)
        bl.addWidget(self.issues)
        lay.addWidget(box)

        cbox = QGroupBox("Conversion")
        cl = QVBoxLayout(cbox)
        self.form = ParamForm(ConvertParams, CONVERT_SPECS)
        cl.addWidget(self.form)
        orow = QHBoxLayout()
        self.ed_out = QLineEdit(settings().value("out_dir", str(Path.home() / "mpsca_data")))
        b_out = QPushButton("…")
        b_out.setMaximumWidth(30)
        b_out.clicked.connect(self._choose_out)
        orow.addWidget(QLabel("Save to"))
        orow.addWidget(self.ed_out)
        orow.addWidget(b_out)
        cl.addLayout(orow)
        self.btn_convert = QPushButton("Convert")
        cl.addWidget(self.btn_convert)
        lay.addWidget(cbox)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)
        lay.addStretch()
        self.runner = Runner(self, self.bar, self.status)

        self.btn_folder.clicked.connect(self._add_folder)
        self.btn_excel.clicked.connect(self._load_excel)
        self.btn_open.clicked.connect(self._open_store)
        self.btn_scan.clicked.connect(self.scan)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_convert.clicked.connect(self.convert)
        self.files.itemChanged.connect(self._files_edited)
        self.form.widget("spatial_binning").valueChanged.connect(self._reset_factors)
        self.form.widget("target_camera_binning").valueChanged.connect(self._reset_factors)
        self._buttons = [self.btn_folder, self.btn_excel, self.btn_open, self.btn_scan, self.btn_convert]

    # ---- loading ------------------------------------------------------------
    def set_meta(self, meta: ExperimentMeta):
        self.meta = meta
        self.report = None
        self.ed_name.setText(meta.name)
        self.cb_genotype.setCurrentText(meta.genotype)
        self.files.blockSignals(True)
        self.files.setRowCount(len(meta.files))
        for r, fe in enumerate(meta.files):
            vals = [fe.filename, fe.path.parent.name if fe.path else "MISSING", str(fe.block),
                    fe.application or "", fe.phase or ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c < 2:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.files.setItem(r, c, it)
            seg = QTableWidgetItem()
            seg.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            seg.setCheckState(Qt.Checked if fe.use_for_seg else Qt.Unchecked)
            self.files.setItem(r, 5, seg)
        self.files.blockSignals(False)
        self.blocks.setRowCount(0)
        self.issues.clear()
        missing = [fe.filename for fe in meta.files if fe.path is None]
        self.status.setText(f"{len(meta.files)} files" + (f", {len(missing)} missing" if missing else "")
                            + ". Next: Scan files.")

    def load_folder(self, folder: str | Path):
        meta = entries_from_folder(folder)
        if not meta.files:
            raise ValueError(f"no TIFF files under {folder}")
        self.set_meta(meta)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Experiment folder", settings().value("data_dir", ""))
        if folder:
            settings().setValue("data_dir", str(Path(folder).parent))
            try:
                self.load_folder(folder)
            except ValueError as e:
                QMessageBox.warning(self, "No files", str(e))

    def load_excel(self, xlsx: str | Path, sheet: str, data_dir: str | Path):
        meta = next(m for m in parse_excel(xlsx) if m.sheet == sheet)
        exp_dir = Path(data_dir) / meta.name
        if not exp_dir.is_dir():
            raise FileNotFoundError(exp_dir)
        resolve_paths(exp_dir, meta)
        self.set_meta(meta)

    def _load_excel(self):
        xlsx, _ = QFileDialog.getOpenFileName(self, "Metadata workbook", settings().value("excel", ""),
                                              "Excel (*.xlsx)")
        if not xlsx:
            return
        settings().setValue("excel", xlsx)
        sheets = [m.sheet for m in parse_excel(xlsx)]
        sheet, ok = QInputDialog.getItem(self, "Experiment", "Sheet", sheets, 0, False)
        if not ok:
            return
        data_dir = settings().value("data_dir", "")
        meta = next(m for m in parse_excel(xlsx) if m.sheet == sheet)
        if not data_dir or not (Path(data_dir) / meta.name).is_dir():
            data_dir = QFileDialog.getExistingDirectory(
                self, f"Folder that contains {meta.name}", data_dir)
            if not data_dir:
                return
        settings().setValue("data_dir", data_dir)
        try:
            self.load_excel(xlsx, sheet, data_dir)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Not found", f"Experiment folder not found:\n{e}")

    def _open_store(self):
        path = QFileDialog.getExistingDirectory(self, "Open converted experiment (.mpsca.zarr)",
                                                settings().value("out_dir", ""))
        if path:
            try:
                self.session.open_store(path)
            except Exception as e:
                QMessageBox.warning(self, "Cannot open", str(e))

    def _choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Save converted data to", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def _preview(self):
        row = self.files.currentRow()
        if self.meta is None or row < 0 or self.meta.files[row].path is None:
            QMessageBox.information(self, "Preview", "Select a file in the table first.")
            return
        fe = self.meta.files[row]
        data = open_tiff_lazy(fe.path)
        lo, hi = np.percentile(np.asarray(data[0]), [1, 99.8])
        self.session.viewer.add_image(data, name=f"raw {fe.filename}", contrast_limits=(lo, hi + 1e-3))

    def remove_selected(self):
        rows = sorted({i.row() for i in self.files.selectedIndexes()}, reverse=True)
        if self.meta is None or not rows:
            return
        self.apply_table()
        for r in rows:
            del self.meta.files[r]
        self.set_meta(self.meta)

    # ---- table edits ----------------------------------------------------------
    def apply_table(self):
        """Copy the edited table back into the file entries."""
        if self.meta is None:
            return
        for r, fe in enumerate(self.meta.files):
            try:
                fe.block = int(self.files.item(r, 2).text())
            except ValueError:
                pass
            fe.application = self.files.item(r, 3).text().strip()
            fe.phase = self.files.item(r, 4).text().strip()
            fe.use_for_seg = self.files.item(r, 5).checkState() == Qt.Checked
        self.meta.name = self.ed_name.text().strip() or self.meta.name
        self.meta.genotype = self.cb_genotype.currentText().strip() or infer_genotype(self.meta.name)

    def _files_edited(self, _item):
        self.apply_table()
        if self.report is not None:
            self._fill_blocks(keep_factors=True)

    # ---- scan -------------------------------------------------------------------
    def scan(self):
        if self.meta is None:
            QMessageBox.information(self, "Scan", "Add a folder or load an Excel sheet first.")
            return
        self.apply_table()
        fps = self.form.value().source_fps
        self.runner.run(scan_experiment, self.meta, fps, done=self._scanned,
                        message="Reading file headers…", lock=self._buttons)

    def _scanned(self, report: ScanReport):
        self.report = report
        times = [f.info.times[np.isfinite(f.info.times)] for f in report.files]
        dts = np.concatenate([np.diff(t) for t in times if len(t) > 2]) if times else []
        if len(dts):
            fps = 1 / float(np.median(dts))
            # timestamps have millisecond resolution: snap 3.98 -> 4
            self.form.widget("source_fps").setValue(round(fps) if abs(fps - round(fps)) < 0.02 * fps
                                                    else round(fps, 2))
        self._fill_blocks()
        self.issues.clear()
        for msg in report.issues or ["No problems found."]:
            self.issues.addItem(msg)
        p = self.form.value()
        if report.needs_decision(p.spatial_binning, p.target_camera_binning):
            self._ask_resolution()

    def _ask_resolution(self):
        r = self.report
        lines = []
        for _, b in r.block_summary().iterrows():
            lines.append(f"Block {b['block']} ({b['application']}): {b['size']}, camera binning {b['camera_binning']}")
        p = self.form.value()
        factors = r.proposed_factors(p.spatial_binning, p.target_camera_binning)
        out = next(iter(r.output_shape(factors)))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Spatial binning needed")
        why = ("The blocks of this experiment were recorded at different resolutions:"
               if r.resolution_mismatch else
               "These recordings have less camera binning than the common grid:")
        box.setText(why + "\n\n" + "\n".join(lines)
                    + f"\n\nProposal: bin so every block is {out[0]}×{out[1]} "
                    f"(spatial factors {', '.join(f'block {k}: {v}' for k, v in factors.items())}).")
        box.setInformativeText("You can change the factor of each block in the scan table.")
        use = box.addButton("Use proposal", QMessageBox.AcceptRole)
        box.addButton("Edit factors", QMessageBox.RejectRole)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        if box.clickedButton() is not use:
            self.blocks.setFocus()

    def _reset_factors(self):
        if self.report is not None:
            self._fill_blocks(keep_factors=False)

    def _fill_blocks(self, keep_factors: bool = False):
        r = self.report
        old = {b: box.value() for b, box in self._factor_boxes.items()} if keep_factors else {}
        p = self.form.value()
        proposed = r.proposed_factors(p.spatial_binning, p.target_camera_binning)
        summary = r.block_summary()
        self.blocks.setRowCount(len(summary))
        self._factor_boxes = {}
        for row, (_, b) in enumerate(summary.iterrows()):
            cond = b["application"] + (f" ({b['phase']})" if b["phase"] else "")
            vals = [b["block"], cond, b["frames"], "" if b["expected"] is None or b["expected"] != b["expected"]
                    else int(b["expected"]), b["size"], b["camera_binning"]]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 2 and vals[3] != "" and int(vals[3]) != int(b["frames"]):
                    it.setForeground(Qt.red)
                self.blocks.setItem(row, c, it)
            spin = QSpinBox()
            spin.setRange(1, 32)
            spin.setValue(old.get(int(b["block"]), proposed[int(b["block"])]))
            spin.valueChanged.connect(self._update_output_sizes)
            self._factor_boxes[int(b["block"])] = spin
            self.blocks.setCellWidget(row, 6, spin)
        self._update_output_sizes()

    def block_factors(self) -> dict[int, int]:
        return {b: s.value() for b, s in self._factor_boxes.items()}

    def _update_output_sizes(self):
        if self.report is None:
            return
        f = self.block_factors()
        sizes = {}
        for fs in self.report.files:
            k = fs.entry.block
            if k in f:
                sizes.setdefault(k, (fs.height // f[k], fs.width // f[k]))
        consistent = len(set(sizes.values())) <= 1
        for row in range(self.blocks.rowCount()):
            k = int(self.blocks.item(row, 0).text())
            h, w = sizes.get(k, (0, 0))
            it = QTableWidgetItem(f"{h}x{w}")
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            if not consistent:
                it.setForeground(Qt.red)
            self.blocks.setItem(row, 7, it)
        self.btn_convert.setToolTip("" if consistent else "Output sizes differ: adjust the spatial factors")

    # ---- convert ----------------------------------------------------------------
    def convert_params(self) -> ConvertParams:
        p = self.form.value()
        p.block_spatial_factors = self.block_factors() or None
        return p

    def convert(self, overwrite: bool | None = None):
        if self.meta is None:
            QMessageBox.information(self, "Convert", "Add a folder or load an Excel sheet first.")
            return
        self.apply_table()
        if self.report is None:
            QMessageBox.information(self, "Convert", "Scan the files first.")
            return
        params = self.convert_params()
        sizes = self.report.output_shape(params.factors_for(self.report))
        if len(sizes) > 1:
            QMessageBox.warning(self, "Convert", f"The spatial factors give different frame sizes: {sorted(sizes)}")
            return
        out_dir = Path(self.ed_out.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        settings().setValue("out_dir", str(out_dir))
        out = out_dir / f"{self.meta.name}.mpsca.zarr"
        if out.exists() and overwrite is None:
            ans = QMessageBox.question(self, "Overwrite?", f"{out} exists. Replace it?")
            if ans != QMessageBox.Yes:
                return
            overwrite = True
        self.runner.run(convert_experiment, self.meta, out, params, overwrite=bool(overwrite),
                        report=self.report, done=lambda store: self.session.open_store(store.path),
                        message=f"Converting into {out.name}…", lock=self._buttons)
