"""Recordings: which files belong together and how they are described.

A *recording* is one video, possibly split over many files and made of several
condition *blocks* (e.g. one block per solution). Recordings are found by searching an
input folder for a file pattern; an optional Excel workbook describes them.

Nothing about the experiments is hard-coded. The workbook needs a file-name column and,
when several recordings share file names, a column naming each recording's folder
(or sheets named like the folders). Every other column is free description.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileEntry:
    relpath: str                        # relative to the input folder, "/"-separated
    block: int = 0                      # condition block index within the recording
    condition: str = ""                 # block label shown in plots, e.g. "20mM Glucose · Synchronised"
    use_for_seg: bool = True            # frames of this file enter the segmentation projections
    include: bool = True                # False: listed but not part of the recording
    expected_frames: int | None = None  # source frames listed for the whole block
    info: dict[str, str] = field(default_factory=dict)  # free description of this file's block
    note: str = ""                      # why a file was excluded, etc.
    path: Path | None = None            # resolved absolute path (not saved)

    @property
    def filename(self) -> str:
        return self.relpath.rsplit("/", 1)[-1]

    @property
    def folder(self) -> str:
        return self.relpath.rsplit("/", 1)[0] if "/" in self.relpath else ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("relpath", "block", "condition", "use_for_seg", "include",
                                              "expected_frames", "info", "note")}

    @classmethod
    def from_dict(cls, d: dict) -> "FileEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "path"})


@dataclass
class Recording:
    name: str
    folder: str = ""                    # recording folder relative to the input folder
    files: list[FileEntry] = field(default_factory=list)
    description: dict[str, str] = field(default_factory=dict)
    include: bool = True
    block_factors: dict[int, int] | None = None  # spatial binning per block chosen by the user

    @property
    def active_files(self) -> list[FileEntry]:
        return [f for f in self.files if f.include]

    @property
    def missing(self) -> list[str]:
        return [f.relpath for f in self.active_files if f.path is None]

    @property
    def n_blocks(self) -> int:
        return len({f.block for f in self.active_files})

    def conditions(self) -> dict[int, str]:
        out = {}
        for f in self.active_files:
            out.setdefault(f.block, f.condition)
        return out

    def to_dict(self) -> dict:
        return {"name": self.name, "folder": self.folder, "include": self.include,
                "description": self.description,
                "block_factors": {str(k): v for k, v in self.block_factors.items()} if self.block_factors else None,
                "files": [f.to_dict() for f in self.files]}

    @classmethod
    def from_dict(cls, d: dict) -> "Recording":
        bf = d.get("block_factors")
        return cls(name=d["name"], folder=d.get("folder", ""), include=d.get("include", True),
                   description=dict(d.get("description") or {}),
                   block_factors={int(k): int(v) for k, v in bf.items()} if bf else None,
                   files=[FileEntry.from_dict(f) for f in d.get("files", [])])


# ---- discovery ----------------------------------------------------------------
GROUPINGS = {
    "folder": "Each top-level folder is one recording",
    "root": "The whole input folder is one recording",
    "acquisition": "Each acquisition is its own recording",
}
_TIFF_EXT = re.compile(r"(\.ome)?\.tiff?$", re.IGNORECASE)
# Micro-Manager splits long acquisitions: X_MMStack_Pos.ome.tif, X_MMStack_Pos_1.ome.tif, ...
_MM_PART = re.compile(r"^(.*_MMStack_[^_]*)(_\d+)?$")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def file_stem(name: str) -> str:
    """File name without directory and (.ome).tif(f) / .ome extension, lower-case."""
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = _TIFF_EXT.sub("", name)
    if name.lower().endswith(".ome"):
        name = name[:-4]
    return name.lower()


def acquisition_key(relpath: str) -> str:
    """Files of one split acquisition share this key (folder + base name)."""
    folder = relpath.rsplit("/", 1)[0] if "/" in relpath else ""
    stem = _TIFF_EXT.sub("", relpath.rsplit("/", 1)[-1])
    m = _MM_PART.match(stem)
    return f"{folder}/{m.group(1) if m else stem}"


def find_files(root: str | Path, pattern: str = "*.ome.tif") -> list[str]:
    """Relative paths of files under ``root`` matching ``pattern`` (case-insensitive).

    Several patterns can be given separated by ``;``. Paths are sorted naturally so split
    files stay in acquisition order (``x``, ``x_1``, ..., ``x_10``).
    """
    root = Path(root)
    pats = [p.strip().lower() for p in pattern.split(";") if p.strip()] or ["*"]
    out = []
    for p in root.rglob("*"):
        if p.is_file() and any(fnmatch.fnmatch(p.name.lower(), pat) for pat in pats):
            out.append(p.relative_to(root).as_posix())
    return sorted(out, key=natural_key)


def _blocks_by_acquisition(files: list[str], rec_folder: str) -> list[FileEntry]:
    entries, keys = [], {}
    for rel in files:
        key = acquisition_key(rel)
        block = keys.setdefault(key, len(keys))
        folder = rel.rsplit("/", 1)[0] if "/" in rel else ""
        sub = folder[len(rec_folder):].strip("/") if folder.startswith(rec_folder) else folder
        label = sub or key.rsplit("/", 1)[-1]
        entries.append(FileEntry(relpath=rel, block=block, condition=label))
    return entries


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w.+-]+", "_", s).strip("_") or "recording"


def group_recordings(root: str | Path, files: list[str], grouping: str = "folder") -> list[Recording]:
    """Group files into recordings without a workbook; each acquisition is one block."""
    root = Path(root)
    groups: dict[str, list[str]] = {}
    for rel in files:
        if grouping == "root":
            key = ""
        elif grouping == "acquisition":
            key = acquisition_key(rel)
        elif grouping == "folder":
            key = rel.split("/", 1)[0] if "/" in rel else ""
        else:
            raise ValueError(f"unknown grouping {grouping!r}; use one of {list(GROUPINGS)}")
        groups.setdefault(key, []).append(rel)
    recs = []
    for key, rels in groups.items():
        if grouping == "acquisition":
            folder = key.rsplit("/", 1)[0]
            name = key.replace("/", "_").strip("_")
        else:
            folder = key
            name = key or root.name
        recs.append(Recording(name=_safe_name(name), folder=folder,
                              files=_blocks_by_acquisition(rels, folder)))
    _make_unique(recs)
    return recs


def _make_unique(recs: list[Recording]) -> None:
    seen: dict[str, int] = {}
    for r in recs:
        n = seen.get(r.name, 0)
        seen[r.name] = n + 1
        if n:
            r.name = f"{r.name}_{n + 1}"


# ---- Excel ----------------------------------------------------------------------
# role -> accepted header spellings (lower-case, text before any "(")
_ROLES = {
    "recording": ("main folder", "recording", "experiment", "experiment folder", "folder"),
    "subfolder": ("subfolder", "sub folder", "sub-folder"),
    "file": ("file name", "filename", "file"),
    "frames": ("frames", "n frames", "frame count", "number of frames"),
    "segmentation": ("segmentation", "use for segmentation", "seg"),
}


@dataclass
class SheetRow:
    sheet: str
    recording: str | None
    subfolder: str | None
    file: str
    block: int
    use_for_seg: bool | None
    frames: int | None
    info: dict[str, str]


@dataclass
class Workbook:
    rows: list[SheetRow]
    columns: list[str]                  # description columns, in sheet order
    issues: list[str] = field(default_factory=list)


def _cell_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return s or None


def _parse_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _parse_bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "x", "ja"):
        return True
    if s in ("false", "no", "n", "0", "nein", ""):
        return False
    return None


def _map_headers(header_row) -> tuple[dict[str, int], dict[str, int]]:
    roles, desc = {}, {}
    for i, h in enumerate(header_row):
        text = _cell_text(h)
        if not text:
            continue
        key = text.split("(")[0].strip().lower()
        role = next((r for r, names in _ROLES.items() if key in names and r not in roles), None)
        if role:
            roles[role] = i
        else:
            desc[text] = i
    return roles, desc


def read_workbook(xlsx_path: str | Path) -> Workbook:
    """Read every sheet that has a file-name column.

    Description cells are filled downwards until the next non-empty cell; a row with any
    description, subfolder or frame-count cell filled starts a new block, a filled
    recording cell a new recording. The segmentation flag is per file.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    rows, columns, issues = [], [], []
    for sheet in wb.sheetnames:
        table = list(wb[sheet].iter_rows(values_only=True))
        if not table:
            continue
        roles, desc = _map_headers(table[0])
        if "file" not in roles:
            issues.append(f"sheet {sheet!r} skipped: no file-name column")
            continue
        for c in desc:
            if c not in columns:
                columns.append(c)

        def cell(row, i):
            return _cell_text(row[i]) if i is not None and i < len(row) else None

        recording = sub = frames = None
        info: dict[str, str] = {}
        block = -1
        for row in table[1:]:
            rec = cell(row, roles.get("recording"))
            if rec and rec != recording:
                recording, info, sub, frames, block = rec, {}, None, None, -1
            new_desc = {c: cell(row, i) for c, i in desc.items() if cell(row, i) is not None}
            s = cell(row, roles.get("subfolder"))
            fr = cell(row, roles.get("frames"))
            if new_desc or s is not None or fr is not None:
                block += 1
            block = max(block, 0)
            info.update(new_desc)
            if s is not None:
                sub = s
            if fr is not None:
                frames = _parse_int(fr)
            fname = cell(row, roles.get("file"))
            if not fname:
                continue
            rows.append(SheetRow(sheet=sheet, recording=recording, subfolder=sub, file=fname,
                                 block=block, frames=frames, info=dict(info),
                                 use_for_seg=_parse_bool(row[roles["segmentation"]])
                                 if "segmentation" in roles and roles["segmentation"] < len(row) else None))
    wb.close()
    return Workbook(rows, columns, issues)


def condition_label(info: dict[str, str], columns: list[str]) -> str:
    return " · ".join(info[c] for c in columns if info.get(c))


def default_condition_columns(wb: Workbook) -> list[str]:
    """Description columns whose value changes between blocks of some recording."""
    out = []
    for c in wb.columns:
        per_rec: dict[tuple, set] = {}
        for r in wb.rows:
            per_rec.setdefault((r.sheet, r.recording), set()).add(r.info.get(c))
        if any(len(v) > 1 for v in per_rec.values()):
            out.append(c)
    return out or wb.columns[:1]


def _find_folder(name: str, files: list[str]) -> str | None:
    """Shallowest folder called ``name`` (case-insensitive) that contains matching files."""
    target = name.strip().lower()
    best = None
    for rel in files:
        parts = rel.split("/")[:-1]
        for depth, part in enumerate(parts):
            if part.lower() == target:
                folder = "/".join(parts[:depth + 1])
                if best is None or folder.count("/") < best.count("/"):
                    best = folder
                break
    return best


def match_workbook(root: str | Path, files: list[str], wb: Workbook,
                   condition_columns: list[str] | None = None) -> tuple[list[Recording], list[str]]:
    """Build recordings from the workbook rows, matched against the files found on disk.

    Returns the recordings and a list of problems (missing files, unmatched folders).
    Files on disk that the workbook does not list stay in the recording, excluded.
    """
    issues = list(wb.issues)
    cols = condition_columns if condition_columns is not None else default_condition_columns(wb)
    grouped: dict[tuple[str, str], list[SheetRow]] = {}
    for r in wb.rows:
        grouped.setdefault((r.sheet, r.recording or r.sheet), []).append(r)
    several = len(grouped) > 1
    recs = []
    for (sheet, rec_name), rows in grouped.items():
        folder = _find_folder(rec_name, files)
        if folder is None and not several:
            folder = ""  # a single recording: search the whole input folder
        if folder is None:
            issues.append(f"{rec_name} (sheet {sheet!r}): no folder with this name under the input folder")
            rec = Recording(name=_safe_name(rec_name), folder="", include=False,
                            description={"sheet": sheet})
            for r in rows:
                rec.files.append(FileEntry(relpath=r.file, block=r.block, info=r.info,
                                           condition=condition_label(r.info, cols),
                                           use_for_seg=r.use_for_seg if r.use_for_seg is not None else True,
                                           expected_frames=r.frames, note="folder not found"))
            recs.append(rec)
            continue
        prefix = f"{folder}/" if folder else ""
        here = [f for f in files if f.startswith(prefix)]
        by_stem: dict[str, list[str]] = {}
        for f in here:
            by_stem.setdefault(file_stem(f), []).append(f)
        used, entries = set(), []
        for r in rows:
            cands = [c for c in by_stem.get(file_stem(r.file), []) if c not in used]
            if len(cands) > 1 and r.subfolder:
                in_sub = [c for c in cands if c.rsplit("/", 2)[-2:-1] == [r.subfolder]]
                cands = in_sub or cands
            if len(cands) > 1:
                issues.append(f"{rec_name}: {r.file} matches {len(cands)} files, using {cands[0]}")
            rel = cands[0] if cands else f"{prefix}{r.file}"
            if not cands:
                issues.append(f"{rec_name}: {r.file} not found")
            used.add(rel)
            entries.append(FileEntry(relpath=rel, block=r.block, info=r.info,
                                     condition=condition_label(r.info, cols),
                                     use_for_seg=r.use_for_seg if r.use_for_seg is not None else True,
                                     expected_frames=r.frames))
        for f in here:
            if f not in used:
                entries.append(FileEntry(relpath=f, block=max((e.block for e in entries), default=-1) + 1,
                                         condition="(not in the workbook)", include=False,
                                         use_for_seg=False, note="not listed in the workbook"))
        description = {"sheet": sheet} if sheet != rec_name else {}
        # columns with one value for the whole recording also describe the recording
        for c in wb.columns:
            vals = {r.info.get(c) for r in rows}
            if len(vals) == 1 and None not in vals:
                description[c] = vals.pop()
        recs.append(Recording(name=_safe_name(rec_name), folder=folder, files=entries,
                              description=description))
    _make_unique(recs)
    return recs, issues


def resolve(root: str | Path, recs: list[Recording]) -> None:
    """Set ``FileEntry.path`` for every file that exists under ``root``."""
    root = Path(root)
    for rec in recs:
        for f in rec.files:
            p = root / f.relpath
            f.path = p if p.is_file() else None
