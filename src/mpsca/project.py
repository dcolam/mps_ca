"""A project: all recordings of a study, how they are described and how they are processed.

The project file (``mpsca_project.json`` in the output folder) keeps the input folder,
the file pattern, the optional workbook, every recording with its files and description,
the conversion parameters and the processing settings. File paths are stored relative to
the input folder, so a project still works when the data drive has another letter.

Typical use: find recordings -> scan all -> convert all -> try the processing settings on
a few recordings -> run them on all recordings.
"""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import pipeline
from .io.convert import ConvertParams, convert_recording
from .io.metadata import (Recording, find_files, group_recordings, match_workbook, read_workbook,
                          resolve)
from .io.scan import ScanReport, scan_recording
from .io.store import ExperimentStore
from .pipeline import PipelineSettings

log = logging.getLogger(__name__)
PROJECT_FILE = "mpsca_project.json"
Progress = Callable[[float], None] | None
Say = Callable[[str], None] | None


@dataclass
class Project:
    input_dir: str
    output_dir: str
    pattern: str = "*.ome.tif"
    excel: str | None = None
    grouping: str = "folder"                    # used when there is no workbook
    condition_columns: list[str] | None = None  # workbook columns that name a block; None = auto
    recordings: list[Recording] = field(default_factory=list)
    convert: ConvertParams = field(default_factory=ConvertParams)
    settings: PipelineSettings = field(default_factory=PipelineSettings)
    issues: list[str] = field(default_factory=list)

    # ---- files ----------------------------------------------------------------
    @property
    def path(self) -> Path:
        return Path(self.output_dir) / PROJECT_FILE

    def save(self) -> Path:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        conv = asdict(self.convert)
        conv["block_spatial_factors"] = None  # chosen per recording
        data = {"format": 1, "input_dir": self.input_dir, "output_dir": self.output_dir,
                "pattern": self.pattern, "excel": self.excel, "grouping": self.grouping,
                "condition_columns": self.condition_columns, "issues": self.issues,
                "convert": conv, "settings": self.settings.to_dict(),
                "recordings": [r.to_dict() for r in self.recordings]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    @classmethod
    def load(cls, path: str | Path, input_dir: str | Path | None = None) -> "Project":
        """Open a project file (or the output folder holding it).

        ``input_dir`` replaces the stored input folder, e.g. on another computer.
        """
        path = Path(path)
        if path.is_dir():
            path = path / PROJECT_FILE
        d = json.loads(path.read_text(encoding="utf-8"))
        conv = {k: v for k, v in (d.get("convert") or {}).items() if k in ConvertParams.__dataclass_fields__}
        proj = cls(input_dir=str(input_dir or d["input_dir"]), output_dir=str(path.parent),
                   pattern=d.get("pattern", "*.ome.tif"), excel=d.get("excel"),
                   grouping=d.get("grouping", "folder"), condition_columns=d.get("condition_columns"),
                   recordings=[Recording.from_dict(r) for r in d.get("recordings", [])],
                   convert=ConvertParams(**conv), settings=PipelineSettings.from_dict(d.get("settings")),
                   issues=list(d.get("issues", [])))
        proj.resolve()
        return proj

    def resolve(self) -> None:
        resolve(self.input_dir, self.recordings)

    @property
    def input_found(self) -> bool:
        return Path(self.input_dir).is_dir()

    # ---- recordings -------------------------------------------------------------
    def get(self, name: str) -> Recording:
        for r in self.recordings:
            if r.name == name:
                return r
        raise KeyError(name)

    def select(self, names=None) -> list[Recording]:
        """Included recordings, optionally only those named."""
        if names:
            wanted = set(names)
            unknown = wanted - {r.name for r in self.recordings}
            if unknown:
                raise KeyError(f"unknown recording(s): {sorted(unknown)}")
            return [r for r in self.recordings if r.name in wanted]
        return [r for r in self.recordings if r.include]

    def store_path(self, rec: Recording) -> Path:
        return Path(self.output_dir) / f"{rec.name}.mpsca.zarr"

    def description_keys(self) -> list[str]:
        keys: list[str] = []
        for r in self.recordings:
            for k in r.description:
                if k not in keys:
                    keys.append(k)
        return keys

    def status(self, rec: Recording) -> str:
        return store_status(self.store_path(rec))


def store_status(path: str | Path) -> str:
    """Last completed step of a recording's store."""
    path = Path(path)
    if not path.exists():
        return "not converted"
    try:
        store = ExperimentStore.open(path, mode="r")
    except Exception:
        return "unreadable"
    if not store.complete:
        return "incomplete"
    if store.group.attrs.get("analysis_summary"):
        return "analysed"
    if store.has("traces/dff"):
        return "traces"
    act = store.group.attrs.get("active_labels")
    if act:
        return "ROIs (edited)" if act == "edited" else "ROIs"
    if store.has("projections/mean"):
        return "projections"
    return "converted"


def find_recordings(input_dir: str | Path, pattern: str = "*.ome.tif", excel: str | Path | None = None,
                    grouping: str = "folder", condition_columns: list[str] | None = None
                    ) -> tuple[list[Recording], list[str]]:
    files = find_files(input_dir, pattern)
    if not files:
        return [], [f"no files matching {pattern!r} under {input_dir}"]
    if excel:
        recs, issues = match_workbook(input_dir, files, read_workbook(excel), condition_columns)
    else:
        recs, issues = group_recordings(input_dir, files, grouping), []
    resolve(input_dir, recs)
    return recs, issues


def create_project(input_dir: str | Path, output_dir: str | Path, pattern: str = "*.ome.tif",
                   excel: str | Path | None = None, grouping: str = "folder",
                   condition_columns: list[str] | None = None, previous: Project | None = None) -> Project:
    """Find the recordings; edits of ``previous`` (descriptions, choices) are kept by name."""
    recs, issues = find_recordings(input_dir, pattern, excel, grouping, condition_columns)
    proj = Project(input_dir=str(input_dir), output_dir=str(output_dir), pattern=pattern,
                   excel=str(excel) if excel else None, grouping=grouping,
                   condition_columns=condition_columns, recordings=recs, issues=issues)
    if previous is not None:
        proj.convert, proj.settings = previous.convert, previous.settings
        old = {r.name: r for r in previous.recordings}
        for r in recs:
            if r.name in old:
                o = old[r.name]
                r.description = {**r.description, **o.description}
                r.include, r.block_factors = o.include, o.block_factors
    return proj


# ---- batch steps ----------------------------------------------------------------
def _sub(progress: Progress, i: int, n: int) -> Progress:
    return (lambda f: progress((i + f) / n)) if progress else None


def scan_all(project: Project, recs: list[Recording] | None = None, progress: Progress = None,
             say: Say = None) -> dict[str, ScanReport]:
    """Read the headers of every file (no pixel data); the preflight before converting."""
    say = say or log.info
    recs = project.select() if recs is None else recs
    reports = {}
    for i, rec in enumerate(recs):
        say(f"[{i + 1}/{len(recs)}] scanning {rec.name}")
        reports[rec.name] = scan_recording(rec, project.convert.source_fps, progress=_sub(progress, i, len(recs)))
    project.save()  # the scan may have excluded single images
    return reports


def needs_decision(project: Project, rec: Recording, report: ScanReport) -> bool:
    p = project.convert
    return bool(report.files) and rec.block_factors is None and report.needs_decision(
        p.spatial_binning, p.target_camera_binning)


def accept_proposals(project: Project, reports: dict[str, ScanReport]) -> list[str]:
    """Store the proposed spatial factors for every recording that needs a decision."""
    p, done = project.convert, []
    for name, rep in reports.items():
        rec = project.get(name)
        if needs_decision(project, rec, rep):
            rec.block_factors = rep.proposed_factors(p.spatial_binning, p.target_camera_binning)
            done.append(name)
    project.save()
    return done


def convert_all(project: Project, recs: list[Recording] | None = None,
                reports: dict[str, ScanReport] | None = None, overwrite: bool = False,
                progress: Progress = None, say: Say = None) -> dict[str, str]:
    """Convert recordings one after the other; a failure does not stop the others.

    Recordings already converted are skipped (unless ``overwrite``); interrupted ones
    are converted again.
    """
    say = say or log.info
    recs = project.select() if recs is None else recs
    reports = dict(reports or {})
    out = {}
    for i, rec in enumerate(recs):
        path = project.store_path(rec)
        status = store_status(path)
        prefix = f"[{i + 1}/{len(recs)}] {rec.name}"
        if status not in ("not converted", "incomplete", "unreadable") and not overwrite:
            out[rec.name] = "skipped (already converted)"
            say(f"{prefix}: already converted, skipped")
            continue
        try:
            if rec.missing:
                raise ValueError(f"{len(rec.missing)} file(s) missing, e.g. {rec.missing[0]}; "
                                 "exclude them or fix the input folder")
            rep = reports.get(rec.name) or scan_recording(rec, project.convert.source_fps)
            if not rep.files:
                raise ValueError("no readable video files")
            if needs_decision(project, rec, rep):
                raise ValueError("spatial binning not confirmed (block sizes differ or are finer than "
                                 "the common grid); accept the proposal first")
            say(f"{prefix}: converting {len(rep.files)} files")
            store = convert_recording(rec, path, project.convert, overwrite=path.exists(),
                                      progress=_sub(progress, i, len(recs)), report=rep)
            out[rec.name] = f"ok, movie {tuple(store.movie.shape)}"
            say(f"{prefix}: done, movie {tuple(store.movie.shape)}")
        except Exception as e:
            log.debug("conversion failed", exc_info=True)
            out[rec.name] = f"failed: {e}"
            say(f"{prefix}: FAILED: {e}")
    project.save()
    return out


def run_all(project: Project, recs: list[Recording] | None = None, steps=pipeline.STEPS,
            progress: Progress = None, say: Say = None) -> dict[str, str]:
    """Run processing steps with the project settings on each converted recording."""
    from .segment.cellpose_seg import free_gpu

    say = say or log.info
    recs = project.select() if recs is None else recs
    out = {}
    for i, rec in enumerate(recs):
        prefix = f"[{i + 1}/{len(recs)}] {rec.name}"
        path = project.store_path(rec)
        try:
            if store_status(path) in ("not converted", "incomplete", "unreadable"):
                raise ValueError("not converted")
            store = ExperimentStore.open(path)
            store.set_meta(description=rec.description)
            say(f"{prefix}: processing")
            res = pipeline.run_steps(store, project.settings, steps, progress=_sub(progress, i, len(recs)),
                                     say=lambda m, p=prefix: say(f"{p}: {m}"))
            out[rec.name] = "ok " + ", ".join(f"{k} {v}" for k, v in res.items() if k != "export")
        except Exception as e:
            log.debug("%s", traceback.format_exc())
            out[rec.name] = f"failed: {e}"
            say(f"{prefix}: FAILED: {e}")
        finally:
            free_gpu()
    return out
