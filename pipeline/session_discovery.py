"""
Discovers video sessions and experiment groups from a root data directory.

Two models are supported:

  VideoSession (legacy)
    One directory containing AVI files = one pipeline run.

  ExperimentGroup (primary)
    One parent directory containing subdirectory recordings of the SAME FOV.
    All recording conditions are merged into a single virtual session so that
    MPS extracts ROIs consistently across all conditions.

    Expected structure:
        root_dir/
          ExperimentDir/            ← one ExperimentGroup
            1_1/                    ← RecordingCondition (same cells)
              chunk001.tif
              chunk002.tif
            1_2/                    ← RecordingCondition
              chunk001.tif
            40X/                    ← EXCLUDED (reference images)
            40X BF/                 ← EXCLUDED (brightfield reference)
"""
import os
import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Exclusion patterns ────────────────────────────────────────────────────────
# Directories whose names match are skipped during discovery.
_EXCLUDE_DIR_RE = re.compile(r'^(40[Xx]|\.)', re.IGNORECASE)


def _is_excluded_dir(name: str) -> bool:
    """Return True for reference-image dirs (40X, 40X BF) and hidden dirs."""
    return bool(_EXCLUDE_DIR_RE.match(name))


# ── Natural sort helper ───────────────────────────────────────────────────────

def _sort_key_natural(name: str) -> list:
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ── FPS parsing ───────────────────────────────────────────────────────────────

def _parse_fps_from_name(name: str) -> Optional[float]:
    """
    Parse FPS from a directory or file name.

    Examples:
      '250ms_Exp' → 4.0    (1000 / 250)
      '500ms'    → 2.0
      '30fps'    → 30.0
    """
    m = re.search(r'(\d+)ms', name, re.IGNORECASE)
    if m:
        interval_ms = float(m.group(1))
        return round(1000.0 / interval_ms, 2)
    m = re.search(r'(\d+)\s*fps', name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _safe_id(name: str) -> str:
    """Convert an arbitrary directory name into a safe identifier."""
    return re.sub(r'[^\w\-]', '_', name)


# ── ExperimentGroup model ─────────────────────────────────────────────────────

@dataclass
class RecordingCondition:
    """One subfolder within an experiment — all TIF chunks from one recording."""
    directory: str
    condition_name: str       # basename, e.g. "1_1"
    tif_files: List[str]      # sorted TIF chunk paths
    avi_dir: Optional[str] = None   # if set, AVIs live here instead of alongside TIFs

    @property
    def avi_files(self) -> List[str]:
        """
        AVI paths for this condition.

        When avi_dir is set (separate output directory), AVIs are looked for
        there with the same basenames as the TIFs.  Otherwise they are expected
        alongside the source TIFs (same directory, .avi extension).
        """
        base = self.avi_dir or self.directory
        return [
            os.path.join(base, Path(os.path.basename(t)).stem + ".avi")
            for t in self.tif_files
        ]

    @property
    def n_tif_chunks(self) -> int:
        return len(self.tif_files)

    def __repr__(self) -> str:
        return f"RecordingCondition({self.condition_name!r}, {self.n_tif_chunks} chunks)"


@dataclass
class ExperimentGroup:
    """
    One experiment = one FOV imaged across multiple recording conditions.

    The experiment_dir (e.g. '05_11_25_Min6_C23_KO_250ms_Exp1') contains
    subdirectories (1_1, 1_2, ...) that are all recordings of the SAME CELLS
    under different conditions (baseline, drug, washout, ...).

    ROI extraction runs once on all conditions concatenated via a merged
    staging directory → one A matrix shared by every condition.
    """
    experiment_dir: str
    output_root: str
    experiment_id: str              # safe name derived from experiment_dir basename
    conditions: List[RecordingCondition]
    fps: float = 4.0

    # ── Interface matching VideoSession (so SessionRunner duck-types) ──────────

    @property
    def label(self) -> str:
        return self.experiment_id

    @property
    def dataset_output_path(self) -> str:
        return os.path.join(self.output_root, self.experiment_id)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.dataset_output_path, "cache_data")

    @property
    def log_path(self) -> str:
        return os.path.join(self.dataset_output_path, "pipeline.log")

    @property
    def merged_dir(self) -> str:
        """Staging directory — symlinks to all AVIs across all conditions."""
        return os.path.join(self.dataset_output_path, "merged_input")

    @property
    def all_avi_files(self) -> List[str]:
        """All AVI paths across all conditions, in recording order."""
        files = []
        for cond in self.conditions:
            files.extend(cond.avi_files)
        return files

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)

    @property
    def n_total_chunks(self) -> int:
        return sum(c.n_tif_chunks for c in self.conditions)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup_merged_dir(self) -> str:
        """
        Populate merged_input/ with sequentially numbered symlinks that point
        to all AVI files across all conditions (in recording order).

        MPS step2a scans a single directory for AVIs; this lets it treat all
        conditions as one continuous session for ROI extraction.

        Also writes chunk_manifest.json — maps each chunk filename back to its
        source condition and original AVI path, so frames can be traced back
        to conditions after analysis.

        Falls back to copies when symlinks are unsupported (e.g. drvfs mounts).

        Returns the merged directory path.
        """
        import json
        import shutil
        os.makedirs(self.merged_dir, exist_ok=True)
        all_avis = self.all_avi_files

        manifest = []
        for i, avi_path in enumerate(all_avis):
            link_name = f"chunk_{i:06d}.avi"
            link_path = os.path.join(self.merged_dir, link_name)

            # Find which condition this AVI belongs to
            condition_name = None
            for cond in self.conditions:
                if avi_path in cond.avi_files:
                    condition_name = cond.condition_name
                    break

            manifest.append({
                "chunk":          link_name,
                "chunk_index":    i,
                "condition":      condition_name,
                "source_avi":     avi_path,
                "source_tif":     self._avi_to_tif(avi_path),
            })

            if os.path.exists(link_path) or os.path.islink(link_path):
                continue
            if not os.path.exists(avi_path):
                logger.warning(f"AVI not found (convert first?): {avi_path}")
                continue
            try:
                os.symlink(avi_path, link_path)
            except OSError:
                shutil.copy2(avi_path, link_path)
                logger.debug(f"Symlink unsupported; copied {os.path.basename(avi_path)}")

        # Write manifest even if links already existed
        manifest_path = os.path.join(self.merged_dir, "chunk_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        n_links = len([f for f in os.listdir(self.merged_dir) if f.endswith(".avi")])
        logger.info(f"Merged dir ready: {self.merged_dir}  ({n_links} AVI files)")
        logger.info(f"Chunk manifest:   {manifest_path}")
        return self.merged_dir

    def _avi_to_tif(self, avi_path: str) -> str:
        """Best-guess reverse mapping from AVI path to source TIF path."""
        for cond in self.conditions:
            for tif, avi in zip(cond.tif_files, cond.avi_files):
                if avi == avi_path:
                    return tif
        return ""

    def __repr__(self) -> str:
        return (f"ExperimentGroup(id={self.experiment_id!r}, "
                f"conditions={self.n_conditions}, chunks={self.n_total_chunks}, fps={self.fps})")


# ── ExperimentGroup discovery ─────────────────────────────────────────────────

def discover_experiment_groups(
    root_dir: str,
    output_root: str,
    tif_extensions: Optional[set] = None,
    default_fps: float = 4.0,
    avi_root: Optional[str] = None,
) -> List[ExperimentGroup]:
    """
    Walk root_dir and return one ExperimentGroup per experiment directory.

    An "experiment directory" is any directory whose subdirectories contain
    TIF files.  Subdirectories matching _EXCLUDE_DIR_RE (40X, hidden) are
    pruned from the walk entirely.

    FPS is parsed from the experiment directory name when possible
    (e.g. '250ms_Exp' → 4 Hz), falling back to *default_fps*.

    Args:
        root_dir:       Root folder containing experiment directories.
        output_root:    Base directory for pipeline outputs.
        tif_extensions: File suffixes to treat as TIF source files.
        default_fps:    Frame rate assumed when not parseable from directory name.
        avi_root:       If the AVIs were converted into a separate directory
                        (e.g. with ``convert_to_avi.py --output_dir``), pass that
                        directory here.  The folder structure is assumed to mirror
                        root_dir.  When None, AVIs are expected alongside the TIFs.

    Returns:
        Sorted list of ExperimentGroup objects.
    """
    if tif_extensions is None:
        tif_extensions = {".tif", ".tiff"}

    # Pass 1: walk, prune excluded dirs, collect TIF-containing dirs
    dirs_with_tifs: Dict[str, List[str]] = {}
    for dirpath, subdirs, filenames in os.walk(root_dir, topdown=True):
        subdirs[:] = sorted(d for d in subdirs if not _is_excluded_dir(d))
        tifs = sorted(
            os.path.join(dirpath, f)
            for f in filenames
            if Path(f).suffix.lower() in tif_extensions
        )
        if tifs:
            dirs_with_tifs[dirpath] = tifs

    if not dirs_with_tifs:
        logger.warning(f"No TIF files found under {root_dir}")
        return []

    # Pass 2: group by parent directory (parent = experiment dir)
    parent_to_conditions: Dict[str, Dict[str, Tuple]] = defaultdict(dict)
    for dirpath, tifs in dirs_with_tifs.items():
        parent = str(Path(dirpath).parent)
        condition_name = os.path.basename(dirpath)
        parent_to_conditions[parent][condition_name] = (dirpath, tifs)

    # Pass 3: build ExperimentGroup objects
    groups: List[ExperimentGroup] = []
    for parent_dir, conditions_dict in sorted(parent_to_conditions.items()):
        sorted_conditions = sorted(
            conditions_dict.items(),
            key=lambda kv: _sort_key_natural(kv[0]),
        )

        conditions = []
        for cname, (cdir, tifs) in sorted_conditions:
            avi_dir = None
            if avi_root:
                rel = os.path.relpath(cdir, root_dir)
                avi_dir = os.path.join(avi_root, rel)
            conditions.append(RecordingCondition(
                directory=cdir,
                condition_name=cname,
                tif_files=tifs,
                avi_dir=avi_dir,
            ))

        fps = _parse_fps_from_name(os.path.basename(parent_dir)) or default_fps
        exp_id = _safe_id(os.path.basename(parent_dir))

        group = ExperimentGroup(
            experiment_dir=parent_dir,
            output_root=output_root,
            experiment_id=exp_id,
            conditions=conditions,
            fps=fps,
        )
        groups.append(group)
        logger.info(f"Discovered {group}")

    logger.info(f"Total experiments found: {len(groups)}")
    return groups


def completed_steps_experiment(group: ExperimentGroup) -> List[str]:
    """Return completed step names for an ExperimentGroup."""
    done = []
    for step, markers in _STEP_MARKERS.items():
        if any(os.path.exists(os.path.join(group.cache_dir, m)) for m in markers):
            done.append(step)
    return done


# ── VideoSession model (legacy) ───────────────────────────────────────────────

@dataclass
class VideoSession:
    """Represents one recording session ready for pipeline processing."""
    input_dir: str
    output_root: str
    animal_id: int
    session_id: int
    video_files: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"A{self.animal_id:04d}_S{self.session_id:04d}"

    @property
    def dataset_output_path(self) -> str:
        return os.path.join(self.output_root, f"{self.animal_id}_{self.session_id}_Processed")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.dataset_output_path, "cache_data")

    @property
    def log_path(self) -> str:
        return os.path.join(self.dataset_output_path, "pipeline.log")

    def __repr__(self) -> str:
        return (f"VideoSession(animal={self.animal_id}, session={self.session_id}, "
                f"n_files={len(self.video_files)}, dir={self.input_dir})")


# ── ID extraction (for VideoSession discovery) ────────────────────────────────

def _extract_ids(path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract numeric animal and session IDs from a directory path.

    Supported conventions (first match wins):
      .../1234/T5/...   → animal=1234, session=5
      .../1234_5_...    → animal=1234, session=5
      path part is a pure integer → animal candidate
      path part matches T<digits> → session candidate
    """
    parts = Path(os.path.normpath(path)).parts
    animal: Optional[int] = None
    session: Optional[int] = None

    for part in parts:
        m = re.match(r'^(\d+)[_\-](\d+)', part)
        if m and animal is None:
            animal = int(m.group(1))
            session = int(m.group(2))
            continue

        m = re.match(r'^[Tt](\d+)$', part)
        if m and session is None:
            session = int(m.group(1))
            continue

        m = re.match(r'^[a-zA-Z]+[_\-](\d+)$', part)
        if m:
            n = int(m.group(1))
            if re.match(r'^[Ss]ession', part) and session is None:
                session = n
            elif animal is None:
                animal = n
            continue

        if re.match(r'^\d{1,6}$', part) and animal is None:
            animal = int(part)

    return animal, session


def discover_sessions(
    root_dir: str,
    output_root: str,
    video_pattern: str = r".*\.avi$",
    fallback_animal: int = 1,
    fallback_session_start: int = 1,
) -> List[VideoSession]:
    """
    Walk root_dir recursively and return one VideoSession per directory that
    contains AVI files matching *video_pattern*.
    """
    pattern = re.compile(video_pattern, re.IGNORECASE)
    sessions: List[VideoSession] = []
    fallback_session = fallback_session_start
    seen: set = set()

    for dirpath, _, filenames in os.walk(root_dir):
        avi_files = sorted([
            os.path.join(dirpath, f)
            for f in filenames
            if pattern.search(f)
        ])
        if not avi_files or dirpath in seen:
            continue
        seen.add(dirpath)

        animal, session = _extract_ids(dirpath)
        if animal is None:
            animal = fallback_animal
        if session is None:
            session = fallback_session
            fallback_session += 1

        sess = VideoSession(
            input_dir=dirpath,
            output_root=output_root,
            animal_id=animal,
            session_id=session,
            video_files=avi_files,
        )
        sessions.append(sess)
        logger.info(f"Discovered {sess}")

    sessions.sort(key=lambda s: (s.animal_id, s.session_id))
    logger.info(f"Total sessions found: {len(sessions)}")
    return sessions


# ── Checkpoint helpers ────────────────────────────────────────────────────────

_STEP_MARKERS = {
    "step2a": ["step2a_varr.zarr"],
    "step2b": ["step2b_varr_ref.zarr"],
    "step2c": ["step2c_motion.zarr"],
    "step2d": ["step2d_varr_ref.zarr"],
    "step2e": ["step2e_Y_fm_chk.zarr", "step2e_Y_hw_chk.zarr"],
    "step3a": ["step3a_Y_fm_cropped.zarr"],
    "step3b": ["A_init.zarr"],
    "step4a": ["step4a_watershed_params.json"],
    "step4b": ["step4b_separated_components.zarr"],
    "step4c": ["step4c_merged_components.zarr"],
    "step4d": ["step4d_components_with_temporal.zarr"],
    "step4e": ["step4e_A.zarr", "step4e_C.zarr"],
    "step4f": ["step4f_A_clean.zarr"],
    "step4g": ["step4g_A_merged.zarr"],
}


def completed_steps(session: VideoSession) -> List[str]:
    """Return completed step names for a VideoSession."""
    done = []
    for step, markers in _STEP_MARKERS.items():
        if any(os.path.exists(os.path.join(session.cache_dir, m)) for m in markers):
            done.append(step)
    return done


def last_completed_step(session: VideoSession) -> Optional[str]:
    """Return the name of the most recently completed step, or None."""
    step_order = list(_STEP_MARKERS.keys())
    done = completed_steps(session)
    if not done:
        return None
    return max(done, key=lambda s: step_order.index(s) if s in step_order else -1)
