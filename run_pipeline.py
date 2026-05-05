"""
MPS Headless Pipeline — main entry point.

Usage:
    python run_pipeline.py \\
        --data_dir   /mnt/z/ephacoffice/DColameo/Ca_Anand_AllData \\
        --output_dir /mnt/z/ephacoffice/DColameo/Ca_Anand_Processed \\
        --mps_root   /mnt/c/Users/DColameo/Documents/dev/MPS_1.0.0 \\
        --config     configs/default_config.json \\
        --workers    4 \\
        --resume

Data structure expected (default --mode experiment):
    data_dir/
      ExperimentDir/           ← one pipeline run; ROIs shared across conditions
        1_1/                   ← recording condition (same FOV)
          chunk001.avi ...
        1_2/                   ← another condition
        40X/                   ← ignored automatically (reference images)

Arguments:
    --data_dir    Root folder containing experiment directories.
    --output_dir  Where processed results are written.
    --mps_root    Path to the MPS_1.0.0 source directory.
    --config      JSON config file (default: configs/default_config.json).
    --workers     Number of experiments to process in parallel (default: 1).
    --resume      Skip steps whose output zarr files already exist.
    --dry_run     Print discovered experiments and exit without processing.
    --experiment  Only process a specific experiment ID.
    --fps         Override frame rate (Hz). Default: parsed from folder name
                  or 4 Hz (250ms exposure).
    --mode        'experiment' (default) or 'session' (legacy per-AVI-dir mode).
    --log_level   DEBUG, INFO, WARNING (default: INFO).

Notes:
    • Each experiment runs in its own subprocess (isolated Dask cluster, memory).
    • With --workers N, up to N experiments run in parallel.
    • On a 30-core / 1 TB workstation, use --workers 4 and step1.n_workers=8.
    • AVI files must exist before running this script (see convert_to_avi.py).
    • In experiment mode, a merged_input/ staging directory is created inside
      each experiment's output folder — this requires no extra storage beyond
      the symlinks themselves.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = _HERE / "configs" / "default_config.json"
DEFAULT_MPS_ROOT = "/mnt/c/Users/DColameo/Documents/dev/MPS_1.0.0"


def _setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig strips Windows BOM
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _merge_configs(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


# ── Worker functions (run in subprocesses) ────────────────────────────────────

def _run_experiment(group_data: dict, config: dict, mps_root: str, resume: bool) -> tuple:
    """Worker for ProcessPoolExecutor — runs one ExperimentGroup."""
    from pipeline.session_discovery import ExperimentGroup, RecordingCondition
    from pipeline.runner import ExperimentRunner

    # Reconstruct dataclasses from plain dicts (subprocess pickling)
    conditions = [RecordingCondition(**c) for c in group_data.pop("conditions")]
    group = ExperimentGroup(**group_data, conditions=conditions)

    runner = ExperimentRunner(group=group, config=config, mps_root=mps_root, resume=resume)
    success = runner.run()
    return group.label, success


def _run_session(session_data: dict, config: dict, mps_root: str, resume: bool) -> tuple:
    """Worker for ProcessPoolExecutor — runs one VideoSession (legacy mode)."""
    from pipeline.session_discovery import VideoSession
    from pipeline.runner import SessionRunner

    session = VideoSession(**session_data)
    runner = SessionRunner(session=session, config=config, mps_root=mps_root, resume=resume)
    success = runner.run()
    return session.label, success


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MPS headless pipeline — batch ROI extraction from calcium imaging data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mps_root",   default=DEFAULT_MPS_ROOT)
    parser.add_argument("--config",     default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers",    type=int, default=1)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--experiment", default=None,
                        help="Process only this experiment ID (experiment mode).")
    parser.add_argument("--session",    default=None,
                        help="Process only this session label (session mode).")
    parser.add_argument("--fps",        type=float, default=None,
                        help="Override FPS for all experiments (default: parse from folder name).")
    parser.add_argument("--avi_dir",    default=None,
                        help="Directory where converted AVIs live, if different from --data_dir. "
                             "Pass the same value you used for convert_to_avi.py --output_dir.")
    parser.add_argument("--mode",       default="experiment",
                        choices=["experiment", "session"],
                        help="Discovery mode (default: experiment).")
    parser.add_argument("--pattern",    default=r".*\.avi$",
                        help="AVI regex pattern for session mode.")
    parser.add_argument("--log_level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    _setup_logging(args.log_level)
    log = logging.getLogger("run_pipeline")

    for p, name in [(args.data_dir, "data_dir"), (args.mps_root, "mps_root")]:
        if not os.path.isdir(p):
            log.error(f"--{name} does not exist: {p}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    base_cfg = _load_config(str(DEFAULT_CONFIG))
    if args.config != str(DEFAULT_CONFIG) and os.path.isfile(args.config):
        config = _merge_configs(base_cfg, _load_config(args.config))
        log.info(f"Config merged: base + {args.config}")
    else:
        config = base_cfg

    # ── Discover ──────────────────────────────────────────────────────────────

    if args.mode == "experiment":
        from pipeline.session_discovery import (
            discover_experiment_groups, completed_steps_experiment,
        )

        log.info(f"Scanning for experiments in: {args.data_dir}")
        groups = discover_experiment_groups(
            root_dir=args.data_dir,
            output_root=args.output_dir,
            default_fps=args.fps or 4.0,
            avi_root=args.avi_dir,
        )

        if args.fps is not None:
            for g in groups:
                g.fps = args.fps

        if not groups:
            log.warning("No experiments found. Run explore_data.py to inspect the directory.")
            sys.exit(0)

        if args.experiment:
            groups = [g for g in groups if g.label == args.experiment]
            if not groups:
                log.error(f"Experiment '{args.experiment}' not found.")
                sys.exit(1)

        log.info(f"Found {len(groups)} experiment(s):")
        for g in groups:
            done = completed_steps_experiment(g)
            conds = ", ".join(c.condition_name for c in g.conditions)
            log.info(
                f"  {g.label}  ({len(done)}/14 steps done)  "
                f"conditions=[{conds}]  fps={g.fps}"
            )

        if args.dry_run:
            log.info("--dry_run: exiting without processing.")
            sys.exit(0)

        # Serialize for subprocess pickling
        def _group_to_dict(g):
            return dict(
                experiment_dir=g.experiment_dir,
                output_root=g.output_root,
                experiment_id=g.experiment_id,
                fps=g.fps,
                conditions=[
                    dict(
                        directory=c.directory,
                        condition_name=c.condition_name,
                        tif_files=c.tif_files,
                        avi_dir=c.avi_dir,
                    )
                    for c in g.conditions
                ],
            )

        items = [(g.label, _group_to_dict(g)) for g in groups]
        worker_fn = _run_experiment

    else:
        # Legacy session mode
        from pipeline.session_discovery import discover_sessions, completed_steps

        log.info(f"Scanning for AVI sessions in: {args.data_dir}")
        sessions = discover_sessions(
            root_dir=args.data_dir,
            output_root=args.output_dir,
            video_pattern=args.pattern,
        )

        if not sessions:
            log.warning("No sessions found.")
            sys.exit(0)

        if args.session:
            sessions = [s for s in sessions if s.label == args.session]
            if not sessions:
                log.error(f"Session '{args.session}' not found.")
                sys.exit(1)

        log.info(f"Found {len(sessions)} session(s):")
        for s in sessions:
            done = completed_steps(s)
            log.info(f"  {s.label}  ({len(done)}/14 steps done)  {s.input_dir}")

        if args.dry_run:
            log.info("--dry_run: exiting without processing.")
            sys.exit(0)

        def _session_to_dict(s):
            return dict(
                input_dir=s.input_dir,
                output_root=s.output_root,
                animal_id=s.animal_id,
                session_id=s.session_id,
                video_files=s.video_files,
            )

        items = [(s.label, _session_to_dict(s)) for s in sessions]
        worker_fn = _run_session

    # ── Run ───────────────────────────────────────────────────────────────────

    results = {}
    n_workers = min(args.workers, len(items))

    if n_workers == 1:
        for label, data in items:
            _, ok = worker_fn(data, config, args.mps_root, args.resume)
            results[label] = ok
    else:
        log.info(f"Processing {len(items)} item(s) with {n_workers} parallel workers")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(worker_fn, data, config, args.mps_root, args.resume): label
                for label, data in items
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    _, ok = future.result()
                    results[label] = ok
                except Exception as exc:
                    log.error(f"{label}: {exc}")
                    results[label] = False

    # ── Summary ───────────────────────────────────────────────────────────────

    n_ok  = sum(1 for ok in results.values() if ok)
    n_err = len(results) - n_ok
    log.info("=" * 60)
    log.info(f"Pipeline complete: {n_ok} succeeded, {n_err} failed")
    for label, ok in sorted(results.items()):
        log.info(f"  {label}: {'OK' if ok else 'FAILED'}")
    log.info("=" * 60)

    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    main()
