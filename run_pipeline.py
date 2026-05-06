"""
MPS Headless Pipeline — main entry point.

Usage:
    python run_pipeline.py \
        --data_dir   "Z:\\ephacoffice\\DColameo\\Ca_Anand_AllData" \
        --avi_dir    "Z:\\ephacoffice\\DColameo\\Ca_Anand_AVI" \
        --output_dir "D:\\DC_Ca-Data\\Ca_Anand_Processed" \
        --mps_root   "C:\\Users\\d.colameo\\dev\\MPS" \
        --config     configs\\workstation.json \
        --workers    28 \
        --resume

Data structure expected (default --mode experiment):
    data_dir/
      ExperimentDir/           <- one pipeline run; ROIs shared across conditions
        1_1/                   <- recording condition (same FOV)
          chunk001.tif ...
        1_2/                   <- another condition
        40X/                   <- ignored automatically (reference images)

Arguments:
    --data_dir    Root folder containing experiment directories.
    --output_dir  Where processed results are written.
    --mps_root    Path to the MPS source directory.
    --config      JSON config file (default: configs/default_config.json).
    --workers     Dask workers per experiment (overrides config step1.n_workers).
                  All cores are given to one experiment at a time.
    --resume      Skip steps whose output zarr files already exist.
    --dry_run     Print discovered experiments and exit without processing.
    --experiment  Only process a specific experiment ID.
    --fps         Override frame rate (Hz). Default: parsed from folder name.
    --mode        'experiment' (default) or 'session' (legacy per-AVI-dir mode).
    --log_level   DEBUG, INFO, WARNING (default: INFO).

Notes:
    - Experiments are processed sequentially one at a time.
    - All --workers cores are given to each experiment's Dask cluster.
    - AVI files must exist before running (see convert_to_avi.py).
"""

import argparse
import json
import logging
import os
import sys
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
    with open(path, encoding="utf-8-sig") as f:
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
    parser.add_argument("--workers",    type=int, default=None,
                        help="Dask workers per experiment (overrides config step1.n_workers).")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--dry_run",    action="store_true")
    parser.add_argument("--experiment", default=None,
                        help="Process only this experiment ID.")
    parser.add_argument("--session",    default=None,
                        help="Process only this session label (session mode).")
    parser.add_argument("--fps",        type=float, default=None,
                        help="Override FPS for all experiments.")
    parser.add_argument("--avi_dir",    default=None,
                        help="Directory where converted AVIs live, if different from --data_dir.")
    parser.add_argument("--mode",       default="experiment",
                        choices=["experiment", "session"])
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

    if args.workers is not None:
        config["step1"]["n_workers"] = args.workers
        log.info(f"Dask workers per experiment: {args.workers}")

    # ── Discover ──────────────────────────────────────────────────────────────

    results = {}

    if args.mode == "experiment":
        from pipeline.session_discovery import (
            discover_experiment_groups, completed_steps_experiment,
        )
        from pipeline.runner import ExperimentRunner

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

        for group in groups:
            runner = ExperimentRunner(
                group=group, config=config,
                mps_root=args.mps_root, resume=args.resume,
            )
            results[group.label] = runner.run()

    else:
        from pipeline.session_discovery import discover_sessions, completed_steps
        from pipeline.runner import SessionRunner

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

        for session in sessions:
            runner = SessionRunner(
                session=session, config=config,
                mps_root=args.mps_root, resume=args.resume,
            )
            results[session.label] = runner.run()

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
