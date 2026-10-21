"""Command line: ``mpsca gui | new | list | scan | convert | run | info``.

Projects are usually set up and tested in the GUI; the CLI runs the long batch steps
(convert, run) with the settings saved in the project file.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .io import ConvertParams, ExperimentStore
from .io.metadata import GROUPINGS
from .project import (Project, accept_proposals, convert_all, create_project, needs_decision, run_all,
                      scan_all)


def _progress_printer(label: str):
    last = [-1]

    def report(frac: float) -> None:
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r  {label}: {pct:3d}%", end="", file=sys.stderr, flush=True)
            if pct == 100:
                print(file=sys.stderr)
    return report


def _say(msg: str) -> None:
    print(f"\r{msg}", flush=True)


def _load(args) -> Project:
    proj = Project.load(args.project, input_dir=getattr(args, "input", None))
    if not proj.input_found:
        raise SystemExit(f"input folder not found: {proj.input_dir} (use --input to point to it)")
    return proj


def _print_recordings(proj: Project) -> None:
    keys = proj.description_keys()
    for r in proj.recordings:
        desc = ", ".join(f"{k}={r.description[k]}" for k in keys if k in r.description)
        mark = "x" if r.include else " "
        print(f"[{mark}] {r.name}: {len(r.active_files)} files, {r.n_blocks} blocks, "
              f"{proj.status(r)}" + (f"  ({desc})" if desc else ""))
        if r.missing:
            print(f"     missing: {', '.join(r.missing)}")
        for b, cond in sorted(r.conditions().items()):
            print(f"     block {b}: {cond}")
    for msg in proj.issues:
        print(f"  ! {msg}")


def _cmd_new(args) -> int:
    proj = create_project(args.input, args.out, args.pattern, args.excel, args.grouping,
                          args.conditions.split(",") if args.conditions else None)
    proj.convert = ConvertParams(source_fps=args.source_fps, target_fps=args.target_fps,
                                 target_camera_binning=args.grid, spatial_binning=args.binning)
    print(f"project saved to {proj.save()}")
    _print_recordings(proj)
    return 0 if proj.recordings else 1


def _cmd_list(args) -> int:
    _print_recordings(_load(args))
    return 0


def _scan(proj: Project, args) -> dict:
    reports = scan_all(proj, proj.select(args.only), progress=_progress_printer("scan"), say=_say)
    p = proj.convert
    for name, rep in reports.items():
        rec = proj.get(name)
        print(f"\n{name}")
        if rep.files:
            summary = rep.block_summary()
            summary["factor"] = summary["block"].map(p.factors_for(rep))
            print(summary.drop(columns=["seg_files"]).to_string(index=False))
        for msg in rep.issues:
            print(f"  ! {msg}")
        if needs_decision(proj, rec, rep):
            print(f"  ? spatial binning needed, proposal {p.factors_for(rep)}")
    if args.accept_proposal:
        done = accept_proposals(proj, reports)
        if done:
            print(f"\nproposed spatial factors stored for: {', '.join(done)}")
    return reports


def _cmd_scan(args) -> int:
    _scan(_load(args), args)
    return 0


def _cmd_convert(args) -> int:
    proj = _load(args)
    reports = _scan(proj, args)
    pending = [n for n, rep in reports.items() if needs_decision(proj, proj.get(n), rep)]
    if pending:
        print(f"\nspatial binning not confirmed for {', '.join(pending)}: re-run with --accept-proposal "
              "or choose the factors in the GUI")
        return 1
    print()
    res = convert_all(proj, proj.select(args.only), reports, overwrite=args.overwrite,
                      progress=_progress_printer("convert"), say=_say)
    return 1 if any(v.startswith("failed") for v in res.values()) else 0


def _cmd_run(args) -> int:
    from .pipeline import STEPS

    proj = _load(args)
    steps = args.steps.split(",") if args.steps else STEPS
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"unknown step(s) {sorted(unknown)}; choose from {','.join(STEPS)}")
    res = run_all(proj, proj.select(args.only), steps, progress=_progress_printer("run"), say=_say)
    print()
    for name, r in res.items():
        print(f"{name}: {r}")
    return 1 if any(v.startswith("failed") for v in res.values()) else 0


def _cmd_gui(args) -> int:
    from .gui import launch
    launch(args.path)
    return 0


def _cmd_info(args) -> int:
    store = ExperimentStore.open(args.store, mode="r")
    if not store.complete:
        print("WARNING: incomplete store (conversion interrupted)")
    meta = store.meta
    print(f"recording  : {meta.get('name')}")
    for k, v in (meta.get("description") or {}).items():
        print(f"  {k}: {v}")
    print(f"movie      : {store.movie.shape} {store.movie.dtype}")
    frames = store.read_frames()
    print(f"frames     : {len(frames)} ({int(frames['is_bad'].sum())} bad, "
          f"{int(frames['use_for_seg'].sum())} for segmentation)")
    blocks = frames.groupby("block", observed=True).agg(
        condition=("condition", "first"), n=("time_s", "size"), t0=("time_s", "min"), t1=("time_s", "max"))
    print(blocks.to_string())
    for w in meta.get("frame_warnings", []):
        print(f"WARNING: {w}")
    for line in store.group.attrs.get("log", []):
        print(f"log: {line}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mpsca")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gui", help="open the napari GUI")
    g.add_argument("path", nargs="?", help="project (file or output folder) or recording store to open")
    g.set_defaults(func=_cmd_gui)

    n = sub.add_parser("new", help="find recordings and create a project")
    n.add_argument("--input", required=True, help="folder that holds the recordings (searched recursively)")
    n.add_argument("--out", required=True, help="output folder: project file and converted recordings")
    n.add_argument("--pattern", default="*.ome.tif", help="file pattern, several separated by ';'")
    n.add_argument("--excel", help="optional workbook describing the files")
    n.add_argument("--grouping", choices=list(GROUPINGS), default="folder",
                   help="without a workbook: " + "; ".join(f"{k} = {v}" for k, v in GROUPINGS.items()))
    n.add_argument("--conditions", help="workbook columns that name a block, e.g. Application,Phase "
                                        "(default: the columns that change between blocks)")
    n.add_argument("--source-fps", type=float, default=4.0)
    n.add_argument("--target-fps", type=float, default=1.0)
    n.add_argument("--grid", type=int, default=2,
                   help="common grid as camera binning: finer blocks are binned to it "
                        "(default 2; 0 = coarsest block of each recording)")
    n.add_argument("--binning", type=int, default=1, help="extra spatial binning on top (default 1)")
    n.set_defaults(func=_cmd_new)

    def project_args(sp):
        sp.add_argument("project", help="project file or its output folder")
        sp.add_argument("--input", help="input folder, if it moved (e.g. another drive letter)")
        sp.add_argument("--only", nargs="+", help="only these recordings")

    ls = sub.add_parser("list", help="recordings of a project and their status")
    project_args(ls)
    ls.set_defaults(func=_cmd_list)

    sc = sub.add_parser("scan", help="read file headers of all recordings and report problems")
    project_args(sc)
    sc.add_argument("--accept-proposal", action="store_true",
                    help="store the proposed spatial binning where block sizes differ")
    sc.set_defaults(func=_cmd_scan)

    c = sub.add_parser("convert", help="convert all recordings (already converted ones are skipped)")
    project_args(c)
    c.add_argument("--accept-proposal", action="store_true",
                   help="use the proposed spatial binning where block sizes differ")
    c.add_argument("--overwrite", action="store_true", help="convert again even if already converted")
    c.set_defaults(func=_cmd_convert)

    r = sub.add_parser("run", help="process converted recordings with the project settings")
    project_args(r)
    r.add_argument("--steps", help="comma-separated subset of: bad_frames,projections,segment,traces,"
                                   "analysis,export (default: all)")
    r.set_defaults(func=_cmd_run)

    i = sub.add_parser("info", help="summarise one converted recording")
    i.add_argument("store")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
