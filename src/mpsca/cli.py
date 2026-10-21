"""Command line: ``mpsca gui | scan | convert | process | info``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .io import (ConvertParams, ExperimentMeta, ExperimentStore, convert_experiment, parse_excel,
                 resolve_paths, scan_experiment)
from .io.metadata import entries_from_files, entries_from_folder, infer_genotype


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


def _experiments(args) -> list[ExperimentMeta]:
    """Experiments selected by --excel/--sheet, --folder or --files, with paths resolved."""
    if args.excel:
        if not args.base_dir:
            raise SystemExit("--base-dir is required with --excel")
        experiments = parse_excel(args.excel)
        if args.sheet:
            wanted = set(args.sheet)
            experiments = [m for m in experiments if m.sheet in wanted or m.name in wanted]
            if not experiments:
                raise SystemExit(f"no sheet matches {sorted(wanted)}")
        for meta in experiments:
            exp_dir = Path(args.base_dir) / meta.name
            if exp_dir.is_dir():
                resolve_paths(exp_dir, meta)
            else:
                print(f"[{meta.sheet}] not found: {exp_dir}", file=sys.stderr)
                for fe in meta.files:
                    fe.path = None
        return experiments
    if args.folder:
        return [entries_from_folder(args.folder)]
    name = args.name or Path(args.files[0]).parent.name
    return [ExperimentMeta(name=name, genotype=infer_genotype(name), files=entries_from_files(args.files))]


def _parse_factors(text: str | None) -> dict[int, int] | None:
    """``"0:2,1:2,3:1"`` -> {0: 2, 1: 2, 3: 1}"""
    if not text:
        return None
    return {int(k): int(v) for k, v in (item.split(":") for item in text.split(","))}


def _cmd_scan(args) -> int:
    for meta in _experiments(args):
        print(f"[{meta.sheet or meta.name}] {meta.name}")
        if not any(fe.path for fe in meta.files):
            continue
        report = scan_experiment(meta, args.source_fps, progress=_progress_printer("scan"))
        summary = report.block_summary()
        summary["factor"] = summary["block"].map(report.proposed_factors(args.binning, args.grid))
        print(summary.to_string(index=False))
        for msg in report.issues:
            print(f"  ! {msg}")
    return 0


def _cmd_convert(args) -> int:
    params = ConvertParams(source_fps=args.source_fps, target_fps=args.target_fps,
                           spatial_binning=args.binning, low_signal_thresh=args.bad_thresh,
                           chunk_frames=args.chunk_frames, target_camera_binning=args.grid,
                           block_spatial_factors=_parse_factors(args.block_factors))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for meta in _experiments(args):
        print(f"[{meta.sheet or meta.name}] {meta.name}")
        if not any(fe.path for fe in meta.files):
            failed += 1
            continue
        for fe in meta.files:
            if fe.path is None:
                print(f"  MISSING: {fe.filename}")
        out = out_dir / f"{meta.name}.mpsca.zarr"
        try:
            report = scan_experiment(meta, params.source_fps, progress=_progress_printer("scan"))
            if (report.needs_decision(params.spatial_binning, params.target_camera_binning)
                    and not params.block_spatial_factors and not args.accept_proposal):
                print(report.block_summary().to_string(index=False))
                print(f"  spatial binning needed (block sizes differ or below the common grid); "
                      f"proposed factors {params.factors_for(report)}. Re-run with "
                      "--accept-proposal or --block-factors (e.g. 0:2,1:2,2:2,3:1)")
                failed += 1
                continue
            store = convert_experiment(meta, out, params, overwrite=args.overwrite,
                                       progress=_progress_printer("convert"), report=report)
        except Exception as e:  # keep going with the remaining experiments
            print(f"  ERROR: {e}")
            failed += 1
            continue
        frames = store.read_frames()
        print(f"  -> {out}  movie {store.movie.shape}, {int(frames['is_bad'].sum())} bad frames")
        for w in store.meta.get("frame_warnings", []):
            print(f"  WARNING: {w}")
    return 1 if failed else 0


def _cmd_process(args) -> int:
    from . import pipeline
    from .analysis import AnalysisParams
    from .traces import DffParams

    store = ExperimentStore.open(args.store)
    if not store.complete:
        raise SystemExit(f"{args.store} is incomplete; convert again")
    pipeline.set_bad_frames(store, args.bad_thresh)
    pipeline.run_projections(store, progress=_progress_printer("projections"))
    if args.method == "cellpose":
        from .segment.cellpose_seg import CellposeParams
        labels = pipeline.run_cellpose(store, args.channels.split(","),
                                       CellposeParams(diameter=args.diameter or None))
    elif args.method == "classic":
        labels = pipeline.run_classic(store).labels
    else:
        act = pipeline.active_labels(store)
        if act is None:
            raise SystemExit("no saved ROIs in this store; use --method cellpose or classic")
        labels = act[1]
    print(f"  {int(labels.max())} ROIs")
    pipeline.run_traces(store, DffParams(offset=args.offset), progress=_progress_printer("traces"))
    res = pipeline.run_analysis(store, AnalysisParams())
    out = pipeline.export_results(store, args.out, res)
    print(f"  {res.summary['n_bursts']} bursts, mean r {res.summary['mean_corr']} -> {out}")
    return 0


def _cmd_gui(args) -> int:
    from .gui import launch
    launch(args.store)
    return 0


def _cmd_info(args) -> int:
    store = ExperimentStore.open(args.store, mode="r")
    if not store.complete:
        print("WARNING: incomplete store (conversion interrupted)")
    meta = store.meta
    print(f"experiment : {meta.get('name')}  (sheet {meta.get('sheet')}, genotype {meta.get('genotype')})")
    print(f"movie      : {store.movie.shape} {store.movie.dtype}")
    frames = store.read_frames()
    print(f"frames     : {len(frames)} ({int(frames['is_bad'].sum())} bad, "
          f"{int(frames['use_for_seg'].sum())} for segmentation)")
    blocks = frames.groupby("block", observed=True).agg(
        application=("application", "first"), phase=("phase", "first"),
        n=("time_s", "size"), t0=("time_s", "min"), t1=("time_s", "max"))
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
    g.add_argument("store", nargs="?", help="experiment store to open")
    g.set_defaults(func=_cmd_gui)

    def sources(sp):
        src = sp.add_mutually_exclusive_group(required=True)
        src.add_argument("--excel", help="metadata workbook (one sheet per experiment)")
        src.add_argument("--folder", help="experiment folder; each subfolder is one condition block")
        src.add_argument("--files", nargs="+", help="TIFF files of one experiment, in order")
        sp.add_argument("--base-dir", help="folder containing the experiment folders (with --excel)")
        sp.add_argument("--sheet", nargs="+", help="only these sheets / experiment folders")
        sp.add_argument("--name", help="experiment name (with --files)")
        sp.add_argument("--source-fps", type=float, default=4.0)
        sp.add_argument("--grid", type=int, default=2,
                        help="common grid as camera binning: unbinned blocks are binned to it "
                             "(default 2; 0 = coarsest block of the experiment)")
        sp.add_argument("--binning", type=int, default=1,
                        help="extra spatial binning on top of the common grid (default 1)")

    sc = sub.add_parser("scan", help="read file headers and report blocks, sizes and problems")
    sources(sc)
    sc.set_defaults(func=_cmd_scan)

    c = sub.add_parser("convert", help="bin raw TIFFs into experiment stores")
    sources(c)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--target-fps", type=float, default=1.0)
    c.add_argument("--block-factors", help="spatial factor per block, e.g. 0:2,1:2,2:2,3:1")
    c.add_argument("--accept-proposal", action="store_true",
                   help="when block sizes differ, use the proposed factors without asking")
    c.add_argument("--bad-thresh", type=float, default=0.5)
    c.add_argument("--chunk-frames", type=int, default=16)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=_cmd_convert)

    pr = sub.add_parser("process", help="projections, segmentation, traces, analysis and export")
    pr.add_argument("store")
    pr.add_argument("--method", choices=["cellpose", "classic", "saved"], default="cellpose")
    pr.add_argument("--channels", default="mean,corr", help="Cellpose input images")
    pr.add_argument("--diameter", type=float, default=0, help="cell diameter in px (0 = auto)")
    pr.add_argument("--offset", type=float, help="camera offset (default: estimated)")
    pr.add_argument("--bad-thresh", type=float, default=0.5)
    pr.add_argument("--out", help="results folder (default: next to the store)")
    pr.set_defaults(func=_cmd_process)

    i = sub.add_parser("info", help="summarise an experiment store")
    i.add_argument("store")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
