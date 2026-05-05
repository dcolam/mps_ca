"""
explore_data.py — Catalog a calcium imaging data directory.

Shows what experiments exist, what recording conditions each has, what file
formats are present, total size, FPS (parsed from folder names), and whether
AVI conversion is needed.

The expected directory structure is:
    data_dir/
      ExperimentDir/           ← one experiment (same FOV across conditions)
        1_1/                   ← recording condition
          chunk001.tif ...
        1_2/                   ← another condition
        40X/                   ← reference image — skipped automatically
        40X BF/                ← brightfield reference — skipped

Usage:
    python explore_data.py --data_dir /mnt/z/ephacoffice/DColameo/Ca_Anand_AllData
    python explore_data.py --data_dir /mnt/z/... --save report.txt --json sessions.json

If the Z drive is not mounted in WSL, run first:
    sudo mkdir -p /mnt/z
    sudo mount -t drvfs Z: /mnt/z
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

TIF_EXTENSIONS   = {".tif", ".tiff"}
AVI_EXTENSIONS   = {".avi"}
VIDEO_EXTENSIONS = TIF_EXTENSIONS | AVI_EXTENSIONS | {
    ".nd2", ".czi", ".lif", ".isxd", ".mp4", ".mkv", ".mov", ".bmp", ".png",
}

_EXCLUDE_DIR_RE = re.compile(r'^(40[Xx]|\.)', re.IGNORECASE)


def _is_excluded(name: str) -> bool:
    return bool(_EXCLUDE_DIR_RE.match(name))


def _parse_fps(name: str) -> Optional[float]:
    m = re.search(r'(\d+)ms', name, re.IGNORECASE)
    if m:
        return round(1000.0 / float(m.group(1)), 2)
    m = re.search(r'(\d+)\s*fps', name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _sort_key(name: str) -> list:
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _size_str(mb: float) -> str:
    return f"{mb/1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


# ── Data collection ───────────────────────────────────────────────────────────

def _dir_size_mb(paths: List[str]) -> float:
    total = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    return total / (1024 ** 2)


def _scan_condition(dirpath: str) -> Dict:
    """Return a summary dict for one recording condition directory."""
    files_by_ext = defaultdict(list)
    for fname in os.listdir(dirpath):
        ext = Path(fname).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            files_by_ext[ext].append(os.path.join(dirpath, fname))

    # Sort each extension's file list
    for ext in files_by_ext:
        files_by_ext[ext].sort(key=lambda p: _sort_key(os.path.basename(p)))

    counts = {ext: len(fs) for ext, fs in files_by_ext.items() if fs}
    dominant = max(counts, key=counts.get) if counts else None
    all_files = [f for fs in files_by_ext.values() for f in fs]

    return {
        "directory":    dirpath,
        "name":         os.path.basename(dirpath),
        "files_by_ext": dict(files_by_ext),
        "dominant_ext": dominant,
        "n_files":      sum(counts.values()),
        "size_mb":      _dir_size_mb(all_files),
        "has_avi":      ".avi" in files_by_ext and len(files_by_ext[".avi"]) > 0,
        "needs_conversion": dominant not in AVI_EXTENSIONS if dominant else True,
    }


def discover_experiments(root: str) -> List[Dict]:
    """
    Walk root and return one dict per experiment directory.
    Each dict contains a list of condition dicts.
    """
    # Find all directories that contain TIF/video files (excluding 40X dirs)
    dirs_with_files: Dict[str, List[str]] = {}
    for dirpath, subdirs, filenames in os.walk(root, topdown=True):
        subdirs[:] = sorted(d for d in subdirs if not _is_excluded(d))
        vfiles = [
            os.path.join(dirpath, f)
            for f in filenames
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        ]
        if vfiles:
            dirs_with_files[dirpath] = vfiles

    # Group by parent
    parent_to_children = defaultdict(list)
    for d in dirs_with_files:
        parent_to_children[str(Path(d).parent)].append(d)

    experiments = []
    for parent, children in sorted(parent_to_children.items()):
        parent_name = os.path.basename(parent)
        fps = _parse_fps(parent_name) or 4.0

        conditions = []
        for child in sorted(children, key=lambda p: _sort_key(os.path.basename(p))):
            cond = _scan_condition(child)
            conditions.append(cond)

        total_size = sum(c["size_mb"] for c in conditions)
        total_files = sum(c["n_files"] for c in conditions)
        needs_conv = any(c["needs_conversion"] for c in conditions)
        all_avi = all(c["has_avi"] for c in conditions)

        experiments.append({
            "experiment_dir":    parent,
            "experiment_name":   parent_name,
            "fps":               fps,
            "conditions":        conditions,
            "total_size_mb":     total_size,
            "total_files":       total_files,
            "needs_conversion":  needs_conv,
            "all_avi_ready":     all_avi,
        })

    return experiments


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(experiments: List[Dict], root: str, file=None):
    def p(*args, **kwargs):
        print(*args, **kwargs, file=file)

    total_size = sum(e["total_size_mb"] for e in experiments)
    n_ready = sum(1 for e in experiments if e["all_avi_ready"])
    n_needs = len(experiments) - n_ready

    p("=" * 72)
    p("CALCIUM IMAGING DATA EXPLORER")
    p(f"Root:        {root}")
    p(f"Experiments: {len(experiments)}")
    p(f"Total size:  {_size_str(total_size)}")
    p("=" * 72)

    p(f"\nSUMMARY")
    p(f"  AVI ready (pipeline can run): {n_ready}")
    p(f"  Need conversion:              {n_needs}")

    # Extension breakdown
    ext_counts: Dict[str, int] = defaultdict(int)
    for exp in experiments:
        for cond in exp["conditions"]:
            if cond["dominant_ext"]:
                ext_counts[cond["dominant_ext"]] += 1
    p(f"\n  File formats across all recording conditions:")
    for ext, count in sorted(ext_counts.items()):
        p(f"    {ext:<8} in {count} condition(s)")

    p("\n" + "─" * 72)
    p("EXPERIMENT DETAILS")
    p("─" * 72)

    for exp in experiments:
        name  = exp["experiment_name"]
        fps   = exp["fps"]
        size  = _size_str(exp["total_size_mb"])
        n_cnd = len(exp["conditions"])
        n_fls = exp["total_files"]
        status = "✓ AVI ready" if exp["all_avi_ready"] else "⚠ CONVERT"

        rel = os.path.relpath(exp["experiment_dir"], root)
        p(f"\n  {rel}")
        p(f"    {n_cnd} condition(s)  |  {n_fls} total files  |  {size}  |  {status}  |  FPS={fps}")

        for cond in exp["conditions"]:
            cname = cond["name"]
            ext   = cond["dominant_ext"] or "?"
            n     = cond["n_files"]
            csz   = _size_str(cond["size_mb"])
            cst   = "✓" if cond["has_avi"] else f"→ convert ({ext})"
            p(f"      {cname:<12} {n:>4} × {ext:<6}  {csz:>8}  {cst}")

        # Duration estimate
        n_chunks = sum(cond["n_files"] for cond in exp["conditions"])
        est_frames = n_chunks * 1000
        est_min = est_frames / fps / 60
        p(f"    Est. ~{est_frames:,} frames @ {fps} Hz → ~{est_min:.1f} min total recording")

    p("\n" + "─" * 72)
    p("NEXT STEPS")
    p("─" * 72)

    if n_needs:
        p(f"\n  1. Convert TIF → AVI (fps=4 for 250ms exposure):")
        p(f"       python convert_to_avi.py \\")
        p(f"           --data_dir {root} \\")
        p(f"           --fps 4 \\")
        p(f"           --workers 4")
        p(f"       # (or --fps 2 for 500ms, --fps 10 for 100ms, etc.)")

    p(f"\n  {'2.' if n_needs else '1.'} Run the MPS pipeline:")
    p(f"       python run_pipeline.py \\")
    p(f"           --data_dir  {root} \\")
    p(f"           --output_dir <output_path> \\")
    p(f"           --workers 4 --resume")
    p(f"       # Each experiment's ROIs are shared across all its conditions.")

    if experiments:
        p(f"\n  Fiji conversion (manual, single file):")
        p(f"       fiji --headless -macro imagej_macros/convert_to_avi.ijm \\")
        p(f'            "path/to/chunk.tif|path/to/chunk.avi|4"')

    p("\n" + "=" * 72)


def export_json(experiments: List[Dict], path: str):
    out = []
    for exp in experiments:
        out.append({
            "experiment_dir":   exp["experiment_dir"],
            "experiment_name":  exp["experiment_name"],
            "fps":              exp["fps"],
            "total_size_mb":    round(exp["total_size_mb"], 1),
            "needs_conversion": exp["needs_conversion"],
            "all_avi_ready":    exp["all_avi_ready"],
            "conditions": [
                {
                    "name":      c["name"],
                    "directory": c["directory"],
                    "n_files":   c["n_files"],
                    "dominant_ext": c["dominant_ext"],
                    "size_mb":   round(c["size_mb"], 1),
                    "has_avi":   c["has_avi"],
                }
                for c in exp["conditions"]
            ],
        })
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Session list saved to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore calcium imaging data directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Root data directory.")
    parser.add_argument("--save", default=None,
                        help="Save text report to this file.")
    parser.add_argument("--json", default=None,
                        help="Export experiment list as JSON.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: directory not found: {args.data_dir}")
        print("\nIf this is a network drive, mount it first:")
        print("  sudo mkdir -p /mnt/z")
        print("  sudo mount -t drvfs Z: /mnt/z")
        sys.exit(1)

    print(f"Scanning {args.data_dir} ...")
    experiments = discover_experiments(args.data_dir)

    if not experiments:
        print("No video files found. Check the path.")
        sys.exit(0)

    print_report(experiments, args.data_dir)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            print_report(experiments, args.data_dir, file=f)
        print(f"\nReport saved to: {args.save}")

    if args.json:
        export_json(experiments, args.json)


if __name__ == "__main__":
    main()
