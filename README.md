# MPS Calcium Imaging Pipeline

Headless batch pipeline for automated ROI extraction from calcium imaging recordings using [Miniscope Processing Suite (MPS)](https://github.com/your-mps-link).

Replaces the MPS GUI workflow with a scriptable, parallelisable pipeline that runs on a server without a display. Steps 1–4g are executed programmatically, producing spatial components (ROI masks, A matrix) for each experiment.

---

## Overview

### Data structure expected

```
Ca_Anand_AllData/
  05_11_25_Min6_C23_KO_250ms_Exp1/    ← one experiment (one FOV)
    1_1/                               ← recording condition (baseline, drug, etc.)
      chunk001.tif
      chunk002.tif
      ...
    1_2/                               ← same cells, different condition
      chunk001.tif
    40X/                               ← reference image — skipped automatically
    40X BF/                            ← brightfield reference — skipped
  05_11_25_Min7_WT_250ms_Exp2/
    ...
```

Each **experiment directory** corresponds to one field of view (FOV). Its subdirectories (`1_1`, `1_2`, …) are different recording conditions of the **same cells**. ROI extraction runs once on all conditions concatenated, producing one consistent set of ROIs for the whole experiment.

### Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| Explore | `explore_data.py` | Catalog data directory, show formats and sizes |
| Convert | `convert_to_avi.py` | TIF stacks → uncompressed AVI via Fiji/Bio-Formats |
| Extract ROIs | `run_pipeline.py` | MPS steps 1–4g headlessly, parallel across experiments |

---

## Requirements

- **Python 3.8+** (must match the MPS installation)
- **MPS 1.0.0** source directory (not installed as a package — path passed via `--mps_root`)
- **Fiji/ImageJ** with Bio-Formats plugin (for TIF → AVI conversion)
- Python packages: `numpy`, `xarray`, `zarr`, `dask`, `opencv-python`, `tifffile`

Install dependencies:
```bash
pip install numpy xarray zarr dask opencv-python tifffile
```

---

## Usage

### 1. Explore your data

```bash
python explore_data.py --data_dir /mnt/z/ephacoffice/DColameo/Ca_Anand_AllData
```

Shows experiments found, recording conditions, file formats, estimated duration, FPS parsed from folder names, and whether conversion is needed. Optionally save as text or JSON:

```bash
python explore_data.py \
    --data_dir /mnt/z/.../Ca_Anand_AllData \
    --save report.txt \
    --json sessions.json
```

---

### 2. Convert TIF stacks to AVI

Requires Fiji. AVIs are written to a separate output directory, mirroring the source folder structure. `40X` reference image folders are skipped automatically.

```bash
python convert_to_avi.py \
    --fiji       "/mnt/c/Users/DColameo/Desktop/Fiji.app/ImageJ-win64.exe" \
    --data_dir   /mnt/z/ephacoffice/DColameo/Ca_Anand_AllData \
    --output_dir /mnt/z/ephacoffice/DColameo/Ca_Anand_AVI \
    --fps        4 \
    --workers    4
```

**FPS guide** (from exposure time in folder name):

| Folder name contains | Exposure | FPS |
|---|---|---|
| `250ms` | 250 ms | `--fps 4` |
| `500ms` | 500 ms | `--fps 2` |
| `100ms` | 100 ms | `--fps 10` |

Options:
```
--fiji        Path to Fiji executable (required)
--data_dir    Root TIF directory
--output_dir  Where to write AVIs (mirrors folder structure)
--fps         Frame rate — must match acquisition settings
--workers     Parallel Fiji instances (default: 1, use 4–8 on workstation)
--ext         Only convert one extension, e.g. --ext .tif
--overwrite   Re-convert even if AVI already exists
--dry_run     Show what would be converted without doing it
```

**Single file:**
```bash
python convert_to_avi.py \
    --fiji "/mnt/c/.../ImageJ-win64.exe" \
    --file /mnt/z/.../chunk001.tif \
    --fps 4
```

---

### 3. Run the ROI extraction pipeline

```bash
python run_pipeline.py \
    --data_dir   /mnt/z/ephacoffice/DColameo/Ca_Anand_AllData \
    --avi_dir    /mnt/z/ephacoffice/DColameo/Ca_Anand_AVI \
    --output_dir /mnt/z/ephacoffice/DColameo/Ca_Anand_Processed \
    --mps_root   /mnt/c/Users/DColameo/Documents/dev/MPS_1.0.0 \
    --workers    4 \
    --resume
```

Options:
```
--data_dir    Root TIF directory (used to discover experiment structure)
--avi_dir     Directory containing converted AVIs (from --output_dir above)
--output_dir  Where pipeline results are written
--mps_root    Path to MPS_1.0.0 source directory
--workers     Parallel experiments (default: 1; use 4–8 on a workstation)
--resume      Skip steps whose zarr outputs already exist (safe to re-run)
--dry_run     List experiments that would be processed and exit
--experiment  Process only one experiment by ID
--fps         Override FPS for all experiments
--config      Custom JSON config (merged on top of configs/default_config.json)
--log_level   DEBUG / INFO / WARNING (default: INFO)
```

Each experiment's output is written to:
```
Ca_Anand_Processed/
  05_11_25_Min6_C23_KO_250ms_Exp1/
    cache_data/
      step4g_A_merged.zarr    ← final ROI masks (A matrix)
      step4e_C.zarr           ← temporal components
      ...
    merged_input/             ← symlinks to all AVIs (staging, auto-created)
    pipeline.log
```

---

## Configuration

Edit `configs/default_config.json` to tune MPS parameters. Key settings:

```jsonc
{
  "step1": {
    "n_workers": 8,          // Dask workers per experiment
    "memory_limit": "200GB"
  },
  "step3a": {
    "radius_factor": 0.75,   // circular crop radius as fraction of frame size
    "y_offset": 0,           // shift crop centre vertically (pixels)
    "x_offset": 0,
    "use_full_frame": false  // set true to skip cropping entirely
  },
  "step3b": {
    "n_components": 100      // max number of ROI candidates
  },
  "step4g": {
    "temporal_corr_threshold": 0.75   // merge threshold for duplicate ROIs
  }
}
```

Pass a custom config with overrides:
```bash
python run_pipeline.py ... --config my_config.json
```
Only the keys present in `my_config.json` override the defaults — you don't need to copy the whole file.

---

## Project structure

```
mps_ca/
├── run_pipeline.py              # main CLI — discovers experiments, runs pipeline
├── convert_to_avi.py            # TIF → AVI conversion via Fiji
├── explore_data.py              # data directory explorer / report generator
├── configs/
│   └── default_config.json      # all MPS step parameters with defaults
├── imagej_macros/
│   └── convert_to_avi.ijm       # ImageJ macro called by convert_to_avi.py
└── pipeline/
    ├── session_discovery.py     # ExperimentGroup / RecordingCondition data model
    ├── headless.py              # tkinter mock — lets MPS modules import without display
    ├── step_runner.py           # one run_stepXx() function per MPS step
    ├── runner.py                # SessionRunner / ExperimentRunner orchestrators
    └── __init__.py
```

---

## How it works

MPS is a tkinter GUI application. Running it headlessly requires three adaptations:

1. **Tkinter mocking** (`pipeline/headless.py`): fake tkinter modules are injected into `sys.modules` before any MPS import, so step classes can be imported without a display.
2. **Bypassing widget initialisation** (`object.__new__()`): MPS step classes inherit from `ttk.Frame`. We instantiate them without calling `__init__`, then wire only the attributes their processing threads actually read.
3. **Step 3a crop computation**: the GUI normally computes the crop region interactively. Headless mode computes it from `radius_factor` / `y_offset` / `x_offset` config parameters and the first 100 frames of step 2e output.

For multi-condition experiments, all AVIs from all conditions are linked into a single `merged_input/` directory. MPS step 2a loads them in order, treating all conditions as one continuous recording for ROI extraction.

---

## Running on a remote Windows workstation

**Recommended: WSL2**
```powershell
# Admin PowerShell on the workstation — one-time
wsl --install
```
Then SSH in and run everything as above.

**Native Windows** (if WSL2 unavailable): use Windows-format paths:
```powershell
python convert_to_avi.py `
    --fiji "C:\Users\DColameo\Desktop\Fiji.app\ImageJ-win64.exe" `
    --data_dir "Z:\ephacoffice\DColameo\Ca_Anand_AllData" `
    --output_dir "Z:\ephacoffice\DColameo\Ca_Anand_AVI" `
    --fps 4 --workers 4
```
The code is cross-platform; symlinks fall back to file copies automatically on Windows.

**Keep running after disconnect:**
```bash
screen -S pipeline
python run_pipeline.py ...
# Ctrl+A D to detach
screen -r pipeline   # reattach
```

---

## Workstation sizing guide

| Resource | Recommendation |
|---|---|
| `--workers` (experiments in parallel) | 4–8 |
| `step1.n_workers` (Dask per experiment) | 6–8 |
| `step1.memory_limit` | e.g. `"200GB"` |
| Disk for AVIs | ~1–5 GB per experiment (uncompressed) |

On a 30-core / 1 TB machine: `--workers 4` with `n_workers: 8` uses all cores while keeping memory bounded.
