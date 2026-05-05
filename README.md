# MPS Calcium Imaging Pipeline

Headless batch pipeline for automated ROI extraction from calcium imaging recordings using [Miniscope Processing Suite (MPS)](https://github.com/your-mps-link).

Replaces the MPS GUI workflow with a scriptable, parallelisable pipeline that runs on a server without a display. MPS steps 1–4g are executed programmatically, producing spatial components (ROI masks, A matrix) for each experiment.

---

## Data structure

```
data_dir/
  ExperimentA_250ms_Exp1/        ← one experiment = one dish / one FOV
    1_1/                          ← recording condition (e.g. baseline)
      recording_chunk_001.tif
      recording_chunk_002.tif
    1_2/                          ← same cells, different condition (e.g. drug)
      recording_chunk_001.tif
    1_3/                          ← washout, etc.
    40X/                          ← single reference image — skipped automatically
    40X BF/                       ← brightfield reference — skipped
  ExperimentB_250ms_Exp2/
    ...
```

Each **experiment directory** (first level) corresponds to one field of view — one dish, one set of cells. Its subdirectories are different **recording conditions** (liquid exchange periods, timepoints) of the **same cells**. ROI extraction runs once on all conditions concatenated, producing one consistent set of ROIs shared across all conditions.

The FPS is parsed automatically from the folder name (e.g. `250ms` → 4 Hz).

---

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| Explore | `explore_data.py` | Catalog data directory, show formats, sizes, FPS |
| Convert | `convert_to_avi.py` | TIF stacks → uncompressed AVI via Fiji/Bio-Formats |
| Extract ROIs | `run_pipeline.py` | MPS steps 1–4g headlessly, parallel across experiments |

---

## MPS steps explained

The pipeline runs MPS steps 1 through 4g. Here is what each step does:

### Step 1 — Project setup
Initialises the output directory structure, sets environment variables, and starts the Dask cluster for parallel computation within the experiment.

### Step 2a — Video loading
Loads all AVI files from the merged input directory (all conditions concatenated in order), stores the raw video as a chunked zarr array. This is the main data ingestion step.

### Step 2b — Background removal & denoising
Applies spatial denoising (default: median filter) followed by background subtraction (default: top-hat morphological filter). This removes slow fluorescence drift and out-of-focus background, leaving only local fluorescence changes.

### Step 2c — Motion correction
Estimates per-frame rigid translation shifts to correct for movement of the dish or microscope stage during recording. Produces a motion vector for each frame.

### Step 2d — Erroneous frame detection
Identifies and optionally drops frames that are statistical outliers (e.g. frames corrupted by vibration, focus loss, or illumination spikes). Controlled by `threshold_factor`.

### Step 2e — Transformation
Applies the motion correction shifts computed in step 2c to produce the final motion-corrected video. Saves two chunked zarr layouts: one chunked by frame (for temporal operations) and one chunked by spatial position (for spatial operations).

### Step 3a — Spatial cropping
Crops the video to a circular region centred on the imaging field. The crop radius is set by `radius_factor` (fraction of frame size). Use `use_full_frame: true` to skip cropping entirely. This reduces the data volume for all subsequent steps.

### Step 3b — NNDSVD initialisation
Runs Non-negative Double Singular Value Decomposition (NNDSVD) on the cropped video to find an initial set of spatial components (candidate ROIs). `n_components` controls the maximum number of candidates. This is the computationally heaviest step.

### Step 4a — Watershed parameter search
Grid-searches over combinations of `min_distance`, `threshold_rel`, and `sigma` to find the watershed segmentation parameters that best separate individual cells. A sample of frames is used to keep this fast.

### Step 4b — Watershed segmentation
Applies the best watershed parameters found in step 4a to segment the spatial components into individual cell ROIs. Each connected region above the threshold becomes a candidate ROI.

### Step 4c — Spatial merging
Merges ROIs that overlap spatially or are too close together (controlled by `distance_threshold` and `size_ratio_threshold`). Removes ROIs that are too small or too large.

### Step 4d — Temporal signal extraction
For each spatial ROI (from step 4c), extracts the corresponding temporal fluorescence trace from the motion-corrected video. This produces the raw C matrix (one trace per ROI).

### Step 4e — A/C initialisation
Normalises the spatial (A) and temporal (C) components and prepares them for the quality-control steps. Removes background components if `skip_bg` is true.

### Step 4f — Component quality filter
Drops components that are NaN, spatially empty, or temporally flat (no signal). These are artefacts from the segmentation rather than real cells.

### Step 4g — Temporal merging
Merges ROIs whose temporal traces are highly correlated (controlled by `temporal_corr_threshold`) and spatially overlapping (`spatial_overlap_threshold`). These are likely the same cell detected twice. The output — `step4g_A_merged.zarr` — is the final ROI mask matrix.

---

## Output structure

```
output_dir/
  ExperimentA_250ms_Exp1/
    cache_data/
      step2a_stream_tmp.zarr
      step2b_varr_ref.zarr
      step2c_motion.zarr
      step2d_varr_ref.zarr
      step2e_Y_fm_chk.zarr
      step2e_Y_hw_chk.zarr
      step3a_Y_fm_cropped.zarr
      A_init.zarr
      step4a_watershed_params.json
      step4b_separated_components.zarr
      step4c_merged_components.zarr
      step4d_components_with_temporal.zarr
      step4e_A.zarr
      step4e_C.zarr
      step4f_A_clean.zarr
      step4g_A_merged.zarr        ← final ROI masks (A matrix)
    merged_input/
      chunk_000000.avi            ← symlink → condition 1_1, chunk 1
      chunk_000001.avi            ← symlink → condition 1_1, chunk 2
      chunk_000008.avi            ← symlink → condition 1_2, chunk 1
      chunk_manifest.json         ← maps every chunk back to its condition and source file
    pipeline.log
```

The **chunk manifest** (`chunk_manifest.json`) records which chunk file corresponds to which recording condition and source TIF, so frames can be traced back to conditions for downstream analysis.

---

## Requirements

- **Python 3.8+** (must match the MPS installation)
- **MPS 1.0.0** source directory (not installed as a package — path passed via `--mps_root`)
- **Fiji/ImageJ** with Bio-Formats plugin (for TIF → AVI conversion)
- Python packages: `numpy`, `xarray`, `zarr`, `dask`, `opencv-python`, `tifffile`

```bash
pip install numpy xarray zarr dask opencv-python tifffile
```

---

## Usage

### 1. Explore your data

```bash
python explore_data.py --data_dir /path/to/data_dir
python explore_data.py --data_dir /path/to/data_dir --save report.txt --json sessions.json
```

Shows experiments found, recording conditions, file formats, estimated duration, FPS, and whether conversion is needed.

---

### 2. Convert TIF stacks to AVI

Requires Fiji. AVIs are written to a separate output directory, mirroring the source folder structure. `40X` reference image folders are skipped automatically.

```bash
python convert_to_avi.py \
    --fiji       /path/to/Fiji.app/ImageJ-linux64 \
    --data_dir   /path/to/data_dir \
    --output_dir /path/to/avi_dir \
    --fps        4 \
    --workers    4
```

**FPS guide** (from exposure time in folder name):

| Folder name contains | Exposure | FPS |
|---|---|---|
| `250ms` | 250 ms | `--fps 4` |
| `500ms` | 500 ms | `--fps 2` |
| `100ms` | 100 ms | `--fps 10` |

Key options:
```
--fiji        Path to Fiji executable (required)
--data_dir    Root TIF directory
--output_dir  Where to write AVIs (mirrors folder structure; if omitted, writes alongside TIFs)
--fps         Frame rate in Hz — must match acquisition settings
--workers     Parallel Fiji instances (default: 1)
--overwrite   Re-convert even if AVI already exists
--dry_run     Show what would be converted without doing it
```

**Windows (PowerShell):**
```powershell
python convert_to_avi.py --fiji "C:\path\to\Fiji.app\ImageJ-win64.exe" --data_dir "Z:\data_dir" --output_dir "Z:\avi_dir" --fps 4 --workers 4
```

---

### 3. Run the ROI extraction pipeline

```bash
python run_pipeline.py \
    --data_dir   /path/to/data_dir \
    --avi_dir    /path/to/avi_dir \
    --output_dir /path/to/output_dir \
    --mps_root   /path/to/MPS_1.0.0 \
    --workers    4 \
    --resume
```

Key options:
```
--data_dir    Root TIF directory (used to discover experiment structure)
--avi_dir     Directory containing converted AVIs
--output_dir  Where pipeline results are written
--mps_root    Path to MPS_1.0.0 source directory
--workers     Experiments processed in parallel (default: 1)
--resume      Skip steps whose zarr outputs already exist — safe to re-run
--dry_run     List experiments that would be processed and exit
--experiment  Process only one experiment by ID
--fps         Override FPS for all experiments
--config      Custom JSON config merged on top of default_config.json
--log_level   DEBUG / INFO / WARNING (default: INFO)
```

**Watch the logs live:**
```bash
tail -f /path/to/output_dir/ExperimentA/pipeline.log
```
```powershell
Get-Content "Z:\output_dir\ExperimentA\pipeline.log" -Wait
```

---

## Configuration

All algorithmic parameters live in `configs/default_config.json`. Create a small override file and pass it with `--config` — only the keys you specify are overridden:

```json
{
  "step1": {
    "n_workers": 10,
    "memory_limit": "13GB"
  },
  "step3a": {
    "use_full_frame": true
  },
  "step3b": {
    "n_components": 150
  }
}
```

Key parameters to tune:

| Parameter | Default | Effect |
|---|---|---|
| `step1.n_workers` | 8 | Dask workers per experiment |
| `step1.memory_limit` | `"200GB"` | Memory per Dask worker |
| `step3a.radius_factor` | 0.75 | Circular crop radius as fraction of frame |
| `step3a.use_full_frame` | false | Skip cropping entirely |
| `step3b.n_components` | 100 | Max ROI candidates from NNDSVD |
| `step4g.temporal_corr_threshold` | 0.75 | Merge threshold for duplicate ROIs |

---

## Chunk manifest and frame-to-condition mapping

When the pipeline runs, it creates `merged_input/chunk_manifest.json` for each experiment. This maps every merged chunk file back to its source condition and TIF file:

```json
[
  {"chunk": "chunk_000000.avi", "chunk_index": 0, "condition": "1_1", "source_avi": "...", "source_tif": "..."},
  {"chunk": "chunk_000007.avi", "chunk_index": 7, "condition": "1_2", "source_avi": "...", "source_tif": "..."}
]
```

The ordering is always deterministic: conditions are sorted naturally (1_1 → 1_2 → 1_3), files within each condition alphabetically. You can always reconstruct the mapping by counting chunks per condition even without the manifest.

To generate manifests for all experiments after the pipeline has run:

```bash
python -c "
from pipeline.session_discovery import discover_experiment_groups
groups = discover_experiment_groups('/path/to/data_dir', '/path/to/output_dir', avi_root='/path/to/avi_dir')
for g in groups:
    g.setup_merged_dir()
"
```

---

## Project structure

```
mps_ca/
├── run_pipeline.py              # main CLI — discovers experiments, runs pipeline
├── convert_to_avi.py            # TIF → uncompressed AVI via Fiji
├── explore_data.py              # data directory explorer and report generator
├── configs/
│   └── default_config.json      # all MPS step parameters with defaults
├── imagej_macros/
│   └── convert_to_avi.ijm       # ImageJ macro called by convert_to_avi.py
└── pipeline/
    ├── session_discovery.py     # ExperimentGroup / RecordingCondition data model
    ├── headless.py              # tkinter mock — lets MPS modules import without a display
    ├── step_runner.py           # one run_stepXx() function per MPS step
    ├── runner.py                # ExperimentRunner orchestrator with resume logic
    └── __init__.py
```

---

## How it works

MPS is a tkinter GUI application. Running it headlessly requires three adaptations:

1. **Tkinter mocking** (`pipeline/headless.py`): fake tkinter modules are injected into `sys.modules` before any MPS import, so step classes load without a display.
2. **Bypassing widget initialisation**: MPS step classes inherit from `ttk.Frame`. Steps are instantiated with `object.__new__()` to skip `__init__`, then only the attributes the processing threads actually read are wired up.
3. **Step 3a crop computation**: the GUI computes the crop region interactively via a preview. Headless mode computes it from `radius_factor` / `y_offset` / `x_offset` and the first 100 frames of step 2e output.

For multi-condition experiments, all AVIs from all conditions are linked into a single `merged_input/` directory in sorted order. MPS step 2a loads them sequentially, treating all conditions as one continuous recording for ROI extraction. The `chunk_manifest.json` records the mapping so frames can be attributed to conditions afterwards.

---

## Running on a remote Windows workstation

**Native Windows (PowerShell):** the code is cross-platform. Use Windows paths directly:
```powershell
python run_pipeline.py --data_dir "Z:\data_dir" --avi_dir "Z:\avi_dir" --output_dir "Z:\output_dir" --mps_root "C:\path\to\MPS_1.0.0" --workers 6 --resume
```

Symlinks fall back to file copies automatically on Windows.

**Keep running after disconnect** (WSL/Linux):
```bash
screen -S pipeline
python run_pipeline.py ...
# Ctrl+A D to detach — job keeps running
screen -r pipeline   # reattach later
```

---

## Workstation sizing guide

| Setting | Formula | Example (40 cores, 800 GB) |
|---|---|---|
| `--workers` | = number of experiments | 6 |
| `step1.n_workers` | ≈ total cores / workers | 10 |
| `step1.memory_limit` | ≈ total RAM / (workers × n_workers) | `"13GB"` |

Going beyond ~12 Dask workers per experiment gives diminishing returns — most MPS steps are sequentially bottlenecked between Dask operations.
