"""
SessionRunner / ExperimentRunner: orchestrate the full steps 1–4g pipeline.

  SessionRunner   — for a single VideoSession (one directory of AVI files).
  ExperimentRunner — for an ExperimentGroup (multiple recording conditions
                     from the same FOV that share ROIs).

Usage (called from run_pipeline.py via multiprocessing):

    runner = ExperimentRunner(group, config, mps_root)
    runner.run()
"""
import os
import logging
import time
import traceback
from typing import Any, Dict

from .session_discovery import VideoSession, ExperimentGroup, _STEP_MARKERS

logger = logging.getLogger(__name__)

_STEP_ORDER = [
    "step2a", "step2b", "step2c", "step2d", "step2e",
    "step3a", "step3b",
    "step4a", "step4b", "step4c", "step4d", "step4e", "step4f", "step4g",
]


class SessionRunner:
    """
    Runs the full MPS steps 1–4g pipeline for one VideoSession.

    Args:
        session:  VideoSession describing inputs/outputs.
        config:   Parameter dictionary (see configs/default_config.json).
        mps_root: Absolute path to MPS_1.0.0 directory.
        resume:   If True, skip steps whose output zarr files already exist.
    """

    def __init__(
        self,
        session: VideoSession,
        config: Dict[str, Any],
        mps_root: str,
        resume: bool = True,
    ):
        self.session = session
        self.config = config
        self.mps_root = mps_root
        self.resume = resume

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> bool:
        """Execute the pipeline. Returns True on success, False on error."""
        self._configure_logging()
        logger.info(f"=== Starting pipeline for {self.session.label} ===")
        logger.info(f"Output: {self.session.dataset_output_path}")

        t0 = time.time()
        try:
            self._run_pipeline()
            elapsed = time.time() - t0
            logger.info(f"=== {self.session.label} COMPLETE in {elapsed/60:.1f} min ===")
            return True
        except Exception:
            elapsed = time.time() - t0
            logger.error(f"=== {self.session.label} FAILED after {elapsed/60:.1f} min ===")
            logger.error(traceback.format_exc())
            return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _configure_logging(self):
        os.makedirs(self.session.dataset_output_path, exist_ok=True)
        fh = logging.FileHandler(self.session.log_path, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(fh)

    def _skip_step(self, step_key: str) -> bool:
        if not self.resume:
            return False
        markers = _STEP_MARKERS.get(step_key, [])
        if any(os.path.exists(os.path.join(self.session.cache_dir, m)) for m in markers):
            logger.info(f"[{step_key}] Already complete — skipping.")
            return True
        return False

    def _run_pipeline(self):
        from .headless import setup_headless_env, HeadlessController
        setup_headless_env(self.mps_root)
        from . import step_runner as sr

        c = self.config
        ctrl = HeadlessController(state={"results": {}})

        sr.run_step1(
            ctrl,
            animal_id=self.session.animal_id,
            session_id=self.session.session_id,
            input_dir=self.session.input_dir,
            output_dir=self.session.output_root,
            n_workers=c["step1"]["n_workers"],
            memory_limit=c["step1"]["memory_limit"],
            dask_local_dir=c["step1"].get("dask_local_dir"),
        )

        self._run_steps_2_to_4g(ctrl, sr, c)

    def _run_steps_2_to_4g(self, ctrl, sr, c):
        """Execute steps 2a through 4g with skip/reload logic."""

        # ── Step 2a: video loading ────────────────────────────────────────────
        if not self._skip_step("step2a"):
            sr.run_step2a(
                ctrl,
                pattern=c["step2a"]["pattern"],
                downsample=c["step2a"]["downsample"],
                downsample_strategy=c["step2a"]["downsample_strategy"],
                detect_line_splitting=c["step2a"]["detect_line_splitting"],
            )
        else:
            self._reload_step("step2a", ctrl)

        # ── Step 2b: background removal ───────────────────────────────────────
        if not self._skip_step("step2b"):
            sr.run_step2b(
                ctrl,
                denoise_method=c["step2b"]["denoise_method"],
                kernel_size=c["step2b"]["kernel_size"],
                bg_method=c["step2b"]["bg_method"],
                bg_window=c["step2b"]["bg_window"],
                processing_order=c["step2b"]["processing_order"],
            )
        else:
            self._reload_step("step2b", ctrl)

        # ── Step 2c: motion correction ────────────────────────────────────────
        if not self._skip_step("step2c"):
            sr.run_step2c(
                ctrl,
                dim=c["step2c"]["dim"],
                subset_mc=c["step2c"]["subset_mc"],
            )
        else:
            self._reload_step("step2c", ctrl)

        # ── Step 2d: erroneous frame detection ────────────────────────────────
        if not self._skip_step("step2d"):
            sr.run_step2d(
                ctrl,
                threshold_factor=c["step2d"]["threshold_factor"],
                drop_frames=c["step2d"]["drop_frames"],
            )
        else:
            self._reload_step("step2d", ctrl)

        # ── Step 2e: transformation ───────────────────────────────────────────
        if not self._skip_step("step2e"):
            sr.run_step2e(
                ctrl,
                fill_value=c["step2e"]["fill_value"],
                create_frame_chunks=c["step2e"]["create_frame_chunks"],
                create_spatial_chunks=c["step2e"]["create_spatial_chunks"],
            )
        else:
            self._reload_step("step2e", ctrl)

        # ── Step 3a: cropping ─────────────────────────────────────────────────
        if not self._skip_step("step3a"):
            sr.run_step3a(
                ctrl,
                radius_factor=c["step3a"]["radius_factor"],
                y_offset=c["step3a"]["y_offset"],
                x_offset=c["step3a"]["x_offset"],
                use_full_frame=c["step3a"]["use_full_frame"],
            )
        else:
            self._reload_step("step3a", ctrl)

        # ── Step 3b: NNDSVD ───────────────────────────────────────────────────
        if not self._skip_step("step3b"):
            sr.run_step3b(
                ctrl,
                n_components=c["step3b"]["n_components"],
                n_power_iter=c["step3b"]["n_power_iter"],
                sparsity_threshold=c["step3b"]["sparsity_threshold"],
                spatial_reg=c["step3b"]["spatial_reg"],
                chunk_size=c["step3b"]["chunk_size"],
            )
        else:
            self._reload_step("step3b", ctrl)

        # ── Step 4a: watershed search ─────────────────────────────────────────
        if not self._skip_step("step4a"):
            sr.run_step4a(
                ctrl,
                min_distances=c["step4a"]["min_distances"],
                threshold_rels=c["step4a"]["threshold_rels"],
                sigmas=c["step4a"]["sigmas"],
                sample_size=c["step4a"]["sample_size"],
                include_bg=c["step4a"]["include_bg"],
            )
        else:
            self._reload_step("step4a", ctrl)

        # ── Step 4b: segmentation ─────────────────────────────────────────────
        if not self._skip_step("step4b"):
            sr.run_step4b(
                ctrl,
                min_distance=c["step4b"].get("min_distance"),
                threshold_rel=c["step4b"].get("threshold_rel"),
                sigma=c["step4b"].get("sigma"),
                min_size=c["step4b"]["min_size"],
            )
        else:
            self._reload_step("step4b", ctrl)

        # ── Step 4c: merging units ────────────────────────────────────────────
        if not self._skip_step("step4c"):
            sr.run_step4c(
                ctrl,
                distance_threshold=c["step4c"]["distance_threshold"],
                size_ratio_threshold=c["step4c"]["size_ratio_threshold"],
                min_size=c["step4c"]["min_size"],
                max_size=c["step4c"]["max_size"],
                use_parallel=c["step4c"]["use_parallel"],
            )
        else:
            self._reload_step("step4c", ctrl)

        # ── Step 4d: temporal extraction ──────────────────────────────────────
        if not self._skip_step("step4d"):
            sr.run_step4d(
                ctrl,
                batch_size=c["step4d"]["batch_size"],
                frame_chunk_size=c["step4d"]["frame_chunk_size"],
                component_limit=c["step4d"].get("component_limit"),
                use_managed_memory=c["step4d"]["use_managed_memory"],
                use_garbage_collection=c["step4d"]["use_garbage_collection"],
            )
        else:
            self._reload_step("step4d", ctrl)

        # ── Step 4e: A/C init ─────────────────────────────────────────────────
        if not self._skip_step("step4e"):
            sr.run_step4e(
                ctrl,
                spatial_norm=c["step4e"]["spatial_norm"],
                min_size=c["step4e"]["min_size"],
                max_components=c["step4e"].get("max_components"),
                skip_bg=c["step4e"]["skip_bg"],
                check_nan=c["step4e"]["check_nan"],
            )
        else:
            self._reload_step("step4e", ctrl)

        # ── Step 4f: drop bad components ──────────────────────────────────────
        if not self._skip_step("step4f"):
            sr.run_step4f(
                ctrl,
                max_components=c["step4f"].get("max_components"),
                check_nan=c["step4f"]["check_nan"],
                check_empty=c["step4f"]["check_empty"],
                check_flat=c["step4f"]["check_flat"],
            )
        else:
            self._reload_step("step4f", ctrl)

        # ── Step 4g: temporal merging ─────────────────────────────────────────
        if not self._skip_step("step4g"):
            sr.run_step4g(
                ctrl,
                temporal_corr_threshold=c["step4g"]["temporal_corr_threshold"],
                spatial_overlap_threshold=c["step4g"]["spatial_overlap_threshold"],
                max_size=c["step4g"]["max_size"],
                input_type=c["step4g"]["input_type"],
                max_components=c["step4g"].get("max_components"),
            )

        logger.info(f"[{self.session.label}] Pipeline finished. ROIs in {self.session.cache_dir}")

    def _reload_step(self, step_key: str, ctrl) -> None:
        """Reload a previously completed step's zarr outputs into state['results']."""
        import xarray as xr

        cache = ctrl.state["cache_path"]
        results = ctrl.state.setdefault("results", {})
        step_dict = results.setdefault(step_key, {})

        markers = _STEP_MARKERS.get(step_key, [])
        for marker in markers:
            zarr_path = os.path.join(cache, marker)
            if not os.path.isdir(zarr_path):
                continue
            var_name = marker.replace(".zarr", "")
            try:
                ds = xr.open_zarr(zarr_path)
                if var_name in ds:
                    step_dict[var_name] = ds[var_name]
                else:
                    var = list(ds.data_vars)[0]
                    step_dict[var_name] = ds[var]
                logger.info(f"[{step_key}] Reloaded {marker}")
            except Exception as e:
                logger.warning(f"[{step_key}] Could not reload {marker}: {e}")

        for var_name, val in list(step_dict.items()):
            results[var_name] = val


# ── ExperimentRunner ──────────────────────────────────────────────────────────

class ExperimentRunner(SessionRunner):
    """
    Runs the pipeline for one ExperimentGroup.

    Before step 1, sets up the merged_input/ directory (symlinks to all AVIs
    across all recording conditions).  Step 1 is then pointed at merged_input/
    and uses the experiment's output path directly (not animal_id/session_id).

    Steps 2a–4g are identical to SessionRunner.
    """

    def __init__(
        self,
        group: ExperimentGroup,
        config: Dict[str, Any],
        mps_root: str,
        resume: bool = True,
    ):
        self.session = group      # duck-typed: ExperimentGroup has the same interface
        self.config = config
        self.mps_root = mps_root
        self.resume = resume

    def _run_pipeline(self):
        # Merge all AVI files from all conditions into a single staging directory
        self.session.setup_merged_dir()

        from .headless import setup_headless_env, HeadlessController
        setup_headless_env(self.mps_root)
        from . import step_runner as sr

        c = self.config
        ctrl = HeadlessController(state={"results": {}})

        # Use output_path_override so the output goes to the experiment directory,
        # not to a {animal_id}_{session_id}_Processed subdirectory.
        sr.run_step1(
            ctrl,
            animal_id=0,
            session_id=0,
            input_dir=self.session.merged_dir,
            output_dir=self.session.output_root,
            n_workers=c["step1"]["n_workers"],
            memory_limit=c["step1"]["memory_limit"],
            output_path_override=self.session.dataset_output_path,
            dask_local_dir=c["step1"].get("dask_local_dir"),
        )

        self._run_steps_2_to_4g(ctrl, sr, c)
