"""
Headless execution of MPS steps 1–4g.

Each `run_stepXx()` function:
  1. Creates a bare HeadlessStep instance (bypassing tkinter widget creation).
  2. Wires up only the attributes that the processing thread reads from `self`.
  3. Calls the processing thread directly (synchronously, not as a daemon thread).
  4. Returns True on success, raises on failure.

Call setup_headless_env() from pipeline.headless BEFORE importing this module.
"""
import os
import sys
import json
import logging
import importlib
import numpy as np
from typing import Any, Dict, Optional

from .headless import HeadlessController, HeadlessStep, _StringVar, _IntVar, _DoubleVar, _BooleanVar, _MockWidget

logger = logging.getLogger(__name__)


# ── Module-level step class imports (lazy, resolved at first use) ─────────────
# We import lazily so that setup_headless_env() is guaranteed to have run first.

def _import_step(module_name: str, class_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _new_step(controller: HeadlessController, step_label: str) -> HeadlessStep:
    """Return a HeadlessStep bound to *controller*."""
    return HeadlessStep(controller, step_label)


# ── Step 1: Project Configuration ─────────────────────────────────────────────

def run_step1(
    controller: HeadlessController,
    animal_id: int,
    session_id: int,
    input_dir: str,
    output_dir: str,
    n_workers: int = 8,
    memory_limit: str = "200GB",
    output_path_override: Optional[str] = None,
    dask_local_dir: Optional[str] = None,
) -> None:
    """
    Replicates Step1Setup._initialize_thread without tkinter.
    Sets up state, creates output directories, and sets env vars.

    Args:
        output_path_override: When set, use this as dataset_output_path instead
            of the default "{output_dir}/{animal_id}_{session_id}_Processed".
            Used by ExperimentRunner to place outputs at the experiment level.
        dask_local_dir: Local directory for Dask worker scratch files. Should
            point to a fast local disk (e.g. C:/temp/dask-scratch), not a
            network drive. Avoids the "scratch directories taking surprisingly
            long time" warning and associated nanny crashes on Windows.
    """
    logger.info("[Step 1] Initializing project configuration")

    controller.state["animal"] = animal_id
    controller.state["session"] = session_id
    controller.state["input_dir"] = input_dir
    controller.state["output_dir"] = output_dir
    controller.state["n_workers"] = n_workers
    controller.state["memory_limit"] = memory_limit
    controller.state["dask_local_dir"] = dask_local_dir
    controller.state["video_percent"] = 100
    controller.state["initialized"] = True
    controller.state.setdefault("results", {})

    dataset_output_path = output_path_override or os.path.join(
        output_dir, f"{animal_id}_{session_id}_Processed"
    )
    cache_path = os.path.join(dataset_output_path, "cache_data")
    os.makedirs(cache_path, exist_ok=True)

    controller.state["dataset_output_path"] = dataset_output_path
    controller.state["cache_path"] = cache_path

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CACHE_PATH"] = cache_path

    logger.info(f"[Step 1] cache_path = {cache_path}")


# ── Dask configuration ────────────────────────────────────────────────────────

def _configure_dask(local_dir: Optional[str] = None) -> None:
    """
    Configure Dask before step2a creates the LocalCluster.

    1. Sets the worker scratch directory via dask.config so workers don't
       write temp files to the network drive (fixes the
       "scratch directories taking surprisingly long time" warning and the
       associated nanny crashes on Windows).
    2. Monkey-patches LocalCluster to use dashboard_address=":0" so that
       parallel experiments don't all fight over port 8787.
    """
    try:
        import dask

        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            dask.config.set({"temporary-directory": local_dir})
            logger.info(f"[Dask] temporary-directory → {local_dir}")

        # Also patch the dashboard port — dask.config has no clean key for this,
        # so we wrap LocalCluster once per process.
        import dask.distributed as _dd
        _Orig = _dd.LocalCluster
        if getattr(_dd.LocalCluster, "_headless_patched", False):
            return

        def _patched_cluster(*args, **kwargs):
            kwargs["dashboard_address"] = ":0"
            return _Orig(*args, **kwargs)

        _patched_cluster._headless_patched = True
        _dd.LocalCluster = _patched_cluster
        # Also patch the top-level distributed module (same object, but be safe)
        try:
            import distributed as _dist
            if _dist.LocalCluster is _Orig:
                _dist.LocalCluster = _patched_cluster
        except Exception:
            pass

        logger.info("[Dask] LocalCluster patched — dashboard_address=:0")
    except Exception as exc:
        logger.warning(f"[Dask] configuration failed: {exc}")


# ── Step 2a: Video Loading ─────────────────────────────────────────────────────

def run_step2a(
    controller: HeadlessController,
    pattern: str = r".*\.avi$",
    downsample: Optional[Dict] = None,
    downsample_strategy: str = "subset",
    detect_line_splitting: bool = True,
) -> None:
    """Run Step 2a: load AVI files, write zarr, store xarray in state."""
    logger.info("[Step 2a] Loading videos")

    _configure_dask(controller.state.get("dask_local_dir"))

    ds = downsample or {"frame": 1, "height": 1, "width": 1}
    param_load_videos = {
        "pattern": pattern,
        "dtype": np.uint8,
        "downsample": ds,
        "downsample_strategy": downsample_strategy,
        "cache_path": controller.state["cache_path"],
    }

    Step2aVideoLoading = _import_step("step2a_video_loading", "Step2aVideoLoading")
    step = _new_step(controller, "2a")
    Step2aVideoLoading._load_videos_thread(
        step,
        controller.state["input_dir"],
        param_load_videos,
        100,  # video_percent
        controller.state["cache_path"],
        detect_line_splitting,
    )
    _assert_step_result(controller, "step2a", "step2a: video loading")


# ── Step 2b: Background Removal & Denoising ────────────────────────────────────

def run_step2b(
    controller: HeadlessController,
    denoise_method: str = "median",
    kernel_size: int = 3,
    bg_method: str = "tophat",
    bg_window: int = 30,
    processing_order: str = "bg_first",
) -> None:
    """Run Step 2b: background removal and denoising."""
    logger.info("[Step 2b] Background removal and denoising")

    Step2bProcessing = _import_step("step2b_processing", "Step2bProcessing")
    step = _new_step(controller, "2b")
    Step2bProcessing._process_thread(
        step,
        denoise_method,
        kernel_size,
        bg_method,
        bg_window,
        processing_order,
    )
    _assert_step_result(controller, "step2b", "step2b: processing")


# ── Step 2c: Motion Correction ────────────────────────────────────────────────

def run_step2c(
    controller: HeadlessController,
    dim: str = "frame",
    subset_mc: bool = False,
) -> None:
    """Run Step 2c: motion estimation."""
    logger.info("[Step 2c] Motion correction")

    Step2cMotionEstimation = _import_step("step2c_motion_estimation", "Step2cMotionEstimation")
    step = _new_step(controller, "2c")
    Step2cMotionEstimation._estimate_motion_thread(step, dim, subset_mc)
    _assert_step_result(controller, "step2c", "step2c: motion estimation")


# ── Step 2d: Erroneous Frame Detection ───────────────────────────────────────

def run_step2d(
    controller: HeadlessController,
    threshold_factor: float = 3.0,
    drop_frames: bool = True,
) -> None:
    """Run Step 2d: detect and optionally drop erroneous frames."""
    logger.info("[Step 2d] Erroneous frame detection")

    Step2dErroneousFrames = _import_step("step2d_erroneous_frames", "Step2dErroneousFrames")
    step = _new_step(controller, "2d")

    # Inspect the thread signature to pass the right positional args
    import inspect
    sig = inspect.signature(Step2dErroneousFrames._detect_frames_thread)
    params = list(sig.parameters.keys())  # skip 'self'

    kwargs = {
        "step2d_threshold_factor": threshold_factor,
        "step2d_drop_frames": drop_frames,
    }
    # Fill any extra params the thread might take with safe defaults
    call_args = [kwargs.get(p, None) for p in params[1:]]
    Step2dErroneousFrames._detect_frames_thread(step, *call_args)
    _assert_step_result(controller, "step2d", "step2d: erroneous frames")


# ── Step 2e: Transformation / Validation ─────────────────────────────────────

def run_step2e(
    controller: HeadlessController,
    fill_value: float = 0.0,
    create_frame_chunks: bool = True,
    create_spatial_chunks: bool = True,
) -> None:
    """Run Step 2e: apply motion transformation and save chunked zarr."""
    logger.info("[Step 2e] Data transformation")

    Step2eTransformation = _import_step("step2e_transformation", "Step2eTransformation")
    step = _new_step(controller, "2e")
    Step2eTransformation._transform_thread(
        step,
        fill_value,
        create_frame_chunks,
        create_spatial_chunks,
    )
    _assert_step_result(controller, "step2e", "step2e: transformation")


# ── Step 3a: Spatial Cropping ────────────────────────────────────────────────

def run_step3a(
    controller: HeadlessController,
    radius_factor: float = 0.75,
    y_offset: int = 0,
    x_offset: int = 0,
    use_full_frame: bool = False,
) -> None:
    """
    Run Step 3a: crop the video to the circular ROI.

    For headless operation, we compute *current_crop_info* from the configured
    radius/offsets without calling the interactive preview.
    """
    logger.info("[Step 3a] Spatial cropping")

    Step3aCropping = _import_step("step3a_cropping", "Step3aCropping")

    # Compute crop slices from the step2e data already in state
    results = controller.state.get("results", {})
    step2e = results.get("step2e", {})
    Y_hw = step2e.get("step2e_Y_hw_chk") or results.get("step2e_Y_hw_chk")

    if Y_hw is None:
        raise RuntimeError("Step 3a requires step2e_Y_hw_chk in state['results'].")

    if use_full_frame:
        H, W = Y_hw.sizes["height"], Y_hw.sizes["width"]
        crop_slices = {"height": slice(0, H), "width": slice(0, W)}
        half_h, half_w = H // 2, W // 2
    else:
        # Replicate the preview_crop calculation
        n_frames_sample = min(100, Y_hw.sizes["frame"])
        activity = Y_hw.isel(frame=slice(0, n_frames_sample)).mean("frame").compute()
        H, W = activity.sizes["height"], activity.sizes["width"]
        center_y = H // 2 + y_offset
        center_x = W // 2 + x_offset
        half_size = max(10, int(min(H, W) * radius_factor / 2))
        y_start = max(0, center_y - half_size)
        y_stop  = min(H, center_y + half_size)
        x_start = max(0, center_x - half_size)
        x_stop  = min(W, center_x + half_size)
        crop_slices = {
            "height": slice(y_start, y_stop),
            "width":  slice(x_start, x_stop),
        }
        half_h = y_stop - y_start
        half_w = x_stop - x_start

    step = _new_step(controller, "3a")
    # _crop_thread reads self.current_crop_info
    step.current_crop_info = {
        "center_radius_factor": radius_factor,
        "y_offset": y_offset,
        "x_offset": x_offset,
        "crop_slices": crop_slices,
        "reduction": 0.0,
        "final_height": half_h,
        "final_width": half_w,
        "optimal_chunk_size": None,
    }
    # Also provide the get_step2e_data helper that _crop_thread calls
    def _get_step2e_data():
        r = controller.state.get("results", {})
        se = r.get("step2e", {})
        hw = se.get("step2e_Y_hw_chk") or r.get("step2e_Y_hw_chk")
        fm = se.get("step2e_Y_fm_chk") or r.get("step2e_Y_fm_chk")
        return hw, fm

    step.get_step2e_data = _get_step2e_data

    # _crop_thread uses self._get_common_chunk_size and self._calculate_spatial_chunk_size
    # Provide them from the class
    step._get_common_chunk_size = Step3aCropping._get_common_chunk_size.__get__(step, type(step))
    step._calculate_spatial_chunk_size = Step3aCropping._calculate_spatial_chunk_size.__get__(step, type(step))

    Step3aCropping._crop_thread(step)
    _assert_step_result(controller, "step3a", "step3a: cropping")


# ── Step 3b: NNDSVD Initialisation ───────────────────────────────────────────

def run_step3b(
    controller: HeadlessController,
    n_components: int = 100,
    n_power_iter: int = 5,
    sparsity_threshold: float = 0.05,
    spatial_reg: float = 1.0,
    chunk_size: int = 1000,
) -> None:
    """Run Step 3b: NNDSVD spatial component initialisation."""
    logger.info("[Step 3b] NNDSVD initialisation")

    Step3bSvd = _import_step("step3b_svd", "Step3bSvd")
    step = _new_step(controller, "3b")

    import inspect
    sig = inspect.signature(Step3bSvd._nndsvd_thread)
    params = list(sig.parameters.keys())[1:]  # skip 'self'

    # Build the positional argument list from known parameters
    arg_map = {
        "n_components": n_components,
        "n_power_iter": n_power_iter,
        "sparsity_threshold": sparsity_threshold,
        "spatial_reg": spatial_reg,
        "chunk_size": chunk_size,
    }
    call_args = [arg_map.get(p, None) for p in params]
    Step3bSvd._nndsvd_thread(step, *call_args)
    _assert_step_result(controller, "step3b", "step3b: NNDSVD")


# ── Step 4a: Watershed Parameter Search ───────────────────────────────────────

def run_step4a(
    controller: HeadlessController,
    min_distances: list = None,
    threshold_rels: list = None,
    sigmas: list = None,
    sample_size: int = 20,
    include_bg: bool = False,
) -> None:
    """Run Step 4a: grid search for optimal watershed parameters."""
    logger.info("[Step 4a] Watershed parameter search")

    min_distances  = min_distances  or [10, 20, 30]
    threshold_rels = threshold_rels or [0.1, 0.2]
    sigmas         = sigmas         or [1.0, 2.0]

    Step4aWatershedSearch = _import_step("step4a_watershed_search", "Step4aWatershedSearch")
    step = _new_step(controller, "4a")
    Step4aWatershedSearch._parameter_search_thread(
        step, min_distances, threshold_rels, sigmas, sample_size, include_bg
    )
    _assert_step_result(controller, "step4a", "step4a: watershed search")


# ── Step 4b: Watershed Segmentation ───────────────────────────────────────────

def run_step4b(
    controller: HeadlessController,
    min_distance: Optional[int] = None,
    threshold_rel: Optional[float] = None,
    sigma: Optional[float] = None,
    min_size: int = 9,
    apply_filter: bool = True,
) -> None:
    """
    Run Step 4b: apply best watershed parameters.
    If min_distance/threshold_rel/sigma are None, use the values suggested by Step 4a.
    """
    logger.info("[Step 4b] Watershed segmentation")

    # Fall back to step4a suggestions when not explicitly provided
    step4a_res = controller.state.get("results", {}).get("step4a", {})
    wp = step4a_res.get("watershed_params", {})
    if min_distance is None:
        min_distance = wp.get("min_distance", 20)
    if threshold_rel is None:
        threshold_rel = wp.get("threshold_rel", 0.1)
    if sigma is None:
        sigma = wp.get("sigma", 1.0)

    Step4bWatershedSegmentation = _import_step(
        "step4b_watershed_segmentation", "Step4bWatershedSegmentation"
    )
    step = _new_step(controller, "4b")

    import inspect
    sig = inspect.signature(Step4bWatershedSegmentation._segmentation_thread)
    params = list(sig.parameters.keys())[1:]

    arg_map = {
        "min_distance": min_distance,
        "threshold_rel": threshold_rel,
        "sigma": sigma,
        "min_size": min_size,
        "apply_filter": apply_filter,
    }
    call_args = [arg_map.get(p, None) for p in params]
    Step4bWatershedSegmentation._segmentation_thread(step, *call_args)
    _assert_step_result(controller, "step4b", "step4b: segmentation")


# ── Step 4c: Merging Spatial Units ────────────────────────────────────────────

def run_step4c(
    controller: HeadlessController,
    distance_threshold: float = 25.0,
    size_ratio_threshold: float = 5.0,
    min_size: int = 9,
    max_size: int = 10000,
    use_parallel: bool = False,
) -> None:
    """Run Step 4c: merge spatially overlapping components."""
    logger.info("[Step 4c] Merging spatial units")

    Step4cMergingUnits = _import_step("step4c_merging_units", "Step4cMergingUnits")
    step = _new_step(controller, "4c")
    Step4cMergingUnits._merging_thread(
        step,
        distance_threshold,
        size_ratio_threshold,
        min_size,
        max_size,
        use_parallel,
    )
    _assert_step_result(controller, "step4c", "step4c: merging units")


# ── Step 4d: Temporal Signal Extraction ───────────────────────────────────────

def run_step4d(
    controller: HeadlessController,
    batch_size: int = 10,
    frame_chunk_size: int = 10000,
    component_limit: Optional[int] = None,
    use_managed_memory: bool = True,
    use_garbage_collection: bool = True,
) -> None:
    """Run Step 4d: extract temporal signals for each spatial component."""
    logger.info("[Step 4d] Temporal signal extraction")

    Step4dTemporalSignals = _import_step("step4d_temporal_signals", "Step4dTemporalSignals")
    step = _new_step(controller, "4d")

    import inspect
    sig = inspect.signature(Step4dTemporalSignals._extraction_thread)
    params = list(sig.parameters.keys())[1:]

    arg_map = {
        "batch_size": batch_size,
        "frame_chunk_size": frame_chunk_size,
        "component_limit": component_limit,
        "use_managed_memory": use_managed_memory,
        "use_garbage_collection": use_garbage_collection,
    }
    call_args = [arg_map.get(p, None) for p in params]
    Step4dTemporalSignals._extraction_thread(step, *call_args)
    _assert_step_result(controller, "step4d", "step4d: temporal extraction")


# ── Step 4e: A/C Initialisation ───────────────────────────────────────────────

def run_step4e(
    controller: HeadlessController,
    spatial_norm: str = "max",
    min_size: int = 1,
    max_components: Optional[int] = None,
    skip_bg: bool = True,
    check_nan: bool = True,
) -> None:
    """Run Step 4e: prepare A and C matrices for CNMF."""
    logger.info("[Step 4e] A/C initialisation")

    Step4eAcInitialization = _import_step("step4e_ac_initialization", "Step4eAcInitialization")
    step = _new_step(controller, "4e")
    Step4eAcInitialization._initialization_thread(
        step, spatial_norm, min_size, max_components, skip_bg, check_nan
    )
    _assert_step_result(controller, "step4e", "step4e: AC init")


# ── Step 4f: Drop NaN / empty / flat components ───────────────────────────────

def run_step4f(
    controller: HeadlessController,
    max_components: Optional[int] = None,
    check_nan: bool = True,
    check_empty: bool = True,
    check_flat: bool = True,
) -> None:
    """Run Step 4f: quality-control filter before expensive CNMF steps."""
    logger.info("[Step 4f] Dropping bad components")

    Step4fDroppingNans = _import_step("step4f_dropping_nans", "Step4fDroppingNans")
    step = _new_step(controller, "4f")
    Step4fDroppingNans._processing_thread(
        step, max_components, check_nan, check_empty, check_flat
    )
    _assert_step_result(controller, "step4f", "step4f: component filter")


# ── Step 4g: Temporal Merging ─────────────────────────────────────────────────

def run_step4g(
    controller: HeadlessController,
    temporal_corr_threshold: float = 0.75,
    spatial_overlap_threshold: float = 0.3,
    max_size: int = 10000,
    input_type: str = "clean",
    max_components: Optional[int] = None,
) -> None:
    """Run Step 4g: merge temporally correlated spatial duplicates."""
    logger.info("[Step 4g] Temporal merging")

    Step4gTemporalMerging = _import_step("step4g_temporal_merging", "Step4gTemporalMerging")
    step = _new_step(controller, "4g")
    Step4gTemporalMerging._merging_thread(
        step,
        temporal_corr_threshold,
        spatial_overlap_threshold,
        max_size,
        input_type,
        max_components,
    )
    _assert_step_result(controller, "step4g", "step4g: temporal merging")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_step_result(controller: HeadlessController, step_key: str, label: str):
    """Raise RuntimeError if a step failed to write results to state."""
    results = controller.state.get("results", {})
    if step_key not in results:
        raise RuntimeError(
            f"{label} did not produce any output in state['results']['{step_key}']. "
            "Check logs above for the underlying error."
        )
    logger.info(f"[{step_key}] Results written to state. Keys: {list(results[step_key].keys()) if isinstance(results[step_key], dict) else type(results[step_key])}")
