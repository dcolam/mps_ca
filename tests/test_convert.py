import dask.array as da
import numpy as np
import pytest

from conftest import load_recording, v2_proc
from mpsca.io import ConvertParams, ExperimentStore, convert_recording, open_tiff_lazy, scan_recording
from mpsca.preprocess import bin_movie


def test_open_tiff_lazy_reads_own_pages(synthetic_experiment):
    stem = synthetic_experiment["order"][0]
    path = synthetic_experiment["exp"] / "1_1" / f"{stem}.tif"
    stores = []
    arr = open_tiff_lazy(path, stores)
    assert isinstance(arr, da.Array)
    np.testing.assert_array_equal(arr.compute(), synthetic_experiment["movies"][stem])
    stores[0].close()


@pytest.mark.parametrize("t,s", [(1, 1), (4, 2), (3, 3)])
def test_bin_movie_matches_v2(t, s):
    a = np.random.default_rng(1).integers(0, 4000, size=(50, 33, 47), dtype=np.uint16)
    out = bin_movie(da.from_array(a, chunks=(7, 33, 47)), t, s).compute()
    np.testing.assert_allclose(out, v2_proc(a, t, s), rtol=1e-6)


def _convert(synthetic_experiment, tmp_path, **kw):
    rec = load_recording(synthetic_experiment["root"], synthetic_experiment["xlsx"])
    seen = []
    kw.setdefault("spatial_binning", 2)  # v2 binned 2x2 in software
    store = convert_recording(rec, tmp_path / "exp.mpsca.zarr", ConvertParams(**kw),
                              progress=seen.append)
    return store, seen


def test_convert_matches_v2_and_masks_bad_frames(synthetic_experiment, tmp_path):
    store, seen = _convert(synthetic_experiment, tmp_path)
    movies = synthetic_experiment["movies"]
    ref = np.concatenate([v2_proc(movies[s], 4, 2) for s in synthetic_experiment["order"]])
    np.testing.assert_allclose(store.movie.compute(), ref, rtol=1e-6)
    assert seen and seen[-1] == pytest.approx(1.0)

    frames = store.read_frames()
    assert len(frames) == ref.shape[0]  # bad frames are kept, only flagged
    bad = np.flatnonzero(frames["is_bad"])
    assert bad.tolist() == [5 + 1]  # 5 binned frames from file 1, then binned frame 1 of file 2
    # time follows the raw frame position: file 1 has 23 raw frames (3 trimmed by binning)
    np.testing.assert_allclose(frames["time_s"][:7], [0, 1, 2, 3, 4, 5.75, 6.75])
    assert list(frames["condition"].unique()) == ["2mM Glucose · Starvation · 10s", "20mM Glucose · Sync · 5s"]
    assert frames["use_for_seg"].tolist() == [True] * 5 + [False] * 4 + [True] * 5


def test_frame_count_warnings_and_reopen(synthetic_experiment, tmp_path):
    store, _ = _convert(synthetic_experiment, tmp_path)
    warnings = store.meta["frame_warnings"]
    assert len(warnings) == 1 and "20mM Glucose" in warnings[0]  # 21 on disk, 24 listed
    reopened = ExperimentStore.open(tmp_path / "exp.mpsca.zarr", mode="r")
    assert reopened.meta["description"]["Binning"] == "2*2"
    assert [b["condition"] for b in reopened.meta["blocks"]] == ["2mM Glucose · Starvation · 10s", "20mM Glucose · Sync · 5s"]
    assert reopened.params("convert")["spatial_binning"] == 2
    assert reopened.read_frames().equals(store.read_frames())


def test_refuses_to_overwrite(synthetic_experiment, tmp_path):
    _convert(synthetic_experiment, tmp_path)
    with pytest.raises(Exception):
        _convert(synthetic_experiment, tmp_path)


def test_convert_with_many_chunks(synthetic_experiment, tmp_path, monkeypatch):
    """Real frames are ~3.5 MB, so binning and the store see many irregular chunks."""
    import mpsca.preprocess.binning as binning
    monkeypatch.setattr(binning, "_TARGET_CHUNK_BYTES", 3 * 30 * 41 * 2)  # ~3 raw frames
    store, _ = _convert(synthetic_experiment, tmp_path, chunk_frames=4)
    movies = synthetic_experiment["movies"]
    ref = np.concatenate([v2_proc(movies[s], 4, 2) for s in synthetic_experiment["order"]])
    assert store.group["movie"].chunks[0] == 4
    np.testing.assert_allclose(store.movie.compute(), ref, rtol=1e-6)


def test_geometry_recorded(synthetic_experiment, tmp_path):
    store, _ = _convert(synthetic_experiment, tmp_path, pixel_size_um=0.65)
    meta = store.meta
    assert meta["raw_shapes"] == [[23, 30, 41], [17, 30, 41], [21, 30, 41]]
    assert meta["camera_binning"] == []  # taken from Micro-Manager page metadata; none here
    assert meta["pixel_size_um"] == pytest.approx(0.65)


def test_mixed_resolution_and_timestamps(mixed_resolution_experiment, tmp_path):
    d = mixed_resolution_experiment
    rec = load_recording(d["exp"])
    store = convert_recording(rec, tmp_path / "m.mpsca.zarr")

    # default: bring the unbinned block down to the camera-binned size, nothing more
    ref = np.concatenate([v2_proc(d["hi"], 4, 2), v2_proc(d["lo"], 4, 1)])
    np.testing.assert_allclose(store.movie.compute(), ref, rtol=1e-6)

    frames = store.read_frames()
    assert frames["spatial_factor"].tolist() == [2] * 4 + [1] * 3
    assert frames["camera_binning"].tolist() == [1] * 4 + [2] * 3
    # raw frame 9 has no metadata (filled from its neighbours); block 1 starts after the pause
    np.testing.assert_allclose(frames["time_s"], [0, 1, 2, 3, 604.5, 605.5, 606.5], atol=1e-3)
    assert any("frame size differs" in w for w in store.meta["frame_warnings"])
    assert store.meta["pixel_size_um"] == pytest.approx(0.65 * 2)
    assert store.complete


def test_spatial_factors():
    from mpsca.io.convert import spatial_factors
    assert spatial_factors([(2200, 3200), (1100, 1600)], 2) == [4, 2]
    with pytest.raises(ValueError):
        spatial_factors([(2200, 3200), (1000, 1600)], 2)


def test_fill_missing_timestamps():
    from mpsca.io.convert import _fill_missing
    ts = np.array([np.nan, 10.0, np.nan, np.nan, 11.0, np.nan])
    np.testing.assert_allclose(_fill_missing(ts, 4.0), [9.75, 10, 10.25, 10.75, 11, 11.25])


def test_scan_and_block_overrides(mixed_resolution_experiment, tmp_path):
    d = mixed_resolution_experiment
    rec = load_recording(d["exp"])
    report = scan_recording(rec)
    assert report.resolution_mismatch
    assert report.proposed_factors() == {0: 2, 1: 1}
    assert report.proposed_factors(2) == {0: 4, 1: 2}
    summary = report.block_summary()
    assert summary["frames"].tolist() == [18, 12]
    assert summary["size"].tolist() == ["20x24", "10x12"]

    # user picks 4x4 / 2x2 explicitly
    params = ConvertParams(block_spatial_factors={0: 4, 1: 2})
    store = convert_recording(rec, tmp_path / "o.mpsca.zarr", params, report=report)
    assert store.movie.shape[1:] == (5, 6)
    # inconsistent choice is refused before anything is written
    with pytest.raises(ValueError, match="different frame sizes"):
        convert_recording(rec, tmp_path / "bad.mpsca.zarr",
                           ConvertParams(block_spatial_factors={0: 2, 1: 2}), report=report)
    assert not (tmp_path / "bad.mpsca.zarr").exists()


def test_unbinned_experiment_goes_to_common_grid(tmp_path):
    """Like WT Exp1: every block unbinned -> proposal bins 2x2 and the user is asked."""
    from conftest import write_mm_tiff

    rng = np.random.default_rng(5)
    for i, sub in enumerate(["1_1", "1_2"]):
        (tmp_path / "e" / sub).mkdir(parents=True)
        write_mm_tiff(tmp_path / "e" / sub / "a.ome.tif",
                      rng.integers(90, 110, (8, 20, 24), dtype=np.uint16), 1.7e9 + 100 * i, binning=1)
    report = scan_recording(load_recording(tmp_path / "e"))
    assert not report.resolution_mismatch
    assert report.proposed_factors() == {0: 2, 1: 2}
    assert report.needs_decision()
    assert report.proposed_factors(target_camera_binning=0) == {0: 1, 1: 1}
    assert not report.needs_decision(target_camera_binning=0)


def test_detected_fps_jitter_is_tolerated():
    assert ConvertParams(source_fps=3.98, target_fps=1).temporal_factor == 4
    assert ConvertParams(source_fps=4.02, target_fps=1).temporal_factor == 4
    with pytest.raises(ValueError):
        ConvertParams(source_fps=4.5, target_fps=1).temporal_factor


def test_blocks_put_in_time_order(tmp_path):
    """Folder names need not sort in acquisition order; timestamps decide."""
    from conftest import write_mm_tiff

    rng = np.random.default_rng(6)
    for name, start in [("1200", 0), ("40X BF", 5000), ("2400", 1000)]:
        (tmp_path / "e" / name).mkdir(parents=True)
        write_mm_tiff(tmp_path / "e" / name / f"{name}_MMStack_Default.ome.tif",
                      rng.integers(90, 110, (8, 10, 12), dtype=np.uint16), 1.7e9 + start)
    rec = load_recording(tmp_path / "e")
    assert [rec.conditions()[b] for b in sorted(rec.conditions())] == ["40X BF", "1200", "2400"]
    report = scan_recording(rec)
    assert [f.entry.condition for f in report.files] == ["1200", "2400", "40X BF"]
    assert any("acquisition order" in m for m in report.issues)


def test_single_images_are_excluded(tmp_path):
    import tifffile
    from conftest import write_mm_tiff

    (tmp_path / "e" / "1_1").mkdir(parents=True)
    (tmp_path / "e" / "40X").mkdir()
    write_mm_tiff(tmp_path / "e" / "1_1" / "a.ome.tif", np.ones((8, 10, 12), np.uint16), 1.7e9)
    tifffile.imwrite(tmp_path / "e" / "40X" / "40X_MMStack_Default.ome.tif", np.ones((10, 12), np.uint16))
    rec = load_recording(tmp_path / "e")
    assert rec.n_blocks == 2
    report = scan_recording(rec)
    assert len(report.files) == 1 and rec.n_blocks == 1
    assert [f.note for f in rec.files if not f.include] == ["single image, not a video"]
