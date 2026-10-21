from pathlib import Path

import numpy as np
import openpyxl
import pytest
import tifffile

REPO = Path(__file__).resolve().parents[1]

HEADER = ["Main folder", "Subfolder", "File name", "Application", "Phase", "Time",
          "Binning", "Frames (4fps)", "Segmentation"]


@pytest.fixture
def synthetic_experiment(tmp_path):
    """Two condition blocks, three files, with a few dark frames in the second file.

    Layout mirrors the real data: ``<exp>/<subfolder>/<name>.ome.tif`` and a sheet
    whose Subfolder column holds a frame count on the first row (a quirk of Exp1).
    """
    rng = np.random.default_rng(0)
    exp = tmp_path / "data" / "01_01_25_Min6WT_Exp9"
    shape = (21, 30, 41)  # odd sizes exercise trimming
    movies = {}
    layout = [("1_1", "1_1_MMStack_Default.ome", 23), ("1_1", "1_1_MMStack_Default_1.ome", 17),
              ("1_2", "1_2_MMStack_Default.ome", 21)]
    for sub, stem, n in layout:
        a = rng.integers(900, 1100, size=(n, *shape[1:]), dtype=np.uint16)
        if stem.endswith("_1.ome"):
            a[4:8] = 50  # dark frames -> bad after 4x temporal binning (binned frame 1)
        (exp / sub).mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(exp / sub / f"{stem}.tif", a, ome=True, metadata={"axes": "TYX"})
        movies[stem] = a

    xlsx = tmp_path / "meta.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Min6 WT Exp9"
    ws.append(HEADER)
    ws.append([exp.name, 40, "1_1_MMStack_Default.ome", "2mM Glucose", "Starvation", "10s", "2*2", 40, True])
    ws.append([None, None, "1_1_MMStack_Default_1.ome", None, None, None, None, None, False])
    ws.append([None, "1_2", "1_2_MMStack_Default.ome", "20mM Glucose", "Sync", "5s", None, 24, True])
    wb.save(xlsx)
    return {"root": exp.parent, "exp": exp, "xlsx": xlsx, "movies": movies,
            "order": [stem for _, stem, _ in layout]}


def load_recording(root, xlsx=None, grouping="root"):
    """The single recording under ``root``, described by ``xlsx`` when given."""
    from mpsca.io import find_files, group_recordings, match_workbook, read_workbook, resolve

    files = find_files(root)
    if xlsx:
        recs, issues = match_workbook(root, files, read_workbook(xlsx))
        assert not issues, issues
    else:
        recs = group_recordings(root, files, grouping)
    resolve(root, recs)
    assert len(recs) == 1
    return recs[0]


def v2_proc(arr, ratio, b):
    """Reference binning copied from ca_pipeline_v2.ipynb (``_proc``)."""
    a = arr.astype(np.float32)
    T, H, W = a.shape
    Tt = (T // ratio) * ratio
    a = a[:Tt].reshape(Tt // ratio, ratio, H, W).mean(axis=1)
    if b > 1:
        T2, H2, W2 = a.shape
        Hb, Wb = H2 // b, W2 // b
        a = a[:, :Hb * b, :Wb * b].reshape(T2, Hb, b, Wb, b).mean(axis=(2, 4))
    return a


def write_mm_tiff(path, frames, t0, fps=4.0, binning=1, missing_md=()):
    """Write frames page by page with Micro-Manager-style per-page JSON (tag 51123)."""
    import json
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=1))
    with tifffile.TiffWriter(path) as w:
        for i, frame in enumerate(frames):
            tags = []
            if i not in missing_md:
                ts = datetime.fromtimestamp(t0 + i / fps, tz)
                md = {"ReceivedTime": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " +0100",
                      "Binning": binning, "PixelSizeUm": 0.65 * binning}
                tags = [(51123, "s", 0, json.dumps(md), True)]
            w.write(frame, contiguous=False, metadata=None, extratags=tags)


@pytest.fixture
def mixed_resolution_experiment(tmp_path):
    """Like Exp1 KO: block 0 unbinned (2× size), block 1 camera-binned, 600 s pause between."""
    rng = np.random.default_rng(2)
    exp = tmp_path / "exp_mixed"
    t0 = 1_762_348_906.0
    hi = rng.integers(900, 1100, size=(18, 20, 24), dtype=np.uint16)
    lo = rng.integers(900, 1100, size=(12, 10, 12), dtype=np.uint16)
    (exp / "1_1").mkdir(parents=True)
    (exp / "1_2").mkdir()
    write_mm_tiff(exp / "1_1" / "a.ome.tif", hi, t0, binning=1, missing_md={9})
    write_mm_tiff(exp / "1_2" / "b.ome.tif", lo, t0 + 18 / 4 + 600, binning=2)
    return {"exp": exp, "hi": hi, "lo": lo}


def make_cell_movie(n_cells=12, shape=(96, 128), T=240, seed=3, offset=100.0, noise=1.5):
    """Disk-shaped cells with independent calcium-like transients on a camera offset."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    truth = np.zeros(shape, np.int32)
    centres = []
    while len(centres) < n_cells:
        c = rng.uniform([10, 10], [shape[0] - 10, shape[1] - 10])
        if all(np.hypot(*(c - d)) > 16 for d in centres):
            centres.append(c)
    for i, (cy, cx) in enumerate(centres, 1):
        truth[(yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2] = i
    traces = np.zeros((n_cells, T), np.float32)
    kernel = np.exp(-np.arange(20) / 4.0)
    for i in range(n_cells):
        spikes = (rng.random(T) < 0.03).astype(np.float32) * rng.uniform(4, 8, T)
        traces[i] = np.convolve(spikes, kernel)[:T]
    movie = np.full((T, *shape), offset, np.float32) + 2.0
    for i in range(n_cells):
        movie[:, truth == i + 1] += 2.0 + traces[i][:, None]
    movie += rng.normal(0, noise, movie.shape).astype(np.float32)
    return movie, truth, traces


@pytest.fixture
def synthetic_store(tmp_path):
    import pandas as pd
    from mpsca.io import ExperimentStore

    movie, truth, traces = make_cell_movie()
    T = movie.shape[0]
    movie[100] = 20.0  # one dark frame
    store = ExperimentStore.create(tmp_path / "syn.mpsca.zarr")
    arr = store.create_movie(movie.shape, (16, *movie.shape[1:]))
    arr[...] = movie
    time_s = np.arange(T, dtype=float)
    time_s[120:] += 600  # pause between the two blocks
    store.write_frames(pd.DataFrame({
        "source_file": "syn.tif", "source_frame": np.arange(T, dtype=np.int32),
        "block": np.where(np.arange(T) < 120, 0, 1).astype(np.int16),
        "condition": np.where(np.arange(T) < 120, "2mM Glucose", "20mM Glucose"),
        "use_for_seg": True, "time_s": time_s,
        "frame_mean": movie.mean(axis=(1, 2)), "is_bad": False,
    }))
    store.set_meta(name="syn_1", description={"group": "A"}, fps=1.0)
    store.set_params("convert", {"target_fps": 1.0})
    store.mark_complete()
    return {"store": store, "truth": truth, "traces": traces}


@pytest.fixture
def two_recordings(tmp_path):
    """R1 camera-binned (2x2), R2 unbinned; two blocks each, recorded at 4 fps."""
    root = tmp_path / "data"
    movie, truth, _ = make_cell_movie(T=160, shape=(48, 64), n_cells=6)
    raw = np.repeat(movie, 4, axis=0).round().astype(np.uint16)  # 4 fps raw -> 1 fps working
    t0 = 1.7e9
    for rec, binning in (("R1", 2), ("R2", 1)):
        frames = raw if binning == 2 else np.repeat(np.repeat(raw, 2, axis=1), 2, axis=2)
        for i, sub in enumerate(("1_1", "1_2")):
            (root / rec / sub).mkdir(parents=True)
            part = frames[i * 320:(i + 1) * 320]
            write_mm_tiff(root / rec / sub / f"{sub}_MMStack_Default.ome.tif", part,
                          t0 + i * 400, binning=binning)
    (root / "R1" / "results").mkdir()
    (root / "R1" / "results" / "mean_image.tif").write_bytes(b"not matched by the pattern")
    return {"root": root, "out": tmp_path / "out", "truth": truth}
