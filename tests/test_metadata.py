import numpy as np
import openpyxl
import pytest
import tifffile

from conftest import REPO, load_recording
from mpsca.io import find_files, group_recordings, match_workbook, read_workbook
from mpsca.io.metadata import acquisition_key, file_stem


def _touch(root, rels):
    for rel in rels:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(root / rel, np.zeros((2, 4, 4), np.uint16))


def _workbook(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


def test_synthetic_workbook(synthetic_experiment):
    rec = load_recording(synthetic_experiment["root"], synthetic_experiment["xlsx"])
    assert rec.name == synthetic_experiment["exp"].name
    assert [f.block for f in rec.files] == [0, 0, 1]
    assert [f.condition for f in rec.files] == ["2mM Glucose · Starvation · 10s"] * 2 + ["20mM Glucose · Sync · 5s"]
    assert [f.use_for_seg for f in rec.files] == [True, False, True]
    assert [f.expected_frames for f in rec.files] == [40, 40, 24]
    assert rec.files[2].info == {"Application": "20mM Glucose", "Phase": "Sync", "Time": "5s", "Binning": "2*2"}
    assert rec.description == {"sheet": "Min6 WT Exp9", "Binning": "2*2"}
    assert all(f.path is not None for f in rec.files)


def test_names_and_acquisitions():
    assert file_stem("1_1_MMStack_Default.ome") == file_stem("x/1_1_MMStack_Default.ome.tif") == "1_1_mmstack_default"
    assert acquisition_key("e/1_2/1_2_MMStack_Default_7.ome.tif") == "e/1_2/1_2_MMStack_Default"
    assert acquisition_key("e/a_1.ome.tif") != acquisition_key("e/a_2.ome.tif")  # not Micro-Manager parts


def test_folder_grouping_and_order(tmp_path):
    _touch(tmp_path, ["A/1_2/b_MMStack_Default.ome.tif", "A/1_10/c_MMStack_Default.ome.tif",
                      "A/1_2/b_MMStack_Default_1.ome.tif", "A/1_1/a_MMStack_Default.ome.tif",
                      "B/x.ome.tif", "B/notes.tif", "A/out/mean_image.tif"])
    files = find_files(tmp_path, "*.ome.tif")
    assert "B/notes.tif" not in files and "A/out/mean_image.tif" not in files
    recs = group_recordings(tmp_path, files, "folder")
    assert [r.name for r in recs] == ["A", "B"]
    a = recs[0]
    assert [f.relpath for f in a.files] == [
        "A/1_1/a_MMStack_Default.ome.tif", "A/1_2/b_MMStack_Default.ome.tif",
        "A/1_2/b_MMStack_Default_1.ome.tif", "A/1_10/c_MMStack_Default.ome.tif"]
    assert [f.block for f in a.files] == [0, 1, 1, 2]
    assert a.conditions() == {0: "1_1", 1: "1_2", 2: "1_10"}
    assert len(group_recordings(tmp_path, files, "root")) == 1
    assert len(group_recordings(tmp_path, files, "acquisition")) == 4
    assert len(find_files(tmp_path, "*.ome.tif; *.tif")) == 7


def test_workbook_matches_repeated_names_by_folder(tmp_path):
    """Every recording has a 1_1_MMStack_Default; the recording column tells them apart."""
    _touch(tmp_path / "data", ["E1/1_1/1_1_MMStack_Default.ome.tif", "E1/1_2/1_2_MMStack_Default.ome.tif",
                               "E1/40X/40X_MMStack_Default.ome.tif",
                               "E2/1_1/1_1_MMStack_Default.ome.tif", "E2/1_2/1_2_MMStack_Default.ome.tif"])
    hdr = ["Recording", "File name", "Treatment", "Line", "Segmentation"]
    xlsx = _workbook(tmp_path / "m.xlsx", {"all": [
        hdr,
        ["E1", "1_1_MMStack_Default.ome", "low", "mutant", "yes"],
        [None, "1_2_MMStack_Default.ome", "high", None, "no"],
        ["E2", "1_1_MMStack_Default.ome", "low", "control", 1],
        [None, "1_2_MMStack_Default.ome", "high", None, 0],
        ["E3", "1_1_MMStack_Default.ome", "low", "control", 1],
    ]})
    root = tmp_path / "data"
    recs, issues = match_workbook(root, find_files(root), read_workbook(xlsx))
    e1, e2, e3 = recs
    assert e1.files[0].relpath == "E1/1_1/1_1_MMStack_Default.ome.tif"
    assert e2.files[0].relpath == "E2/1_1/1_1_MMStack_Default.ome.tif"
    assert e1.description == {"sheet": "all", "Line": "mutant"}  # constant within the recording
    assert [f.condition for f in e1.active_files] == ["low", "high"]  # Line does not change
    assert [f.use_for_seg for f in e1.active_files] == [True, False]
    assert [(f.relpath, f.include) for f in e1.files if not f.include] == [("E1/40X/40X_MMStack_Default.ome.tif", False)]
    assert not e3.include and any("E3" in m for m in issues)


def test_sheet_name_matches_folder_without_recording_column(tmp_path):
    _touch(tmp_path / "data", ["E1/a.ome.tif", "E2/a.ome.tif"])
    xlsx = _workbook(tmp_path / "m.xlsx", {"E1": [["File", "Drug"], ["a.ome.tif", "none"]],
                                           "E2": [["File", "Drug"], ["a.ome.tif", "MFQ"]]})
    root = tmp_path / "data"
    recs, issues = match_workbook(root, find_files(root), read_workbook(xlsx))
    assert not issues
    assert [(r.name, r.files[0].relpath, r.files[0].info["Drug"]) for r in recs] == [
        ("E1", "E1/a.ome.tif", "none"), ("E2", "E2/a.ome.tif", "MFQ")]


@pytest.mark.skipif(not (REPO / "Application_period.xlsx").exists(), reason="workbook not present")
def test_real_workbook_structure(tmp_path):
    wb = read_workbook(REPO / "Application_period.xlsx")
    assert wb.columns == ["Application", "Phase", "Time", "Binning"]
    # recreate the folder layout from the workbook names, plus the 40X snapshots
    files = []
    for r in wb.rows:
        sub = {"1_5": "1_4"}.get(r.file[:3], r.file[:3])  # KO Exp1 keeps 1_5 files in folder 1_4
        files.append(f"{r.recording}/{sub}/{r.file}.tif")
    files += [f"{rec}/40X/40X_MMStack_Default.ome.tif" for rec in {r.recording for r in wb.rows}]
    recs, issues = match_workbook(tmp_path, sorted(files), wb)
    assert not issues
    assert len(recs) == 6
    for rec in recs:
        assert rec.n_blocks == 4, rec.name
        assert rec.description["sheet"].startswith("Min6")
        assert [f.relpath.split("/")[1] for f in rec.files if not f.include] == ["40X"]
    ko1 = next(r for r in recs if r.description["sheet"] == "Min6 KO Exp1")
    assert len(ko1.active_files) == 27
    block1 = [f.use_for_seg for f in ko1.active_files if f.block == 1]
    assert block1 == [True, True] + [False] * 6  # the per-file flag never splits a block
