import pytest

from conftest import REPO
from mpsca.io import parse_excel, resolve_paths


def test_parse_synthetic(synthetic_experiment):
    (meta,) = parse_excel(synthetic_experiment["xlsx"])
    assert meta.name == synthetic_experiment["exp"].name
    assert meta.genotype == "WT"
    assert [f.block for f in meta.files] == [0, 0, 1]
    assert [f.application for f in meta.files] == ["2mM Glucose", "2mM Glucose", "20mM Glucose"]
    assert [f.use_for_seg for f in meta.files] == [True, False, True]
    assert meta.files[0].subfolder is None  # numeric Subfolder cell is ignored
    assert meta.files[2].subfolder == "1_2"
    assert meta.files[0].filename.endswith(".ome.tif")
    assert meta.files[1].expected_frames == 40  # forward-filled within the block


def test_resolve_paths(synthetic_experiment):
    (meta,) = parse_excel(synthetic_experiment["xlsx"])
    assert resolve_paths(synthetic_experiment["exp"], meta) == []
    assert all(f.path.exists() for f in meta.files)
    meta.files[0].filename = "nope.tif"
    assert resolve_paths(synthetic_experiment["exp"], meta) == ["nope.tif"]


@pytest.mark.skipif(not (REPO / "Application_period.xlsx").exists(), reason="workbook not present")
def test_parse_real_workbook():
    exps = parse_excel(REPO / "Application_period.xlsx")
    assert len(exps) == 6
    assert {m.genotype for m in exps} == {"WT", "KO"}
    for m in exps:
        assert m.files, m.sheet
        assert all(f.application for f in m.files)
        blocks = [f.block for f in m.files]
        assert blocks == sorted(blocks)


def test_folder_import_skips_reference_images(tmp_path):
    import numpy as np
    import tifffile
    from mpsca.io.metadata import entries_from_folder

    for rel in ["1_2/b_MMStack_Default.ome.tif", "1_10/c.ome.tif", "1_2/b_MMStack_Default_1.ome.tif",
                "40X/40X_MMStack_Default.ome.tif", "1_1/a.ome.tif"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(tmp_path / rel, np.zeros((2, 4, 4), np.uint16))
    meta = entries_from_folder(tmp_path)
    assert [f"{f.subfolder}/{f.filename}" for f in meta.files] == [
        "1_1/a.ome.tif", "1_2/b_MMStack_Default.ome.tif", "1_2/b_MMStack_Default_1.ome.tif",
        "1_10/c.ome.tif"]
    assert [f.block for f in meta.files] == [0, 1, 1, 2]
