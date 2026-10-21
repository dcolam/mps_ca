import json

import numpy as np
import pytest

from mpsca.cli import main
from mpsca.io import ExperimentStore
from mpsca.project import (Project, accept_proposals, convert_all, create_project, run_all, scan_all,
                           store_status)


def test_project_workflow(two_recordings):
    d = two_recordings
    proj = create_project(d["root"], d["out"])
    assert [r.name for r in proj.recordings] == ["R1", "R2"]
    assert [r.n_blocks for r in proj.recordings] == [2, 2]
    proj.recordings[0].description["group"] = "control"
    proj.recordings[1].description["group"] = "treated"
    proj.save()

    # paths are stored relative to the input folder
    saved = json.loads(proj.path.read_text())
    assert saved["recordings"][0]["files"][0]["relpath"] == "R1/1_1/1_1_MMStack_Default.ome.tif"
    proj = Project.load(d["out"], input_dir=d["root"])
    assert proj.get("R2").description == {"group": "treated"}
    assert not proj.get("R1").missing

    reports = scan_all(proj)
    # R2 is finer than the common grid: converting it needs a decision, R1 still converts
    res = convert_all(proj, reports=reports)
    assert res["R1"].startswith("ok") and "not confirmed" in res["R2"]
    assert accept_proposals(proj, reports) == ["R2"]
    assert proj.get("R2").block_factors == {0: 2, 1: 2}
    res = convert_all(proj, reports=reports)
    assert res["R1"].startswith("skipped") and res["R2"].startswith("ok")
    s1 = ExperimentStore.open(proj.store_path(proj.get("R1")), mode="r")
    s2 = ExperimentStore.open(proj.store_path(proj.get("R2")), mode="r")
    assert s1.movie.shape == s2.movie.shape == (160, 48, 64)
    assert s1.read_frames()["time_s"].iloc[80] == pytest.approx(400)  # pause between blocks kept

    # try settings on one recording, then run all
    proj.settings.seg_method = "classic"
    proj.settings.analysis.robust_noise = True
    res = run_all(proj, [proj.get("R1")], steps=["bad_frames", "projections", "segment"])
    assert res["R1"].startswith("ok") and proj.status(proj.get("R1")) == "ROIs"
    res = run_all(proj)
    assert all(v.startswith("ok") for v in res.values()), res
    assert {proj.status(r) for r in proj.recordings} == {"analysed"}
    s2 = ExperimentStore.open(proj.store_path(proj.get("R2")), mode="r")
    assert s2.meta["description"] == {"group": "treated"}
    assert s2.params("analysis")["robust_noise"] is True


def test_failures_do_not_stop_the_batch(two_recordings):
    d = two_recordings
    proj = create_project(d["root"], d["out"])
    proj.convert.target_camera_binning = 0  # no common grid: nothing to decide
    proj.get("R1").files[0].relpath = "R1/1_1/gone.ome.tif"
    proj.resolve()
    res = convert_all(proj)
    assert res["R2"].startswith("ok")
    assert proj.status(proj.get("R2")) == "converted"
    res = run_all(proj, steps=["bad_frames"])
    assert res["R1"].startswith("failed") and res["R2"].startswith("ok")


def test_interrupted_conversion_is_redone(two_recordings):
    d = two_recordings
    proj = create_project(d["root"], d["out"])
    proj.convert.target_camera_binning = 0
    rec = proj.get("R1")
    ExperimentStore.create(proj.store_path(rec))  # left behind by an interrupted run
    assert store_status(proj.store_path(rec)) == "incomplete"
    assert convert_all(proj, [rec])["R1"].startswith("ok")
    assert proj.status(rec) == "converted"


def test_cli_batch(two_recordings, capsys):
    d = two_recordings
    root, out = str(d["root"]), str(d["out"])
    assert main(["new", "--input", root, "--out", out]) == 0
    assert "R1: 2 files, 2 blocks, not converted" in capsys.readouterr().out
    assert main(["convert", out]) == 1  # R2 needs a decision
    assert "--accept-proposal" in capsys.readouterr().out
    assert main(["convert", out, "--accept-proposal"]) == 0
    proj = Project.load(out)
    proj.settings.seg_method = "classic"
    proj.save()
    assert main(["run", out, "--only", "R2", "--steps", "bad_frames,projections,segment,traces"]) == 0
    assert main(["list", out]) == 0
    text = capsys.readouterr().out
    assert "R1: 2 files, 2 blocks, converted" in text and "R2: 2 files, 2 blocks, traces" in text
    with pytest.raises(SystemExit):
        main(["run", out, "--steps", "nonsense"])
    store = proj.store_path(proj.get("R2"))
    assert main(["info", str(store)]) == 0
    assert "1_2" in capsys.readouterr().out
