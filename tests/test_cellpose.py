import numpy as np
import pytest

pytest.importorskip("cellpose")

from mpsca import pipeline  # noqa: E402
from mpsca.segment.cellpose_seg import CellposeParams  # noqa: E402
from test_pipeline import _matched  # noqa: E402


@pytest.mark.slow
def test_cellpose_on_projections(synthetic_store):
    store, truth = synthetic_store["store"], synthetic_store["truth"]
    pipeline.run_projections(store)
    labels = pipeline.run_cellpose(store, ["mean", "corr"], CellposeParams(diameter=11, gpu=False))
    assert labels.shape == truth.shape
    assert _matched(labels, truth) >= 9
    assert pipeline.active_labels(store)[0] == "cellpose"
