"""The open experiment store and the napari layers that show it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, QSettings, Signal

from .. import pipeline
from ..io.store import ExperimentStore

ROI_LAYER = "ROIs"
MOVIE_LAYER = "movie"


def settings() -> QSettings:
    return QSettings(QSettings.defaultFormat(), QSettings.UserScope, "mpsca", "mpsca")


class Session(QObject):
    store_changed = Signal()
    labels_changed = Signal()
    traces_changed = Signal()

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.store: ExperimentStore | None = None

    # ---- store ------------------------------------------------------------
    def open_store(self, path: str | Path) -> ExperimentStore:
        store = ExperimentStore.open(path)
        if not store.complete:
            raise ValueError(f"{path} is incomplete (the conversion was interrupted); convert again")
        self.store = store
        settings().setValue("last_store", str(path))
        self._load_layers()
        self.store_changed.emit()
        if pipeline.load_traces(store) is not None:
            self.traces_changed.emit()
        return store

    @property
    def name(self) -> str:
        return self.store.meta.get("name", "") if self.store else ""

    # ---- layers -------------------------------------------------------------
    def _remove_ours(self):
        for layer in list(self.viewer.layers):
            if layer.metadata.get("mpsca"):
                self.viewer.layers.remove(layer)

    def _load_layers(self):
        self._remove_ours()
        movie = self.store.movie
        first = np.asarray(movie[0])
        lo, hi = np.percentile(first, [1, 99.8])
        self.viewer.add_image(movie, name=MOVIE_LAYER, contrast_limits=(float(lo), float(hi) + 1e-3),
                              metadata={"mpsca": True}, colormap="gray")
        proj = pipeline.load_projections(self.store)
        for name in ("mean", "max", "std", "corr"):
            if name in proj:
                self.show_image(name, proj[name], visible=name == "corr")
        act = pipeline.active_labels(self.store)
        if act is not None:
            self.show_labels(act[1])
        self.viewer.reset_view()

    def show_image(self, name: str, data: np.ndarray, visible: bool = True, **kwargs):
        lo, hi = np.percentile(data, [1, 99.8])
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = data
            layer.contrast_limits = (float(lo), float(hi) + 1e-6)
        else:
            layer = self.viewer.add_image(data, name=name, contrast_limits=(float(lo), float(hi) + 1e-6),
                                          metadata={"mpsca": True}, visible=visible,
                                          colormap="magma" if name == "corr" else "gray", **kwargs)
        return layer

    def show_labels(self, labels: np.ndarray):
        if ROI_LAYER in self.viewer.layers:
            layer = self.viewer.layers[ROI_LAYER]
            layer.data = labels.astype(np.int32)
        else:
            layer = self.viewer.add_labels(labels.astype(np.int32), name=ROI_LAYER, opacity=0.45,
                                           metadata={"mpsca": True})
        self.viewer.layers.selection.active = layer
        self.labels_changed.emit()
        return layer

    def roi_layer(self):
        return self.viewer.layers[ROI_LAYER] if ROI_LAYER in self.viewer.layers else None
