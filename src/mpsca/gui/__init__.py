"""napari GUI for mpsca."""
from pathlib import Path


def launch(path: str | None = None):
    """Open napari with the mpsca dock widget, optionally with a project or a recording store."""
    import napari

    from ._main import MainWidget

    viewer = napari.Viewer(title="mpsca")
    widget = MainWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Calcium pipeline", area="right")
    if path:
        p = Path(path)
        if p.suffix == ".zarr" or (p / "zarr.json").exists():
            widget.session.open_store(p)
        else:
            widget.project_tab.open_project(p)
    napari.run()
