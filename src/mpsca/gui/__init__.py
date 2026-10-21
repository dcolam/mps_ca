"""napari GUI for mpsca."""


def launch(store: str | None = None):
    """Open napari with the mpsca dock widget (and optionally an experiment store)."""
    import napari

    from ._main import MainWidget

    viewer = napari.Viewer(title="mpsca")
    widget = MainWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Calcium pipeline", area="right")
    if store:
        widget.session.open_store(store)
    napari.run()
