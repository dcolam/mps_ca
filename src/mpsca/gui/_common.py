"""Shared Qt helpers: background workers with progress, figure panels, parameter forms."""
from __future__ import annotations

import threading
import traceback
from dataclasses import fields
from typing import Any, Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                            QMessageBox, QProgressBar, QSpinBox, QVBoxLayout, QWidget)


class _Bridge(QObject):
    progress = Signal(float)
    message = Signal(str)


class Runner(QObject):
    """Run one function at a time in a worker thread, with a progress bar and error dialog.

    The function receives ``progress=callable(fraction)`` when ``with_progress`` is set,
    and ``say=callable(text)`` when ``with_say`` is set (shown in ``log``, a list widget).
    With ``with_stop`` it receives ``should_stop=callable()``, which turns True after ``stop()``.
    Widgets listed in ``lock`` are disabled while it runs.
    """
    busy = Signal(bool)

    def __init__(self, parent: QWidget, bar: QProgressBar, status: QLabel | None = None, log=None):
        super().__init__(parent)
        self._parent, self._bar, self._status, self._log = parent, bar, status, log
        self._worker = None
        self._stop = threading.Event()
        self._bridge = _Bridge()
        self._bridge.progress.connect(lambda f: self._bar.setValue(int(round(f * 100))))
        self._bridge.message.connect(self._show_message)

    def _show_message(self, text: str) -> None:
        if self._status is not None:
            self._status.setText(text)
        if self._log is not None:
            self._log.addItem(text)
            self._log.scrollToBottom()

    @property
    def running(self) -> bool:
        return self._worker is not None

    def stop(self) -> None:
        """Ask a job started ``with_stop`` to finish early."""
        self._stop.set()

    def run(self, fn: Callable, *args, done: Callable[[Any], None] | None = None,
            message: str = "", with_progress: bool = True, with_say: bool = False,
            with_stop: bool = False, lock=(), **kwargs):
        from napari.qt.threading import create_worker

        if self._worker is not None:
            QMessageBox.information(self._parent, "Busy", "Another step is still running.")
            return None
        if with_progress:
            kwargs["progress"] = self._bridge.progress.emit
        if with_say:
            kwargs["say"] = self._bridge.message.emit
        if with_stop:
            self._stop.clear()
            kwargs["should_stop"] = self._stop.is_set
        self._bar.setRange(0, 100 if with_progress else 0)  # 0..0 = busy indicator
        self._bar.setValue(0)
        self._bar.setVisible(True)
        if self._status is not None:
            self._status.setText(message)
        for w in lock:
            w.setEnabled(False)

        worker = None

        def _finish():
            if self._worker is not worker:  # already released (a follow-up job may be running)
                return
            self._worker = None
            self._bar.setVisible(False)
            for w in lock:
                w.setEnabled(True)
            self.busy.emit(False)

        def _error(exc):
            if self._status is not None:
                self._status.setText(f"Failed: {exc}")
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            box = QMessageBox(QMessageBox.Critical, "Step failed", str(exc), parent=self._parent)
            box.setDetailedText(detail)
            box.show()

        def _done(result):
            _finish()  # release first, so ``done`` may start the next job
            if self._status is not None:
                self._status.setText("Done.")
            if done is not None:
                done(result)

        worker = create_worker(fn, *args, _start_thread=False, **kwargs)
        worker.returned.connect(_done)
        worker.errored.connect(_error)
        worker.finished.connect(_finish)
        self._worker = worker
        self.busy.emit(True)
        worker.start()
        return worker


class FigurePanel(QWidget):
    """Holds one matplotlib figure; ``set_figure`` swaps it."""

    def __init__(self, parent=None, placeholder: str = "", toolbar: bool = True):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._toolbar_enabled = toolbar
        self._placeholder = QLabel(placeholder)
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)
        self.canvas = None
        self._toolbar = None

    def set_figure(self, fig) -> FigureCanvasQTAgg:
        for w in (self.canvas, self._toolbar):
            if w is not None:
                self._layout.removeWidget(w)
                w.hide()
                w.deleteLater()
        self._placeholder.setVisible(False)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumHeight(int(fig.get_figheight() * 60))
        if self._toolbar_enabled:
            self._toolbar = NavigationToolbar2QT(self.canvas, self)
            self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self.canvas)
        self.canvas.draw_idle()
        return self.canvas


class ParamForm(QWidget):
    """Form generated from a parameter dataclass.

    ``specs`` maps field names to options: label, min, max, step, decimals, tooltip,
    optional (the minimum is shown as "auto" and returned as None), skip.
    """

    def __init__(self, cls, specs: dict[str, dict] | None = None, parent=None):
        super().__init__(parent)
        self._cls = cls
        self._specs = specs or {}
        self._widgets: dict[str, QWidget] = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        default = cls()
        for f in fields(cls):
            spec = self._specs.get(f.name, {})
            if spec.get("skip"):
                continue
            value = getattr(default, f.name)
            w = self._make(value, spec)
            if w is None:
                continue
            if spec.get("tooltip"):
                w.setToolTip(spec["tooltip"])
            self._widgets[f.name] = w
            form.addRow(spec.get("label", f.name.replace("_", " ").capitalize()), w)

    def _make(self, value, spec):
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
            return w
        if isinstance(value, int) and not spec.get("float"):
            w = QSpinBox()
            w.setRange(int(spec.get("min", 0)), int(spec.get("max", 1_000_000)))
            w.setSingleStep(int(spec.get("step", 1)))
            w.setValue(value)
            return w
        if isinstance(value, float) or value is None or spec.get("float"):
            w = QDoubleSpinBox()
            w.setDecimals(spec.get("decimals", 2))
            w.setRange(spec.get("min", 0.0), spec.get("max", 1e6))
            w.setSingleStep(spec.get("step", 0.1))
            if spec.get("optional"):
                w.setSpecialValueText("auto")
            w.setValue(w.minimum() if value is None else float(value))
            return w
        if isinstance(value, str):
            return QLineEdit(value)
        return None  # tuples/dicts keep their defaults

    def widget(self, name: str) -> QWidget:
        return self._widgets[name]

    def value(self):
        kwargs = {}
        for name, w in self._widgets.items():
            spec = self._specs.get(name, {})
            if isinstance(w, QCheckBox):
                v = w.isChecked()
            elif isinstance(w, QSpinBox):
                v = w.value()
            elif isinstance(w, QDoubleSpinBox):
                v = w.value()
                if spec.get("optional") and v == w.minimum():
                    v = None
            else:
                v = w.text().strip()
            kwargs[name] = v
        return self._cls(**kwargs)

    def set_value(self, obj) -> None:
        for name, w in self._widgets.items():
            v = getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)
            if v is None and not isinstance(w, QDoubleSpinBox):
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(w.minimum() if v is None else float(v))
            else:
                w.setText(str(v))


def hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: #8a8a8a;")
    return lab
