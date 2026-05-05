"""
Headless infrastructure for running MPS steps without a display or tkinter.

Must be imported (and setup_headless_env() called) BEFORE importing any MPS step modules.
"""
import sys
import os
import types
import logging

logger = logging.getLogger(__name__)

# ── Minimal tkinter mock ──────────────────────────────────────────────────────

class _MockWidget:
    """No-op widget base. All method calls and attribute accesses are absorbed."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return _MockWidget()
    def __getitem__(self, key):
        return _MockWidget()
    def __setitem__(self, key, value):
        pass
    def __getattr__(self, name):
        return _MockWidget()
    def get(self):
        return None
    def set(self, value):
        pass
    def pack(self, *a, **k): pass
    def grid(self, *a, **k): pass
    def config(self, *a, **k): pass
    def configure(self, *a, **k): pass
    def bind(self, *a, **k): pass
    def destroy(self): pass
    def insert(self, *a, **k): pass
    def delete(self, *a, **k): pass
    def see(self, *a, **k): pass
    def update(self): pass
    def update_idletasks(self): pass
    def winfo_width(self): return 800
    def winfo_height(self): return 600
    def winfo_screenwidth(self): return 1920
    def winfo_screenheight(self): return 1080
    def create_window(self, *a, **k): return 0
    def itemconfig(self, *a, **k): pass
    def bbox(self, *a, **k): return (0, 0, 100, 100)
    def yview(self, *a, **k): pass
    def xview(self, *a, **k): pass


class _StringVar:
    def __init__(self, value='', *a, **k): self._v = str(value) if value is not None else ''
    def get(self): return self._v
    def set(self, v): self._v = str(v) if v is not None else ''
    def trace_add(self, *a, **k): pass
    def trace(self, *a, **k): pass


class _IntVar:
    def __init__(self, value=0, *a, **k): self._v = int(value)
    def get(self): return self._v
    def set(self, v): self._v = int(v)


class _DoubleVar:
    def __init__(self, value=0.0, *a, **k): self._v = float(value)
    def get(self): return self._v
    def set(self, v): self._v = float(v)


class _BooleanVar:
    def __init__(self, value=False, *a, **k): self._v = bool(value)
    def get(self): return self._v
    def set(self, v): self._v = bool(v)


def _make_tk_module():
    """Build a minimal tkinter mock module."""
    mod = types.ModuleType('tkinter')
    mod.Frame = _MockWidget
    mod.Label = _MockWidget
    mod.Entry = _MockWidget
    mod.Button = _MockWidget
    mod.Checkbutton = _MockWidget
    mod.Scale = _MockWidget
    mod.Text = _MockWidget
    mod.Canvas = _MockWidget
    mod.Scrollbar = _MockWidget
    mod.Toplevel = _MockWidget
    mod.Spinbox = _MockWidget
    mod.Tk = _MockWidget
    mod.Menu = _MockWidget
    mod.Menubutton = _MockWidget
    mod.StringVar = _StringVar
    mod.IntVar = _IntVar
    mod.DoubleVar = _DoubleVar
    mod.BooleanVar = _BooleanVar
    # Common constants
    for name in ('END', 'LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'X', 'Y', 'BOTH',
                 'HORIZONTAL', 'VERTICAL', 'NORMAL', 'DISABLED', 'ACTIVE',
                 'NW', 'NE', 'SW', 'SE', 'N', 'S', 'E', 'W',
                 'WORD', 'CHAR', 'FLAT', 'RAISED', 'SUNKEN', 'GROOVE', 'RIDGE'):
        setattr(mod, name, name.lower())
    mod.END = 'end'
    mod.LEFT = 'left'
    mod.RIGHT = 'right'
    mod.BOTH = 'both'
    mod.NORMAL = 'normal'
    mod.DISABLED = 'disabled'
    return mod


def _make_ttk_module(tk_mod):
    """Build a minimal ttk mock module that reuses _MockWidget."""
    mod = types.ModuleType('tkinter.ttk')
    for name in ('Frame', 'Label', 'Entry', 'Button', 'Checkbutton', 'Scale',
                 'Scrollbar', 'Combobox', 'Progressbar', 'LabelFrame', 'Spinbox',
                 'Notebook', 'Treeview', 'Separator'):
        setattr(mod, name, _MockWidget)
    return mod


def setup_headless_env(mps_root: str):
    """
    Inject tkinter mocks and configure matplotlib for headless operation.
    Must be called once before importing any MPS step modules.

    Args:
        mps_root: Absolute path to the MPS_1.0.0 directory.
    """
    # ── Mock tkinter ─────────────────────────────────────────────────────────
    _tk_mod = _make_tk_module()
    _ttk_mod = _make_ttk_module(_tk_mod)

    for name in ('tkinter', 'tkinter.ttk', 'tkinter.filedialog',
                 'tkinter.messagebox', 'tkinter.simpledialog',
                 'tkinter.colorchooser', 'tkinter.font'):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            # Reuse the same mock widget classes
            mod.__dict__.update({
                'Frame': _MockWidget, 'Label': _MockWidget, 'Entry': _MockWidget,
                'Button': _MockWidget, 'Checkbutton': _MockWidget,
                'Scale': _MockWidget, 'Text': _MockWidget, 'Canvas': _MockWidget,
                'Scrollbar': _MockWidget, 'Toplevel': _MockWidget, 'Tk': _MockWidget,
                'Menu': _MockWidget, 'Spinbox': _MockWidget,
                'StringVar': _StringVar, 'IntVar': _IntVar,
                'DoubleVar': _DoubleVar, 'BooleanVar': _BooleanVar,
                'END': 'end', 'LEFT': 'left', 'RIGHT': 'right', 'BOTH': 'both',
                'NORMAL': 'normal', 'DISABLED': 'disabled',
                'askdirectory': lambda **k: '', 'askopenfilename': lambda **k: '',
                'asksaveasfilename': lambda **k: '',
                'showinfo': lambda *a, **k: None, 'showerror': lambda *a, **k: None,
                'showwarning': lambda *a, **k: None,
                'askyesno': lambda *a, **k: True,
                'LabelFrame': _MockWidget, 'Combobox': _MockWidget,
                'Progressbar': _MockWidget,
            })
            if name == 'tkinter':
                mod = _tk_mod
            elif name == 'tkinter.ttk':
                mod = _ttk_mod
            sys.modules[name] = mod

    # ── Mock matplotlib tkagg backend ─────────────────────────────────────────
    if 'matplotlib.backends.backend_tkagg' not in sys.modules:
        _mpl_tkagg = types.ModuleType('matplotlib.backends.backend_tkagg')
        _mpl_tkagg.FigureCanvasTkAgg = _MockWidget
        _mpl_tkagg.NavigationToolbar2Tk = _MockWidget
        sys.modules['matplotlib.backends.backend_tkagg'] = _mpl_tkagg

    # ── Configure matplotlib for headless rendering ────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass

    # ── Add MPS source directories to sys.path ────────────────────────────────
    for subdir in ('steps', 'utils', ''):
        path = os.path.join(mps_root, subdir)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    logger.info(f"Headless environment configured. MPS root: {mps_root}")


# ── Headless controller and step base ─────────────────────────────────────────

class HeadlessController:
    """
    Mimics the MPS GUI MainApplication controller for headless operation.
    Provides the state dict and no-op GUI methods expected by step classes.
    """

    def __init__(self, state: dict):
        self.state = state
        # GUI stubs
        self.status_var = _StringVar()
        self.next_button = _MockWidget()
        self.prev_button = _MockWidget()
        self.autorun_enabled = False
        self.autorun_indicator = _MockWidget()
        self.progress = _MockWidget()
        self.loaded_parameters = None

    def after(self, delay, fn=None, *args):
        """GUI event loop scheduling — no-op in headless mode."""
        pass

    def after_idle(self, fn=None, *args):
        pass

    def get_step_parameters(self, step_name):
        """Return pre-loaded parameters for a step, if any."""
        if self.loaded_parameters and 'steps' in self.loaded_parameters:
            return self.loaded_parameters['steps'].get(step_name)
        return None

    def auto_save_parameters(self):
        pass

    def on_step_complete(self, step_name):
        logger.info(f"Step complete: {step_name}")


class HeadlessStep:
    """
    Lightweight stand-in for an MPS step frame class.
    Provides the interface that step processing threads call on `self`.
    """

    def __init__(self, controller: HeadlessController, name: str = "step"):
        self.controller = controller
        self._name = name
        self.processing_complete = False
        self.status_var = _StringVar()
        self.progress = _MockWidget()
        self.run_button = _MockWidget()

    def log(self, message: str):
        logger.info(f"[{self._name}] {message}")

    def after(self, delay, fn=None, *args):
        pass

    def after_idle(self, fn=None, *args):
        pass

    def update_idletasks(self):
        pass

    def update(self):
        pass

    def update_progress(self, value: float):
        pass

    def winfo_width(self): return 800
    def winfo_height(self): return 600
