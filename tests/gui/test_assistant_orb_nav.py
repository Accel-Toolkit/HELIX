"""Assistant state orb + tab-navigation wiring (offscreen)."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


# ---- the animated orb ------------------------------------------------
def test_orb_states_and_paint_do_not_crash(qapp):
    from PyQt6.QtGui import QPixmap, QPainter
    from linac_gen_gui.interphase.dialogs.assistant_orb import AssistantOrb
    orb = AssistantOrb()
    orb.resize(200, 200)
    for st in ("idle", "thinking", "responding", "listening",
               "awaiting-confirm", "error", "starting"):
        orb.set_state(st)
        orb.set_level(0.5)
        orb._tick()                       # advance animation
        pm = QPixmap(200, 200)
        p = QPainter(pm)
        orb.render(p)                     # exercise paintEvent paths
        p.end()
    orb.stop()


# ---- InterphaseWindow.show_tab resolution ----------------------------
def test_show_tab_resolves_index_id_and_title(qapp):
    from PyQt6.QtWidgets import QTabWidget, QWidget
    from linac_gen_gui.interphase.app import InterphaseWindow
    tabs = QTabWidget()
    for label in ("Beam", "Lattice", "Matching", "Results"):
        tabs.addTab(QWidget(), label)
    fake = type("F", (), {"_tabs": tabs})()
    show = InterphaseWindow.show_tab
    assert show(fake, 0) == "Beam" and tabs.currentIndex() == 0
    assert show(fake, "Results") == "Results" and tabs.currentIndex() == 3
    assert show(fake, "results") == "Results"          # case-insensitive
    assert show(fake, "beam") == "Beam"                # stable-id == label
    assert show(fake, "latt") == "Lattice"             # substring
    assert show(fake, "nonsuch") is None               # no match
    assert InterphaseWindow.assistant_tab_labels(fake) == [
        "Beam", "Lattice", "Matching", "Results"]


# ---- panel navigation: navigate_requested -> app.show_tab ------------
def test_panel_navigate_calls_app_show_tab(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    panel = ap.AssistantPanel(None, AppState())
    try:
        seen = []
        panel._app = type("A", (), {
            "show_tab": lambda self, t, sub=None: seen.append((t, sub)),
            "show_result_plot": lambda self, k: seen.append(("plot", k)),
        })()
        panel._on_navigate("Results")
        panel._on_navigate("Lattice", "Breakdown")
        panel._on_plot("phase")
        assert ("Results", None) in seen
        assert ("Lattice", "Breakdown") in seen
        assert ("plot", "phase") in seen
        # the orb/lamp exist and a state change updates the lamp text
        assert panel._orb is not None
        panel._set_state("responding")
        assert "responding" in panel._lamp.text()
    finally:
        panel.shutdown()


# ---- context navigation: resolves + emits the queued signal ----------
def test_context_show_tab_resolves_and_emits(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    panel = ap.AssistantPanel(None, AppState())
    try:
        panel.tab_labels = ["Beam", "Lattice", "Matching", "Results"]
        panel.subtab_map = {"Lattice": ["Sequence", "Breakdown"]}
        panel.plot_catalog = [("phase", "Phase space (4-panel)"),
                              ("rms", "RMS σ (x · y · z)"),
                              ("tune_depr", "Tune depression η = σ/σ₀")]
        emitted = []
        panel.navigate_requested.connect(
            lambda t, s: emitted.append((t, s)))
        plots = []
        panel.plot_requested.connect(plots.append)
        ctx = ap._make_context(AppState(), str(tmp_path), nav=panel)
        assert ctx.available_tabs() == panel.tab_labels
        assert ctx.show_tab("results") == "Results"     # fuzzy -> canonical
        assert ctx.show_tab("lattice") == "Lattice"     # stable id
        # subtab resolves + describes
        assert ctx.show_tab("Lattice", "breakdown") == "Lattice › Breakdown"
        assert ctx.show_tab("nope") is None
        # plots resolve by fuzzy name -> canonical label, emit the KEY
        assert ctx.available_plots() == [lab for _k, lab in panel.plot_catalog]
        assert ctx.open_plot("phase space") == "Phase space (4-panel)"
        assert ctx.open_plot("tune depression") == "Tune depression η = σ/σ₀"
        assert ctx.open_plot("no such plot") is None
        qapp.processEvents()
        assert emitted == [("Results", ""), ("Lattice", ""),
                           ("Lattice", "breakdown")]
        assert plots == ["phase", "tune_depr"]
    finally:
        panel.shutdown()


def test_app_show_result_plot_and_subtab(qapp):
    from PyQt6.QtWidgets import QTabWidget, QWidget
    from linac_gen_gui.interphase.app import InterphaseWindow

    tabs = QTabWidget()
    lattice_page = QWidget()
    sub = QTabWidget(lattice_page)         # nested subtab widget on the page
    sub.addTab(QWidget(), "Sequence")
    sub.addTab(QWidget(), "Breakdown")
    for w, label in ((QWidget(), "Beam"), (lattice_page, "Lattice"),
                     (QWidget(), "Results")):
        tabs.addTab(w, label)

    opened = []

    class _Results:
        def open_plot(self, k): opened.append(k); return True
        def plot_catalog(self): return [("phase", "Phase space (4-panel)")]

    fake = type("F", (), {})()
    fake._tabs = tabs
    # _assistant_pages must align with tab indices [Beam, Lattice, Results]
    fake.beam_tab = tabs.widget(0)
    fake.lattice_tab = lattice_page
    fake.results_tab = _Results()
    fake.matching_tab = fake.convergence_tab = fake.surrogates_tab = None
    fake.errors_tab = fake.failures_tab = None

    def _pages(self=fake):
        return [fake.beam_tab, fake.lattice_tab, fake.results_tab]
    fake._assistant_pages = _pages
    fake._page_subtab_widget = lambda page: InterphaseWindow._page_subtab_widget(fake, page)

    # subtab navigation
    got = InterphaseWindow.show_tab(fake, "Lattice", "Breakdown")
    assert got == "Lattice › Breakdown"
    assert sub.currentIndex() == 1
    # result plot
    lab = InterphaseWindow.show_result_plot(fake, "phase")
    assert lab == "Phase space (4-panel)"
    assert opened == ["phase"]
    assert tabs.currentIndex() == 2         # switched to Results


# ---- Phase 1: GUI symbiosis (highlight / cursor / gui-context) --------
def _mini_state(qapp):
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen_gui.interphase.state import AppState
    lat = Lattice()
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QF1", 50.0, gradient=5.0, aperture=20.0))
    lat.add(Drift("D2", 300.0))
    st = AppState()
    st.set_lattice(lat, "<test>")
    return st


def test_highlight_and_cursor_signal_chain(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    st = _mini_state(qapp)
    panel = ap.AssistantPanel(None, st)
    try:
        tabs_shown = []
        panel._app = type("A", (), {
            "show_tab": lambda self, t, sub=None: tabs_shown.append(t)})()
        ctx = ap._make_context(st, str(tmp_path), nav=panel)
        # context hook emits; queued slot runs on processEvents
        assert ctx.highlight_element(1, 200.0) is True
        assert ctx.set_cursor(450.0) is True
        qapp.processEvents()
        assert tabs_shown == ["Lattice"]              # switched to Lattice
        assert getattr(st.selected, "name", "") == "QF1"
        assert st.s_cursor == 450.0                   # last cursor request
    finally:
        panel.shutdown()


def test_gui_context_snapshot(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    st = _mini_state(qapp)
    from linac_gen_gui.interphase.state import TABS
    st.set_tab([t for t, _ in TABS].index("results"))  # Results
    st.set_selected(st.lattice.elements[1])
    st.set_s_cursor(123.0)
    panel = ap.AssistantPanel(None, st)
    try:
        panel._app = type("A", (), {
            "show_tab": lambda self, t, sub=None: None,
            "assistant_open_plots": lambda self: ["Phase space (4-panel)"],
        })()
        ctx = ap._make_context(st, str(tmp_path), nav=panel)
        snap = ctx.gui_context()   # on GUI thread -> direct path, no hang
        assert snap["lattice_loaded"] is True
        assert snap["selected_element"] == "QF1"
        assert snap["s_cursor_m"] == 0.123
        assert snap["current_tab"] == "Results"
        assert snap["open_plot_windows"] == ["Phase space (4-panel)"]
    finally:
        panel.shutdown()


def test_run_on_gui_roundtrip_from_worker_thread(qapp, monkeypatch, tmp_path):
    """The generic GUI-thread bridge: a worker thread gets fn()'s result;
    the GUI thread must keep processing events (no deadlock)."""
    import threading
    import time
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    panel = ap.AssistantPanel(None, _mini_state(qapp))
    try:
        out = {}

        def worker():
            out["r"] = panel.run_on_gui(lambda: 40 + 2, timeout=5.0)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        deadline = time.time() + 5.0
        while t.is_alive() and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        t.join(timeout=1.0)
        assert out.get("r") == 42
    finally:
        panel.shutdown()


def test_open_plot_respects_disabled_cards(qapp):
    """Adversarial M3: a disabled card (e.g. IBS for non-H⁻) must not be
    openable by the assistant either."""
    from PyQt6.QtCore import pyqtSignal, QObject
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    class _Card(QObject):
        clicked = pyqtSignal(str)
        def __init__(self, key, enabled):
            super().__init__()
            self._key = key
            self._enabled = enabled
        def isEnabled(self): return self._enabled

    fake = type("F", (), {})()
    clicks = []
    on = _Card("rms", True); on.clicked.connect(clicks.append)
    off = _Card("ibs", False); off.clicked.connect(clicks.append)
    fake._cards = [on, off]
    assert ResultsTab.open_plot(fake, "rms") is True
    assert ResultsTab.open_plot(fake, "ibs") is False    # gated off
    assert ResultsTab.open_plot(fake, "nonsuch") is False
    assert clicks == ["rms"]


# ---- run_in_gui: press the real Run buttons --------------------------
def test_app_assistant_run_simulation_guards(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow

    class _St:
        def __init__(self): self.running = False; self.lattice = 1; self.beam_config = 1
    fake = type("F", (), {})()
    fake.state = _St()
    pressed = []
    fake._run_mp = lambda: (pressed.append("mp"),
                            setattr(fake.state, "running", True))
    fake._run_envelope = lambda: (pressed.append("env"),
                                  setattr(fake.state, "running", True))
    run = InterphaseWindow.assistant_run_simulation

    assert run(fake, "mp") == "started" and pressed == ["mp"]
    assert run(fake, "mp") == "busy"                  # already running
    fake.state.running = False
    assert run(fake, "envelope") == "started" and pressed[-1] == "env"
    fake.state.running = False; fake.state.lattice = None
    assert run(fake, "mp") == "no lattice is loaded in the GUI"
    fake.state.lattice = 1; fake.state.beam_config = None
    assert run(fake, "mp") == "no beam configuration is set in the GUI"


def test_context_run_gui_simulation_roundtrip(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    panel = ap.AssistantPanel(None, AppState())
    try:
        got = []
        panel._app = type("A", (), {
            "assistant_run_simulation":
                lambda self, k: got.append(k) or "started"})()
        ctx = ap._make_context(AppState(), str(tmp_path), nav=panel)
        assert ctx.run_gui_simulation("mp") == "started"   # on GUI thread
        assert got == ["mp"]
    finally:
        panel.shutdown()
