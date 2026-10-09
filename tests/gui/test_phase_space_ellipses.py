"""Phase Space popup: the "ellipses" overlay — the beam's rms ellipse, the
design ellipse (same emittance) and the BMAG mismatch factor per panel.

Pins: the drawn rms ellipse IS the Courant–Snyder ellipse of the plotted
(folded) particles; BMAG in the panel title equals ``mismatch_factor`` of
the beam and design Twiss (analytic cases); the design comes from the
Beam-tab input Twiss at s = 0 and from the error-free envelope (companion
probe cache or a background probe) elsewhere; a probe truncated by a run
is never cached; the overlay follows the fold and the (z, δ) basis; it is
off by default, remembered, and leaves Ctrl+S exports unchanged when off.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")

KEY = "linac_gen/phase_space/ellipses"
FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))


@pytest.fixture(autouse=True)
def _clean_cache_and_setting():
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen_gui.interphase.app_settings import make_settings
    rt._COMPANION_PROBE.clear()
    s = make_settings("linac_gen", "linac_gen_gui"); s.remove(KEY); s.sync()
    yield
    rt._COMPANION_PROBE.clear()
    s = make_settings("linac_gen", "linac_gen_gui"); s.remove(KEY); s.sync()


# ----------------------------------------------------------------- helpers
def _exact_cov(n, cov2, centre, rng):
    """(n, 2) sample whose 1/N sample covariance equals ``cov2`` exactly
    and whose mean equals ``centre`` (whitening + Cholesky)."""
    z = rng.normal(size=(n, 2))
    z -= z.mean(axis=0)
    c = np.cov(z, rowvar=False, bias=True)
    w = z @ np.linalg.inv(np.linalg.cholesky(c)).T
    return w @ np.linalg.cholesky(np.asarray(cov2, float)).T + np.asarray(centre, float)


def _cs_cov(alpha, beta, eps):
    g = (1.0 + alpha * alpha) / beta
    return eps * np.array([[beta, -alpha], [-alpha, g]])


def _particles(planes, n=600, seed=0, centre=None):
    """(n, 6) array with exact Twiss per plane: planes = {"x": (α, β, ε), ...}."""
    rng = np.random.default_rng(seed)
    centre = centre or {}
    p = np.zeros((n, 6))
    for pl, (i, j) in (("x", (0, 1)), ("y", (2, 3)), ("z", (4, 5))):
        a, b, e = planes.get(pl, (0.0, 1.0, 1.0))
        p[:, (i, j)] = _exact_cov(n, _cs_cov(a, b, e), centre.get(pl, (0.0, 0.0)), rng)
    return p


def _ref(s=0.0):
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    return ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5, s=s)


def _recorder(p, ref_s=0.0, length=400.0, continuous=False, snapshots=None):
    from linac_gen.core.beam import Beam
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rec = DiagnosticRecorder()
    rec.s = [0.0, length]
    rec.element_names = ["INPUT", "Q1"]
    for s_mm, q in (snapshots or {}).items():
        b = Beam(ref=_ref(s_mm), n_particles=len(q), current=0.0)
        b.particles[:] = q
        rec.save_snapshot(b, s_mm)
    beam = Beam(ref=_ref(ref_s), n_particles=len(p), current=0.0)
    beam.particles[:] = p
    beam.continuous = continuous
    rec.beam = beam
    return rec


def _cfg(**kw):
    from linac_gen.core.config import BeamConfig
    base = dict(species="H-", energy=2.12, frequency=162.5, n_particles=10,
                emit_nx=0.25, emit_ny=0.25, emit_z=0.3,
                alpha_x=0.0, beta_x=1.0, alpha_y=0.0, beta_y=1.0,
                alpha_z=0.0, beta_z=2.0)
    base.update(kw)
    return BeamConfig(**base)


def _fodo(n_cells=3):
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=10.0))
    return lat


def _state(p=None, ref_s=0.0, cfg=None, lattice=None, **rec_kw):
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    if lattice is not None:
        st.set_lattice(lattice)
    if cfg is not None:
        st.set_beam_config(cfg)
    if p is not None:
        st.set_results(_recorder(p, ref_s=ref_s, **rec_kw))
    return st


def _popup(st, on=True):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, st)
    dlg.refresh(st.results)
    if on:
        dlg._ellipses.setChecked(True)
    return dlg


def _curves(panel):
    out = {}
    for it in panel.plot.listDataItems():
        name = getattr(it, "name", lambda: None)()
        if name:
            x, y = it.getData()
            out[name] = (np.asarray(x, float), np.asarray(y, float))
    return out


def _half_extent(curve, axis=0):
    v = curve[axis]
    return 0.5 * (v.max() - v.min())


def _bmag_from_title(panel):
    t = panel._title
    assert "BMAG" in t, t
    return t.split("BMAG")[-1].strip()


def _shoelace(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ------------------------------------------------------------------- tests
def test_checkbox_default_off_and_no_curves(qapp):
    dlg = _popup(_state(_particles({})), on=False)
    try:
        assert dlg._ellipses.isChecked() is False
        for panel in (dlg._panel_xx, dlg._panel_yy, dlg._panel_phi):
            assert "rms ellipse" not in _curves(panel)
        assert dlg._panel_xx._title == "x – x'"
        assert dlg._panel_phi._title == "φ – W"
    finally:
        dlg.deleteLater()


def test_rms_ellipse_matches_particle_twiss(qapp):
    from linac_gen.diagnostics.moments import compute_twiss_from_particles
    planes = {"x": (0.7, 2.5, 1.3), "y": (-1.1, 0.8, 0.4), "z": (0.3, 20.0, 0.05)}
    p = _particles(planes, centre={"x": (1.5, -0.2), "y": (-0.4, 0.1), "z": (3.0, -0.01)})
    dlg = _popup(_state(p))
    try:
        for pl, panel in (("x", dlg._panel_xx), ("y", dlg._panel_yy), ("z", dlg._panel_phi)):
            cur = _curves(panel)["rms ellipse"]
            tw = compute_twiss_from_particles(p, pl)
            i, j = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}[pl]
            assert 0.5 * (cur[0].max() + cur[0].min()) == pytest.approx(p[:, i].mean(), abs=1e-9)
            assert 0.5 * (cur[1].max() + cur[1].min()) == pytest.approx(p[:, j].mean(), abs=1e-9)
            assert _half_extent(cur, 0) == pytest.approx(math.sqrt(tw["emittance"] * tw["beta"]), rel=1e-9)
            # u′ extremum lies between the 200 samples when α ≠ 0: (π/200)²/2 ≈ 1e-4
            assert _half_extent(cur, 1) == pytest.approx(math.sqrt(tw["emittance"] * tw["gamma_t"]), rel=5e-4)
            assert cur[0][0] == pytest.approx(cur[0][-1]) and cur[1][0] == pytest.approx(cur[1][-1])
        assert "rms ellipse" not in _curves(dlg._panel_xy)
    finally:
        dlg.deleteLater()


def test_bmag_analytic_input_twiss_at_s0(qapp):
    """Design = Beam-tab input Twiss at s = 0: closed-form BMAG values."""
    cfg = _cfg(alpha_x=0.0, beta_x=1.0, alpha_y=0.0, beta_y=1.0, alpha_z=0.0, beta_z=2.0)
    p = _particles({"x": (0.0, 2.0, 1.0), "y": (1.0, 1.0, 1.0), "z": (0.0, 8.0, 0.1)})
    dlg = _popup(_state(p, ref_s=0.0, cfg=cfg))
    try:
        assert _bmag_from_title(dlg._panel_xx) == "1.250"
        assert _bmag_from_title(dlg._panel_yy) == "1.500"
        assert _bmag_from_title(dlg._panel_phi) == "2.125"
        assert dlg._probe_worker is None
        assert "design ellipse" in _curves(dlg._panel_xx)
        assert "input Twiss" in dlg._ellipses.toolTip()
    finally:
        dlg.deleteLater()


def test_design_ellipse_same_area_centre_and_design_extent(qapp):
    cfg = _cfg(alpha_x=0.5, beta_x=3.0)
    p = _particles({"x": (0.0, 2.0, 1.7)}, centre={"x": (2.0, 0.3)})
    dlg = _popup(_state(p, ref_s=0.0, cfg=cfg))
    try:
        c = _curves(dlg._panel_xx)
        rms, des = c["rms ellipse"], c["design ellipse"]
        assert _shoelace(*rms) == pytest.approx(math.pi * 1.7, rel=1e-3)
        assert _shoelace(*des) == pytest.approx(math.pi * 1.7, rel=1e-3)
        assert 0.5 * (des[0].max() + des[0].min()) == pytest.approx(2.0, abs=1e-9)
        assert _half_extent(des, 0) == pytest.approx(math.sqrt(1.7 * 3.0), rel=1e-9)
    finally:
        dlg.deleteLater()


def _fake_env(lat, cfg_rows):
    """EnvelopeResults with one Σ row per (s, twiss dict)."""
    from linac_gen.tracking.envelope import EnvelopeResults, _build_sigma_matrix
    env = EnvelopeResults()
    for s_mm, tw in cfg_rows:
        env.s.append(float(s_mm))
        env.sigma_matrix.append(_build_sigma_matrix(*tw))
    env.element_exit_idx = list(range(1, len(lat.elements) + 1))
    return env


def test_design_from_companion_cache(qapp):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen.analysis.phase_advance import mismatch_factor
    from linac_gen.diagnostics.moments import compute_twiss_from_particles
    from linac_gen.tracking.envelope import _sigma_to_twiss
    lat = _fodo(1)                                     # 4 elements, 300 mm
    cfg = _cfg()
    p_exit = _particles({"x": (0.2, 1.5, 1.0), "y": (0.0, 1.0, 1.0)}, seed=1)
    p_snap = _particles({"x": (-0.4, 0.9, 1.0), "y": (0.0, 1.0, 1.0)}, seed=2)
    st = _state(p_exit, ref_s=300.0, cfg=cfg, lattice=lat, length=300.0,
                snapshots={150.0: p_snap})
    rows = [(0.0, (0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.1)),
            (100.0, (0.1, 1.2, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.1)),
            (150.0, (0.3, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.1)),
            (250.0, (0.2, 1.4, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.1)),
            (300.0, (-0.5, 0.7, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.1))]
    env = _fake_env(lat, rows)
    rt._COMPANION_PROBE.clear()
    rt._COMPANION_PROBE.update({"sig": rt._probe_sig(lat, cfg), "results": env})
    dlg = _popup(st)
    try:
        assert dlg._probe_worker is None               # cache hit: no worker
        tw = compute_twiss_from_particles(p_exit, "x")
        d = _sigma_to_twiss(env.sigma_matrix[4], "x")
        want = mismatch_factor(tw["alpha"], tw["beta"], d["alpha"], d["beta"])
        assert _bmag_from_title(dlg._panel_xx) == f"{want:.3f}"
        assert "envelope" in dlg._ellipses.toolTip()
        dlg._location.setCurrentIndex(dlg._location.findData(150.0))
        tw = compute_twiss_from_particles(p_snap, "x")
        d = _sigma_to_twiss(env.sigma_matrix[2], "x")
        want = mismatch_factor(tw["alpha"], tw["beta"], d["alpha"], d["beta"])
        assert _bmag_from_title(dlg._panel_xx) == f"{want:.3f}"
        assert dlg._probe_worker is None
    finally:
        dlg.deleteLater()


def test_cache_miss_starts_probe_and_fills_cache(qapp):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    lat = _fodo()
    cfg = _cfg(current=0.0)
    p = _particles({"x": (0.0, 1.0, 1.0), "y": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, ref_s=900.0, cfg=cfg, lattice=lat, length=900.0))
    try:
        assert dlg._probe_worker is not None
        assert dlg._panel_xx._title.endswith("BMAG …")
        assert dlg._probe_worker.wait(120000)
        for _ in range(3):
            qapp.processEvents()
        val = float(_bmag_from_title(dlg._panel_xx))
        assert math.isfinite(val) and val >= 1.0
        assert "design ellipse" in _curves(dlg._panel_xx)
        assert rt._COMPANION_PROBE.get("sig") == rt._probe_sig(lat, cfg)
        assert len(rt._COMPANION_PROBE["results"].element_exit_idx) == len(lat.elements)
    finally:
        dlg.close(); dlg.deleteLater()


def test_truncated_probe_is_not_cached(qapp):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen.tracking.envelope import EnvelopeResults
    lat = _fodo()
    cfg = _cfg()
    p = _particles({"x": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, ref_s=900.0, cfg=cfg, lattice=lat, length=900.0), on=False)
    try:
        sig = rt._probe_sig(lat, cfg)
        dlg._probe_sig_pending = sig
        env = EnvelopeResults(); env.s = [0.0]; env.element_exit_idx = []
        dlg._on_probe_ready(env)
        assert rt._COMPANION_PROBE.get("results") is None
        assert dlg._design_failed_sig is None
    finally:
        dlg.close(); dlg.deleteLater()


def test_probe_failure_latches_and_recovers_on_config_change(qapp):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    lat = _fodo()
    cfg = _cfg()
    p = _particles({"x": (0.0, 1.0, 1.0)})
    st = _state(p, ref_s=900.0, cfg=cfg, lattice=lat, length=900.0)
    dlg = _popup(st)
    try:
        w = dlg._probe_worker
        assert w is not None
        w.request_stop(); w.wait(60000)
        dlg._probe_worker = None
        dlg._probe_sig_pending = rt._probe_sig(lat, cfg)
        dlg._on_probe_failed("boom")
        assert dlg._panel_xx._title.endswith("BMAG n/a")
        assert "boom" in dlg._ellipses.toolTip()
        dlg._redraw()
        assert dlg._probe_worker is None                # latched: no relaunch
        st.set_beam_config(_cfg(beta_x=1.7))            # new signature → retry
        dlg._redraw()
        assert dlg._probe_worker is not None
    finally:
        dlg.close(); dlg.deleteLater()


def test_worker_path_through_real_window(win):
    """Real entry path: FODO run → Results tab → Phase space → tick ellipses."""
    import linac_gen_gui.interphase.tabs.results_tab as rt
    win.open_lattice(FODO)
    win.beam_tab._npart.setValue(300)
    win.beam_tab._apply(); win.pump()
    win._run_mp(); win.wait_worker(win._mp_worker)
    assert win.results_tab.open_plot("phase")
    dlg = win.results_tab._popups["phase"]
    dlg._ellipses.setChecked(True)
    win.wait_popup_idle(dlg); win.pump(0.3)
    for panel in (dlg._panel_xx, dlg._panel_yy):
        val = float(_bmag_from_title(panel))
        assert math.isfinite(val) and val >= 1.0
        assert {"rms ellipse", "design ellipse"} <= set(_curves(panel))
    assert "rms ellipse" not in _curves(dlg._panel_xy)
    assert rt._COMPANION_PROBE.get("results") is not None
    assert win.message_boxes == []
    dlg._ellipses.setChecked(False)
    assert "rms ellipse" not in _curves(dlg._panel_xx)
    assert dlg._panel_xx._title == "x – x'"


def test_checkbox_off_export_unchanged(qapp):
    p = _particles({"x": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, cfg=_cfg()), on=False)
    try:
        def names():
            return [str(c.name) for pn in dlg._collect_panels() for c in (getattr(pn, "curves", None) or [])]
        before = names()
        dlg._ellipses.setChecked(True)
        on = names()
        assert "rms ellipse" in on and "design ellipse" in on
        assert not any("value" in n.lower() for n in on)
        dlg._ellipses.setChecked(False)
        assert names() == before
    finally:
        dlg.deleteLater()


def _train(n=900, seed=3):
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 6))
    p[:, 0] = rng.normal(0, 0.5, n); p[:, 1] = rng.normal(0, 0.5, n)
    p[:, 2] = rng.normal(0, 0.5, n); p[:, 3] = rng.normal(0, 0.5, n)
    p[:, 4] = rng.normal(0, 4.0, n); p[:, 5] = rng.normal(0, 0.02, n)
    p[300:600, 4] += 360.0; p[600:, 4] -= 360.0
    return p


@pytest.mark.parametrize("periodic", [False, True])
def test_fold_regimes(qapp, periodic):
    st = _state(_train())
    st.results.periodic_phase = periodic
    dlg = _popup(st)
    try:
        ext = _half_extent(_curves(dlg._panel_phi)["rms ellipse"], 0)
        if periodic:                      # fold control disabled: raw train
            assert ext > 100.0
        else:
            assert ext == pytest.approx(4.0, rel=0.4)
            dlg._wrap_phi.setChecked(False)
            assert _half_extent(_curves(dlg._panel_phi)["rms ellipse"], 0) > 100.0
    finally:
        dlg.deleteLater()


def test_tracewin_basis_scales_longitudinal_ellipse(qapp):
    from linac_gen.core.constants import C_LIGHT
    from linac_gen.core.particle import H_MINUS
    cfg = _cfg()
    p = _particles({"z": (0.4, 6.0, 0.2)})
    dlg = _popup(_state(p, cfg=cfg))
    try:
        ours = _curves(dlg._panel_phi)["rms ellipse"]
        b0 = _bmag_from_title(dlg._panel_phi)
        dlg._basis.setCurrentIndex(dlg._basis.findData("tracewin"))
        tw = _curves(dlg._panel_phi)["rms ellipse"]
        gamma = 1 + cfg.energy / H_MINUS.mass
        beta = math.sqrt(1 - 1 / gamma ** 2)
        wl_m = C_LIGHT / (cfg.frequency * 1e6)
        assert _half_extent(tw, 0) == pytest.approx(_half_extent(ours, 0) * beta * wl_m / 360.0, rel=1e-9)
        assert _half_extent(tw, 1) == pytest.approx(_half_extent(ours, 1) / (beta * beta * gamma * H_MINUS.mass), rel=1e-9)
        assert _bmag_from_title(dlg._panel_phi) == b0
        dlg._basis.setCurrentIndex(dlg._basis.findData("ours"))
        back = _curves(dlg._panel_phi)["rms ellipse"]
        assert _half_extent(back, 0) == pytest.approx(_half_extent(ours, 0), rel=1e-12)
    finally:
        dlg.deleteLater()


def test_dc_record_has_no_longitudinal_ellipse(qapp):
    p = _particles({"x": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, cfg=_cfg(), continuous=True))
    try:
        assert "rms ellipse" in _curves(dlg._panel_xx)
        assert "rms ellipse" not in _curves(dlg._panel_phi)
        assert dlg._panel_phi._title == "φ – W"
    finally:
        dlg.deleteLater()
    # regenerated-input route with a DC config: same rule
    from linac_gen_gui.interphase.state import AppState
    st = AppState(); st.set_beam_config(_cfg(continuous=True, dc_energy_spread_keV=1.0))
    dlg = _popup(st)
    try:
        assert "rms ellipse" in _curves(dlg._panel_xx)
        assert "rms ellipse" not in _curves(dlg._panel_phi)
    finally:
        dlg.deleteLater()


def test_empty_and_degenerate_inputs(qapp):
    from linac_gen_gui.interphase.state import AppState
    dlg = _popup(AppState())                           # nothing at all
    try:
        assert all("rms ellipse" not in _curves(pn) for pn in (dlg._panel_xx, dlg._panel_yy, dlg._panel_phi))
        dlg.refresh(None)
    finally:
        dlg.deleteLater()
    p = _particles({"y": (0.0, 1.0, 1.0)}); p[:, 1] = 0.0   # x′ ≡ 0: rank-1 x plane
    dlg = _popup(_state(p))
    try:
        assert "rms ellipse" not in _curves(dlg._panel_xx)
        assert dlg._panel_xx._title == "x – x'"
        assert "rms ellipse" in _curves(dlg._panel_yy)
    finally:
        dlg.deleteLater()
    dlg = _popup(_state(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])))
    try:
        dlg._redraw()                                   # 2 particles: no crash
    finally:
        dlg.deleteLater()


def test_no_lattice_gives_na_and_no_worker(qapp):
    p = _particles({"x": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, ref_s=400.0, cfg=_cfg()))   # exit beam, no lattice
    try:
        assert dlg._panel_xx._title.endswith("BMAG n/a")
        assert dlg._probe_worker is None
        assert "no lattice" in dlg._ellipses.toolTip()
        assert "rms ellipse" in _curves(dlg._panel_xx)
        assert "design ellipse" not in _curves(dlg._panel_xx)
    finally:
        dlg.deleteLater()


def test_scatter_colour_mode_keeps_overlay(qapp):
    p = _particles({"x": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, cfg=_cfg()))
    try:
        dlg._colour.setCurrentIndex(dlg._colour.findData("dw"))
        assert {"rms ellipse", "design ellipse"} <= set(_curves(dlg._panel_xx))
    finally:
        dlg.deleteLater()


def test_setting_persists_to_a_new_popup(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    st = _state(_particles({}))
    dlg = _popup(st)
    dlg.deleteLater()
    dlg2 = _PhaseSpacePopup(None, st)
    try:
        assert dlg2._ellipses.isChecked() is True
        dlg2._ellipses.setChecked(False)
        dlg3 = _PhaseSpacePopup(None, st)
        assert dlg3._ellipses.isChecked() is False
        dlg3.deleteLater()
    finally:
        dlg2.deleteLater()


def test_close_with_running_probe_stops_worker(qapp):
    import time
    lat = _fodo(20)
    p = _particles({"x": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, ref_s=6000.0, cfg=_cfg(), lattice=lat, length=6000.0))
    assert dlg._probe_worker is not None
    dlg.close()
    t0 = time.time()
    while dlg._live_workers() and time.time() - t0 < 5.0:
        qapp.processEvents(); time.sleep(0.05)
    assert dlg._live_workers() == []
    dlg.deleteLater()


def test_legend_lists_both_ellipses_and_hides_when_off(qapp):
    """Each overlaid panel carries a legend naming the two ellipses; the
    legend is hidden when the overlay is off or the plane has no ellipse,
    and the legend never changes the exported curve names."""
    p = _particles({"x": (0.0, 1.0, 1.0), "y": (0.0, 1.0, 1.0)})
    dlg = _popup(_state(p, cfg=_cfg()))
    try:
        for panel in (dlg._panel_xx, dlg._panel_yy, dlg._panel_phi):
            leg = dlg._ell_items[id(panel)]["legend"]
            assert leg.isVisible()
            labels = [lab.text for _sample, lab in leg.items]
            assert labels == ["beam rms ellipse", "design ellipse (same emittance)"]
        assert set(_curves(dlg._panel_xx)) == {"rms ellipse", "design ellipse"}
        dlg._ellipses.setChecked(False)
        for panel in (dlg._panel_xx, dlg._panel_yy, dlg._panel_phi):
            leg = dlg._ell_items[id(panel)]["legend"]
            assert not leg.isVisible() and leg.items == []
    finally:
        dlg.deleteLater()
    # no design available: only the beam ellipse is listed
    dlg = _popup(_state(p, ref_s=400.0, cfg=_cfg()))          # exit beam, no lattice
    try:
        leg = dlg._ell_items[id(dlg._panel_xx)]["legend"]
        assert [lab.text for _s, lab in leg.items] == ["beam rms ellipse"]
    finally:
        dlg.deleteLater()
