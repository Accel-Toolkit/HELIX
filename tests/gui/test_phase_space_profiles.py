"""Phase Space popup: the projected profiles beside each density map.

Pins: each profile is the histogram of the plotted coordinate over the map's
range (same bins), scaled to its peak; the dashed rms Gaussian has the
particles' mean and rms and the same normalisation; the strips follow the
map's range one way (zoom, re-arm) and line up with it; they follow the
basis, the location and the colour mode; the "profiles" / "rms Gaussian"
toggles hide them and are remembered, and hidden the panel is laid out as
one built without them; particles left out of the statistics
(``beam.stats_excluded``) are left out of the plot too, with a note; the
Beam tab's panels have no profiles; the density colour map comes back
after a scatter colour mode.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

KEYS = ("linac_gen/phase_space/profiles",
        "linac_gen/phase_space/profile_gaussian",
        "linac_gen/phase_space/ellipses")


@pytest.fixture(autouse=True)
def _clean_settings():
    from linac_gen_gui.interphase.app_settings import make_settings
    s = make_settings("linac_gen", "linac_gen_gui")
    for k in KEYS:
        s.remove(k)
    s.sync()
    yield
    s = make_settings("linac_gen", "linac_gen_gui")
    for k in KEYS:
        s.remove(k)
    s.sync()


# ----------------------------------------------------------------- helpers
def _particles(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 6))
    p[:, 0] *= 1.5            # x (mm)
    p[:, 1] *= 3.0            # x' (mrad)
    p[:, 2] *= 0.8
    p[:, 3] *= 2.0
    p[:, 4] = 6.0 * p[:, 4]   # phase (deg)
    p[:, 5] = 0.007 * p[:, 5] + 0.025   # dW (MeV)
    return p


def _ref(s=0.0):
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    return ReferenceParticle(species=H_MINUS, w_kin=2.0985, frequency=162.5, s=s)


def _recorder(p, excluded=None, snapshots=None):
    from linac_gen.core.beam import Beam
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rec = DiagnosticRecorder()
    rec.s = [0.0, 400.0]
    rec.element_names = ["INPUT", "Q1"]
    for s_mm, (q, ex) in (snapshots or {}).items():
        b = Beam(ref=_ref(s_mm), n_particles=len(q), current=0.0)
        b.particles[:] = q
        if ex is not None:
            b.stats_excluded = ex
        rec.save_snapshot(b, s_mm)
    beam = Beam(ref=_ref(400.0), n_particles=len(p), current=0.0)
    beam.particles[:] = p
    if excluded is not None:
        beam.stats_excluded = excluded
    rec.beam = beam
    return rec


def _state(p, **kw):
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    st.set_beam_config(BeamConfig(species="H-", energy=0.03, frequency=162.5,
                                  n_particles=10))
    st.set_results(_recorder(p, **kw))
    return st


def _popup(st):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, st)
    dlg.refresh(st.results)
    return dlg


def _expected(q, rng, bins):
    h, edges = np.histogram(q, bins=bins, range=rng)
    return edges, h / h.max()


def _range(q):
    return (q.mean() - 4.5 * q.std(), q.mean() + 4.5 * q.std())


# -------------------------------------------------------------------- tests
def test_profiles_are_the_histograms_of_the_plotted_coordinates(qapp):
    p = _particles()
    dlg = _popup(_state(p))
    try:
        for panel, (i, j) in ((dlg._panel_xx, (0, 1)), (dlg._panel_yy, (2, 3)),
                              (dlg._panel_xy, (0, 2)), (dlg._panel_phi, (4, 5))):
            assert panel.profiles_visible()
            for curve, col in ((panel._px_curve, i), (panel._py_curve, j)):
                x, y = curve.getData()
                edges, want = _expected(p[:, col], _range(p[:, col]), panel._bins)
                assert np.allclose(x, edges) and np.allclose(y, want)
        assert dlg._panel_xx._px_curve.name() == "x profile"
        assert dlg._panel_xx._py_curve.name() == "x' profile"
        assert dlg._panel_phi._py_curve.name() == "ΔW profile"
    finally:
        dlg.deleteLater()


def test_rms_gaussian_has_the_particles_mean_and_rms(qapp):
    p = _particles()
    dlg = _popup(_state(p))
    try:
        panel = dlg._panel_xx
        q = p[:, 0]
        lo, hi = _range(q)
        g_x, g = panel._px_gauss.getData()
        h, _ = np.histogram(q, bins=panel._bins, range=(lo, hi))
        width = (hi - lo) / panel._bins
        want = (q.size * width / (q.std() * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((g_x - q.mean()) / q.std()) ** 2)) / h.max()
        assert np.allclose(g, want)
        assert g_x[np.argmax(g)] == pytest.approx(q.mean(), abs=2 * (hi - lo) / 400)
        # a Gaussian beam's profile lies on its rms Gaussian (to sampling noise)
        x, y = panel._px_curve.getData()
        centres = 0.5 * (x[1:] + x[:-1])
        g_at = np.interp(centres, g_x, g)
        core = g_at > 0.3
        assert np.max(np.abs(y[core] - g_at[core])) < 0.12
        assert panel._px_gauss.isVisible()
    finally:
        dlg.deleteLater()


def test_profiles_follow_the_map_range_one_way(qapp):
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.plot_style import rearm_autorange
    dlg = _popup(_state(_particles()))
    try:
        panel = dlg._panel_xx
        main = panel.plot.getViewBox()
        main.setXRange(-1.0, 2.0, padding=0)
        main.setYRange(-4.0, 3.0, padding=0)
        assert panel._prof_x.getViewBox().viewRange()[0] == pytest.approx([-1.0, 2.0])
        assert panel._prof_y.getViewBox().viewRange()[1] == pytest.approx([-4.0, 3.0])
        # re-arming: the map auto-ranges again, the strips copy it
        rearm_autorange(dlg)
        assert main.autoRangeEnabled()[0] and main.autoRangeEnabled()[1]
        main.updateAutoRange()
        assert (panel._prof_x.getViewBox().viewRange()[0]
                == pytest.approx(main.viewRange()[0]))
        assert (panel._prof_y.getViewBox().viewRange()[1]
                == pytest.approx(main.viewRange()[1]))
        # the strips never auto-range their followed axis themselves
        assert not panel._prof_x.getViewBox().autoRangeEnabled()[0]
        assert not panel._prof_y.getViewBox().autoRangeEnabled()[1]
        assert not panel._prof_x.getViewBox().state["mouseEnabled"][0]
        assert pg.ViewBox.XAxis == 0
    finally:
        dlg.deleteLater()


def test_strips_line_up_with_the_map(qapp):
    dlg = _popup(_state(_particles()))
    try:
        dlg.resize(1300, 1000)
        dlg.show()
        for _ in range(5):
            qapp.processEvents()
        for panel in dlg._density_panels():
            m = panel.plot.getViewBox().sceneBoundingRect()
            bx = panel._prof_x.getViewBox().sceneBoundingRect()
            ly = panel._prof_y.getViewBox().sceneBoundingRect()
            assert abs(bx.left() - m.left()) <= 1.0 and abs(bx.right() - m.right()) <= 1.0
            assert abs(ly.top() - m.top()) <= 1.0 and abs(ly.bottom() - m.bottom()) <= 1.0
            assert bx.top() >= m.bottom() and ly.right() <= m.left()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_toggles_hide_and_are_remembered(qapp):
    from linac_gen_gui.interphase.plots.plot_style import DensityPanel
    st = _state(_particles())
    dlg = _popup(st)
    try:
        dlg._gauss_cb.setChecked(False)
        assert not dlg._panel_phi._px_gauss.isVisible()
        dlg._profiles_cb.setChecked(False)
        assert not dlg._gauss_cb.isEnabled()
        for panel in dlg._density_panels():
            assert not panel.profiles_visible()
            assert panel._prof_x not in panel.ci.items
            assert panel.plot.getAxis("left").style["showValues"]
            assert panel.plot.getAxis("bottom").labelText
    finally:
        dlg.deleteLater()
    # hidden, a panel is laid out as one built without profiles (same size,
    # same data, so the same tick labels)
    q = _particles(n=3000)
    off = DensityPanel(title="x – x'", xlabel="x", xunits="mm",
                       ylabel="x'", yunits="mrad", profiles=True)
    plain = DensityPanel(title="x – x'", xlabel="x", xunits="mm",
                         ylabel="x'", yunits="mrad")
    try:
        off.set_profiles_visible(False)
        for w in (off, plain):
            w.set_data(q[:, 0], q[:, 1])
            w.resize(600, 450)
            w.show()
        for _ in range(5):
            qapp.processEvents()
        a = off.plot.getViewBox().sceneBoundingRect()
        b = plain.plot.getViewBox().sceneBoundingRect()
        for u, v in ((a.left(), b.left()), (a.top(), b.top()),
                     (a.width(), b.width()), (a.height(), b.height())):
            assert abs(u - v) <= 1.0, (a, b)
    finally:
        off.close(); plain.close()
        off.deleteLater(); plain.deleteLater()
    again = _popup(st)
    try:
        assert not again._profiles_cb.isChecked() and not again._gauss_cb.isChecked()
        assert not again._panel_xx.profiles_visible()
        again._profiles_cb.setChecked(True)
        assert again._panel_xx.profiles_visible() and again._gauss_cb.isEnabled()
    finally:
        again.deleteLater()


def test_basis_location_and_colour_mode_recompute_the_profiles(qapp):
    p = _particles()
    snap = _particles(n=5000, seed=7) * 0.5
    dlg = _popup(_state(p, snapshots={300.0: (snap, None)}))
    try:
        panel = dlg._panel_phi
        dlg._basis.setCurrentIndex(dlg._basis.findData("tracewin"))
        assert panel._px_curve.name() == "z profile"
        assert "z" in panel._prof_x.getAxis("bottom").labelText
        x_tw, _ = panel._px_curve.getData()
        dlg._basis.setCurrentIndex(dlg._basis.findData("ours"))
        assert panel._px_curve.name() == "Δφ profile"
        x_ours, _ = panel._px_curve.getData()
        assert not np.allclose(x_tw, x_ours)
        # location: the snapshot's own histogram
        dlg._location.setCurrentIndex(dlg._location.findData(300.0))
        x, y = dlg._panel_xx._px_curve.getData()
        edges, want = _expected(snap[:, 0], _range(snap[:, 0]), dlg._panel_xx._bins)
        assert np.allclose(x, edges) and np.allclose(y, want)
        # scatter colour mode: over the points' own extent
        dlg._location.setCurrentIndex(0)
        dlg._colour.setCurrentIndex(dlg._colour.findData("dw"))
        x, y = dlg._panel_xx._px_curve.getData()
        edges, want = _expected(p[:, 0], (p[:, 0].min(), p[:, 0].max()),
                                dlg._panel_xx._bins)
        assert np.allclose(x, edges) and np.allclose(y, want)
        assert dlg._panel_xx.profiles_visible()
    finally:
        dlg.deleteLater()


def test_tracewin_basis_uses_the_reference_at_the_plotted_location(qapp):
    """z = −Δφ·βλ/360 and δ = ΔW/(β²γm) with β, γ, λ of the plotted beam's
    own reference: 2.0985 MeV at the exit, 1 MeV at the snapshot — never the
    Beam tab's 30 keV input.  The rms ellipse is scaled the same way."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    p = _particles()
    st = _state(p)
    snap = _particles(n=5000, seed=7)
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=1.0,
                                   frequency=162.5, s=300.0),
             n_particles=len(snap), current=0.0)
    b.particles[:] = snap
    st.results.save_snapshot(b, 300.0)
    dlg = _popup(st)
    try:
        panel = dlg._panel_phi
        dlg._ellipses.setChecked(True)
        dlg._basis.setCurrentIndex(dlg._basis.findData("tracewin"))
        for index, q, w in ((0, p, 2.0985),
                            (dlg._location.findData(300.0), snap, 1.0)):
            dlg._location.setCurrentIndex(index)
            ref = ReferenceParticle(species=H_MINUS, w_kin=w, frequency=162.5)
            z = -q[:, 4] * ref.beta * ref.wavelength * 1e-3 / 360.0
            d = q[:, 5] / (ref.beta ** 2 * ref.gamma * H_MINUS.mass)
            for curve, c in ((panel._px_curve, z), (panel._py_curve, d)):
                x, y = curve.getData()
                edges, want = _expected(c, _range(c), panel._bins)
                assert np.allclose(x, edges) and np.allclose(y, want)
            ex, _ey = dlg._ell_items[id(panel)]["rms"].getData()
            assert (ex.max() - ex.min()) / 2 == pytest.approx(z.std(),
                                                               rel=1e-9)
    finally:
        dlg.deleteLater()


def test_particles_left_out_of_the_statistics_are_left_out_of_the_plot(qapp):
    p = _particles()
    p[:3, 5] = -2.07                       # three unaccelerated particles
    ex = np.zeros(len(p), bool)
    ex[:3] = True
    snap_ex = np.zeros(len(p), bool)
    snap_ex[:2] = True
    dlg = _popup(_state(p, excluded=ex, snapshots={300.0: (p.copy(), snap_ex)}))
    try:
        assert dlg._n_left_out == 3
        assert not dlg._left_out_note.isHidden()
        assert dlg._left_out_note.text().startswith("3 unaccelerated particles")
        kept = p[3:, 5]
        x, y = dlg._panel_phi._py_curve.getData()
        edges, want = _expected(kept, _range(kept), dlg._panel_phi._bins)
        assert np.allclose(x, edges) and np.allclose(y, want)
        # the Beam-parameters table: statistics of the shown particles,
        # but every alive particle counts as transmitted
        dlg._params_btn.setChecked(True)
        rows = {dlg._table.item(r, 0).text().strip(): dlg._table.item(r, 1).text()
                for r in range(dlg._table.rowCount())
                if dlg._table.item(r, 0) is not None and dlg._table.item(r, 1) is not None}
        assert rows["macroparticles"] == f"{len(p)} / {len(p)}"
        assert float(rows["transmission"]) == pytest.approx(100.0)
        assert rows["left out of the statistics"] == "3"
        dlg._params_btn.setChecked(False)
        # snapshot: its own marks
        dlg._location.setCurrentIndex(dlg._location.findData(300.0))
        assert dlg._n_left_out == 2
        assert dlg._left_out_note.text().startswith("2 unaccelerated particles")
    finally:
        dlg.deleteLater()
    plain = _popup(_state(_particles()))
    try:
        assert plain._n_left_out == 0 and plain._left_out_note.isHidden()
    finally:
        plain.deleteLater()


def test_beam_tab_panels_have_no_profiles(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    tab = BeamTab(AppState())
    try:
        for name in ("_panel_xx", "_panel_yy", "_panel_zp", "_panel_xy"):
            panel = getattr(tab, name)
            assert panel._prof_x is None and not panel.profiles_visible()
    finally:
        tab.deleteLater()


def test_density_colour_map_returns_after_a_scatter_mode(qapp):
    from linac_gen_gui.interphase.plots.plot_style import DensityPanel
    panel = DensityPanel(title="t", profiles=True)
    try:
        lut0 = np.array(panel._image.lut, copy=True)
        x = np.random.default_rng(0).normal(size=500)
        panel.set_scatter(x, x, c=x)
        assert not np.array_equal(np.asarray(panel._image.lut), lut0)
        panel.set_data(x, x)
        assert np.array_equal(np.asarray(panel._image.lut), lut0)
    finally:
        panel.deleteLater()


def test_empty_and_degenerate_data_clear_the_profiles(qapp):
    from linac_gen_gui.interphase.plots.plot_style import DensityPanel
    panel = DensityPanel(title="t", profiles=True)
    try:
        panel.set_data([], [])
        assert panel._px_curve.getData()[0] is None or len(panel._px_curve.getData()[0]) == 0
        panel.set_data(np.ones(50), np.ones(50))      # zero spread
        assert len(panel._px_gauss.getData()[0] or []) == 0
        panel.set_scatter([], [])
        assert len(panel._py_curve.getData()[0] or []) == 0
    finally:
        panel.deleteLater()
