"""Results-tab solenoid / cavity charts see every field map.

The Peak |B_z|, ∫B²·dz, E_acc and V₀ charts skipped maps inside a
SUPERPOSE_MAP cluster — every PIP-II HWR/SSR solenoid carries two
superposed correctors, so a verbatim TraceWin SCL deck showed NO solenoid
at all — and the Peak |B_z| popup also skipped 3-D solenoid maps (29 of
the 37 PIP-II solenoids).  A cluster is now one stem valued from the SUMMED
on-axis field of its children; zero-amplitude children add nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

_noop = lambda *a, **k: None                                  # noqa: E731


def _write_1d(path, vals, zmax_m):
    path.write_text(f"{len(vals) - 1} {zmax_m!r}\n1\n"
                    + "\n".join(repr(float(v)) for v in vals) + "\n",
                    encoding="utf-8")


def _write_3d(prefix, bz_axis, zmax_m, n=4, half=0.016):
    """3-D static-B map: Bz(z) uniform across the (x, y) grid, Bx = By = 0."""
    nz = len(bz_axis) - 1
    hdr = f"{nz} {zmax_m!r}\n{n} {-half!r} {half!r}\n{n} {-half!r} {half!r}\n1\n"
    for comp in ("bsx", "bsy", "bsz"):
        with open(f"{prefix}.{comp}", "w", encoding="utf-8") as f:
            f.write(hdr)
            for k in range(nz + 1):
                v = bz_axis[k] if comp == "bsz" else 0.0
                f.write(("%r\n" % float(v)) * ((n + 1) * (n + 1)))


@pytest.fixture()
def deck(tmp_path):
    z = np.linspace(0.0, 0.3, 61)
    bump = 1.0 / (1.0 + ((z - 0.15) / 0.05) ** 4)
    _write_1d(tmp_path / "sol.bsz", bump, 0.3)
    _write_1d(tmp_path / "cav.edz", np.sin(np.pi * z / 0.3), 0.3)
    _write_3d(str(tmp_path / "cx"), 0.02 * np.ones(61), 0.3)
    _write_3d(str(tmp_path / "sol3"), bump, 0.3)
    p = tmp_path / "d.dat"
    p.write_text(
        "FREQ 162.5\n"
        # 1: solenoid + two zero-amplitude 3-D correctors (PIP-II idiom)
        "SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 0 0 0 0 cx\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 0.0 0 0 0 cx\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.3 0 0 0 sol\n"
        "DRIFT 50 16 0\n"
        # 2: standalone 3-D solenoid map
        "FIELD_MAP 70 300 0 16 1.5 0 0 0 sol3\n"
        "DRIFT 50 16 0\n"
        # 3: two half-strength solenoids superposed == one full
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 50 16 0\n"
        # 4: two half-strength cavities superposed == one full
        "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
        "DRIFT 50 16 0\n"
        "FIELD_MAP 100 300 0 16 0 1.0 0 0 cav\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "END\n", encoding="utf-8")
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, meta = parse_tracewin(str(p))
    return lat


def _stems(deck, key, qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    state = AppState()
    state.set_lattice(deck, path=None)
    tab = ResultsTab(state, _noop, _noop, _noop)
    tab._open_popup(key)
    pop = tab._popups[key]
    import pyqtgraph as pg
    sc = [it for it in pop._plot.listDataItems()
          if isinstance(it, pg.ScatterPlotItem)]
    x, y = (sc[0].getData() if sc else (np.array([]), np.array([])))
    pop.close()
    return np.asarray(x), np.asarray(y), tab


def test_cluster_values_equal_their_contributing_maps(deck):
    from linac_gen_gui.interphase.tabs import results_tab as rt
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    els = list(deck.elements)
    clusters = [e for e in els if isinstance(e, SuperposedFieldMap)]
    assert len(clusters) == 3
    sol_cl, half_sol, half_cav = clusters
    full_cav, full_sol = els[-2], els[-1]
    sol_child = sol_cl.children[2][1]
    # unpowered correctors add nothing: EXACTLY the solenoid's own value
    assert rt._fieldmap_bpeak_T(sol_cl) == rt._fieldmap_bpeak_T(sol_child)
    assert rt._fieldmap_int_b2(sol_cl) == rt._fieldmap_int_b2(sol_child)
    # superposition: two halves == one full (same grid ⇒ to round-off)
    assert rt._fieldmap_bpeak_T(half_sol) == pytest.approx(
        rt._fieldmap_bpeak_T(full_sol), rel=1e-12)
    assert rt._fieldmap_int_b2(half_sol) == pytest.approx(
        rt._fieldmap_int_b2(full_sol), rel=1e-12)
    assert rt._fieldmap_eacc_MV_per_m(half_cav) == pytest.approx(
        rt._fieldmap_eacc_MV_per_m(full_cav), rel=1e-12)
    assert rt._fieldmap_vgap_MV(half_cav) == pytest.approx(
        rt._fieldmap_vgap_MV(full_cav), rel=1e-12)
    # solenoid clusters are not cavities and vice versa
    assert rt._fieldmap_eacc_MV_per_m(sol_cl) is None
    assert rt._fieldmap_bpeak_T(half_cav) is None


@pytest.mark.parametrize("key,n", [("bpeak", 4), ("int_b2", 4),
                                   ("eacc", 2), ("rf_volt", 2)])
def test_popups_show_every_solenoid_and_cavity(deck, qapp, key, n):
    """Through the real Results-tab popups: solenoid cluster, standalone
    3-D solenoid, half+half cluster and the plain 1-D solenoid (4); the
    half+half cavity cluster and the plain cavity (2)."""
    x, y, _tab = _stems(deck, key, qapp)
    assert len(x) == n, (key, x, y)
    assert np.all(np.isfinite(y)) and np.all(y > 0)


def _parse_text(tmp_path, text):
    from linac_gen.io.tracewin_parser import parse_tracewin
    p = tmp_path / "x.dat"
    p.write_text(text, encoding="utf-8")
    return list(parse_tracewin(str(p))[0].elements)


def test_rf_children_add_as_phasors(deck, tmp_path):
    """Two half-strength cavities 180° apart cancel; ±ke 180° apart add."""
    from linac_gen_gui.interphase.tabs import results_tab as rt
    full = [e for e in deck.elements][-2]           # plain full cavity
    opp = _parse_text(tmp_path,
        "FREQ 162.5\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 180 16 0 0.5 0 0 cav\nEND\n")[1]
    assert rt._fieldmap_eacc_MV_per_m(opp) < 1e-12
    assert rt._fieldmap_vgap_MV(opp) < 1e-12
    flip = _parse_text(tmp_path,
        "FREQ 162.5\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 180 16 0 -0.5 0 0 cav\nEND\n")[1]
    assert rt._fieldmap_eacc_MV_per_m(flip) == pytest.approx(
        rt._fieldmap_eacc_MV_per_m(full), rel=1e-12)
    assert rt._fieldmap_vgap_MV(flip) == pytest.approx(
        rt._fieldmap_vgap_MV(full), rel=1e-12)


def test_only_the_tracked_span_counts(deck, tmp_path):
    """A solenoid placed at z0 = −150 mm is tracked only from the cluster
    entrance on: ∫B² counts its downstream half (the bump is symmetric)."""
    from linac_gen_gui.interphase.tabs import results_tab as rt
    full_sol = [e for e in deck.elements][-1]       # plain sol, kb = 2.0
    cl = _parse_text(tmp_path,
        "SUPERPOSE_MAP -150\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 0 0 0 0 sol\nEND\n")[0]
    assert cl.length == pytest.approx(300.0)
    assert rt._fieldmap_int_b2(cl) == pytest.approx(
        0.5 * rt._fieldmap_int_b2(full_sol), rel=1e-9)


def test_int_b2_is_continuous_when_a_corrector_is_powered(tmp_path, deck):
    """Powering a corrector from 0 to 1e-9 must not step ∫B² (the old
    union-grid sum jumped by the regridding error, −0.1 % on the FDR HWR
    cluster)."""
    from linac_gen_gui.interphase.tabs import results_tab as rt
    # a corrector on a FINE grid (348 samples, as the PIP-II hwrcx map)
    # against the coarse 61-sample solenoid — the case that regrids
    _write_3d(str(tmp_path / "cxf"), 0.02 * np.ones(348), 0.29967)
    txt = ("SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 {k} 0 0 0 cxf\n"
           "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.3 0 0 0 sol\nEND\n")
    off = _parse_text(tmp_path, txt.format(k=0))[0]
    on = _parse_text(tmp_path, txt.format(k=1e-9))[0]
    a, b = rt._fieldmap_int_b2(off), rt._fieldmap_int_b2(on)
    assert abs(b - a) / a < 1e-8, (a, b)
