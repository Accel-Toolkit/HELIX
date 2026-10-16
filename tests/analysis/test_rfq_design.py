"""RFQ design and run quantities for the Results tab's RFQ panels
(linac_gen.analysis.rfq_design): positions, card geometry, the figures of
merit against hand arithmetic, the Kilpatrick limit, the run-row mapping,
losses, and the surface-field estimate on known fields."""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.analysis import rfq_design as rd
from linac_gen.io.tracewin_parser import parse_tracewin

DEMO = Path(__file__).resolve().parents[2] / "examples" / "rfq_demo" / "rfq_demo.dat"


@pytest.fixture(scope="module")
def demo():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(DEMO))
    return lat


def test_the_chain_and_its_positions(demo):
    ch = rd.find_rfq(demo)
    assert len(ch.cells) == 199 and ch.n_chains == 1
    total = sum(float(e.length) for e in demo.elements)
    assert ch.s0[0] == 0.0 and ch.s1[-1] == pytest.approx(total)
    np.testing.assert_allclose(ch.s1 - ch.s0, [c.length for c in ch.cells])
    assert ch.edges[-1] == pytest.approx(ch.s1[-1] - ch.s0[0])


def test_no_rfq_gives_none():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    lat = Lattice()
    lat.add(Drift("D", length=10.0))
    assert rd.find_rfq(lat) is None
    assert rd.cell_table(lat, frequency_MHz=352.21) is None


def test_focusing_strength_is_the_demo_readme_value(demo):
    t = rd.cell_table(demo, species="proton", frequency_MHz=352.21)
    lam = 299_792_458.0 / 352.21e6
    b = 85_000.0 * lam ** 2 / (938.27208816e6 * 3.40e-3 ** 2)
    assert t["B"][0] == pytest.approx(b, rel=1e-9)
    assert t["B"][0] == pytest.approx(5.7, abs=0.05)        # README: 5.7


def test_kilpatrick_limit_is_the_root_of_the_criterion():
    for f in (162.5, 352.21, 425.0):
        e = rd.kilpatrick_field(f)
        assert 1.643 * e * e * math.exp(-8.5 / e) == pytest.approx(f, rel=1e-9)
    assert rd.kilpatrick_field(352.21) == pytest.approx(18.44, abs=0.01)


def test_figures_of_merit_against_hand_arithmetic():
    V, r0, A, phi, beta, a = 60_000.0, 5.0, 0.3, -30.0, 0.04, 3.5
    mc2, q, lam_mm = 938.272e6, 1.0, 1844.9
    out = rd.figures_of_merit(V, r0, A, phi, beta, a, mc2_eV=mc2, q=q,
                              lam_mm=lam_mm)
    lam, g = lam_mm * 1e-3, 1.0 / math.sqrt(1.0 - beta ** 2)
    s = math.sin(math.radians(phi))
    B = V * lam ** 2 / (mc2 * (r0 * 1e-3) ** 2)
    d = math.pi ** 2 * A * V * s / (2.0 * mc2 * beta ** 2 * g)
    assert out["B"] == pytest.approx(B, rel=1e-12)
    assert out["delta_rf"] == pytest.approx(d, rel=1e-12)
    assert out["sigma0t_deg"] == pytest.approx(
        math.degrees(math.sqrt(B * B / (8 * math.pi ** 2) + d)), rel=1e-12)
    assert out["sigma0l_deg"] == pytest.approx(math.degrees(math.sqrt(
        math.pi ** 2 * A * V * abs(s) / (mc2 * beta ** 2 * g ** 3))), rel=1e-12)
    # σ0l² = −2Δ_rf apart from γ² (Earnshaw)
    assert math.radians(out["sigma0l_deg"]) ** 2 == pytest.approx(
        -2.0 * d / g ** 2, rel=1e-12)
    assert out["E0T_MV_m"] == pytest.approx(
        math.pi * A * V / (2 * beta * lam) * 1e-6, rel=1e-12)
    # current limits, hand-built from the ellipsoid formulas
    b = beta * lam * 1.5 * abs(math.radians(phi)) / (2 * math.pi)
    f = rd._spheroid_form_factor(g * b / (a * 1e-3))
    common = mc2 / q * g ** 3 * (a * 1e-3) ** 2 * b / (3 * rd.Z0_OHM * lam ** 3)
    st, sl = math.radians(out["sigma0t_deg"]), math.radians(out["sigma0l_deg"])
    assert out["I_t_mA"] == pytest.approx(8 * math.pi * common * st ** 2
                                          / (1 - f) * 1e3, rel=1e-12)
    assert out["I_l_mA"] == pytest.approx(4 * math.pi * common * sl ** 2 / f
                                          * 1e3, rel=1e-12)


def test_no_acceleration_no_longitudinal_figures():
    out = rd.figures_of_merit(60_000.0, 5.0, 0.0, -90.0, 0.01, 5.0,
                              mc2_eV=938e6, q=1.0, lam_mm=1844.9)
    assert np.isfinite(out["B"]) and out["delta_rf"] == 0.0
    assert np.isnan(out["sigma0l_deg"]) and np.isnan(out["I_l_mA"])


def test_spheroid_form_factor():
    assert rd._spheroid_form_factor(1.0) == pytest.approx(1 / 3)
    assert rd._spheroid_form_factor(0.999) == pytest.approx(1 / 3, abs=1e-3)
    assert rd._spheroid_form_factor(1.001) == pytest.approx(1 / 3, abs=1e-3)
    assert rd._spheroid_form_factor(20.0) < 0.02         # long: little E_z
    assert rd._spheroid_form_factor(0.05) > 0.9           # flat: mostly E_z


def test_card_tips_and_minimum_aperture(demo):
    from linac_gen.elements.rfq_coefficients import vane_apertures
    cell = next(c for c in rd.find_rfq(demo).cells
                if abs(c.cell_type) == 2 and c.A10 > 0.3)
    L = float(cell.length)
    z = np.linspace(0, L, 5)
    rx, ry = rd.cell_tips(cell, z)
    for k, zz in enumerate(z):
        x, y = vane_apertures(cell.r0_mm, cell.A10, L, cell.cell_type, zz)
        assert (rx[k], ry[k]) == pytest.approx((x, y), rel=1e-12)
    ends = [min(vane_apertures(cell.r0_mm, cell.A10, L, cell.cell_type, zz))
            for zz in (0.0, L)]
    assert rd.min_aperture(cell) == pytest.approx(min(ends), rel=1e-12)
    # the modulation the cards state: widest / narrowest ≈ m
    assert max(rx.max(), ry.max()) / rd.min_aperture(cell) == pytest.approx(
        cell.modulation, rel=0.02)


def test_design_beta_and_phase_from_the_cell_lengths(demo):
    from linac_gen.elements.rfq_coefficients import synchronous_phase_from_lengths
    t = rd.cell_table(demo, species="proton", frequency_MHz=352.21)
    lam = rd.wavelength_mm(352.21)
    acc = np.abs(t["type"]) == 2
    np.testing.assert_allclose(t["beta"][acc], 2 * t["L"][acc] / lam)
    assert np.all(np.isnan(t["beta"][~acc]))
    phi, _ok = synchronous_phase_from_lengths(t["L"], t["A10"], t["V"],
                                              938.27208816, 1.0, lam)
    np.testing.assert_array_equal(t["phi_s_from_L"], phi)
    assert t["run_note"] == "no run" and "trans" not in t
    assert not t["beta_from_run"]


def _fake_results(lattice, *, substeps=1, rename=False):
    """Results with ``substeps`` rows per element (the last at its exit)."""
    names, exit_idx = ["INPUT"], []
    for el in lattice.elements:
        for _ in range(substeps):
            names.append("X" if rename else el.name)
        exit_idx.append(len(names) - 1)
    n = len(names)
    return SimpleNamespace(
        element_names=names, element_exit_idx=exit_idx,
        transmission=list(np.linspace(100, 80, n)),
        ref_beta=list(np.linspace(0.01, 0.06, n)),
        ref_w_kin=list(np.linspace(0.05, 1.9, n)),
        sigma_x=list(np.full(n, 1.0)), sigma_y=list(np.full(n, 1.1)),
        x_max=list(np.full(n, 3.0)), y_max=list(np.full(n, 3.1)),
        sigma_phi=list(np.full(n, 10.0)), sigma_w=list(np.full(n, 0.02)),
        centroid=[np.arange(6.0) for _ in range(n)])


def test_run_values_come_from_each_cell_exit_row(demo):
    for sub in (1, 3):
        res = _fake_results(demo, substeps=sub)
        t = rd.cell_table(demo, frequency_MHz=352.21, results=res)
        rows = np.asarray(res.element_exit_idx)[rd.find_rfq(demo).index]
        np.testing.assert_array_equal(t["trans"],
                                      np.asarray(res.transmission)[rows])
        np.testing.assert_array_equal(t["beta"], np.asarray(res.ref_beta)[rows])
        assert t["beta_from_run"] and "run" in t["beta_source"]
        np.testing.assert_array_equal(t["phi_offset"], np.full(len(rows), 4.0))
        np.testing.assert_array_equal(t["w_offset"], np.full(len(rows), 5.0))
    t = rd.cell_table(demo, frequency_MHz=352.21,
                      results=_fake_results(demo, rename=True))
    assert "do not belong" in t["run_note"] and "trans" not in t


def test_losses_per_cell(demo):
    from linac_gen.analysis.loss_power import watts_per_macroparticle
    from linac_gen.core.beam import LOSS_DTYPE
    ch = rd.find_rfq(demo)
    lt = np.zeros(4, dtype=LOSS_DTYPE)
    lt["element_name"] = [ch.names[0], ch.names[5], ch.names[5], "ELSEWHERE"]
    lt["energy"] = [0.1, 0.5, 0.7, 2.0]
    lt["s"] = [1.0, 50.0, 51.0, 9e3]
    res = SimpleNamespace(loss_table=lt, n_macro=100)
    lo = rd.cell_losses(res, ch, current_mA=10.0, duty_pct=50.0)
    assert lo["count"][0] == 1 and lo["count"][5] == 2 and lo["count"].sum() == 3
    w = watts_per_macroparticle(np.array([0.5, 0.7]), 10.0, 50.0, 100)
    assert lo["watts"][5] == pytest.approx(w.sum())
    assert lo["s"].size == 3                     # the loss outside left out
    assert rd.cell_losses(SimpleNamespace(), ch) is None


def test_cross_section_is_the_electrodes_inside_test(demo):
    ch = rd.find_rfq(demo)
    el = rd.cards_electrodes(demo, ch)
    z = 0.5 * (ch.edges[150] + ch.edges[151])
    xs, ys, metal, tips = rd.cross_section(el, z, 8.0, n=81)
    X, Y = np.meshgrid(xs, ys)
    np.testing.assert_array_equal(metal, el.inside(X, Y, np.full_like(X, z)))
    t = rd.tips_along(el, [z])
    assert tips["R_x"] == pytest.approx(float(t["R_x"][0]))
    assert not metal[40, 40]                       # the axis is in the bore
    assert metal[40, -1] and metal[0, 40]          # the vanes at 8 mm


def test_vane_file_tips_follow_its_pole_numbering(tmp_path):
    """Pole 2 of the file is +x, pole 1 is +y (the synthetic vane of the
    time-domain tests: x opens where y closes)."""
    pytest.importorskip("pyamg")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tc", Path(__file__).resolve().parents[1] / "rfq_td" / "test_chain.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    deck = tc._write_rfq(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    ch = rd.find_rfq(lat)
    el, _scale, src, note = rd.solver_electrodes(lat, ch)
    assert src == "vane file" and note == ""
    z = np.linspace(tc.GAP_IN + 1, ch.edges[-1] - tc.GAP_OUT - 1, 30)
    amp = (tc.M - 1) / (tc.M + 1)
    c = np.cos(np.pi * (z - tc.GAP_IN) / tc.L_CELL)
    tips = rd.tips_along(el, z)
    np.testing.assert_allclose(tips["R_x"], tc.R0 * (1 + amp * c), rtol=1e-3)
    np.testing.assert_allclose(tips["R_y"], tc.R0 * (1 - amp * c), rtol=1e-3)


def test_surface_field_extrapolation_recovers_a_curved_conductor():
    """Field of a line charge on each tip's axis: |E| = A/r around it, so
    the fit E_s ρ_e/(ρ_e + δ) recovers A/ρ on the surface exactly."""
    from linac_gen.rfq_td.geometry import RfqElectrodes
    r0, rho, A = 5.0, 3.75, 1000.0
    z = np.linspace(0.0, 100.0, 11)
    el = RfqElectrodes(z=z, R_x=np.full_like(z, r0), R_y=np.full_like(z, r0),
                       rho_x=np.full_like(z, rho), rho_y=np.full_like(z, rho),
                       V_x=-1.0, V_y=1.0)
    cx = cy = r0 + rho

    def field(x, y, zz):
        dx, dy = np.asarray(x) - cx, np.asarray(y)
        d2x = dx * dx + dy * dy
        ex1, ey1 = A * dx / d2x, A * dy / d2x       # +x tip line charge
        ex2, ey2 = np.asarray(x), np.asarray(y) - cy
        d2y = ex2 * ex2 + ey2 * ey2
        near_x = d2x < d2y
        ex = np.where(near_x, ex1, A * ex2 / d2y)
        ey = np.where(near_x, ey1, A * ey2 / d2y)
        return ex, ey, np.zeros_like(ex)

    ex_tip, ey_tip = rd.peak_surface_field(field, el, [50.0],
                                           offsets_mm=(0.3, 0.5),
                                           x_store_mm=30.0)
    assert ex_tip[0] == pytest.approx(A / rho, rel=1e-9)
    assert ey_tip[0] == pytest.approx(A / rho, rel=1e-9)


def test_surface_field_of_a_plain_quadrupole_is_converged():
    """The estimate on a solved field: an unmodulated four-vane quadrupole
    (ρ/r0 = 0.75) at the field-map step 0.1 mm gives κ = E_s/(V/r0) = 1.24,
    the value a twice finer grid also gives (within 1 %).  For this cross-
    section (r0 = 5.576 mm, ρ = 0.75 r0, 60 kV) a published CST/SUPERFISH
    design study gives a peak field of 13.4 MV/m (G. Romanov et al.,
    "Project X RFQ EM design", IPAC2012, THPPP064, Table 2)."""
    pytest.importorskip("pyamg")
    from linac_gen.rfq_td.fieldmap import RfqFieldMap
    from linac_gen.rfq_td.geometry import RfqElectrodes
    from linac_gen.rfq_td.laplace import Window, solve_block
    r0, rho, V, h = 5.576, 4.183, 60_000.0, 0.1
    z = np.linspace(-200.0, 200.0, 401)
    el = RfqElectrodes(z=z, R_x=np.full_like(z, r0), R_y=np.full_like(z, r0),
                       rho_x=np.full_like(z, rho), rho_y=np.full_like(z, rho),
                       V_x=-V / 2, V_y=V / 2)
    win = Window(z0=-2.0, z1=2.0, hz=0.5, x_max=16.0, x_store=10.0)
    zf, hz, phi = solve_block(el, win, h=h, pad_mm=4.0)
    fmap = RfqFieldMap.from_blocks(h, [(win.z0, win.z1)], [(zf, hz, phi)])
    ex, ey = rd.peak_surface_field(rd._field_fn(fmap), el, [0.0],
                                   offsets_mm=(3 * h, 5 * h),
                                   x_store_mm=10.0 - 2 * h)
    kappa = max(ex[0], ey[0]) / (V / r0)
    assert kappa == pytest.approx(1.24, abs=0.01)
    assert max(ex[0], ey[0]) * 1e-3 == pytest.approx(13.4, rel=0.01)  # MV/m
    assert ex[0] == pytest.approx(ey[0], rel=1e-3)   # the two planes alike


def test_steep_or_stepped_tips_are_left_out():
    from linac_gen.rfq_td.geometry import RfqElectrodes
    z = np.linspace(0.0, 60.0, 241)
    R = np.where(z < 30.0, 5.0, 4.0)                  # a 1 mm step at 30 mm
    el = RfqElectrodes(z=z, R_x=R, R_y=R, rho_x=np.full_like(z, 3.75),
                       rho_y=np.full_like(z, 3.75), V_x=-1.0, V_y=1.0)
    sl = rd.tip_slope(el, [10.0, 29.0, 31.0, 50.0])
    assert sl[0] == 0.0 and sl[3] == 0.0
    assert sl[1] > rd.MAX_TIP_SLOPE and sl[2] > rd.MAX_TIP_SLOPE

    def field(x, y, zz):
        one = np.ones_like(np.asarray(x, float))
        return one, 0 * one, 0 * one
    ex, ey = rd.peak_surface_field(field, el, [10.0, 30.0],
                                   offsets_mm=(0.3, 0.5), x_store_mm=30.0)
    assert np.isfinite(ex[0]) and np.isnan(ex[1]) and np.isnan(ey[1])


def test_cached_vane_profile_belongs_to_its_file(tmp_path):
    from linac_gen.elements.rfq_geometry_profile import (_CACHE_VERSION,
                                                          _cache_path)
    vane = tmp_path / "x.vane"
    vane.write_text("0 0\n")
    assert rd.cached_vane_profile(str(vane)) is None          # no cache
    st = vane.stat()
    good = f"v{_CACHE_VERSION};{st.st_size};{int(st.st_mtime)};nx=61"
    np.savez(_cache_path(vane), z_mm=np.arange(3.0), gx=np.ones(3),
             gy=-np.ones(3), fingerprint=good)
    z, gx, gy = rd.cached_vane_profile(str(vane))
    np.testing.assert_array_equal(gy, -np.ones(3))
    np.savez(_cache_path(vane), z_mm=np.arange(3.0), gx=np.ones(3),
             gy=-np.ones(3), fingerprint=good.replace(f";{st.st_size};", ";9;"))
    assert rd.cached_vane_profile(str(vane)) is None          # other file


def test_aperture_profile_draws_the_rfq_vane_tips(demo):
    from linac_gen.analysis.aperture_profile import aperture_profile
    s, rx, ry = aperture_profile(demo)
    ch = rd.find_rfq(demo)
    k = 120
    m = np.flatnonzero((s >= ch.s0[k] - 1e-9) & (s <= ch.s1[k] + 1e-9))
    m = m[1:-1]              # the neighbours' samples on the two cell edges
    assert m.size == 9
    zl = s[m] - ch.s0[k]
    tx, ty = rd.cell_tips(ch.cells[k], zl)
    np.testing.assert_allclose(rx[m], tx)
    np.testing.assert_allclose(ry[m], ty)
    assert np.all(np.diff(s) >= 0)


def test_solver_electrodes_leave_the_run_notes_alone(demo, caplog):
    import logging
    ch = rd.find_rfq(demo)
    with caplog.at_level(logging.INFO, logger="linac_gen.rfq_td"):
        el, scale, src, note = rd.solver_electrodes(demo, ch)
    assert src == "RFQ_CELL cards" and note == ""
    assert scale == pytest.approx(85_000.0)             # map at 1 V
    assert not getattr(demo, "_rfq_td_notes", None)
    assert not [r for r in caplog.records if "RFQ_GEOM" in r.getMessage()]


def test_solver_electrodes_refuse_what_the_solver_refuses(demo):
    import copy
    lat = copy.deepcopy(demo)
    ch = rd.find_rfq(lat)
    ch.cells[10].voltage_V = 90_000.0
    el, _scale, _src, note = rd.solver_electrodes(lat, ch)
    assert el is None and "different vane voltages" in note


def test_cards_field_follows_each_cells_voltage(demo):
    import copy
    lat = copy.deepcopy(demo)
    ch = rd.find_rfq(lat)
    k = 150
    z = np.linspace(ch.edges[k] + 0.2 * ch.cells[k].length,
                    ch.edges[k + 1] - 0.2 * ch.cells[k].length, 7)
    ez1, g1 = rd.cards_axis_fields(ch, z)
    ch.cells[k].voltage_V *= 2.0
    ez2, g2 = rd.cards_axis_fields(ch, z)
    np.testing.assert_allclose(ez2, 2.0 * ez1, rtol=1e-12)
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-12)
    # mid-cell, where the A10 term's gradient (∝ cos kz) vanishes, the
    # gradient is the quadrupole term's V/r0² (V/mm²)
    assert g1[3] == pytest.approx(85_000.0 / 3.40 ** 2, rel=1e-4)
    np.testing.assert_allclose(g1[:3] + g1[:3:-1], 2 * g1[3], rtol=1e-3)


def test_the_rf_frequency_is_the_one_at_the_rfq(demo):
    ch = rd.find_rfq(demo)
    # no run: the FREQ card before the RFQ, not the beam's frequency
    assert rd.rfq_frequency(demo, ch, None, default_MHz=100.0) == 352.21
    # a run: its reference frequency at the RFQ, not at the end of the run
    res = _fake_results(demo)
    n = len(res.element_names)
    res.ref_frequency = [352.21] * (n - 1) + [704.42]
    assert rd.rfq_frequency(demo, ch, res, default_MHz=100.0) == 352.21
    # neither: the beam's
    from linac_gen.core.lattice import Lattice
    bare = Lattice()
    for c in ch.cells:
        bare.add(c)
    assert rd.rfq_frequency(bare, rd.find_rfq(bare), None, 162.5) == 162.5
    assert rd.rfq_frequency(bare, rd.find_rfq(bare), None, None) is None
