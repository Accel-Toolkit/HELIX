"""NCELLS ("Cavity multi-gap") element: geometry, gap physics, phase modes.

Physics anchors:
  * cell lengths / gap positions per mode (π, 2π);
  * |q| (abs-charge) energy gain — TraceWin's convention (H⁻ and protons alike),
    validated end-to-end against a TraceWin fnalscl run (see
    tests/io/test_ncells_import.py::test_fnalscl_energy_matches_tracewin);
  * π-mode coherence (polarity × running-clock ⇒ every cell accelerates);
  * MP (track_rk4) and envelope (advance_ref / fitted_matrix) agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.elements.ncells import NCells, parse_ttf_tail, _ttf_value, TTFSet
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam

F = 804.96          # MHz — a real fnalscl frequency
BG = 0.456630316    # a real fnalscl βg


def _w_matched(beta, species=H_MINUS):
    g = 1.0 / np.sqrt(1.0 - beta * beta)
    return (g - 1.0) * species.mass


def _ref(w=None, beta=BG, species=H_MINUS, freq=F):
    if w is None:
        w = _w_matched(beta, species)
    return ReferenceParticle(species=species, w_kin=w, frequency=freq)


# ------------------------------------------------------------------ geometry
def test_pi_mode_cell_geometry():
    nc = NCells("c", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=6.8e6,
                theta_s_deg=62.0, aperture_mm=15.0, p_flag=1, frequency_mhz=F)
    lam = nc._wavelength_mm()
    lc = 0.5 * BG * lam                      # π mode: Lc = βλ/2
    assert nc._interior_cell_length(BG) == pytest.approx(lc)
    assert nc.length == pytest.approx(16 * lc)
    assert len(nc._gaps) == 16
    # interior gaps sit at cell centre; spacing = Lc
    zc = [g.z_center_mm for g in nc._gaps]
    assert zc[0] == pytest.approx(0.5 * lc)          # Le = Lc/2 (dzi=0)
    assert np.allclose(np.diff(zc), lc)


def test_2pi_mode_cell_length_doubles():
    nc = NCells("c", mode=0, n_cells=8, beta_g=BG, eot_v_per_m=6.8e6,
                theta_s_deg=0.0, frequency_mhz=F)
    lam = nc._wavelength_mm()
    assert nc._interior_cell_length(BG) == pytest.approx(BG * lam)   # 2π: Lc = βλ
    assert nc.length == pytest.approx(8 * BG * lam)


def test_gap_count_voltage_and_polarity():
    eot = 6.8e6
    nc = NCells("c", mode=1, n_cells=6, beta_g=BG, eot_v_per_m=eot,
                theta_s_deg=0.0, frequency_mhz=F)
    lc = nc._interior_cell_length(BG)
    v_expect = eot * (lc * 1e-3) * 1e-6              # |EoTL| in MV
    assert len(nc._gaps) == 6
    assert abs(nc._gaps[0].voltage_mv) == pytest.approx(v_expect)
    # π mode: polarity alternates +,-,+,-,...
    signs = [np.sign(g.voltage_mv) for g in nc._gaps]
    assert signs == [1, -1, 1, -1, 1, -1]
    # gap classes: first=input, last=output, rest=middle
    assert nc._gaps[0].kind == "input"
    assert nc._gaps[-1].kind == "output"
    assert nc._gaps[2].kind == "middle"


def test_end_cell_displacement_and_field_correction():
    nc = NCells("c", mode=1, n_cells=5, beta_g=BG, eot_v_per_m=1.0e6,
                theta_s_deg=0.0, k_eot_i=0.10, k_eot_o=-0.20,
                dz_i_mm=3.0, dz_o_mm=-2.0, frequency_mhz=F)
    lc = nc._interior_cell_length(BG)
    # first gap displaced by +dzi from the cell centre
    assert nc._gaps[0].z_center_mm == pytest.approx(0.5 * lc + 3.0)
    # input gap field scaled by (1+kEoTi); output by (1+kEoTo)
    base = 1.0e6 * (lc * 1e-3) * 1e-6
    assert abs(nc._gaps[0].voltage_mv) == pytest.approx(base * 1.10)
    assert abs(nc._gaps[-1].voltage_mv) == pytest.approx(base * 0.80)


# --------------------------------------------------------------- gap physics
def test_abscharge_convention_is_species_independent():
    """Energy gain uses the |q| (abs-charge) convention TraceWin/RfqCell use —
    NOT signed charge — so H⁻ and a proton gain the SAME energy at the same
    phase, and θ≈crest (0°) accelerates both.  (Validated against a TraceWin
    fnalscl run; the old signed convention would flip the H⁻ sign.)"""
    kw = dict(mode=1, n_cells=1, beta_g=BG, eot_v_per_m=5.0e6,
              theta_s_deg=0.0, p_flag=0, frequency_mhz=F)
    ref_h = _ref(species=H_MINUS)
    ref_p = _ref(species=PROTON, beta=BG)
    nc_h = NCells("h", **kw); nc_p = NCells("p", **kw)
    w_h0, w_p0 = ref_h.w_kin, ref_p.w_kin
    nc_h.advance_ref(ref_h)
    nc_p.advance_ref(ref_p)
    dW_h = ref_h.w_kin - w_h0
    dW_p = ref_p.w_kin - w_p0
    assert dW_h > 0 and dW_p > 0                     # both accelerate at crest
    assert dW_h == pytest.approx(dW_p, rel=1e-9)     # |q|: species-independent


def test_pi_mode_gaps_accelerate_coherently():
    """At matched β, the polarity flip + ~180°/cell running clock make every
    cell gain the SAME-sign energy (no gap-to-gap cancellation)."""
    nc = NCells("c", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=1.685e6,
                theta_s_deg=179.0, p_flag=1, frequency_mhz=F)
    ref = _ref()
    ref.phi_s = 0.0
    gaps = nc._ensure_gaps(ref)
    nc._phi_s_at_entrance = ref.phi_s
    z = 0.0
    dWs = []
    for g in gaps:
        nc._ref_drift(ref, g.z_center_mm - z)
        phi = nc._phi_gap_rad(ref, g)
        dW = ref.species.charge * g.voltage_mv * nc._ttf_correction(ref.beta, g.kind) * np.cos(phi)
        dWs.append(dW)
        ref.w_kin += dW
        z = g.z_center_mm
    dWs = np.array(dWs)
    assert np.all(dWs > 0) or np.all(dWs < 0), dWs      # all same sign


def test_pflag_absolute_differs_from_relative():
    """With a non-zero entrance clock, P=1 (absolute, global) and P=0
    (relative, cell-local) give different gap phases → different energy gain."""
    common = dict(mode=1, n_cells=4, beta_g=BG, eot_v_per_m=6.8e6,
                  theta_s_deg=62.0, frequency_mhz=F)
    r1 = _ref(); r1.phi_s = 137.0                       # arbitrary accumulated clock
    r0 = _ref(); r0.phi_s = 137.0
    nc1 = NCells("abs", p_flag=1, **common)
    nc0 = NCells("rel", p_flag=0, **common)
    w1, w0 = r1.w_kin, r0.w_kin
    nc1.advance_ref(r1)
    nc0.advance_ref(r0)
    assert (r1.w_kin - w1) != pytest.approx(r0.w_kin - w0, abs=1e-6)


# ---------------------------------------------------------------------- TTF
def test_ttf_tail_parsing_and_betas_zero_is_none():
    assert parse_ttf_tail([]) is None
    assert parse_ttf_tail(["0.0"]) is None              # βs=0 ⇒ no correction
    t = parse_ttf_tail(["0.5", "0.8", "0.1", "0.02"])
    assert t is not None and t.beta_s == 0.5
    assert t.middle.Ts == 0.8 and t.middle.kTp == 0.1


def test_ttf_value_reduces_to_Ts_at_reference():
    s = TTFSet(Ts=0.85, kTp=0.2, k2Tpp=-0.05)
    assert _ttf_value(s, beta=0.6, beta_s=0.6) == pytest.approx(0.85)   # u=0
    # off-reference: monotone in u
    assert _ttf_value(s, beta=0.5, beta_s=0.6) != pytest.approx(0.85)


def test_ttf_correction_unity_without_tail():
    nc = NCells("c", mode=1, n_cells=4, beta_g=BG, eot_v_per_m=6.8e6,
                theta_s_deg=0.0, frequency_mhz=F, ttf=None)
    assert nc._ttf_correction(0.5, "middle") == 1.0


# ------------------------------------------------------ MP / envelope parity
def test_mp_and_advance_ref_agree_on_energy():
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    w0 = _w_matched(BG)
    beam = Beam(ref=_ref(w0), n_particles=200, current=0.0)
    beam.particles[:] = np.random.default_rng(1).normal(0, 0.15, (200, 6))
    lat = Lattice()
    lat.add(NCells("m", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=6.82062e6,
                   theta_s_deg=62.0, aperture_mm=15.0, p_flag=1, frequency_mhz=F))
    Simulation(lat, beam).run()
    w_mp = beam.ref.w_kin

    nc = lat.elements[0]; nc.reset_run_state()
    r = _ref(w0)
    nc.advance_ref(r)
    assert r.w_kin == pytest.approx(w_mp, rel=1e-9)


def test_fitted_matrix_finite_with_damping():
    nc = NCells("m", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=6.82062e6,
                theta_s_deg=62.0, p_flag=1, frequency_mhz=F)
    M = nc.fitted_matrix(_ref())
    assert M.shape == (6, 6) and np.all(np.isfinite(M))
    # accelerating cavity ⇒ transverse adiabatic damping (det of x-block < 1)
    assert 0.0 < np.linalg.det(M[:2, :2]) < 1.0


# ------------------------------------------------------------ βg≤0 + reset
def test_betaG_zero_resolves_geometry_from_running_beta():
    """βg=0: cell length set by the running velocity, not known at construction."""
    nc = NCells("c", mode=1, n_cells=8, beta_g=0.0, eot_v_per_m=1.0e6,
                theta_s_deg=0.0, frequency_mhz=F)
    assert nc._gaps is None                             # lazy until a ref arrives
    r = _ref(beta=0.5)
    nc.advance_ref(r)
    assert nc._gaps is not None and len(nc._gaps) == 8
    lam = nc._wavelength_mm()
    # π-mode geometry keyed off the running β: gap 1 sits Le = βλ/4 from the
    # entrance (β = 0.5 exactly there), and the resolved total is ~8 cells of
    # βλ/2 (slightly longer — the 1 MV/m gaps accelerate the resolving ref).
    assert nc._gaps[0].z_center_mm == pytest.approx(0.25 * 0.5 * lam,
                                                    rel=1e-9)
    assert nc.length == pytest.approx(8 * 0.5 * 0.5 * lam, rel=0.05)
    assert nc.length >= 8 * 0.5 * 0.5 * lam               # acceleration only grows cells


def test_betaG_zero_mp_matches_advance_ref():
    """βg=0 MP tracking must apply every cell-gap (deterministic geometry +
    provisional-length hardening) → same reference energy as advance_ref."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    w0 = _w_matched(0.45)

    def _lat():
        lat = Lattice()
        lat.add(NCells("m0", mode=1, n_cells=16, beta_g=0.0, eot_v_per_m=6.82062e6,
                       theta_s_deg=0.0, aperture_mm=15.0, p_flag=0, frequency_mhz=F))
        return lat

    lat = _lat(); r = ReferenceParticle(species=H_MINUS, w_kin=w0, frequency=F)
    lat.elements[0].advance_ref(r)
    w_ref = r.w_kin

    lat2 = _lat()
    beam = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=w0, frequency=F),
                n_particles=80, current=0.0)
    beam.particles[:] = np.random.default_rng(2).normal(0, 0.1, (80, 6))
    Simulation(lat2, beam).run()
    assert beam.ref.w_kin == pytest.approx(w_ref, rel=1e-6)
    assert beam.ref.w_kin > w0                       # net acceleration


def test_reset_run_state_clears_cursor_and_lazy_geometry():
    nc = NCells("c", mode=1, n_cells=4, beta_g=0.0, eot_v_per_m=1.0e6,
                theta_s_deg=0.0, frequency_mhz=F)
    r = _ref(beta=0.5)
    nc.advance_ref(r)
    nc._step_idx = 3
    nc.reset_run_state()
    assert nc._step_idx == 0
    assert nc._gaps is None                             # βg≤0 → re-resolve per run


# ---------------------------------------- slice/cursor correctness (regression)
def test_mp_energy_correct_across_step_configs():
    """Regression for the slice-boundary double-fire + mixed-ds desync bugs:
    the MP exit energy must equal advance_ref for EVERY step config — including
    ones whose n_int is commensurate with n_cells (gaps land exactly on slice
    boundaries) and ones with a trailing remainder (mixed ds)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.core.step_config import StepConfig
    w0 = _w_matched(BG)

    def _lat():
        lat = Lattice()
        lat.add(NCells("m", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=6.82062e6,
                       theta_s_deg=62.0, aperture_mm=15.0, p_flag=1, frequency_mhz=F))
        return lat

    r = _ref(w0)
    lat = _lat(); lat.elements[0].advance_ref(r); w_ref = r.w_kin
    for cfg in [(94, 94), (50, 3), (82, 4), (100, 50)]:
        lat = _lat(); lat.step_config = StepConfig(*cfg)
        beam = Beam(ref=_ref(w0), n_particles=40, current=0.0)
        beam.particles[:] = np.random.default_rng(0).normal(0, 0.1, (40, 6))
        Simulation(lat, beam).run()
        assert beam.ref.w_kin == pytest.approx(w_ref, abs=1e-6), (cfg, beam.ref.w_kin)


def test_fitted_matrix_composition_matches_full():
    """Composing fitted_matrix_slice with the envelope's explicit z-cursor over
    any partition (incl. bundle+remainder) equals the full fitted_matrix — the
    matrix half of the boundary/mixed-ds fix."""
    nc = NCells("m", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=6.82062e6,
                theta_s_deg=62.0, p_flag=1, frequency_mhz=F)
    M_full = nc.fitted_matrix(_ref())
    L = nc.length
    for part in ([L / 128] * 128, [L / 69 * 13] * 5 + [L / 69 * 4]):
        part = [p * L / sum(part) for p in part]
        nc.reset_run_state(); ref = _ref(); M = np.eye(6); z = 0.0
        for seg in part:
            Mi = nc.fitted_matrix_slice(ref, seg, _z_from_mm=z)
            nc.advance_ref_over(ref, z, z + seg)
            M = Mi @ M; z += seg
        assert np.max(np.abs(M - M_full)) < 1e-9


def test_envelope_reference_energy_matches_mp():
    """The real EnvelopeSolver advances the NCELLS reference energy identically
    to multi-particle tracking, with and without space charge."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.elements.drift import Drift
    w0 = _w_matched(BG)

    def _lat():
        lat = Lattice()
        lat.add(NCells("m", mode=1, n_cells=16, beta_g=BG, eot_v_per_m=6.82062e6,
                       theta_s_deg=62.0, aperture_mm=15.0, p_flag=1, frequency_mhz=F))
        lat.add(Drift("d", 150.0))
        return lat

    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=1e-6, alpha_y=0.0, beta_y=1.0,
                   emit_y=1e-6, alpha_z=0.0, beta_z=1.0, emit_z=1e-6)
    for cur in (0.0, 5.0):
        beam = Beam(ref=_ref(w0), n_particles=1500, current=cur)
        beam.particles[:] = (np.random.default_rng(0).normal(0, 1, (1500, 6))
                             * np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.02]))
        Simulation(_lat(), beam).run()
        res = EnvelopeSolver(_lat(), _ref(w0), initial, current=cur).run()
        assert res.ref_w_kin[-1] == pytest.approx(beam.ref.w_kin, rel=1e-4)


# ------------------------------------------------ adversarial-review regressions
def _mk(sync=False, theta=0.0, n=1, bg=0.5):
    return NCells("R", mode=1, n_cells=n, beta_g=bg, eot_v_per_m=5e6,
                  theta_s_deg=theta, aperture_mm=15, p_flag=0,
                  frequency_mhz=650.0, sync_phase=sync)


def _dw(nc, w=150.0):
    r = ReferenceParticle(species=H_MINUS, w_kin=w, frequency=650.0)
    nc.reset_run_state()
    nc.advance_ref(r)
    return r.w_kin - w


def test_sync_phase_single_cell_equals_relative():
    """SET_SYNC_PHASE with one cell must equal P=0 exactly: the calibrated ψ
    has no β-slippage to absorb.  Regression: the ψ probe was
    entrance-referenced while the application is gap-1-referenced, so ψ
    double-counted the Le drift (90° in π-mode) and a crest request landed
    on the zero crossing (ΔW ≈ 0 instead of the full crest gain)."""
    for th in (0.0, -30.0, -90.0, 45.0):
        assert _dw(_mk(sync=True, theta=th)) == pytest.approx(
            _dw(_mk(sync=False, theta=th)), abs=1e-12)


def test_sync_phase_crest_request_gives_crest():
    d0 = _dw(_mk(sync=True, theta=0.0, n=16))
    assert d0 > _dw(_mk(sync=True, theta=-30.0, n=16))
    assert d0 > _dw(_mk(sync=True, theta=+30.0, n=16))
    assert d0 > 0.9 * 16 * _dw(_mk(sync=True, theta=0.0, n=1))


def test_error_cav_hooks_apply():
    """ERROR_CAV slots: voltage_rel scales every gap kick; phase_offset
    shifts the applied RF phase.  Regression: both were silent no-ops on
    NCELLS (tolerance studies ran with zero RF jitter on NCELLS decks)."""
    # Single cell: dW = |q|·V·T·cosφ is exactly linear in V (multi-cell is
    # not — a harder kick changes the intra-cavity phase slip).
    base = _dw(_mk(theta=0.0, n=1))
    amp = _mk(theta=0.0, n=1)
    amp.voltage_rel = 0.10
    assert _dw(amp) == pytest.approx(1.10 * base, rel=1e-9)
    ph = _mk(theta=0.0, n=1)
    ph.phase_offset = 90.0
    assert abs(_dw(ph)) < 1e-6 * base          # crest -> zero crossing
    # Multi-cell sanity: the amplitude error still increases the gain.
    b4 = _dw(_mk(theta=0.0, n=4))
    a4 = _mk(theta=0.0, n=4)
    a4.voltage_rel = 0.10
    assert _dw(a4) > b4


def test_betaG_zero_cross_energy_rerun_matches_fresh():
    """βg=0 lattice-object reuse: after reset_run_state a re-run at a very
    different energy must equal a fresh element.  Regression: run 1's
    resolved length survived the reset, the tracker sized its slices from
    it, and the tail gaps of the (longer) re-resolved cavity never fired
    (−6 MeV on a 45→235 MeV reuse)."""
    def mp_exit(nc, w):
        ref = ReferenceParticle(species=H_MINUS, w_kin=w, frequency=650.0)
        beam = Beam(ref=ref, n_particles=4, current=0.0)
        beam.particles[:, :] = 0.0
        from linac_gen.core.lattice import Lattice
        from linac_gen.tracking.tracker import Tracker
        lat = Lattice()
        lat.add(nc)
        Tracker(lat, beam).run()
        return beam.ref.w_kin

    reused = NCells("z", mode=1, n_cells=16, beta_g=0.0, eot_v_per_m=5e6,
                    theta_s_deg=0.0, aperture_mm=15, p_flag=0,
                    frequency_mhz=650.0)
    lo = mp_exit(reused, 45.35)                  # run 1 resolves ~β=0.30 cells
    hi_reused = mp_exit(reused, 234.82)          # rerun at ~β=0.60
    fresh = NCells("z", mode=1, n_cells=16, beta_g=0.0, eot_v_per_m=5e6,
                   theta_s_deg=0.0, aperture_mm=15, p_flag=0,
                   frequency_mhz=650.0)
    hi_fresh = mp_exit(fresh, 234.82)
    assert hi_reused == pytest.approx(hi_fresh, abs=1e-9)
    assert lo > 45.35                            # run 1 itself accelerated


def test_envelope_reuse_resets_fieldmap_state():
    """Envelope I=0 pure-linear path must reset stateful field-map elements:
    a second run on the SAME lattice object must reproduce the first.
    Regression: no reset on that path — the second full fitted_matrix call
    started at z=length and no gap ever fired (net energy loss)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat = Lattice()
    lat.add(NCells("e", mode=1, n_cells=16, beta_g=0.0, eot_v_per_m=5e6,
                   theta_s_deg=0.0, aperture_mm=15, p_flag=0,
                   frequency_mhz=650.0))
    ini = dict(alpha_x=0.0, beta_x=5.0, emit_x=0.2,
               alpha_y=0.0, beta_y=5.0, emit_y=0.2,
               alpha_z=0.0, beta_z=60.0, emit_z=0.5)
    def run(w):
        ref = ReferenceParticle(species=H_MINUS, w_kin=w, frequency=650.0)
        return EnvelopeSolver(lat, ref, ini, current=0.0).run().ref_w_kin[-1]
    first = run(45.35)
    second = run(234.82)
    fresh_lat = Lattice()
    fresh_lat.add(NCells("e", mode=1, n_cells=16, beta_g=0.0, eot_v_per_m=5e6,
                         theta_s_deg=0.0, aperture_mm=15, p_flag=0,
                         frequency_mhz=650.0))
    ref = ReferenceParticle(species=H_MINUS, w_kin=234.82, frequency=650.0)
    second_fresh = EnvelopeSolver(fresh_lat, ref, ini,
                                  current=0.0).run().ref_w_kin[-1]
    assert second == pytest.approx(second_fresh, abs=1e-9)
    assert first > 45.35 and second > 234.82


def test_backtrack_warns_linear_fallback_for_ncells():
    """Exact-mode backtracking through NCELLS has no exact inverse — it must
    WARN about the linear fallback, not degrade closure silently."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.tracker import Tracker
    from linac_gen.tracking.backtrack import (backtrack_distribution,
                                              BacktrackWarning)
    lat = Lattice()
    lat.add(_mk(theta=-30.0, n=4))
    ref = ReferenceParticle(species=H_MINUS, w_kin=150.0, frequency=650.0)
    beam = Beam(ref=ref, n_particles=4, current=0.0)
    beam.particles[:, :] = 0.0
    Tracker(lat, beam).run()
    entrance = ReferenceParticle(species=H_MINUS, w_kin=150.0, frequency=650.0)
    with pytest.warns(BacktrackWarning, match="no exact backward"):
        backtrack_distribution(lat, beam, entrance, end=0)


def test_error_study_applies_error_cav_to_ncells(tmp_path):
    """ERROR_CAV_NCPL_STAT on an NCELLS deck must perturb the cavity —
    voltage_rel and phase_offset both land on the element (regression: both
    were dropped, phase silently; tolerance studies ran with zero RF jitter
    on NCELLS decks)."""
    import os
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.errors.error_model import ErrorStudy
    deck = """FREQ 650.0
ERROR_CAV_NCPL_STAT 1 0 0 0 0 0 5.0 10.0 0
NCELLS 1 4 0.5 5.0E6 0 15.0 0.0 0.0 0.0 0.0 0.0
end
"""
    p = os.path.join(tmp_path, "e.dat")
    with open(p, "w") as f:
        f.write(deck)
    lat, _ = parse_tracewin(p)
    assert lat.errors, "ERROR_CAV directive did not register"
    from linac_gen.core.config import BeamConfig
    cfg = BeamConfig(species="H-", energy=150.0, frequency=650.0,
                     current=0.0, n_particles=16)
    study = ErrorStudy(lattice=lat, beam_config=cfg, n_seeds=1)
    perturbed = study._apply_errors(seed=3)
    nc = next(e for e in perturbed.elements if isinstance(e, NCells))
    assert nc.voltage_rel != 0.0, "amplitude error not applied to NCELLS"
    assert nc.phase_offset != 0.0, "phase error not applied to NCELLS"


def test_matrix_path_cross_energy_rerun_matches_fresh():
    """compute_transfer_matrix must reset stateful field-map elements: a
    βg=0 cavity reused at a different energy previously kept run-1's
    resolved geometry (max|ΔM| ≈ 2, tail gaps lost)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    def mk():
        return NCells("mtx", mode=1, n_cells=16, beta_g=0.0,
                      eot_v_per_m=5e6, theta_s_deg=0.0, aperture_mm=15,
                      p_flag=0, frequency_mhz=650.0)
    reused = mk()
    lat = Lattice(); lat.add(reused)
    r1 = ReferenceParticle(species=H_MINUS, w_kin=45.35, frequency=650.0)
    compute_transfer_matrix(lat, r1)                       # run 1 resolves geometry
    r2 = ReferenceParticle(species=H_MINUS, w_kin=234.82, frequency=650.0)
    M_reused = compute_transfer_matrix(lat, r2)
    fresh_lat = Lattice(); fresh_lat.add(mk())
    r3 = ReferenceParticle(species=H_MINUS, w_kin=234.82, frequency=650.0)
    M_fresh = compute_transfer_matrix(fresh_lat, r3)
    np.testing.assert_allclose(M_reused, M_fresh, atol=1e-9,
                               err_msg="matrix path reused stale geometry")


# ── geometric synchronism factor (2026-07-21, TW matrix reverse-eng.) ────
def _cav16(**kw):
    return NCells("CAV", mode=1, n_cells=16, beta_g=BG,
                  eot_v_per_m=6.8e6, theta_s_deg=62.0, aperture_mm=15.0,
                  p_flag=1, frequency_mhz=F, **kw)


def test_sync_factor_normalised_at_synchronism():
    from linac_gen.elements.ncells import _sync_factor
    assert _sync_factor(BG, BG) == pytest.approx(1.0, abs=1e-12)
    # slower particle -> u > pi/2 -> smaller factor; faster -> larger.
    assert _sync_factor(BG * 0.98, BG) < 1.0 < _sync_factor(BG * 1.02, BG)
    # raw form: sin(u)/u with T(sync) = 2/pi.
    u = (np.pi / 2.0) * BG / (BG * 1.05)
    assert _sync_factor(BG * 1.05, BG) == pytest.approx(
        (np.sin(u) / u) / (2.0 / np.pi), rel=1e-12)


def test_gap_matrix_synchronism_det_one_and_asymmetry():
    """M55 = 1+delta, M44 = 1/(1+delta): determinant exactly 1, and the
    synchronism escape hatch recovers the ideal thin-gap (T==1) map."""
    cav = _cav16()
    ref = _ref(w=120.0)
    cav.reset_run_state()
    M = cav.fitted_matrix(ref)
    L = M[np.ix_([4, 5], [4, 5])]
    assert abs(np.linalg.det(L) - 1.0) < 1e-9
    cav2 = _cav16()
    cav2.synchronism = False
    cav2.reset_run_state()
    M0 = cav2.fitted_matrix(_ref(w=120.0))
    L0 = M0[np.ix_([4, 5], [4, 5])]
    # The synchronism terms produce the asymmetric diagonals (M55 up /
    # M66 down vs the ideal map at beam beta != beta_g).
    assert not np.allclose(L, L0, rtol=1e-6)
    assert abs(np.linalg.det(L0) - 1.0) < 1e-9


def test_tracking_map_linearises_to_gap_matrix():
    """The MP kick linearises exactly to the synchronism-OFF matrix;
    the synchronism-ON matrix (TW matrix engine) deliberately differs
    (mode-faithful split).  Jacobian by central differences."""
    eps_phi, eps_w = 1e-3, 1e-4        # deg, MeV
    J = np.zeros((2, 2))
    for col, (dphi0, dw0, eps) in enumerate(
            (((1, 0), (0, 0), eps_phi), ((0, 0), (0, 1), eps_w))):
        outs = []
        for sgn in (+1.0, -1.0):
            cav = _cav16()
            ref = _ref(w=120.0)
            beam = Beam(ref=ref, n_particles=1, current=0.0)
            beam.particles[0, 4] = sgn * eps * dphi0[0] + sgn * eps * dw0[0]
            beam.particles[0, 5] = sgn * eps * dphi0[1] + sgn * eps * dw0[1]
            cav.reset_run_state()
            n = 64
            ds = cav.length / n
            for _ in range(n):
                cav.track_rk4(beam, ds)
            outs.append(beam.particles[0, [4, 5]].copy())
        J[:, col] = (outs[0] - outs[1]) / (2 * eps)
    cav = _cav16()
    cav.reset_run_state()
    M = cav.fitted_matrix(_ref(w=120.0))
    L = M[np.ix_([4, 5], [4, 5])]
    # MODE-FAITHFUL split (mirrors TraceWin's own two engines): the
    # tracking map is the stock thin-gap kick (TW-partran-faithful),
    # the matrix path carries the det-normalised synchronism factor
    # (TW-matrix-engine-faithful).  With synchronism OFF the two must
    # agree exactly; with it ON they deliberately differ.
    cav0 = _cav16(); cav0.synchronism = False
    cav0.reset_run_state()
    L0 = cav0.fitted_matrix(_ref(w=120.0))[np.ix_([4, 5], [4, 5])]
    assert np.allclose(J, L0, rtol=2e-3, atol=2e-4), (J, L0)
    assert not np.allclose(L, L0, rtol=1e-4)


def test_sync_delta_clamped_off_synchronism():
    """A beam grossly below the cavity's geometric beta drives That -> 0
    where the log-derivative diverges (delta ~ -800 while finite).  The
    validity clamp must fall back to the stock matrix (delta = 0), warn
    once, and keep the fitted matrix sane instead of ~1e5 entries."""
    import warnings as _w
    from linac_gen.elements.ncells import _sync_factor

    def _beta(wk):
        g = 1.0 + wk / H_MINUS.mass
        return float(np.sqrt(1.0 - 1.0 / g**2))

    # find the smallest positive That in the diverging region (beta
    # slightly above beta_g/2, u just below pi)
    ws = np.linspace(20.0, 30.0, 20001)
    ts = np.array([_sync_factor(_beta(w), BG) for w in ws])
    pos = ts > 0
    w_bad = float(ws[pos][np.argmin(ts[pos])])

    cav = _cav16()
    cav.reset_run_state()
    ref = _ref(w=w_bad)
    gap = cav._ensure_gaps(ref)[0]
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        d1 = cav._sync_delta(ref, gap)
        d2 = cav._sync_delta(ref, gap)      # second call: no re-warn
    assert d1 == 0.0 and d2 == 0.0
    msgs = [str(r.message) for r in rec if "synchronism" in str(r.message)]
    assert len(msgs) == 1, msgs

    # clamped everywhere -> the fitted matrix falls back bit-identically
    # to the synchronism-OFF (stock thin-gap) matrix, not ~1e5 entries
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        cav2 = _cav16()
        cav2.reset_run_state()
        M = cav2.fitted_matrix(_ref(w=w_bad))
        cav2b = _cav16()
        cav2b.synchronism = False
        cav2b.reset_run_state()
        M0 = cav2b.fitted_matrix(_ref(w=w_bad))
    assert np.array_equal(M, M0)

    # the matched cavity is far inside the validity range: unclamped
    cav3 = _cav16()
    cav3.reset_run_state()
    ref3 = _ref(w=120.0)
    d = cav3._sync_delta(ref3, cav3._ensure_gaps(ref3)[0])
    assert d != 0.0 and abs(d) < cav3._SYNC_DELTA_MAX


def test_sync_particle_energy_unaffected_by_synchronism():
    for flag in (True, False):
        cav = _cav16()
        cav.synchronism = flag
        ref = _ref(w=120.0)
        cav.reset_run_state()
        cav.advance_ref(ref)
        if flag:
            w_on = ref.w_kin
        else:
            assert ref.w_kin == pytest.approx(w_on, rel=1e-12)
