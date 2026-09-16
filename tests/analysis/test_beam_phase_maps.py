"""Beam phase advance accumulated through the phase-probe slice maps.

Pins (2026-09-13, fnalscl finding): the record-grid trapezoid ∫ds/β is
12–38 % high through 1.4–2.1 m cavities on an element-exit grid, even
for a perfectly matched beam; the map-based walk reproduces the cell's
depressed eigenphase (Floquet) to round-off on ANY grid, flags a coarse
trapezoid, keeps the TraceWin writer on the trapezoid (TW's own kx
convention), and fixes the z-plane branch of ``sigma_over_sigma0_z``.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.analysis.period_detect import PeriodicStructure, detect_periods
from linac_gen.analysis.phase_advance import (
    beam_phase_advance,
    beam_phase_advance_along_s,
    beam_phase_advance_from_maps,
    run_phase_probe,
    structure_phase_advance,
)
from linac_gen.cli import common
from linac_gen.cli.common import _envelope_initial, build_ref
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import _with_tilt, compute_twiss, get_element_matrix

ROOT = Path(__file__).resolve().parents[2]
FNALSCL = ROOT / "examples" / "piplattice" / "fnalscl_periods.lgproj"

_INIT0 = dict(alpha_x=0.0, beta_x=0.5, emit_x=1.0, alpha_y=0.0, beta_y=0.5, emit_y=1.0,
              alpha_z=0.0, beta_z=10.0, emit_z=0.1)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _fnalscl_cell(k: int):
    """(lattice, cfg, period, cell lattice, matched state) for FODO cell ``k`` (0-based)."""
    from linac_gen.matching.periodic import find_matched_period_sigma
    with _quiet():
        lat, cfg, conv = common.load_input(str(FNALSCL))
    ref = build_ref(cfg)
    initial = _envelope_initial(cfg, ref)
    period = detect_periods(lat)[0]
    a, b = period.spans()[k]
    one = PeriodicStructure(start=a, end=b, inner_period_length=period.inner_period_length,
                            inner_slice_end=b, n_repeats=1, label=f"cell {k + 1}",
                            source="manual", repeat_spans=((a, b),))
    with _quiet():
        ms = find_matched_period_sigma(lat, ref, one, float(cfg.current), initial)
    assert ms["converged"]
    cell = Lattice()
    for e in lat.elements[a:b]:
        cell.add(e)
    cell.step_config = common.make_step_config(conv, {})
    init = dict(initial)
    init["continuous"] = ms["continuous"]
    return lat, cfg, conv, cell, ms, init


def _matched_probe(cell, cfg, ms, init, *, record_substeps=False):
    with _quiet():
        return EnvelopeSolver(cell, ms["ref_entry"].copy(), init, current=float(cfg.current),
                              initial_sigma=np.asarray(ms["sigma_entry"]),
                              bunch_frequency=ms["bunch_frequency"], sc_factor=ms["sc_factor"],
                              phase_probe=True, record_substeps=record_substeps).run()


def _eigenphases(res):
    M = np.eye(6)
    for m in res.element_maps_dep:
        M = np.asarray(m) @ M
    return {pl: compute_twiss(M, pl, coupling_tol=1e-3)["mu_folded"] for pl in "xyz"}


# ---------------------------------------------------------------------------
# 1. analytic drift: maps exact, element-grid trapezoid −4.5 %, flagged
# ---------------------------------------------------------------------------
def test_drift_maps_exact_trapezoid_biased_and_flagged():
    lat = Lattice()
    lat.add(Drift("D", 500.0))
    ref = ReferenceParticle(species=PROTON, w_kin=10.0, frequency=162.5)
    with _quiet():
        res = EnvelopeSolver(lat, ref.copy(), dict(_INIT0), current=0.0, phase_probe=True).run()
    beta0 = 0.5                      # m == mm/mrad
    L = 0.5                          # m
    exact = math.degrees(math.atan(L / beta0))            # 45°
    beta1 = beta0 + L * L / beta0
    trap = math.degrees(0.5 * L * (1.0 / beta0 + 1.0 / beta1))   # 42.97°
    maps = beam_phase_advance_along_s(res, method="maps")
    tr = beam_phase_advance_along_s(res, method="trapezoid")
    auto = beam_phase_advance_along_s(res)
    for pl in ("x", "y"):
        assert maps[f"mu_{pl}_deg"][-1] == pytest.approx(exact, rel=1e-9)
        assert tr[f"mu_{pl}_deg"][-1] == pytest.approx(trap, rel=1e-6)
        assert auto[f"mu_{pl}_deg"][-1] == pytest.approx(exact, rel=1e-9)
    assert maps["method"] == "maps" and maps["resolution_ok"] and maps["maps_source"] == "own"
    assert tr["method"] == "trapezoid" and not tr["resolution_ok"]
    assert tr["max_step_deg"] >= trap - 1e-9            # the z plane steps further still
    assert auto["method"] == "maps"


def test_no_probe_falls_back_to_trapezoid_and_maps_raises():
    lat = Lattice()
    lat.add(Drift("D", 500.0))
    ref = ReferenceParticle(species=PROTON, w_kin=10.0, frequency=162.5)
    with _quiet():
        res = EnvelopeSolver(lat, ref.copy(), dict(_INIT0), current=0.0, phase_probe=False).run()
    auto = beam_phase_advance_along_s(res)
    tr = beam_phase_advance_along_s(res, method="trapezoid")
    assert auto["method"] == "trapezoid"
    assert np.array_equal(auto["mu_x_deg"], tr["mu_x_deg"], equal_nan=True)
    with pytest.raises(ValueError):
        beam_phase_advance_along_s(res, method="maps")
    with pytest.raises(ValueError):
        beam_phase_advance_along_s(res, method="bogus")


# ---------------------------------------------------------------------------
# 2. Floquet identity on real cavity cells (fnalscl, 23.7 mA)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 12])
def test_matched_cell_maps_equal_eigenphase_trapezoid_biased(k):
    lat, cfg, conv, cell, ms, init = _fnalscl_cell(k)
    res = _matched_probe(cell, cfg, ms, init)
    eig = _eigenphases(res)
    maps = beam_phase_advance_from_maps(res)
    tr = beam_phase_advance_along_s(res, method="trapezoid")
    for pl in "xyz":
        assert maps[f"mu_{pl}_deg"][-1] == pytest.approx(eig[pl], rel=1e-8), pl
    # element-exit grid: > 10 % high in x and y (14–39 % measured); flagged
    for pl in "xy":
        assert tr[f"mu_{pl}_deg"][-1] / eig[pl] > 1.10, pl
    assert not tr["resolution_ok"]
    # substep grid: the trapezoid is already exact to 1e-4 — the value of the
    # map walk is exactness WITHOUT substeps (GUI default, and MP results)
    res_sub = _matched_probe(cell, cfg, ms, init, record_substeps=True)
    tr_sub = beam_phase_advance_along_s(res_sub, method="trapezoid")
    maps_sub = beam_phase_advance_from_maps(res_sub)
    for pl in "xyz":
        assert tr_sub[f"mu_{pl}_deg"][-1] == pytest.approx(eig[pl], rel=1e-4), pl
        assert maps_sub[f"mu_{pl}_deg"][-1] == pytest.approx(eig[pl], rel=1e-8), pl
    assert tr_sub["resolution_ok"]
    assert maps["n_skipped_x"] == maps["n_skipped_y"] == maps["n_skipped_z"] == 0
    assert not maps["projected_only"]


# ---------------------------------------------------------------------------
# 3. fine-grid consistency on the example decks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deck, coupled", [("examples/fodo_cell.dat", False),
                                           ("examples/solenoid_channel.dat", True)])
def test_fine_grid_maps_vs_trapezoid(deck, coupled):
    with _quiet():
        lat, cfg, conv = common.load_input(str(ROOT / deck))
    ref = build_ref(cfg)
    initial = _envelope_initial(cfg, ref)
    period = detect_periods(lat)[0]
    with _quiet():
        res = run_phase_probe(lat, ref, initial, current=0.0, record_substeps=True)
    maps = beam_phase_advance_from_maps(res)
    tr = beam_phase_advance_along_s(res, method="trapezoid")
    a, b = period.spans()[0]
    r0 = 0 if a == 0 else res.element_exit_idx[a - 1]
    r1 = res.element_exit_idx[b - 1]
    assert maps["projected_only"] is coupled
    if not coupled:
        for pl in "xy":
            dm = maps[f"mu_{pl}_deg"][r1] - maps[f"mu_{pl}_deg"][r0]
            dt = tr[f"mu_{pl}_deg"][r1] - tr[f"mu_{pl}_deg"][r0]
            # a pure drift+quad cell at I = 0 has no substeps inside drifts
            # (full-element map), so the trapezoid keeps its drift error
            assert dm == pytest.approx(dt, rel=0.05), pl
    assert np.all(np.diff(maps["mu_x_deg"][np.isfinite(maps["mu_x_deg"])]) >= -1e-12)


# ---------------------------------------------------------------------------
# 4. tilt rotations are recorded by the probe
# ---------------------------------------------------------------------------
def test_tilted_quad_probe_map_matches_matrix_mode():
    lat = Lattice()
    lat.add(Drift("D1", 300.0))
    q = Quadrupole("Q", length=200.0, gradient=5.0)
    q.tilt_deg = 15.0
    lat.add(q)
    lat.add(Drift("D2", 300.0))
    ref = ReferenceParticle(species=PROTON, w_kin=10.0, frequency=162.5)
    init = dict(_INIT0, beta_x=2.0, beta_y=2.0)
    with _quiet():
        res = EnvelopeSolver(lat, ref.copy(), init, current=0.0, phase_probe=True).run()
    Mp = np.asarray(res.element_maps_dep[1])
    Mt = _with_tilt(q, get_element_matrix(q, ref.copy()))
    assert np.abs(Mp - Mt).max() < 1e-12
    assert np.abs(Mp[0:2, 2:4]).max() > 1e-3          # lab-frame map is coupled
    # the recorded Σ after the tilted quad equals the map applied to the entrance Σ
    S_in = np.asarray(res.sigma_matrix[1]); S_out = np.asarray(res.sigma_matrix[2])
    assert np.abs(Mt @ S_in @ Mt.T - S_out).max() < 1e-9 * max(1.0, np.abs(S_out).max())


# ---------------------------------------------------------------------------
# 5. MP results + companion probe maps (absolute RF phases → keep the reference)
# ---------------------------------------------------------------------------
def test_mp_results_with_companion_maps():
    import copy
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    lat, cfg, conv, cell, ms, init = _fnalscl_cell(0)
    probe = _matched_probe(cell, cfg, ms, init)
    eig = _eigenphases(probe)
    Sg = np.asarray(ms["sigma_entry"])

    def tw(i):
        e = math.sqrt(np.linalg.det(Sg[i:i + 2, i:i + 2]))
        return Sg[i, i] / e, -Sg[i, i + 1] / e, e
    bx, ax_, ex = tw(0); by, ay, ey = tw(2); bz, az, ez = tw(4)
    ref_k = ms["ref_entry"]
    cfg_k = copy.deepcopy(cfg)
    cfg_k.n_particles = 2000
    cfg_k.energy = float(ref_k.w_kin); cfg_k.frequency = float(ref_k.frequency)
    cfg_k.beta_x, cfg_k.alpha_x, cfg_k.beta_y, cfg_k.alpha_y = map(float, (bx, ax_, by, ay))
    cfg_k.beta_z, cfg_k.alpha_z = float(bz), float(az)
    bg = float(ref_k.bg)
    cfg_k.emit_nx, cfg_k.emit_ny, cfg_k.emit_z = float(ex * bg), float(ey * bg), float(ez)
    cfg_k.mismatch_x = cfg_k.mismatch_y = cfg_k.mismatch_z = 0.0
    sc = common.make_sc_config(cfg_k, conv, {})
    with _quiet():
        beam = create_beam(cfg_k, seed=42)
        beam.ref = ref_k.copy()                      # the deck's RF phases are absolute
        rec = Simulation(cell, beam, space_charge=sc).run()
    assert not getattr(rec, "probe_M", None)
    out = beam_phase_advance_along_s(rec, maps_from=probe)
    assert out["method"] == "maps" and out["maps_source"] == "companion"
    for pl in "xy":
        assert np.isfinite(out[f"mu_{pl}_deg"][-1])
        assert out[f"mu_{pl}_deg"][-1] == pytest.approx(eig[pl], rel=0.05), pl
    coarse = beam_phase_advance_along_s(rec)
    assert coarse["method"] == "trapezoid" and not coarse["resolution_ok"]
    # per-period API on MP results with the companion maps
    shifted = PeriodicStructure(start=0, end=len(cell.elements), inner_period_length=len(cell.elements),
                                inner_slice_end=len(cell.elements), n_repeats=1, label="cell 1",
                                source="manual", repeat_spans=((0, len(cell.elements)),))
    pp = beam_phase_advance(rec, shifted, maps_from=probe)
    assert pp["method"] == "maps" and pp["maps_source"] == "companion"
    assert pp["mu_x_deg"] == pytest.approx(out["mu_x_deg"][-1], rel=1e-12)
    # a companion whose element count differs is refused → trapezoid
    with _quiet():
        lat2, cfg2, conv2 = common.load_input(str(FNALSCL))
        other = run_phase_probe(lat2, build_ref(cfg2), _envelope_initial(cfg2, build_ref(cfg2)), current=0.0)
    mism = beam_phase_advance_along_s(rec, maps_from=other)
    assert mism["method"] == "trapezoid"


# ---------------------------------------------------------------------------
# 6. rows: exit rows exact, interior rows interpolated and counted
# ---------------------------------------------------------------------------
def test_substep_rows_interpolated_exit_rows_identical():
    lat, cfg, conv, cell, ms, init = _fnalscl_cell(1)
    coarse = beam_phase_advance_from_maps(_matched_probe(cell, cfg, ms, init))
    res_sub = _matched_probe(cell, cfg, ms, init, record_substeps=True)
    fine = beam_phase_advance_from_maps(res_sub)
    exit_rows = [0] + list(res_sub.element_exit_idx)
    assert fine["interpolated_rows"] == len(res_sub.s) - len(exit_rows)
    for pl in "xyz":
        a = fine[f"mu_{pl}_deg"][exit_rows]
        b = coarse[f"mu_{pl}_deg"]
        assert np.allclose(a, b, rtol=0, atol=1e-10), pl
        assert np.all(np.diff(fine[f"mu_{pl}_deg"]) >= -1e-12), pl


# ---------------------------------------------------------------------------
# 7. z-plane denominator: sigma_over_sigma0_z ≈ η_z for a matched beam
# ---------------------------------------------------------------------------
def test_sigma_over_sigma0_z_uses_the_followed_branch():
    lat, cfg, conv, cell, ms, init = _fnalscl_cell(0)
    res = _matched_probe(cell, cfg, ms, init)
    n = len(cell.elements)
    shifted = PeriodicStructure(start=0, end=n, inner_period_length=n, inner_slice_end=n,
                                n_repeats=1, label="cell 1", source="manual", repeat_spans=((0, n),))
    with _quiet():
        s0 = structure_phase_advance(cell, ms["ref_entry"].copy(), shifted)
    assert s0["mu_z_branch_deg"] > 180.0        # the mirrored branch that used to be the denominator
    pp = beam_phase_advance(res, shifted, sigma0=s0)
    assert pp["method"] == "maps"
    for pl in "xyz":
        r = pp[f"sigma_over_sigma0_{pl}"]
        assert r is not None and 0.85 < r < 1.02, (pl, r)      # depressed beam: η ≈ 0.92–0.99


# ---------------------------------------------------------------------------
# 8. the TraceWin writer stays on the trapezoid (TW's kx convention)
# ---------------------------------------------------------------------------
def test_tracewin_writer_keeps_trapezoid_kx(tmp_path):
    from linac_gen.io.tracewin_outputs import write_partran_out
    lat, cfg, conv, cell, ms, init = _fnalscl_cell(0)
    res = _matched_probe(cell, cfg, ms, init)
    tr = beam_phase_advance_along_s(res, method="trapezoid")
    maps = beam_phase_advance_from_maps(res)
    out = write_partran_out(res, lattice=None, beam_cfg=None, path=tmp_path / "p.out")
    rows = [l for l in out.read_text(encoding="utf-8").splitlines() if "\t" in l and not l.startswith("#")]
    s = np.asarray(res.s, float)
    exit_idx = list(res.element_exit_idx)
    # the cavity row (longest element): kx = Δμ/Δs of the trapezoid curve, not
    # of the map walk — TW's column is the endpoint-density average
    j = int(np.argmax([float(getattr(e, "length", 0.0) or 0.0) for e in cell.elements]))
    r, prev = exit_idx[j], (exit_idx[j - 1] if j > 0 else 0)
    kx_written = float(rows[j + 1].split("\t")[22])        # rows[0] is the INPUT row
    kx_trap = (tr["mu_x_deg"][r] - tr["mu_x_deg"][prev]) / (s[r] - s[prev])
    kx_maps = (maps["mu_x_deg"][r] - maps["mu_x_deg"][prev]) / (s[r] - s[prev])
    assert kx_written == pytest.approx(kx_trap, rel=1e-6)   # 8 significant digits in the file
    assert abs(kx_written - kx_maps) > 1e-2 * abs(kx_written)


# ---------------------------------------------------------------------------
# 9. structure walk replays the reference like compute_transfer_matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deck", ["examples/piplattice/fnalscl_periods.lgproj", "examples/mebt_plus_hwr.dat"])
def test_structure_walk_first_cell_equals_period_eigenphase(deck):
    """On decks whose cavity matrices depend on the arrival phase
    (absolute-phase NCELLS, SET_SYNC_PHASE field maps) the along-s
    structure walk used to advance s but not phi_s and never reset the
    cavities: fnalscl cell 1 read 135° for an eigenphase of 50.7°."""
    from linac_gen.analysis.phase_advance import structure_phase_advance_along_s
    with _quiet():
        lat, cfg, conv = common.load_input(str(ROOT / deck))
    ref = build_ref(cfg)
    period = detect_periods(lat)[0]
    with _quiet():
        s0 = structure_phase_advance(lat, ref, period)
        sc = structure_phase_advance_along_s(lat, ref, period, seed=s0)
    a, b = period.spans()[0]
    planes = [pl for pl in ("x", "y", "z") if s0.get(f"mu_{pl}_deg") is not None]
    assert planes, "period seeds no plane"            # mebt_hwr is x-y coupled: z only
    for pl in planes:
        d = sc[f"mu_{pl}_deg"][b] - sc[f"mu_{pl}_deg"][a]
        assert d == pytest.approx(s0[f"mu_{pl}_deg"], rel=1e-6), (pl, d, s0[f"mu_{pl}_deg"])
        # the walk's Twiss must come back to the seed after one period (matched)
        assert sc[f"beta_{pl}"][b] == pytest.approx(s0[f"beta_{pl}"], rel=1e-6), pl


# ---------------------------------------------------------------------------
# 10. sigma_over_sigma0 branch: a period above 180° with a beam depressed below it
# ---------------------------------------------------------------------------
def test_branch_rule_period_above_180_degrees():
    from linac_gen.core.particle import H_MINUS
    from linac_gen.matching.periodic import find_matched_period_sigma
    def fodo(g):
        lat = Lattice()
        for _ in range(9):
            lat.add(Drift("D", 100.0, aperture=10.0)); lat.add(Quadrupole("QF", length=50.0, gradient=g, aperture=10.0))
            lat.add(Drift("D", 100.0, aperture=10.0)); lat.add(Quadrupole("QD", length=50.0, gradient=-g, aperture=10.0))
        return lat
    period = PeriodicStructure(start=0, end=36, inner_period_length=12, inner_slice_end=12, n_repeats=3,
                               label="3-cell period", source="manual", repeat_spans=((0, 12), (12, 24), (24, 36)))
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    lat = s0 = None
    for g in (36.0, 38.0, 40.0, 42.0):       # ~70–78° per cell → 207–235° per period (38 T/m: 220.8°)
        cand = fodo(g)
        with _quiet():
            try:
                cs0 = structure_phase_advance(cand, ref.copy(), period)
            except Exception:                                          # noqa: BLE001
                continue
        # well above 180° so the DEPRESSED period map stays above 180° too
        if cs0.get("mu_x_branch_deg") and cs0["mu_x_branch_deg"] > 205.0 and cs0["mu_x_deg"] < 180.0:
            lat, s0 = cand, cs0
            break
    assert lat is not None, "no gradient gave a 3-cell period above 205°"
    base = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.3, alpha_y=0.0, beta_y=1.0, emit_y=0.3,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    with _quiet():
        ms = find_matched_period_sigma(lat, ref, period, 5.0, base)
    init = dict(base); init["continuous"] = ms["continuous"]
    with _quiet():
        res = EnvelopeSolver(lat, ms["ref_entry"].copy(), init, current=5.0, initial_sigma=np.asarray(ms["sigma_entry"]),
                             bunch_frequency=ms["bunch_frequency"], sc_factor=ms["sc_factor"], phase_probe=True).run()
    Md = np.eye(6); Mb = np.eye(6)
    for j in range(12):
        Md = np.asarray(res.element_maps_dep[j]) @ Md; Mb = np.asarray(res.element_maps_bare[j]) @ Mb
    eta_true = compute_twiss(Md, "x", coupling_tol=1e-3)["mu"] / compute_twiss(Mb, "x", coupling_tol=1e-3)["mu"]
    assert 0.5 < eta_true < 1.0
    pp = beam_phase_advance(res, period, sigma0=s0)
    assert pp["method"] == "maps"
    assert pp["sigma_over_sigma0_x"] == pytest.approx(eta_true, rel=2e-3)
    assert pp["sigma_over_sigma0_x"] < 1.0                             # a "nearest branch" rule gave 1.29 here


# ---------------------------------------------------------------------------
# 11. per-plane resolution verdicts: z never greys out x/y
# ---------------------------------------------------------------------------
def test_resolution_verdicts_are_per_plane():
    with _quiet():
        lat, cfg, conv = common.load_input(str(ROOT / "examples/fodo_cell.dat"))
    ref = build_ref(cfg)
    with _quiet():
        res = run_phase_probe(lat, ref, _envelope_initial(cfg, ref), current=0.0, record_substeps=True)
    tr = beam_phase_advance_along_s(res, method="trapezoid")
    assert tr["resolution_ok_x"] and tr["resolution_ok_y"] and tr["resolution_ok"]
    assert not tr["resolution_ok_z"] and tr["max_step_z_deg"] > 18.0        # no RF: the z "phase" is not resolvable
    assert tr["max_step_deg"] == max(tr["max_step_x_deg"], tr["max_step_y_deg"])


# ---------------------------------------------------------------------------
# 12. MP with substep recording: interior rows interpolated, exit rows unchanged
# ---------------------------------------------------------------------------
def test_mp_substep_rows_interpolated():
    import copy
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    lat, cfg, conv, cell, ms, init = _fnalscl_cell(0)
    probe = _matched_probe(cell, cfg, ms, init)
    cfg_k = copy.deepcopy(cfg); cfg_k.n_particles = 500; cfg_k.current = 0.0
    cfg_k.energy = float(ms["ref_entry"].w_kin); cfg_k.frequency = float(ms["ref_entry"].frequency)
    with _quiet():
        b1 = create_beam(cfg_k, seed=7); b1.ref = ms["ref_entry"].copy()
        rec = Simulation(cell, b1, space_charge=None).run()
        b2 = create_beam(cfg_k, seed=7); b2.ref = ms["ref_entry"].copy()
        rec_sub = Simulation(cell, b2, space_charge=None, record_substeps=True).run()
    coarse = beam_phase_advance_along_s(rec, maps_from=probe)
    fine = beam_phase_advance_along_s(rec_sub, maps_from=probe)
    assert coarse["method"] == fine["method"] == "maps"
    assert coarse["interpolated_rows"] == 0 and fine["interpolated_rows"] > 0
    exit_rows = [0] + list(rec_sub.element_exit_idx)
    # the two MP runs step the quads differently (substep recording), so
    # their Σ differ at the 1e-4 level — compare, don't demand identity
    assert np.allclose(fine["mu_x_deg"][exit_rows], coarse["mu_x_deg"], rtol=2e-3, atol=1e-6)
    assert np.all(np.diff(fine["mu_x_deg"]) >= -1e-12)
