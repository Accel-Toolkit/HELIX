"""Per-cell Hofmann stability analysis (analysis/hofmann_stability.py).

Adapter/driver/margin logic is pure (driven by a channel dict + a light
results object), so most tests are synthetic; dual-regime end-to-end runs
(bunched I>0, I=0, DC, coupled) confirm the wiring off real envelope
probes, and a PIP-II FDR deck test (skipped when the external tree is
absent) exercises a production lattice.

The synthetic growth/flag/margin expectations were cross-verified against
the source manuscript package (PRAB_Hofmann): per-branch growths agree
bitwise and margin onsets/S2-jumps agree with its scalar scan loop.
"""
from __future__ import annotations

import warnings as _warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.analysis.hofmann_dispersion import DispersionSolver
from linac_gen.analysis.hofmann_probabilistic import instability_probability
from linac_gen.analysis.hofmann_stability import (
    FLAG_THRESHOLD,
    S2_GATE,
    anisotropy_margin,
    hofmann_coordinates,
    hofmann_stability,
    per_branch_growth,
)


# =============================================================================
# Synthetic fixtures
# =============================================================================

class _FakeResults:
    continuous = False
    element_maps_dep = [np.eye(6)] * 8
    element_maps_bare = [np.eye(6)] * 8
    element_exit_idx = [1, 2, 3, 4, 5, 6, 7, 8]
    # emit_z (native deg.MeV) deliberately DIFFERS from emit_z_mmmrad so a
    # convention slip in the adapter is caught, not silently equivalent.
    emit_z = [99.0] * 9
    emit_x = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28]
    emit_z_mmmrad = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38]


class _FakePeriod:
    def spans(self):
        return ((0, 4), (4, 8))


def _channel(mu_x_bare, mu_x_dep, mu_z_bare, mu_z_dep):
    n = len(mu_x_bare)
    b = np.asarray(mu_x_bare, float)
    d = np.asarray(mu_x_dep, float)
    zb = np.asarray(mu_z_bare, float)
    zd = np.asarray(mu_z_dep, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "cells": np.arange(1.0, n + 1.0), "coupled_xy": False,
            "mu_x_bare_deg": b, "mu_x_dep_deg": d,
            "mu_z_bare_deg": zb, "mu_z_dep_deg": zd,
            "eta_x": d / b, "eta_z": zd / zb,
        }


CH = _channel([80.0, 82.0], [48.0, 51.0], [70.0, 72.0], [40.0, 44.0])


# =============================================================================
# Adapter (hofmann_coordinates)
# =============================================================================

def test_coordinates_R_is_depressed_ratio():
    c = hofmann_coordinates(_FakeResults(), _FakePeriod(), channel=CH)
    assert c["reason"] is None
    np.testing.assert_allclose(c["R"], [40.0 / 48.0, 44.0 / 51.0], rtol=1e-14)
    np.testing.assert_allclose(c["Y"], [48.0 / 80.0, 51.0 / 82.0], rtol=1e-14)
    # R == (kz0*eta_z)/(kx0*eta_x) identically
    np.testing.assert_allclose(
        c["R"], c["kz0_deg"] * c["eta_z"] / (c["kx0_deg"] * c["eta_x"]),
        rtol=1e-12)


def test_coordinates_eps_sampled_at_each_span_entry_in_mmmrad():
    """eps_ratio uses emit_z_mmmrad (NOT native deg.MeV emit_z), sampled
    at each repeat-span ENTRANCE record via element_record_span."""
    c = hofmann_coordinates(_FakeResults(), _FakePeriod(), channel=CH)
    # span (0,4) -> record 0; span (4,8) -> record exit_idx[3] = 4
    np.testing.assert_allclose(
        c["eps_ratio"], [0.30 / 0.20, 0.34 / 0.24], rtol=1e-14)


def test_coordinates_nan_propagates():
    ch = _channel([80.0, 0.0], [48.0, np.nan], [70.0, 72.0], [40.0, np.nan])
    c = hofmann_coordinates(_FakeResults(), _FakePeriod(), channel=ch)
    assert np.isfinite(c["R"][0])
    assert not np.isfinite(c["R"][1])


def test_coordinates_eta_ge1_warns_but_keeps_cells():
    ch = _channel([80.0, 82.0], [80.0, 82.0], [70.0, 72.0], [70.0, 72.0])
    with pytest.warns(UserWarning, match="eta_x >= 1"):
        c = hofmann_coordinates(_FakeResults(), _FakePeriod(), channel=ch)
    assert c["reason"] is None
    assert np.all(np.isfinite(c["R"]))


def test_coordinates_fold_risk_flag():
    ch = _channel([175.0, 82.0], [100.0, 51.0], [70.0, 172.0], [40.0, 90.0])
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        c = hofmann_coordinates(_FakeResults(), _FakePeriod(), channel=ch)
    assert bool(c["fold_risk"][0]) and bool(c["fold_risk"][1])


def test_coordinates_refuses_coupled():
    c = hofmann_coordinates(
        _FakeResults(), _FakePeriod(),
        channel={"cells": np.array([1.0]), "coupled_xy": True})
    assert c["reason"] is not None and "coupled" in c["reason"]
    assert not np.isfinite(c["R"]).any()


def test_coordinates_refuses_dc():
    class DC(_FakeResults):
        continuous = True
    c = hofmann_coordinates(DC(), _FakePeriod(), channel=CH)
    assert c["reason"] is not None and "DC" in c["reason"]


def test_coordinates_refuses_no_z_tune():
    ch = _channel([80.0, 82.0], [48.0, 51.0],
                  [np.nan, np.nan], [np.nan, np.nan])
    c = hofmann_coordinates(_FakeResults(), _FakePeriod(), channel=ch)
    assert c["reason"] is not None and "longitudinal" in c["reason"]


# =============================================================================
# Driver (hofmann_stability) — expectations cross-verified vs the source
# package: (2.316, 0.490, 1.149) is an in-domain l=3-odd flag (S2 = 7.66,
# gamma = 0.0493); (1.835, 0.197, 2.877) is hot but non-perturbative
# (S2 = 13.2 -> extrapolated flag only).
# =============================================================================

def _coords(R, Y, eps):
    n = len(R)
    return {
        "cells": np.arange(1.0, n + 1.0),
        "R": np.asarray(R, float), "Y": np.asarray(Y, float),
        "eps_ratio": np.asarray(eps, float),
        "kx0_deg": np.full(n, 80.0), "kz0_deg": np.full(n, 60.0),
        "eta_x": np.asarray(Y, float), "eta_z": np.full(n, 0.6),
        "fold_risk": np.zeros(n, dtype=bool),
        "transverse_label": "x", "reason": None,
    }


def _stability_from_coords(coords, **kw):
    """Drive hofmann_stability through a stub channel matching coords."""
    n = coords["cells"].size
    ch = {
        "cells": coords["cells"], "coupled_xy": False,
        "mu_x_bare_deg": coords["kx0_deg"],
        "mu_x_dep_deg": coords["kx0_deg"] * coords["Y"],
        "mu_z_bare_deg": coords["kz0_deg"],
        "mu_z_dep_deg": coords["kx0_deg"] * coords["Y"] * coords["R"],
        "eta_x": coords["eta_x"],
        "eta_z": coords["kx0_deg"] * coords["Y"] * coords["R"]
        / coords["kz0_deg"],
    }

    class _Res(_FakeResults):
        element_maps_dep = [np.eye(6)] * n
        element_maps_bare = [np.eye(6)] * n
        element_exit_idx = list(range(1, n + 1))
        emit_x = [1.0] * (n + 1)
        emit_z = [99.0] * (n + 1)
        emit_z_mmmrad = list(coords["eps_ratio"]) + [1.0]

    class _Per:
        def spans(self):
            return tuple((k, k + 1) for k in range(n))

    return hofmann_stability(_Res(), _Per(), channel=ch, **kw)


def test_stability_flags_and_gate():
    tab = _stability_from_coords(_coords(
        [2.316, 2.411, 1.835], [0.490, 0.535, 0.197],
        [1.149, 1.493, 2.877]))
    assert tab["reason"] is None
    # Cell 1: in-domain l=3-odd flag
    assert tab["valid"][0] and tab["flagged"][0]
    assert tab["g_combined"][0] == pytest.approx(0.049313, abs=1e-5)
    assert tab["g_l3_odd"][0] == pytest.approx(0.049313, abs=1e-5)
    assert tab["g_l2"][0] == 0.0
    # Cell 2: in-domain, quiet
    assert tab["valid"][1] and not tab["flagged"][1]
    assert tab["g_combined"][1] == 0.0
    # Cell 3: hot but non-perturbative -> extrapolated flag only
    assert not tab["valid"][2]
    assert not tab["flagged"][2] and tab["flagged_extrap"][2]
    assert tab["S2"][2] == pytest.approx(13.2296, abs=1e-3)
    # Summary
    assert tab["n_valid"] == 2 and tab["n_flagged"] == 1
    assert tab["worst_cell"] == 1.0
    assert tab["worst_growth"] == pytest.approx(0.049313, abs=1e-5)
    assert tab["threshold"] == FLAG_THRESHOLD
    assert tab["s2_gate"] == S2_GATE
    assert len(tab["solver_fingerprint"]) == 10


def test_stability_l_modes_selects_combined():
    """With l_modes=(2,) the l=3-odd flag must disappear from combined
    (per-branch columns still report it)."""
    tab = _stability_from_coords(
        _coords([2.316], [0.490], [1.149]), l_modes=(2,))
    assert tab["g_combined"][0] == 0.0
    assert tab["g_l3_odd"][0] == pytest.approx(0.049313, abs=1e-5)
    assert not tab["flagged"][0]


def test_stability_l4_odd_off_by_default():
    tab = _stability_from_coords(_coords([2.316], [0.490], [1.149]))
    assert not np.isfinite(tab["g_l4_odd"][0])
    tab2 = _stability_from_coords(_coords([2.316], [0.490], [1.149]),
                                  include_l4_odd=True)
    assert np.isfinite(tab2["g_l4_odd"][0])


def test_per_branch_growth_golden():
    """Bitwise-verified against the source package's per-period driver."""
    g, s2 = per_branch_growth(DispersionSolver(), 1.835, 0.197, 2.877)
    assert s2 == pytest.approx(13.229640462580972, rel=1e-12)
    assert g["l2"] == pytest.approx(0.08743661105884878, rel=1e-9)
    assert g["l3_odd"] == pytest.approx(0.15551342538352686, rel=1e-9)
    assert g["l3_even"] == 0.0
    assert g["l4_even"] == pytest.approx(0.2613287904801357, rel=1e-9)  # -sp^4: now active


# =============================================================================
# Margin scan — onsets cross-verified against the source package's loop
# =============================================================================

def test_margin_onset_and_smooth_summary():
    mg = anisotropy_margin(None, None, coords=_coords(
        [2.411, 2.316], [0.535, 0.490], [1.493, 1.149]))
    assert mg["reason"] is None
    # Cell 1: onset at eps = 2.10 (source-package loop agrees), smooth
    assert mg["onset_eps"][0] == pytest.approx(2.10, abs=1e-9)
    assert mg["margin"][0] == pytest.approx(2.10 / 1.493, rel=1e-9)
    assert not mg["is_seam"][0]
    # Cell 2: already past onset at design (onset 1.05 < 1.149)
    assert mg["onset_eps"][1] == pytest.approx(1.05, abs=1e-9)
    assert mg["margin"][1] < 1.0
    assert mg["n_onsets"] == 2 and mg["n_smooth"] == 2 and mg["n_seam"] == 0
    assert mg["smallest_smooth_margin"] == pytest.approx(1.05 / 1.149,
                                                         rel=1e-9)
    assert mg["smallest_smooth_margin_cell"] == 2.0
    assert mg["earliest_smooth_onset_eps"] == pytest.approx(1.05, abs=1e-9)


def test_margin_skips_nonperturbative_and_undepressed():
    mg = anisotropy_margin(None, None, coords=_coords(
        [1.835, 0.9], [0.197, 1.0], [2.877, 1.5]))
    # cell 1: S2 = 13.2 at design -> skipped; cell 2: Y >= 1 -> skipped
    assert not np.isfinite(mg["onset_eps"]).any()
    assert mg["n_onsets"] == 0


# =============================================================================
# Probabilistic per-cell summary
# =============================================================================

def test_instability_probability_deterministic_and_bounded():
    coords = _coords([2.316, 2.411], [0.490, 0.535], [1.149, 1.493])
    p1 = instability_probability(coords, N_mc=100, seed=7)
    p2 = instability_probability(coords, N_mc=100, seed=7)
    np.testing.assert_array_equal(p1, p2)
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))
    # The flagged cell must carry substantial probability; jitter cannot
    # take it below the deterministic verdict by much.
    assert p1[0] > 0.5


def test_instability_probability_nan_on_refusal():
    coords = _coords([2.316], [0.490], [1.149])
    coords["reason"] = "DC/continuous beam"
    p = instability_probability(coords, N_mc=50)
    assert not np.isfinite(p).any()


def test_instability_probability_zero_without_space_charge():
    coords = _coords([0.9], [0.999], [1.2])
    p = instability_probability(coords, N_mc=100, seed=1)
    assert p[0] == pytest.approx(0.0, abs=0.05)


# =============================================================================
# End-to-end: real envelope probes, dual regimes
# =============================================================================

def _bunched_cell_lattice(n_cells=3):
    """FODO + buncher cell: transverse AND longitudinal focusing."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap

    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(Quadrupole(name="QF", length=40.0, gradient=+50.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(RFGap(name="G", voltage=0.10, phase=-90.0, frequency=162.5))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(Quadrupole(name="QD", length=40.0, gradient=-50.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
    return lat, 7


def _bunched_run(current=5.0):
    from linac_gen.analysis.period_detect import PeriodicStructure
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching.periodic import find_periodic_twiss
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.tracking.matrix_tracking import (
        compute_transfer_matrix, compute_twiss)

    lat, n_elem = _bunched_cell_lattice()
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    M = compute_transfer_matrix(lat, ref, start=0, end=n_elem - 1)
    twz = compute_twiss(M, "z")
    init = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=2.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=2.0,
                alpha_z=twz["alpha"], beta_z=twz["beta"], emit_z=0.1)
    res = EnvelopeSolver(lat, ref.copy(), init, current=current,
                         phase_probe=True).run()
    period = PeriodicStructure(
        start=0, end=3 * n_elem, inner_period_length=n_elem,
        inner_slice_end=n_elem, n_repeats=3, label="cell", source="manual")
    return res, period


@pytest.mark.physics
def test_e2e_bunched_with_current():
    res, period = _bunched_run(current=5.0)
    tab = hofmann_stability(res, period)
    assert tab["reason"] is None
    fin = np.isfinite(tab["R"]) & np.isfinite(tab["Y"])
    assert fin.any()
    assert np.all(tab["R"][fin] > 0)
    assert np.all((tab["Y"][fin] > 0) & (tab["Y"][fin] < 1.0))   # depressed
    assert np.all(tab["eps_ratio"][np.isfinite(tab["eps_ratio"])] > 0)
    g = tab["g_combined"][np.isfinite(tab["g_combined"])]
    assert g.size and np.all(g >= 0.0)
    assert np.isfinite(tab["S2"][fin]).all()
    mg = anisotropy_margin(res, period)
    assert mg["reason"] is None
    p = instability_probability(tab, N_mc=50, seed=3)
    pf = p[np.isfinite(p)]
    assert pf.size and np.all((pf >= 0) & (pf <= 1))


@pytest.mark.physics
def test_e2e_zero_current_trivially_stable():
    res, period = _bunched_run(current=0.0)
    with pytest.warns(UserWarning, match="eta_x >= 1"):
        tab = hofmann_stability(res, period)
    assert tab["reason"] is None
    g = tab["g_combined"][np.isfinite(tab["g_combined"])]
    assert g.size and np.all(g == 0.0)
    assert tab["n_flagged"] == 0


@pytest.mark.physics
def test_e2e_dc_beam_refused():
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.matching.periodic import find_periodic_twiss
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    init = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=15.0,
                         phase_probe=True).run()
    period = detect_periods(lat)[0]
    tab = hofmann_stability(res, period)
    assert tab["reason"] is not None and "DC" in tab["reason"]
    assert tab["n_valid"] == 0 and tab["n_flagged"] == 0


@pytest.mark.physics
def test_e2e_coupled_solenoid_refused():
    from linac_gen.analysis.period_detect import PeriodicStructure
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat = Lattice()
    for _ in range(2):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Solenoid(name="S", length=200.0, field=0.4, aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    init = dict(alpha_x=0.0, beta_x=0.5, emit_x=1.0,
                alpha_y=0.0, beta_y=0.5, emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=5.0,
                         phase_probe=True).run()
    period = PeriodicStructure(start=0, end=6, inner_period_length=3,
                               inner_slice_end=3, n_repeats=2,
                               label="cell", source="manual")
    tab = hofmann_stability(res, period)
    assert tab["reason"] is not None and "coupled" in tab["reason"]


# =============================================================================
# PIP-II FDR deck (external tree; public-cut safe skip)
# =============================================================================

_FDR = Path.home() / (
    "Desktop/Projects/PIP_II/Paper/Lattice_Design_and_Beam_dynamics/"
    "TraceWin/SC_Linac/FDR_Design"
)
_DECK = _FDR / "calculations" / "final_lattice.dat"
_DST = _FDR / "input.dst"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_DECK.is_file() and _DST.is_file() and (_FDR / "Fields").is_dir()),
    reason="PIP-II FDR TraceWin deck not present")
def test_fdr_deck_hofmann_stability(tmp_path):
    """Production-lattice wiring check: the full SC-linac envelope with a
    phase probe feeds the per-cell analysis without refusals or NaN
    storms.  Structural assertions only — the physics cross-validation
    against the manuscript lives in the paper repository."""
    import re

    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.io.tracewin_dst import load_dst
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.tracking.envelope import EnvelopeSolver

    text = _DECK.read_text(encoding="latin-1")
    text = re.sub(r"(?im)^FIELD_MAP_PATH .*$",
                  f"FIELD_MAP_PATH {_FDR / 'Fields'}", text)
    deck = tmp_path / "fdr_final_lattice.dat"
    deck.write_text(text, encoding="latin-1")

    parts, hdr = load_dst(str(_DST))
    cov = np.cov(parts.T)

    def _twiss(c):
        e = float(np.sqrt(max(np.linalg.det(c), 1e-30)))
        return -c[0, 1] / e, c[0, 0] / e, e

    ax, bx, ex = _twiss(cov[np.ix_([0, 1], [0, 1])])
    ay, by, ey = _twiss(cov[np.ix_([2, 3], [2, 3])])
    az, bz, ez = _twiss(cov[np.ix_([4, 5], [4, 5])])
    init = dict(alpha_x=ax, beta_x=bx, emit_x=ex,
                alpha_y=ay, beta_y=by, emit_y=ey,
                alpha_z=az, beta_z=bz, emit_z=ez)
    ref = ReferenceParticle(species=H_MINUS, w_kin=hdr["w_kin_ref"],
                            frequency=hdr["frequency_MHz"])
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
        res = EnvelopeSolver(lat, ref, init, current=hdr["current_mA"],
                             initial_sigma=cov, phase_probe=True).run()

    periods = [p for p in detect_periods(lat)
               if p.source != "fallback" and p.n_repeats >= 3]
    if not periods:
        pytest.skip("no non-fallback periodic structure detected")

    # The SC linac mixes focusing schemes: HWR/SSR sections are
    # solenoid-focused (x-y coupled -> the analysis must refuse with the
    # coupled reason, never crash), LB650/HB650 are quad-doublet sections
    # (must produce a clean table).  Every detected period must land in
    # one of those two outcomes.
    n_clean = 0
    n_coupled = 0
    for period in periods:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            tab = hofmann_stability(res, period)
        if tab["reason"] is None:
            n_clean += 1
            fin = np.isfinite(tab["g_combined"])
            assert fin.any(), f"{period.label}: no finite growth rate"
            assert np.all(tab["g_combined"][fin] >= 0.0)
            assert np.isfinite(tab["S2"][fin]).all()
        else:
            # Solenoid sections refuse as coupled; a period without RF in
            # its span would legitimately refuse for a missing z tune.
            assert ("coupled" in tab["reason"]
                    or "longitudinal" in tab["reason"]), (
                f"{period.label}: unexpected refusal: {tab['reason']}")
            n_coupled += 1
    assert n_clean + n_coupled == len(periods)
    assert n_clean + n_coupled >= 1
