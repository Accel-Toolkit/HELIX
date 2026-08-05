"""LEBT space-charge-compensation analysis (linac_gen/analysis/scc/).

Three layers:

1. Standalone physics pins — behaviors ported from the source tool's own
   suite (no-gas limit exact, pressure monotonicity, signed wells,
   radial Poisson vs the closed form) plus the 45 keV calibration
   anchors and value pins generated from the verified port.
2. Cross-language parity — when the source repo's spec/testvectors.json
   is present (external tree; skips cleanly otherwise), the port must
   reproduce its pinned sections.  The full 15-section contract was
   verified at port time; this re-checks the physics-critical subset.
3. Driver end-to-end — a DC solenoid LEBT run through scc_analysis:
   profiles, cards, taper, build-up, guards (bunched refusal is the
   mirror of the Hofmann diagnostic's DC refusal).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.analysis.scc import (
    BEAM_SPECIES,
    GAS_LIBRARY,
    GasMix,
    RadialGrid,
    SelfConsistentSCC,
    beam_loss_rate_per_length,
    beam_potential_axis,
    compensation_time,
    rms_to_edge_radius,
)
from linac_gen.analysis.scc.constants import A_ION_45KEV, sigma_at_H_equivalent
from linac_gen.analysis.scc.driver import (
    scc_analysis,
    suggested_scc_deck_lines,
)
from linac_gen.analysis.scc.radial import solve_poisson_radial

HMINUS = BEAM_SPECIES["H-"]
HPLUS = BEAM_SPECIES["H+"]

_SCC_REPO = Path.home() / "Desktop/Projects/scc"
_VECTORS = _SCC_REPO / "spec" / "testvectors.json"


# =============================================================================
# 1. Standalone physics pins
# =============================================================================

def test_tau_and_mfp_pins():
    """Value pins generated from the port after the full testvector-contract
    parity run (bitwise on the reference platform)."""
    mix = GasMix(components={"N2": 2e-5, "H2": 1e-6})
    assert compensation_time(mix, HMINUS, 30.0) == pytest.approx(
        4.792669540444739e-06, rel=1e-12)
    assert beam_loss_rate_per_length(mix, HMINUS, 30.0) == pytest.approx(
        0.07694860645411171, rel=1e-12)


def test_calibration_anchors_45keV():
    """The twelve-anchor discipline: ionisation at 45 keV H-equivalent must
    equal the inherited-calibration constants (see the constants docstring —
    these are NOT literature values and must not drift silently)."""
    for g, want in A_ION_45KEV.items():
        got = sigma_at_H_equivalent(GAS_LIBRARY[g].sigma_ion, 45.0)
        assert got == pytest.approx(want, rel=1e-3), g


def test_provenance_tags_present():
    """Every calibration input carries a provenance tag (ported discipline)."""
    import linac_gen.analysis.scc.constants as C
    doc = C.__doc__ or ""
    for tag in ("[SOURCED]", "[DERIVED]", "[ASSUMED]",
                "[INHERITED-CALIBRATION]"):
        assert tag in doc, f"provenance tag {tag} missing from constants doc"


def test_no_gas_means_exactly_zero_compensation():
    grid = RadialGrid(n=120, r_pipe_m=0.06)
    rho = np.where(grid.r <= 0.01, -1e-7 / (math.pi * 0.01 ** 2), 0.0)
    res = SelfConsistentSCC(grid).solve(
        rho, beam_sign=-1.0, production_per_m_per_s=0.0,
        lebt_length_m=2.0, trapped_mass_kg=GAS_LIBRARY["H2"].ion_mass_kg())
    assert res.fc == 0.0


def test_fc_monotonic_with_pressure_and_bounded():
    grid = RadialGrid(n=120, r_pipe_m=0.06)
    lam = -(0.015 / 2.4e6)              # 15 mA H- at ~30 keV velocity scale
    rho = np.where(grid.r <= 0.01, lam / (math.pi * 0.01 ** 2), 0.0)
    fcs = []
    for rate in (1e12, 1e14, 1e16):
        res = SelfConsistentSCC(grid).solve(
            rho, beam_sign=-1.0, production_per_m_per_s=rate,
            lebt_length_m=2.0,
            trapped_mass_kg=GAS_LIBRARY["N2"].ion_mass_kg())
        fcs.append(res.fc)
    assert fcs == sorted(fcs)
    assert all(0.0 <= f <= 1.0 for f in fcs)


def test_positive_beam_traps_electrons_in_positive_well():
    grid = RadialGrid(n=120, r_pipe_m=0.06)
    rho = np.where(grid.r <= 0.01, +1e-7 / (math.pi * 0.01 ** 2), 0.0)
    from linac_gen.analysis.scc.constants import M_E
    res = SelfConsistentSCC(grid).solve(
        rho, beam_sign=+1.0, production_per_m_per_s=1e14,
        lebt_length_m=2.0, trapped_mass_kg=M_E)
    assert res.phi_uncompensated_V > 0.0
    assert 0.0 <= res.fc <= 1.0


def test_radial_poisson_matches_closed_form():
    """Uniform disc in a grounded pipe: solver axis potential vs Chauvin
    Eq. 64 (the same oracle the source tool uses)."""
    pipe, a = 0.06, 0.012
    I, beta = 0.020, 0.0080
    grid = RadialGrid(n=480, r_pipe_m=pipe)
    lam = -I / (beta * 2.99792458e8)
    rho = np.where(grid.r <= a, lam / (math.pi * a * a), 0.0)
    phi = solve_poisson_radial(grid, rho)
    want = beam_potential_axis(I, beta, a, pipe, charge_sign=-1.0)
    assert phi[0] == pytest.approx(want, rel=2e-2)
    # Dirichlet phi=0 sits at the WALL; the last cell CENTER is half a cell
    # inside, so it carries the wall-gradient x dr/2 — small, not zero.
    assert abs(phi[-1]) < 0.01 * abs(phi[0])


# =============================================================================
# 2. Cross-language parity vs the source repo's pinned vectors
# =============================================================================

@pytest.mark.skipif(not _VECTORS.is_file(),
                    reason="SCC source repo testvectors not present")
def test_source_testvector_parity_subset():
    doc = json.loads(_VECTORS.read_text())
    tol = doc["tolerances_rel"]

    def close(got, want, rtol):
        if want == 0.0:
            return abs(got) < 1e-300
        return abs(got - want) <= rtol * abs(want)

    from linac_gen.analysis.scc.constants import (
        sigma_capture, sigma_ionization, sigma_stripping)
    for row in doc["cross_sections"]:
        sp = BEAM_SPECIES[row["species"]]
        for name, fn in (("sigma_ion_m2", sigma_ionization),
                         ("sigma_strip_m2", sigma_stripping),
                         ("sigma_cap_m2", sigma_capture)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = fn(row["gas"], sp, row["T_keV"])
            assert close(got, row[name], tol["cross_sections"]), (name, row)

    for row in doc["tau_scc"]:
        mix = GasMix(components=row["components_mbar"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = compensation_time(mix, BEAM_SPECIES[row["species"]],
                                    row["T_keV"])
        want = row["tau_s"]
        if want is None:
            assert not math.isfinite(got)
        else:
            assert close(got, want, tol["tau_scc"]), row

    for row in doc["scc_balance"]:
        grid = RadialGrid(n=row["n"], r_pipe_m=row["r_pipe_m"])
        a = row["a_m"]
        rho = np.where(grid.r <= a,
                       row["lambda_C_per_m"] / (math.pi * a * a), 0.0)
        total = float((rho * grid.cell_area).sum())
        if total != 0.0:                 # renormalise to the exact discrete
            rho = rho * (row["lambda_C_per_m"] / total)   # line charge
        res = SelfConsistentSCC(
            grid, T_trapped_eV=row["T_trapped_eV"]).solve(
            rho, beam_sign=math.copysign(1.0, row["lambda_C_per_m"]),
            production_per_m_per_s=row["production_per_m_per_s"],
            lebt_length_m=2.0,
            trapped_mass_kg=row["trapped_mass_kg"])
        assert close(res.fc, row["fc"], tol["scc_balance"]), row
        assert close(res.phi_axis_V, row["phi_axis_V"],
                     tol["scc_balance"]), row
        assert res.over_neutralised == row["over_neutralised"], row


# =============================================================================
# 3. Driver end-to-end (DC solenoid LEBT) + guards
# =============================================================================

def _dc_lebt_run(current=15.0):
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat = Lattice()
    lat.add(Drift(name="D1", length=300.0, aperture=60.0))
    lat.add(Solenoid(name="S1", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D2", length=900.0, aperture=60.0))
    lat.add(Solenoid(name="S2", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D3", length=300.0, aperture=60.0))
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5)
    init = dict(alpha_x=0.0, beta_x=0.6, emit_x=3.0,
                alpha_y=0.0, beta_y=0.6, emit_y=3.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=current).run()
    return lat, res


@pytest.mark.physics
def test_driver_end_to_end_computed_mode():
    lat, res = _dc_lebt_run()
    a = scc_analysis(res, lat, species="H-", gas="N2", pressure_mbar=2e-5)
    assert a["reason"] is None
    assert a["fc_source"].startswith("computed balance")
    assert math.isfinite(a["tau_scc_global_us"]) and a["tau_scc_global_us"] > 0
    assert np.all((a["fc"] >= 0.0) & (a["fc"] <= 1.0))
    assert bool(a["slice_converged"].all())
    assert a["phi_min_V"] < 0.0                      # H- well is negative
    assert 0.0 < a["transmission_gas_pct"] <= 100.0
    assert a["mean_ion_mass_amu"] == pytest.approx(28.0, rel=0.05)  # N2+
    # taper: line ends sit below the interior steady state
    assert a["eta_ss"][0] < a["eta_ss"][len(a["eta_ss"]) // 2]
    # cards well-formed, factors within [0, 1]
    assert a["scc_cards"] and all(0.0 <= c["factor"] <= 1.0
                                  for c in a["scc_cards"])
    lines = suggested_scc_deck_lines(a)
    assert any(ln.startswith("SPACE_CHARGE_COMP") for ln in lines)
    # pipe radius picked up from the 60 mm apertures
    assert np.allclose(a["pipe_mm"], 60.0)


@pytest.mark.physics
def test_driver_pressure_monotonicity():
    lat, res = _dc_lebt_run()
    fcs, taus = [], []
    for p_mbar in (2e-6, 8e-6, 3e-5):
        a = scc_analysis(res, lat, species="H-", gas="H2",
                         pressure_mbar=p_mbar, taper=False)
        fcs.append(a["mean_fc"])
        taus.append(a["tau_scc_global_us"])
    assert taus == sorted(taus, reverse=True)        # more gas -> faster
    assert fcs == sorted(fcs)                        # more gas -> more comp.


@pytest.mark.physics
def test_driver_modes_buildup_and_mixture():
    lat, res = _dc_lebt_run()
    a_ass = scc_analysis(res, lat, species="H-", gas="N2",
                         pressure_mbar=2e-5, mode="assumed",
                         eta_assumed=0.5, taper=False)
    assert a_ass["fc_source"] == "assumed eta = 0.5"
    assert np.allclose(a_ass["eta_ss"], 0.5)
    a_t = scc_analysis(res, lat, species="H-", gas="N2",
                       pressure_mbar=2e-5, mode="assumed",
                       eta_assumed=0.5, taper=False, build_up_us=1.0)
    assert a_t["mean_fc"] < a_ass["mean_fc"]         # mid-build-up
    a_mix = scc_analysis(res, lat, species="H-",
                         gas={"Ar": 1e-5, "H2": 4e-6}, taper=False)
    assert a_mix["reason"] is None
    assert set(a_mix["gas_mix_mbar"]) == {"Ar", "H2"}
    assert a_mix["mean_ion_mass_amu"] > 20.0         # Ar-dominated


def test_driver_guards():
    class Bunched:
        continuous = False
    g = scc_analysis(Bunched())
    assert g["reason"] is not None and "bunched" in g["reason"]

    class Empty:
        continuous = True
        s = []
        sigma_x = []
        sigma_y = []
    assert scc_analysis(Empty())["reason"] is not None

    lat, res = _dc_lebt_run()
    with pytest.raises(ValueError, match="species"):
        scc_analysis(res, lat, species="U-238")
    with pytest.raises(ValueError, match="gas"):
        scc_analysis(res, lat, gas="O3")
    # explicit pipe radius overrides lattice apertures
    a = scc_analysis(res, lat, species="H-", pipe_radius_mm=40.0)
    assert np.allclose(a["pipe_mm"], 40.0)


def test_driver_accepts_mp_recorder_style_results():
    """MP recorder results carry per-record continuous_at, not a scalar
    `continuous` — a DC MP run must be accepted (adversarial finding F1)."""
    lat, res = _dc_lebt_run()

    class RecorderStyle:
        # The REAL recorder carries continuous_at (no scalar) and NO
        # current attribute — found live: an MP run silently computed
        # I=0 and f_c=0 everywhere before the explicit current fallback.
        continuous_at = [True] * len(res.s)
        s = res.s
        sigma_x = res.sigma_x
        sigma_y = res.sigma_y
        ref_w_kin = res.ref_w_kin
        transmission = np.full(len(res.s), 100.0)   # numpy array, not list
        element_exit_idx = getattr(res, "element_exit_idx", None)

    # without a current source: honest refusal, never a silent f_c=0
    g = scc_analysis(RecorderStyle(), lat, species="H-", gas="N2",
                     pressure_mbar=2e-5)
    assert g["reason"] is not None and "current" in g["reason"]

    # with the Beam-tab current passed explicitly (the GUI/tool path)
    a = scc_analysis(RecorderStyle(), lat, species="H-", gas="N2",
                     pressure_mbar=2e-5, current_mA=15.0)
    assert a["reason"] is None
    assert 0.0 < a["mean_fc"] <= 1.0
    assert a["phi_min_V"] < 0.0


def test_driver_hdf5_round_trip(tmp_path):
    """A saved-and-reloaded DC run keeps its continuous marker and current
    (adversarial finding F2)."""
    import types

    from linac_gen.io.hdf5_output import (load_results_hdf5,
                                          save_results_hdf5)

    lat, res = _dc_lebt_run()
    path = str(tmp_path / "dc_run.h5")
    save_results_hdf5(res, path)
    loaded = types.SimpleNamespace(**load_results_hdf5(path))
    assert bool(loaded.continuous) is True
    assert float(loaded.current_mA) == 15.0
    a = scc_analysis(loaded, lat, species="H-", gas="N2",
                     pressure_mbar=2e-5)
    assert a["reason"] is None
    assert a["phi_min_V"] < 0.0                  # current survived the trip


def test_driver_rejects_bad_buildup_and_mixture_gas():
    lat, res = _dc_lebt_run()
    with pytest.raises(ValueError, match="build_up_us"):
        scc_analysis(res, lat, build_up_us=-1.0)
    with pytest.raises(ValueError, match="gas"):
        scc_analysis(res, lat, gas={"O3": 1e-6})
    # archives without any continuity marker get the accurate message
    class NoMarker:
        s = res.s
        sigma_x = res.sigma_x
        sigma_y = res.sigma_y
    g = scc_analysis(NoMarker())
    assert "continuous-beam marker" in g["reason"]


def test_driver_run_current_wins_over_explicit_arg():
    """The results' own recorded current is authoritative: the beam sizes
    were computed AT that current, so a differing Beam-tab value must be
    ignored with a note (adversarial finding: a 15 mA HDF5 archive opened
    in a fresh session must not be silently analysed at the tab default)."""
    lat, res = _dc_lebt_run()                       # carries current_mA=15
    a = scc_analysis(res, lat, species="H-", gas="N2", pressure_mbar=2e-5)
    b = scc_analysis(res, lat, species="H-", gas="N2", pressure_mbar=2e-5,
                     current_mA=5.0)
    assert b["reason"] is None
    assert b["mean_fc"] == a["mean_fc"]             # 15 mA used, not 5
    assert any("15" in n_ and "ignored" in n_ for n_ in b["notes"])
    # matching values → no mismatch note
    c = scc_analysis(res, lat, species="H-", gas="N2", pressure_mbar=2e-5,
                     current_mA=15.0)
    assert not any("ignored" in n_ for n_ in c["notes"])


def test_driver_explicit_zero_falls_through_to_run_current():
    """current_mA=0.0 with results that DO carry a current must use the
    run current, not refuse (adversarial finding: the old refusal message
    was false on both clauses for this call)."""
    lat, res = _dc_lebt_run()
    a = scc_analysis(res, lat, species="H-", gas="N2", pressure_mbar=2e-5,
                     current_mA=0.0)
    assert a["reason"] is None and a["mean_fc"] > 0.0


def test_driver_genuine_zero_current_run_is_valid_degenerate():
    """A run genuinely at 0 mA computes (f_c ~ 0, no potential well) with
    an explanatory note — zero is data, not missing data."""
    lat, res = _dc_lebt_run(current=0.0)
    a = scc_analysis(res, lat, species="H-", gas="N2", pressure_mbar=2e-5)
    assert a["reason"] is None
    assert a["mean_fc"] == 0.0
    assert any("0 mA" in n_ for n_ in a["notes"])


def test_cleared_region_forces_residual_both_taper_branches():
    """Ion-cleared element ranges pin f_c to the residual inside, leave
    the line ends untouched — in BOTH taper regimes (dual-regime rule).
    Needs a substep-resolved grid: on the coarse per-element grid the
    window's soft edges are the only samples."""
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.particle import H_MINUS
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat, _ = _dc_lebt_run()
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5)
    init = dict(alpha_x=0.0, beta_x=0.6, emit_x=3.0,
                alpha_y=0.0, beta_y=0.6, emit_y=3.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref, init, current=15.0,
                         record_substeps=True).run()
    for taper_on in (True, False):
        base = scc_analysis(res, lat, species="H-", gas="N2",
                            pressure_mbar=2e-5, taper=taper_on)
        a = scc_analysis(res, lat, species="H-", gas="N2",
                         pressure_mbar=2e-5, taper=taper_on,
                         cleared_regions=[[2, 2]], cleared_residual=0.0)
        assert a["reason"] is None
        mid = int(np.argmin(np.abs(np.asarray(a["z_m"]) - 1.0)))  # D2 core
        assert a["fc"][mid] <= 0.05, f"taper={taper_on}"
        assert base["fc"][mid] > 0.5, f"taper={taper_on}"
        assert abs(a["fc"][0] - base["fc"][0]) < 0.05
        assert abs(a["fc"][-1] - base["fc"][-1]) < 0.05
        assert a["cleared_regions"] == [[2, 2]]
        assert any("cleared" in n_ for n_ in a["notes"])


def test_cleared_region_validation():
    lat, res = _dc_lebt_run()
    with pytest.raises(ValueError, match="outside"):
        scc_analysis(res, lat, cleared_regions=[[2, 99]])
    with pytest.raises(ValueError, match="outside"):
        scc_analysis(res, lat, cleared_regions=[[3, 2]])
    with pytest.raises(ValueError, match="cleared_residual"):
        scc_analysis(res, lat, cleared_regions=[[2, 2]],
                     cleared_residual=1.5)
    with pytest.raises(ValueError, match="pair"):
        scc_analysis(res, lat, cleared_regions=["oops"])
    a = scc_analysis(res, None, cleared_regions=[[2, 2]])
    assert a["reason"] is not None and "lattice" in a["reason"]


def test_cleared_region_ndarray_exit_idx_and_unresolved_note():
    """(a) element_exit_idx as an ndarray (loaded-HDF5 SimpleNamespace
    shape) must not crash the element mapping (adversarial finding:
    `or []` on an ndarray raises); (b) a coarse one-record-per-element
    grid flags the region UNRESOLVED instead of claiming success."""
    lat, res = _dc_lebt_run()

    class NdArrayStyle:
        continuous_at = [True] * len(res.s)
        s = res.s
        sigma_x = res.sigma_x
        sigma_y = res.sigma_y
        ref_w_kin = res.ref_w_kin
        transmission = np.full(len(res.s), 100.0)
        element_exit_idx = np.asarray(res.element_exit_idx)
        current_mA = 15.0

    a = scc_analysis(NdArrayStyle(), lat, species="H-", gas="N2",
                     pressure_mbar=2e-5, cleared_regions=[[2, 2]])
    assert a["reason"] is None
    assert a["cleared_resolved"] is False          # coarse grid: honest
    assert any("UNRESOLVED" in n_ for n_ in a["notes"])


def test_cleared_region_refuses_dense_grid_without_element_map():
    """Substep-dense records with NO element_exit_idx: the identity
    fallback would mislocate the region — refuse in-band instead."""
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.particle import H_MINUS
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat, _ = _dc_lebt_run()
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5)
    init = dict(alpha_x=0.0, beta_x=0.6, emit_x=3.0,
                alpha_y=0.0, beta_y=0.6, emit_y=3.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref, init, current=15.0,
                         record_substeps=True).run()

    class NoMap:
        continuous_at = [True] * len(res.s)
        s = res.s
        sigma_x = res.sigma_x
        sigma_y = res.sigma_y
        ref_w_kin = res.ref_w_kin
        transmission = np.full(len(res.s), 100.0)
        current_mA = 15.0

    a = scc_analysis(NoMap(), lat, species="H-", gas="N2",
                     pressure_mbar=2e-5, cleared_regions=[[2, 2]])
    assert a["reason"] is not None
    assert "element_exit_idx" in a["reason"]


def test_cleared_region_rejects_fractional_indices():
    lat, res = _dc_lebt_run()
    with pytest.raises(ValueError, match="whole"):
        scc_analysis(res, lat, cleared_regions=[[1.9, 2]])
