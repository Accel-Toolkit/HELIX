# tests/diagnostics/test_beam_summary.py
"""summarize_particles — the headless engine behind the phase-space
popup's beam-parameters table."""
import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.diagnostics.beam_summary import summarize_particles
from linac_gen.diagnostics.eigenemittance import eigenemittances
from linac_gen.diagnostics.moments import (
    compute_halo, compute_moments, compute_twiss_from_particles,
)
from linac_gen.diagnostics.recorder import _convert_emit_z_to_mmmrad


def _beam(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 6)) * [2.0, 1.5, 1.8, 1.2, 5.0, 0.05]
    q[:, 1] += 0.3 * q[:, 0]          # x-x' correlation -> alpha != 0
    q[:, 5] -= 0.004 * q[:, 4]        # phi-W correlation
    return q


def _ref(w=100.0):
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=352.21)


def _value(rows, name):
    vals = [v for (_g, n, v, _u) in rows if n == name]
    assert vals, f"row '{name}' missing"
    return vals[0]


def test_values_match_independent_recomputation():
    q = _beam()
    ref = _ref()
    rows = summarize_particles(q, ref, n_total=20000, current_ma=2.0,
                               location="TEST · s=1.000 m")
    mom = compute_moments(q)
    assert float(_value(rows, "σ_x")) == pytest.approx(
        mom["sigma_x"], rel=1e-4)
    assert float(_value(rows, "⟨x'⟩")) == pytest.approx(
        float(mom["mean"][1]), rel=1e-3)
    twx = compute_twiss_from_particles(q, "x")
    assert float(_value(rows, "α_x")) == pytest.approx(
        twx["alpha"], rel=1e-4)
    assert float(_value(rows, "β_x")) == pytest.approx(
        twx["beta"], rel=1e-4)
    assert float(_value(rows, "ε_x (geometric)")) == pytest.approx(
        twx["emittance"], rel=1e-4)
    assert float(_value(rows, "H_y")) == pytest.approx(
        compute_halo(q, "y"), rel=1e-4)
    e1, e2, e3 = eigenemittances(mom["sigma_matrix"])
    assert float(_value(rows, "ε₁ (eigen)")) == pytest.approx(e1, rel=1e-4)
    assert float(_value(rows, "max|Δφ|")) == pytest.approx(
        float(np.max(np.abs(q[:, 4]))), rel=1e-4)
    # transmission row present with n_total
    assert float(_value(rows, "transmission")) == pytest.approx(100.0)


def test_z_conversions_and_normalized_match_formulas():
    q = _beam()
    ref = _ref()
    rows = summarize_particles(q, ref)
    twz = compute_twiss_from_particles(q, "z")
    ez_mm = _convert_emit_z_to_mmmrad(twz["emittance"], ref)
    assert float(_value(rows, "ε_z")) == pytest.approx(
        twz["emittance"], rel=1e-4)
    assert float(_value(rows, "ε_z (geometric)")) == pytest.approx(
        ez_mm, rel=1e-4)
    assert float(_value(rows, "ε_nx (normalized)")) == pytest.approx(
        compute_twiss_from_particles(q, "x")["emittance"] * ref.bg,
        rel=1e-4)
    # sigma_z = sigma_phi/360 * beta*lambda
    mom = compute_moments(q)
    lam = ref.wavelength
    assert float(_value(rows, "σ_z")) == pytest.approx(
        mom["sigma_phi"] / 360.0 * ref.beta * lam, rel=1e-4)
    # sigma_delta = sigma_W / (beta^2 gamma m)
    assert float(_value(rows, "σ_δ (Δp/p)")) == pytest.approx(
        mom["sigma_w"] / (ref.beta**2 * ref.gamma * PROTON.mass),
        rel=1e-4)


def test_ref_none_omits_ref_rows_but_core_present():
    rows = summarize_particles(_beam(), None)
    names = [n for (_g, n, _v, _u) in rows]
    for absent in ("W_kin", "σ_z", "ε_nx (normalized)",
                   "mean kinetic energy", "RF frequency"):
        assert absent not in names
    for present in ("σ_x", "α_z", "ε_z", "H_x", "max|x|"):
        assert present in names


def test_empty_and_none_particles_safe():
    for p in (None, np.zeros((0, 6))):
        rows = summarize_particles(p, _ref(), location="L")
        assert ("Distribution", "status", "no particle data", "") in rows
        assert len(rows) <= 4


def test_species_rows_from_ref():
    rows = summarize_particles(_beam(), _ref())
    assert _value(rows, "species") == "proton"
    assert float(_value(rows, "rest mass")) == pytest.approx(
        PROTON.mass, rel=1e-6)
    assert _value(rows, "charge state") == "+1"


def test_eigen_rows_use_combined_units_never_per_slot():
    """eigenemittances() is MAGNITUDE-ordered, not mode-identified: for
    a beam whose eps_z (deg.MeV) is numerically largest, the largest
    eigen slot IS the longitudinal mode — per-slot units ('mm.mrad' on
    eps1/eps2, 'deg.MeV' on eps3) would mislabel it (adversarial-review
    finding).  All eigen rows must carry the combined-unit hedge."""
    rng = np.random.default_rng(5)
    # eps_z >> transverse: sigma_phi=20 deg, sigma_W=0.5 MeV -> ~10 deg.MeV
    q = rng.standard_normal((30000, 6)) * [2.0, 1.0, 1.0, 1.0, 20.0, 0.5]
    rows = summarize_particles(q, _ref())
    eigen = [(n, u) for (_g, n, _v, u) in rows if "(eigen)" in n]
    assert len(eigen) == 3
    for _n, unit in eigen:
        assert "mm·mrad" in unit and "deg·MeV" in unit, unit


def test_single_particle_no_crash():
    q = np.array([[1.0, 0.5, -1.0, 0.2, 3.0, 0.01]])
    rows = summarize_particles(q, _ref())
    # sigma of one particle is 0; Twiss guards return zeros — no NaN
    for (_g, n, v, _u) in rows:
        if n and v not in ("proton",) and "/" not in v:
            try:
                assert np.isfinite(float(v)), (n, v)
            except ValueError:
                pass                     # non-numeric label rows
