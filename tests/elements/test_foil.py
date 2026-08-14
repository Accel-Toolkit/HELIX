"""Foil element: Highland multiple-Coulomb scattering + Bethe-Bloch loss.

Covers the physics (Highland RMS angle, mean energy loss, straggling) plus
the parser/writer round-trip through the ``; HELIX_FOIL …`` comment.
"""
import math
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.foil import Foil, _MATERIALS, supported_materials


def _make_beam(species, w_kin_MeV, n=10_000, seed=0):
    """A monochromatic, on-axis beam suitable for measuring kick statistics."""
    ref = ReferenceParticle(species=species, w_kin=w_kin_MeV, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    # all (x, xp, y, yp, dphi, dw) start at zero — we measure the post-kick
    # spread directly.
    return beam


# ── construction / validation ────────────────────────────────────────────────

def test_default_construction_succeeds():
    foil = Foil(name="STRIP", material="C", thickness_ug_cm2=600.0)
    assert foil.material == "C"
    assert foil.thickness_ug_cm2 == 600.0
    assert foil.length == 0.0    # ThinKickElement contract
    assert foil.n_steps == 0
    assert foil.straggling == "auto"   # kappa-regime dispatch by default


def test_unsupported_material_rejected():
    with pytest.raises(ValueError, match="not in supported set"):
        Foil(name="STRIP", material="Unobtainium", thickness_ug_cm2=600.0)


def test_unsupported_straggling_rejected():
    with pytest.raises(ValueError, match="straggling"):
        Foil(name="STRIP", material="C", thickness_ug_cm2=600.0,
             straggling="vavilov")


def test_supported_materials_lists_all_table_entries():
    assert set(supported_materials()) == set(_MATERIALS.keys())


# ── Highland scattering ──────────────────────────────────────────────────────

def test_highland_theta_rms_matches_textbook_for_carbon_at_800MeV():
    """Verify the leading 13.6 MeV / βcp · √(x/X0) · log_term formula."""
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=1)

    # Compute expected θ_rms by the same formula in the test for transparency
    p_MeV = beam.ref.bg * beam.ref.species.mass
    beta = beam.ref.beta
    x_X0 = (600.0e-6) / _MATERIALS["C"]["X0_g_cm2"]
    expected = (13.6 / (beta * p_MeV)) * math.sqrt(x_X0) * (
        1.0 + 0.038 * math.log(x_X0)
    )
    measured = foil._highland_theta_rms(beam)
    assert measured == pytest.approx(expected, rel=1e-12)


def test_apply_kick_yields_correct_transverse_angular_spread():
    """Statistical: σ(xp) after kick ≈ Highland θ_rms (within 5 % at N=10000)."""
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=10_000)
    theta_rms_rad = foil._highland_theta_rms(beam)

    foil.apply_kick(beam)

    # xp / yp are stored in mrad in HELIX's particle layout.
    measured_xp_mrad = float(np.std(beam.particles[:, 1], ddof=1))
    measured_yp_mrad = float(np.std(beam.particles[:, 3], ddof=1))
    expected_mrad = theta_rms_rad * 1e3

    assert measured_xp_mrad == pytest.approx(expected_mrad, rel=0.05)
    assert measured_yp_mrad == pytest.approx(expected_mrad, rel=0.05)


def test_apply_kick_preserves_zero_centroid_in_xp_yp():
    """No DC offset on the scattering — only stochastic spread."""
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=123)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=10_000)
    foil.apply_kick(beam)
    # Mean drift is √(σ/N) ≈ θ_rms / √10000 ≈ 0.01 × θ_rms.
    theta_rms_mrad = foil._highland_theta_rms(beam) * 1e3
    assert abs(np.mean(beam.particles[:, 1])) < 0.05 * theta_rms_mrad
    assert abs(np.mean(beam.particles[:, 3])) < 0.05 * theta_rms_mrad


def test_zero_thickness_has_no_effect():
    foil = Foil(name="F", material="C", thickness_ug_cm2=0.0, seed=7)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100)
    beam.particles[:, 1] = 1.23  # arbitrary non-zero baseline
    foil.apply_kick(beam)
    assert np.all(beam.particles[:, 1] == 1.23)
    assert np.all(beam.particles[:, 5] == 0.0)


# ── Bethe-Bloch energy loss ──────────────────────────────────────────────────

def test_mean_energy_loss_matches_table():
    """⟨ΔE⟩ = dE/dx × thickness × |z|², checked against the table value."""
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=1)
    expected = _MATERIALS["C"]["dEdx_min_MeVcm2_g"] * 600.0e-6 * 1.0**2
    measured = foil._mean_energy_loss_MeV(beam)
    assert measured == pytest.approx(expected, rel=1e-12)


def test_apply_kick_subtracts_energy_loss_on_average():
    """⟨dw⟩ after kick ≈ −⟨ΔE⟩ (Gaussian/Bohr mode — back-compat path).

    The paper foil (C 600 μg/cm², 800 MeV) sits deep in the Landau
    regime, so ``straggling="gaussian"`` is forced here to pin the
    legacy Bohr-Gaussian behaviour; Landau-mode statistics are covered
    in the Landau section below.
    """
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42,
                straggling="gaussian")
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=10_000)
    dE_mean = foil._mean_energy_loss_MeV(beam)
    sigma_dE = foil._energy_loss_sigma_MeV(beam)

    foil.apply_kick(beam)

    mean_dw = float(np.mean(beam.particles[:, 5]))
    std_dw  = float(np.std(beam.particles[:, 5], ddof=1))
    # Mean within 3·σ/√N of −dE_mean
    sem = sigma_dE / math.sqrt(10_000)
    assert mean_dw == pytest.approx(-dE_mean, abs=3.0 * sem + 1e-9)
    # Straggling width matches Bohr formula to 5 %
    if sigma_dE > 0:
        assert std_dw == pytest.approx(sigma_dE, rel=0.05)


# ── Landau straggling (thin-absorber regime) ─────────────────────────────────
#
# Paper case: C 600 μg/cm², 800 MeV H⁻ → ξ ≈ 0.0649 keV, T_max ≈ 2.481 MeV,
# κ = ξ/T_max ≈ 2.6e-5 — deep Landau regime.  What apply_kick samples there
# (see Foil._sample_energy_loss_MeV): λ ~ scipy standard Landau truncated at
# λ_max = mode + (T_max − ΔE_mp)/ξ_eff, ΔE = ⟨ΔE⟩_BB + ξ_eff(λ − ⟨λ⟩_trunc)
# with ξ_eff = (π/2)·ξ (scipy's landau is 2/π narrower than Landau's φ) —
# ensemble mean pinned to Bethe-Bloch by construction, shape (mode / FWHM
# 4.02ξ / Rutherford ξ/E tail) Landau.

def test_kappa_deep_landau_for_paper_foil():
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=1)
    k = foil.kappa(beam)
    assert k == pytest.approx(2.62e-5, rel=0.02)
    assert foil.resolved_straggling_model(beam) == "landau"


def test_explicit_straggling_choice_wins_over_dispatch():
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=1)
    gauss = Foil(name="F", material="C", thickness_ug_cm2=600.0,
                 straggling="gaussian")
    assert gauss.resolved_straggling_model(beam) == "gaussian"
    lan = Foil(name="F", material="C", thickness_ug_cm2=600.0,
               straggling="landau")
    assert lan.resolved_straggling_model(beam) == "landau"


def test_dispatch_thick_absorber_resolves_gaussian():
    """κ > 10 (Bohr regime): 0.2 g/cm² W at 5 MeV → Gaussian, no warning."""
    foil = Foil(name="F", material="W", thickness_ug_cm2=200_000.0)
    beam = _make_beam(species=PROTON, w_kin_MeV=5.0, n=1)
    assert foil.kappa(beam) > 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert foil.resolved_straggling_model(beam) == "gaussian"


def test_dispatch_vavilov_regime_warns_and_uses_gaussian():
    """0.01 ≤ κ ≤ 10: Vavilov regime — not implemented, documented fallback."""
    foil = Foil(name="F", material="W", thickness_ug_cm2=600.0)
    beam = _make_beam(species=PROTON, w_kin_MeV=5.0, n=1)
    k = foil.kappa(beam)
    assert 0.01 <= k <= 10.0
    with pytest.warns(UserWarning, match="Vavilov"):
        assert foil.resolved_straggling_model(beam) == "gaussian"


def test_forced_landau_outside_regime_raises():
    """straggling='landau' far outside validity must fail loudly."""
    foil = Foil(name="F", material="W", thickness_ug_cm2=200_000.0,
                seed=1, straggling="landau")
    beam = _make_beam(species=PROTON, w_kin_MeV=5.0, n=100)
    with pytest.raises(ValueError, match="thin-absorber"):
        foil.apply_kick(beam)


def test_landau_mean_pinned_to_bethe_bloch():
    """Ensemble mean loss = tabulated Bethe-Bloch mean, by construction.

    The truncated Landau's sample mean is tail-dominated (sample std
    ≈ 10–13 keV for this foil, seed-dependent), so the tolerance is set
    from that width: ~4·(12 keV/√N) ≈ 0.15 keV on a 1.047 keV mean.
    """
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    dE_mean = foil._mean_energy_loss_MeV(beam)
    foil.apply_kick(beam)
    mean_dw = float(np.mean(beam.particles[:, 5]))
    assert mean_dw == pytest.approx(-dE_mean, abs=0.15e-3)


def test_landau_most_probable_loss_below_mean_right_skew():
    """The pathology fix, part 1: mode < mean (right-skewed loss).

    Checks three mutually consistent things at the paper parameters:
    the analytic mode of the sampled construction, the histogram mode
    of an actual 100k sample, and the median — all below the mean.
    """
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    dE_mean = foil._mean_energy_loss_MeV(beam)      # MeV
    mp_model = foil.most_probable_loss_MeV(beam)    # MeV

    foil.apply_kick(beam)
    loss_keV = -beam.particles[:, 5] * 1e3          # positive = loss

    # analytic mode of the sampled construction sits well below the mean
    assert mp_model < 0.75 * dE_mean
    # histogram mode (50 eV bins over the core) reproduces it
    bins = np.arange(-1.0, 3.0, 0.05)
    hist, edges = np.histogram(loss_keV, bins=bins)
    mode_keV = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    assert mode_keV == pytest.approx(mp_model * 1e3, abs=0.1)
    # right skew: median below mean too
    assert np.median(loss_keV) < dE_mean * 1e3


def test_landau_mode_bookkeeping_and_bethe_consistency():
    """Two checks on the mean-pinned construction's bookkeeping.

    (1) Identity: pinning shifts the whole curve, so the sampled mode
        differs from the standalone ΔE_mp parametrization by exactly
        (raw truncated mean − ⟨ΔE⟩_BB)  [0.253 keV for this foil].
    (2) Physics: the RAW truncated construction's mean,
        ΔE_mp + ξ_eff(⟨λ⟩_trunc − λ_mode), must reproduce the full
        Bethe mean ξ[ln(2m_ec²β²γ²T_max/I²) − 2β²] (δ=0) to O(ξ_eff) —
        Landau theory's internal consistency (measured 0.45·ξ_eff).
        For this foil the tabulated-MIP ⟨ΔE⟩_BB is ~0.25 keV below the
        full Bethe value at 800 MeV; the pinning absorbs that offset
        by design.
    """
    import math as m
    from linac_gen.core.constants import M_ELECTRON
    from linac_gen.elements.foil import (_LANDAU_MODE, _MATERIALS,
                                         _landau_truncated_mean)

    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=1)
    ref = beam.ref
    xi = foil._landau_xi_MeV(ref)
    scale = foil._landau_scale_MeV(ref)
    assert scale == pytest.approx(xi * m.pi / 2.0, rel=1e-12)
    BB = foil._mean_energy_loss_MeV(ref)
    dE_mp = foil._landau_delta_e_mp_MeV(ref)
    mp_model = foil.most_probable_loss_MeV(ref)
    lam_mean = _landau_truncated_mean(round(foil._landau_lam_max(ref), 3))
    raw_mean = dE_mp + scale * (lam_mean - _LANDAU_MODE)
    # (1) algebraic identity of the pinning
    assert dE_mp - mp_model == pytest.approx(raw_mean - BB, rel=1e-9)
    # (2) raw construction reproduces the full Bethe mean to O(ξ_eff)
    I_MeV = _MATERIALS["C"]["I_eV"] * 1e-6
    bethe = xi * (m.log(2 * M_ELECTRON * ref.bg ** 2
                        * foil._t_max_MeV(ref) / I_MeV ** 2)
                  - 2 * ref.beta ** 2)
    assert abs(raw_mean - bethe) < 2.0 * scale


def test_landau_fwhm_matches_canonical_4xi():
    """Core width regression: sampled FWHM = 4.02·ξ (0.261 keV here).

    This pins the scipy-convention scale factor ξ_eff = (π/2)·ξ — with
    unit scale the core comes out 2/π too narrow (0.166 keV), which is
    exactly the bug this test exists to catch.
    """
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    xi_keV = foil._landau_xi_MeV(beam) * 1e3
    foil.apply_kick(beam)
    loss_keV = -beam.particles[:, 5] * 1e3
    bins = np.arange(-0.5, 2.0, 0.02)
    h, e = np.histogram(loss_keV, bins=bins)
    c = 0.5 * (e[:-1] + e[1:])
    above = c[h >= 0.5 * h.max()]
    fwhm = float(above.max() - above.min())
    assert fwhm == pytest.approx(4.018 * xi_keV, rel=0.15)


def test_landau_rutherford_tail_amplitude():
    """Tail regression: P(loss > E) ≈ ξ/(E−⟨ΔE⟩) − ξ/T_max (single-
    collision Rutherford spectrum) for ⟨ΔE⟩ ≪ E ≪ T_max.  Checked at
    E = 45 keV: expectation 1.45e-3 (binomial σ ≈ 0.12e-3 at N=1e5);
    the unit-scale scipy convention would give 2/π of this."""
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    xi = foil._landau_xi_MeV(beam) * 1e3            # keV
    t_max = foil._t_max_MeV(beam) * 1e3             # keV
    BB = foil._mean_energy_loss_MeV(beam) * 1e3     # keV
    foil.apply_kick(beam)
    loss_keV = -beam.particles[:, 5] * 1e3
    expected = xi / (45.0 - BB) - xi / t_max
    measured = float(np.mean(loss_keV > 45.0))
    assert measured == pytest.approx(expected, rel=0.30)


def test_landau_gain_fraction_negligible():
    """The pathology fix, part 2: essentially no net energy GAIN.

    The Gaussian (Bohr) model gives ~44 % of particles a net gain at
    the paper parameters (σ_E = 6.8 keV ≫ ⟨ΔE⟩ = 1.05 keV); the Landau
    left tail decays double-exponentially, so a gain needs λ < −4.1
    (P ≈ 1e-66): the sampled gain fraction must be < 0.5 % (it is 0).
    """
    beam_l = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    Foil(name="F", material="C", thickness_ug_cm2=600.0,
         seed=42).apply_kick(beam_l)
    gain_frac_landau = float(np.mean(beam_l.particles[:, 5] > 0.0))
    assert gain_frac_landau < 0.005

    beam_g = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42,
         straggling="gaussian").apply_kick(beam_g)
    gain_frac_gauss = float(np.mean(beam_g.particles[:, 5] > 0.0))
    assert gain_frac_gauss > 0.30   # the documented pathology


def test_landau_losses_bounded_by_tmax():
    """Kinematic cutoff: no sampled loss above T_max (+ pinning shift)."""
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0, seed=42)
    beam = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100_000)
    t_max = foil._t_max_MeV(beam)
    foil.apply_kick(beam)
    max_loss = float(-beam.particles[:, 5].min())
    assert max_loss <= t_max + 0.01   # pinning shift ≪ 10 keV


def test_landau_mode_constant_matches_scipy():
    """_LANDAU_MODE is the argmax of scipy's standard landau pdf."""
    from scipy.stats import landau
    from linac_gen.elements.foil import _LANDAU_MODE
    p0 = landau.pdf(_LANDAU_MODE)
    assert p0 >= landau.pdf(_LANDAU_MODE - 0.01)
    assert p0 >= landau.pdf(_LANDAU_MODE + 0.01)


# ── reproducibility ──────────────────────────────────────────────────────────

def test_seeded_foil_is_reproducible():
    """Same seed + same beam → identical kicks across two runs."""
    beam1 = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100)
    beam2 = _make_beam(species=H_MINUS, w_kin_MeV=800.0, n=100)
    Foil(name="A", material="C", thickness_ug_cm2=600.0, seed=99).apply_kick(beam1)
    Foil(name="A", material="C", thickness_ug_cm2=600.0, seed=99).apply_kick(beam2)
    assert np.allclose(beam1.particles, beam2.particles)


# ── kick_matrix / matrix-solver contract ─────────────────────────────────────

def test_kick_matrix_is_identity():
    foil = Foil(name="F", material="C", thickness_ug_cm2=600.0)
    ref = ReferenceParticle(species=PROTON, w_kin=800.0, frequency=162.5)
    assert np.allclose(foil.kick_matrix(ref), np.eye(6))


# ── parser / writer round-trip ───────────────────────────────────────────────

def test_dat_round_trip_preserves_foil(tmp_path):
    """Lattice with a Foil round-trips through write_tracewin → parse_tracewin."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    src = Lattice()
    src.add(Drift(name="D1", length=100.0, aperture=10.0))
    src.add(Foil(name="STRIP", material="C", thickness_ug_cm2=600.0))
    src.add(Drift(name="D2", length=50.0, aperture=10.0))

    dat_path = tmp_path / "foil_lattice.dat"
    write_tracewin(src, str(dat_path))

    text = dat_path.read_text()
    assert "; HELIX_FOIL STRIP C 600.0" in text

    parsed, _meta = parse_tracewin(str(dat_path))
    foils = [e for e in parsed.elements if isinstance(e, Foil)]
    assert len(foils) == 1
    assert foils[0].name == "STRIP"
    assert foils[0].material == "C"
    assert foils[0].thickness_ug_cm2 == 600.0


def test_parser_accepts_trailing_comment_after_foil_marker(tmp_path):
    """Authors should be able to annotate the foil marker line, e.g.
    ``; HELIX_FOIL STRIP C 600.0  ; PIP-II BTL->Booster stripper``.

    The trailing-comment branch was missing in the initial regex; this
    test pins it down so future edits don't regress."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "annotated_foil.dat"
    dat.write_text(
        "DRIFT 100.0 10.0\n"
        "; HELIX_FOIL STRIP C 600.0  ; PIP-II BTL->Booster charge-exchange foil\n"
        "DRIFT 50.0 10.0\n"
        "END\n"
    )
    parsed, _meta = parse_tracewin(str(dat))
    foils = [e for e in parsed.elements if isinstance(e, Foil)]
    assert len(foils) == 1
    assert foils[0].material == "C"
    assert foils[0].thickness_ug_cm2 == 600.0


def test_dat_round_trip_preserves_non_default_straggling(tmp_path):
    """A forced straggling model survives write → parse; the default
    ("auto") is omitted from the file so legacy lines stay identical."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    src = Lattice()
    src.add(Drift(name="D1", length=100.0, aperture=10.0))
    src.add(Foil(name="STRIP", material="C", thickness_ug_cm2=600.0,
                 straggling="gaussian"))

    dat_path = tmp_path / "foil_gaussian.dat"
    write_tracewin(src, str(dat_path))
    assert "; HELIX_FOIL STRIP C 600.0 gaussian" in dat_path.read_text()

    parsed, _meta = parse_tracewin(str(dat_path))
    foils = [e for e in parsed.elements if isinstance(e, Foil)]
    assert len(foils) == 1
    assert foils[0].straggling == "gaussian"
