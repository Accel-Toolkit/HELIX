"""Frozen-SC incoherent tune footprint (analysis/footprint.py).

The footprint tracks a sparse amplitude ladder repeatedly through one
period cell with the space-charge field FROZEN from a reference pass of
the matched Gaussian FIELD beam (never the sparse ladder's own std).
The robust, physical claims pinned here:

* I = 0 → every particle has the bare tune, zero spread;
* I > 0 → a real amplitude-dependent spread with the ordering
  core < rms channel tune < bare (the Gaussian core sees ~2× the
  rms-equivalent gradient, so the core is MORE depressed than the rms
  channel value — a core≈centroid identity is NOT claimed);
* tune rises monotonically with amplitude (defocusing SC: core most
  depressed, tails approach the bare tune);
* the frozen field is independent of the probe ladder — the
  smallest-amplitude particle's tune does not change when the aperture
  (which reshapes the ladder's survival but not the matched field) does.

Exact tune values near strong depression sit close to the FFT
resolution floor (~360/n_turns per cell) and are only pinned loosely.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis.footprint import tune_footprint
from linac_gen.analysis.period_detect import detect_periods
from linac_gen.analysis.phase_advance import channel_phase_advance
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver

_CURRENT = 15.0  # mA — depresses the FODO tune to roughly 2/3.
_NT = 128
_NP = 36
_AMP = 2.0


def _fodo(aperture: float = 20.0) -> Lattice:
    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=aperture))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=aperture))
        lat.add(Drift(name="D", length=100.0, aperture=aperture))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=aperture))
    return lat


def _initial(lat: Lattice, ref: ReferenceParticle) -> dict:
    from linac_gen.matching.periodic import find_periodic_twiss
    tw = find_periodic_twiss(lat, ref)
    return dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)


def _channel_bare_dep(lat, ref, init, current):
    res = EnvelopeSolver(lat, ref.copy(), init, current=current,
                         phase_probe=True).run()
    ch = channel_phase_advance(res, detect_periods(lat)[0])
    return (float(np.nanmedian(ch["mu_x_bare_deg"])),
            float(np.nanmedian(ch["mu_x_dep_deg"])))


@pytest.fixture(scope="module")
def ref():
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


@pytest.fixture(scope="module")
def fp_current(ref):
    lat = _fodo()
    init = _initial(lat, ref)
    period = detect_periods(lat)[0]
    fp = tune_footprint(lat, ref.copy(), period, init, current=_CURRENT,
                        n_turns=_NT, n_particles=_NP, amp_max_sigma=_AMP,
                        seed=0)
    bare, dep = _channel_bare_dep(lat, ref, init, _CURRENT)
    return fp, bare, dep


@pytest.fixture(scope="module")
def fp_zero(ref):
    lat = _fodo()
    init = _initial(lat, ref)
    period = detect_periods(lat)[0]
    fp = tune_footprint(lat, ref.copy(), period, init, current=0.0,
                        n_turns=_NT, n_particles=_NP, amp_max_sigma=_AMP,
                        seed=0)
    bare, _ = _channel_bare_dep(lat, ref, init, 0.0)
    return fp, bare


def test_zero_current_is_a_point(fp_zero):
    """No space charge ⇒ zero spread, tune == bare channel tune."""
    fp, bare = fp_zero
    assert fp["mu_x_spread_pp_deg"] < 1.0e-6
    assert fp["mu_y_spread_pp_deg"] < 1.0e-6
    assert fp["mu_x_core_deg"] == pytest.approx(bare, abs=0.5)


def test_all_tunes_finite(fp_current):
    """Runaway particles are retired cleanly; the surviving ladder still
    yields finite tunes for (nearly) every launched particle."""
    fp, _, _ = fp_current
    finite = np.isfinite(fp["qx"]).sum()
    # 0.8, not higher: the outer ladder rungs sit near the chaotic
    # boundary of the resonant scenario, and ulp-level environment
    # differences (numpy/BLAS builds, FFT worker scheduling) legitimately
    # decide their survival.  The assertion pins the MECHANISM — clean
    # retirement to NaN, finite tunes for the surviving majority.
    assert finite >= 0.8 * fp["n_particles"]


def test_core_below_channel_below_bare(fp_current):
    """Gaussian core sees ~2× the rms gradient ⇒ core < rms dep < bare."""
    fp, bare, dep = fp_current
    core = fp["mu_x_core_deg"]
    assert 0.0 < core < dep < bare
    assert dep < bare - 2.0            # the current genuinely depresses


def test_no_tune_exceeds_bare(fp_current):
    """Space charge only DEPRESSES — no particle tune sits above bare."""
    fp, bare, _ = fp_current
    qmax_deg = np.nanmax(fp["qx"]) * 360.0
    assert qmax_deg <= bare + 1.0


def test_real_spread(fp_current):
    """A finite, physically sizeable footprint width exists."""
    fp, _, _ = fp_current
    assert fp["mu_x_spread_pp_deg"] > 2.0
    assert fp["mu_x_spread_rms_deg"] > 0.5


def test_amplitude_tune_correlation_positive(fp_current):
    """Defocusing SC ⇒ tune increases with amplitude (core most
    depressed).  Strong positive correlation on the x-ray subset."""
    fp, _, _ = fp_current
    ax, ay, qx = fp["ax_sigma"], fp["ay_sigma"], fp["qx"]
    xray = (ax > 0) & (ay == 0) & np.isfinite(qx)
    assert xray.sum() >= 5
    corr = float(np.corrcoef(ax[xray], qx[xray])[0, 1])
    assert corr > 0.8


def test_field_independent_of_probe_ladder(ref):
    """The frozen field comes from the matched Gaussian field beam, not
    the probe ladder — so the smallest-amplitude particle sees an
    identical field whether the aperture keeps the tails or not."""
    def smallest_tune(aperture):
        lat = _fodo(aperture)
        init = _initial(lat, ref)
        period = detect_periods(lat)[0]
        fp = tune_footprint(lat, ref.copy(), period, init, current=_CURRENT,
                            n_turns=_NT, n_particles=_NP, amp_max_sigma=_AMP,
                            seed=0)
        i = int(np.argmin(fp["ax_sigma"] + fp["ay_sigma"]))
        return fp["qx"][i]

    q_tight = smallest_tune(10.0)
    q_wide = smallest_tune(50.0)
    assert q_tight == pytest.approx(q_wide, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# Line-density convention of the frozen kicker (strength consistency with
# the matched Σ): DC λ = I/(βc); bunched λ = Q/(√2π·σ_z) at the bunch
# center; SPACE_CHARGE_COMP factor scales the kick; σ_φ = 0 falls back
# to the DC density.
# ---------------------------------------------------------------------------

def _kick_beam(ref, sigma_phi_deg=0.0, current=5.0, n=400):
    from linac_gen.core.beam import Beam
    rng = np.random.default_rng(1)
    beam = Beam(ref.copy(), n_particles=n, current=current)
    beam.particles[:, 0] = rng.normal(0.0, 1.0, n)      # mm
    beam.particles[:, 2] = rng.normal(0.0, 1.2, n)      # mm
    if sigma_phi_deg > 0:
        phi = rng.normal(0.0, sigma_phi_deg, n)
        phi *= sigma_phi_deg / max(np.std(phi), 1e-30)  # exact σ_φ
        beam.particles[:, 4] = phi
    return beam


def test_kicker_dc_line_density(ref):
    from linac_gen.analysis.footprint import FrozenGaussianKicker
    from linac_gen.core.constants import C_LIGHT
    k = FrozenGaussianKicker(continuous=True, f_bunch_MHz=162.5)
    beam = _kick_beam(ref, sigma_phi_deg=5.0)
    k.reset("record")
    k.kick(beam, 10.0)
    lam = k.tape[0][2]
    assert lam == pytest.approx(
        5.0e-3 / (float(ref.beta) * C_LIGHT), rel=1e-12)


def test_kicker_bunched_center_slice_density(ref):
    from linac_gen.analysis.footprint import FrozenGaussianKicker
    from linac_gen.core.constants import C_LIGHT
    sig_phi = 5.0
    k = FrozenGaussianKicker(continuous=False, f_bunch_MHz=162.5)
    beam = _kick_beam(ref, sigma_phi_deg=sig_phi)
    k.reset("record")
    k.kick(beam, 10.0)
    lam = k.tape[0][2]
    beta = float(ref.beta)
    wavelength_m = C_LIGHT / (162.5e6)
    sigma_z_m = sig_phi * beta * wavelength_m / 360.0
    Q = 5.0e-3 / 162.5e6
    lam_ref = Q / (np.sqrt(2.0 * np.pi) * sigma_z_m)
    assert lam == pytest.approx(lam_ref, rel=1e-9)
    # And it exceeds the DC average by the bunching factor.
    assert lam > 3.0 * 5.0e-3 / (beta * C_LIGHT)


def test_kicker_zero_sigphi_falls_back_to_dc(ref):
    from linac_gen.analysis.footprint import FrozenGaussianKicker
    from linac_gen.core.constants import C_LIGHT
    k = FrozenGaussianKicker(continuous=False, f_bunch_MHz=162.5)
    beam = _kick_beam(ref, sigma_phi_deg=0.0)      # no z information
    k.reset("record")
    k.kick(beam, 10.0)
    assert k.tape[0][2] == pytest.approx(
        5.0e-3 / (float(ref.beta) * C_LIGHT), rel=1e-12)


def test_kicker_sc_factor_scales_kick(ref):
    from linac_gen.analysis.footprint import FrozenGaussianKicker

    def deflection(sc_factor):
        k = FrozenGaussianKicker(continuous=True, sc_factor=sc_factor)
        beam = _kick_beam(ref)
        before = beam.particles[:, 1].copy()
        k.reset("record")
        k.kick(beam, 10.0)
        return beam.particles[:, 1] - before

    full = deflection(1.0)
    half = deflection(0.5)
    assert np.allclose(half, 0.5 * full, rtol=1e-12)
    assert np.abs(full).max() > 0
