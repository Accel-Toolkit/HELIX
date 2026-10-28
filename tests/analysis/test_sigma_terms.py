"""The 21 independent terms of Σ along a run (linac_gen.analysis.sigma_terms):
rms values and correlations from the recorded matrices, TraceWin's (z, δ)
basis row by row (frequency jumps included), and NaN wherever a term is
not defined (zero variance, no particles, continuous beam)."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis import sigma_terms as st
from linac_gen.core.constants import C_LIGHT
from linac_gen.tracking.longitudinal_coords import sigma_to_tracewin_custom

MASS = 939.29420
RMS = np.array([1.2, 2.5, 0.8, 1.7, 9.0, 4e-3])       # mm, mrad, mm, mrad, deg, MeV


def _sigma(rng, rms=RMS):
    """A positive-definite Σ with every correlation non-zero."""
    a = rng.normal(size=(6, 6))
    c = a @ a.T + 6 * np.eye(6)
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)
    return corr * np.outer(rms, rms)


class _Run:
    def __init__(self, mats, *, s=None, freq=162.5, beta=0.0667, gamma=1.0022,
                 **extra):
        n = len(mats)
        self.sigma_matrix = [np.asarray(m) for m in mats]
        self.s = list(np.linspace(0.0, 1000.0, n)) if s is None else list(s)
        self.ref_beta = [beta] * n
        self.ref_gamma = [gamma] * n
        self.ref_frequency = list(np.broadcast_to(freq, (n,)))
        self.mass_mev = MASS
        self.element_names = [f"E{k}" for k in range(n)]
        for k, v in extra.items():
            setattr(self, k, v)


def test_rms_and_correlations_of_known_matrices():
    rng = np.random.default_rng(1)
    mats = [_sigma(rng) for _ in range(4)]
    t = st.sigma_terms(_Run(mats))
    assert len(t) == 4 and t.basis == "ours" and t.converted
    np.testing.assert_allclose(t.s, np.linspace(0.0, 1.0, 4))       # mm → m
    for k, S in enumerate(mats):
        d = np.sqrt(np.diag(S))
        np.testing.assert_allclose(t.rms[k], d * [1, 1, 1, 1, 1, 1e3], rtol=1e-14)
        np.testing.assert_allclose(t.corr[k], S / np.outer(d, d), rtol=1e-12,
                                   atol=1e-15)
    assert len(st.TERMS) == 21 and len(st.COUPLING) == 12
    assert t.label(1, 0) == "r(x, x′)" and t.label(5, 5) == "σ(ΔW)"
    assert t.units[5] == "keV" and t.covariance_units(5, 0) == "mm·keV"


def test_tracewin_basis_row_by_row_across_a_frequency_jump():
    rng = np.random.default_rng(2)
    mats = [_sigma(rng) for _ in range(4)]
    freq = [162.5, 162.5, 325.0, 325.0]
    beta = 0.0667
    gamma = 1.0022
    run = _Run(mats, freq=np.array(freq), beta=beta, gamma=gamma)
    t = st.sigma_terms(run, "tracewin")
    ours = st.sigma_terms(run, "ours")
    scale = np.array([1, 1, 1, 1, 1e3, 1e3])
    for k, S in enumerate(mats):
        want = sigma_to_tracewin_custom(
            S, beta=beta, gamma=gamma, mass_MeV=MASS,
            wavelength_mm=C_LIGHT / (freq[k] * 1e6) * 1e3)
        np.testing.assert_allclose(t.sigma[k] / np.outer(scale, scale), want,
                                   rtol=1e-13)
    # same correlations, except that z = −k·Δφ flips the sign of the z ones
    sign = np.ones((6, 6)); sign[4, :] *= -1; sign[:, 4] *= -1
    np.testing.assert_allclose(t.corr, ours.corr * sign, rtol=1e-12, atol=1e-15)
    # σ_z in mm does not jump with the frequency; σ_Δφ in deg does (×2)
    run2 = _Run([mats[0]] * 2, freq=np.array([162.5, 325.0]), beta=beta, gamma=gamma)
    run2.sigma_matrix[1] = mats[0].copy()
    run2.sigma_matrix[1][4, :] *= 2.0
    run2.sigma_matrix[1][:, 4] *= 2.0
    tz = st.sigma_terms(run2, "tracewin")
    assert tz.rms[1, 4] == pytest.approx(tz.rms[0, 4], rel=1e-12)
    assert t.units[4] == "mm" and t.units[5] == "‰"


def test_undefined_terms_are_nan_not_zero():
    rng = np.random.default_rng(3)
    S = _sigma(rng)
    no_spread = S.copy(); no_spread[5, :] = 0.0; no_spread[:, 5] = 0.0
    lost = np.zeros((6, 6))
    t = st.sigma_terms(_Run([S, no_spread, lost]))
    assert t.rms[1, 5] == 0.0                         # a real zero spread
    assert np.isnan(t.corr[1, 5, :]).all() and np.isnan(t.corr[1, :, 5]).all()
    assert np.isfinite(t.corr[1, :5, :5]).all()
    assert np.isnan(t.rms[2]).all() and np.isnan(t.corr[2]).all()   # no particles
    assert np.isfinite(t.corr[0]).all()


def test_continuous_rows_blank_the_phase_terms():
    rng = np.random.default_rng(4)
    mats = [_sigma(rng) for _ in range(4)]
    t = st.sigma_terms(_Run(mats, continuous_at=[True, True, False, False]))
    assert list(t.continuous) == [True, True, False, False]
    assert np.isnan(t.rms[:2, 4]).all() and np.isnan(t.corr[:2, 4, :]).all()
    assert np.isfinite(t.rms[:2, 5]).all()            # ΔW stays: energy spread
    assert np.isfinite(t.corr[2:]).all()
    # envelope runs: one flag for the run, zero Δφ block until the first RF
    env = [m.copy() for m in mats]
    for m in env[:3]:
        m[4, :] = 0.0; m[:, 4] = 0.0; m[5, :] = 0.0; m[:, 5] = 0.0
    te = st.sigma_terms(_Run(env, continuous=True))
    assert list(te.continuous) == [True, True, True, False]
    assert np.isnan(te.rms[:3, 4]).all() and np.isfinite(te.rms[3, 4])
    # a bunched envelope run never has continuous rows
    assert not st.sigma_terms(_Run(env)).continuous.any()


def test_loaded_lists_equal_live_arrays_and_missing_data():
    rng = np.random.default_rng(5)
    mats = [_sigma(rng) for _ in range(3)]
    live = _Run(mats)
    loaded = _Run([m.tolist() for m in mats])         # a results file read back
    a, b = st.sigma_terms(live, "tracewin"), st.sigma_terms(loaded, "tracewin")
    np.testing.assert_array_equal(a.sigma, b.sigma)
    assert st.sigma_terms(None) is None
    assert st.sigma_terms(_Run([])) is None
    assert st.sigma_terms(_Run(mats, s=[0.0, 1.0])) is None      # not one per row
    bare = _Run(mats)
    del bare.ref_frequency
    tb = st.sigma_terms(bare, "tracewin")             # no reference: no z, δ
    assert not tb.converted and np.isnan(tb.sigma[:, 4:, :]).all()
    np.testing.assert_allclose(tb.rms[:, :4], a.rms[:, :4])
    with pytest.raises(ValueError):
        st.sigma_terms(live, "z")


def test_largest_coupling():
    S = np.diag(RMS ** 2).astype(float)
    S[0, 3] = S[3, 0] = 0.7 * RMS[0] * RMS[3]         # x–y′
    S[0, 1] = S[1, 0] = 0.95 * RMS[0] * RMS[1]        # in-plane: not coupling
    S2 = np.diag(RMS ** 2).astype(float)
    S2[0, 4] = S2[4, 0] = 0.9 * RMS[0] * RMS[4]       # x–Δφ on a continuous row
    out = st.largest_coupling(_Run([S, S2], continuous_at=[False, True]))
    assert out[0] == pytest.approx(0.7) and out[1] == 0.0
    assert st.largest_coupling(None) is None


def test_plane_summary():
    eps, alpha, beta = 0.25, -1.3, 0.8                 # mm·mrad, –, mm/mrad
    g = (1 + alpha ** 2) / beta
    S = np.diag(RMS ** 2).astype(float)
    S[0:2, 0:2] = eps * np.array([[beta, -alpha], [-alpha, g]])
    ez, az, bz = 0.044, 0.5, 2000.0                    # deg·MeV, –, deg/MeV
    gz = (1 + az ** 2) / bz
    S[4:6, 4:6] = ez * np.array([[bz, -az], [-az, gz]])
    t = st.sigma_terms(_Run([S]))
    x, _y, z = st.plane_summary(t, 0)
    assert (x["eps"], x["alpha"], x["beta"]) == pytest.approx((eps, alpha, beta))
    assert (z["eps"], z["alpha"], z["beta"]) == pytest.approx((ez, az, bz))
    assert z["eps_units"] == "deg·MeV" and z["units"] == ("deg", "keV")
    assert np.isnan(st.plane_twiss(1.0, 1.0, 1.0)[0])  # no area


def test_matches_the_particle_covariance_of_a_real_record():
    """Through the recorder: the correlations are those of the particles."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rng = np.random.default_rng(6)
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5),
             n_particles=4000, current=0.0)
    mix = np.eye(6) + 0.3 * rng.normal(size=(6, 6))
    b.particles[:] = (rng.normal(size=(4000, 6)) @ mix.T) * [1, 2, 1, 2, 5, 0.004]
    b.lost[:100] = True
    rec = DiagnosticRecorder()
    rec.record(b, 0.0)
    t = st.sigma_terms(rec)
    want = np.corrcoef(b.particles[~b.lost].T)
    np.testing.assert_allclose(t.corr[0], want, rtol=1e-9, atol=1e-12)
    assert np.abs(t.corr[0][np.tril_indices(6, -1)]).min() > 0.0
