"""Ring-mode coordinate conversions: exact, and pinned to a reference code's
particle model.

The energy variables (delta, ptau, rpp, rvv) and the reference change are
compared with reference code A's particle model, in the same canonical
coordinates ring mode uses (``tests/ring/fixtures/refs/coords_ref.json``, from
a dev-only generator that never imports HELIX); the time-coordinate signs of
HELIX, the time-lag coordinates and MAD-X are pinned by construction; the
H⁻ → p charge exchange by the velocity.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.ring import coords as co
from linac_gen.ring.reference import RingReference


def _ref(w=800.0, species="proton", constants="helix"):
    return RingReference.from_kinetic(species, w, constants=constants)


DELTAS = np.array([-3e-3, -1e-4, 0.0, 2.5e-5, 4e-3])
REF_FILE = Path(__file__).resolve().parent / "fixtures" / "refs" / "coords_ref.json"
REF = json.loads(REF_FILE.read_text())


def test_reference_kinematics():
    r = _ref()
    assert r.kinetic0 == pytest.approx(800.0, rel=1e-15)
    assert r.gamma0 == pytest.approx(1.0 + 800.0 / r.mass, rel=1e-15)
    assert r.beta0 * r.gamma0 == pytest.approx(r.beta0gamma0, rel=1e-15)
    # Brho = p/q: 800 MeV proton, 4.881... T m
    assert r.brho == pytest.approx(r.p0c * 1e6 / 299792458.0, rel=1e-15)


def test_reference_numbers_belong_to_these_references():
    manifest = json.loads(REF_FILE.with_suffix(".manifest.json").read_text())
    # (line ends hashed as LF: a Windows checkout may turn them into CRLF)
    sha = hashlib.sha256(REF_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert manifest["output_sha256"] == sha
    # (the reference code was given HELIX's own momenta and mass)
    assert REF["mass_mev"] == _ref().mass
    assert REF["p0c_mev"] == _ref(800.0).p0c and REF["p0c_new_mev"] == _ref(812.5).p0c
    assert REF["deltas"] == DELTAS.tolist()


def test_energy_variables_match_the_reference():
    r = _ref()
    # the reference forms ptau as sqrt(...) - 1/beta0 (cancellation at small
    # delta); ours avoids it, so the two agree to a few round-off units of O(1)
    # quantities
    np.testing.assert_allclose(co.ptau_from_delta(DELTAS, r.beta0), REF["ptau"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(co.delta_from_ptau(np.asarray(REF["ptau"]), r.beta0), DELTAS,
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(co.rpp_from_delta(DELTAS), REF["rpp"], rtol=1e-15)
    np.testing.assert_allclose(co.rvv_from_delta(DELTAS, r.beta0), REF["rvv"], rtol=1e-15)


def test_change_reference_matches_the_reference():
    old, new = _ref(800.0), _ref(812.5)
    n = DELTAS.size
    X = np.vstack([np.full(n, 1e-3), np.full(n, 2e-4), np.full(n, -5e-4), np.full(n, 1e-4),
                   np.linspace(-0.3, 0.3, n), DELTAS])
    assert X.tolist() == REF["particles"]
    Y = co.change_reference(X, old=old, new=new)
    for i, k in enumerate(("x", "px", "y", "py", "zeta", "delta")):
        np.testing.assert_allclose(Y[i], REF["changed"][k], rtol=1e-14, atol=1e-15, err_msg=k)


def test_helix_round_trip_and_time_sign():
    r = _ref()
    f = 44.7044
    P = np.array([[1.0, 0.5, -2.0, 0.1, 10.0, 0.3],       # late by 10 deg, +0.3 MeV
                  [-0.5, -1.0, 0.0, 0.0, -5.0, -0.2]])
    X = co.from_helix(P, w_ref=800.0, mass=r.mass, frequency_mhz=f, ring_ref=r)
    assert X[4, 0] < 0 and X[4, 1] > 0                       # late -> zeta < 0
    assert X[4, 0] == pytest.approx(-r.beta0 * 299792458.0 * 10.0 / (360 * f * 1e6), rel=1e-15)
    assert X[5, 0] > 0 and X[5, 1] < 0
    back = co.to_helix(X, ring_ref=r, w_ref=800.0, frequency_mhz=f)
    np.testing.assert_allclose(back, P, rtol=0, atol=1e-12)


def test_time_lag_and_madx_signs_and_round_trips():
    r = _ref()
    b0 = r.beta0
    t, pt = np.array([0.2, -0.1]), np.array([-1e-3, 2e-4])   # time lag: t > 0 late, pt = -dE/(P0 c)
    X = co.from_time_lag(np.zeros(2), np.zeros(2), t, np.zeros(2), np.zeros(2), pt, beta0=b0)
    assert X[4, 0] == pytest.approx(-b0 * 0.2) and X[5, 0] > 0      # late and above the reference energy
    np.testing.assert_allclose(co.to_time_lag(X, beta0=b0)[2], t, rtol=1e-15)
    np.testing.assert_allclose(co.to_time_lag(X, beta0=b0)[5], pt, rtol=1e-12)
    T = -t                                                           # MAD-X T = -c dt
    Xm = co.from_madx(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), T, -pt, beta0=b0)
    np.testing.assert_allclose(Xm, X, rtol=1e-15, atol=1e-18)


def test_slopes_and_momenta_are_inverse():
    xp_, yp_ = np.array([1e-3, -2e-2]), np.array([5e-4, 1e-2])
    d = np.array([1e-3, -2e-3])
    px, py = co.momenta_from_slopes(xp_, yp_, d)
    a, b = co.slopes_from_momenta(px, py, d)
    np.testing.assert_allclose(a, xp_, rtol=1e-14)
    np.testing.assert_allclose(b, yp_, rtol=1e-14)


def test_hminus_stripping_conserves_velocity():
    hm = _ref(800.0, "H-")
    p_ref = _ref(799.0)                     # any proton reference
    X = np.zeros((6, 1))
    Y = co.strip_to_proton(X, hminus_ref=hm, ring_ref=p_ref)
    pc_p = p_ref.p0c * (1.0 + Y[5, 0])
    m = p_ref.mass
    w_p = pc_p * pc_p / (math.hypot(pc_p, m) + m)
    assert w_p == pytest.approx(800.0 * m / hm.mass, rel=1e-13)        # same gamma
    assert w_p == pytest.approx(799.1296, abs=1e-4)                      # 800 MeV H- -> 799.1296 MeV p


def test_energy_variables_are_exact_to_round_off():
    """Independent pin: 50-digit arithmetic (mpmath) for delta <-> ptau."""
    mp = pytest.importorskip("mpmath")
    r = _ref()
    with mp.workdps(50):                  # (local: the session's precision is untouched)
        b0 = mp.mpf(r.beta0)
        inv_bg2 = 1 / b0**2 - 1
        for d in [1e-9, -2.5e-6, 3e-3, -0.02]:
            dd = mp.mpf(d)
            exact_ptau = mp.sqrt((1 + dd) ** 2 + inv_bg2) - 1 / b0
            got = co.ptau_from_delta(np.array([d]), r.beta0)[0]
            assert abs(mp.mpf(got) - exact_ptau) <= (2e-16 * max(abs(exact_ptau), mp.mpf("1e-300"))
                                                     + 1e-30)
            back = co.delta_from_ptau(np.array([float(exact_ptau)]), r.beta0)[0]
            assert abs(mp.mpf(back) - dd) <= 4e-16 * abs(dd) + 1e-30
