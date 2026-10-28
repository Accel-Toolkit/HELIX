"""Ring mode's curvilinear model against reference code B (L1).

``tests/ring/fixtures/refs/simple_booster_curvilinear_ref.npz`` holds the
reference code's results on the simple Booster (the deck in
``tests/ring/fixtures``, BSD): 1000 protons tracked two turns, its tunes and
chromaticity from its own tracking, and particles through single elements.
The generator is dev-only and never imports HELIX.

**How close, and why not closer.**  Against a 40-digit evaluation of the same
maps, HELIX is exact to rounding (~1e-16 m in x after a turn) and the
reference is off by ~3e-12 m: the two-turn agreement is bounded by the
reference's own floating-point rounding, ~5e-12 m in x (5e-10 of the beam
size), 1e-14 m in y.  Part of it is the form of its sector bend and
curvilinear multipoles, which cancel terms of order rc ≈ 40 m (HELIX
evaluates the same expressions regrouped); the rest is the rounding of its
compiled C++ (fused multiply-adds, its math library).  The residual has no
structure in the particles' coordinates — a cubic polynomial in them explains
no more of it than of noise — so it is rounding, not a difference of model;
bit-level agreement (1e-12 relative) would need the reference's rounding
reproduced.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring import coords as C
from linac_gen.ring.compile import blocks as B
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.kernels import curvilinear as curv_kernels
from linac_gen.ring.kernels import nodes as N
from linac_gen.ring.optics import _energy, twiss
from linac_gen.ring.reference import RingReference
from tests.ring import mp_kernels
from tests.ring.kernel_harness import run

FIX = Path(__file__).resolve().parent / "fixtures"
DECK = FIX / "simple_booster.madx"
REF = FIX / "refs" / "simple_booster_curvilinear_ref.npz"
DATA = np.load(REF)
MAN = json.loads(REF.with_suffix(".manifest.json").read_text())
MASS, BETA, _GAMMA, BG, _PT = MAN["reference_mass_MeV_beta_gamma_bg_pt"]
P0C = BG * MASS
#: the reference's integration settings (sextupoles as the other magnets)
REFERENCE_SETTINGS = dict(model="curvilinear", integrator="symp6", bend_steps=6,
                          magnet_integrator="symp6", magnet_steps=6)
RF_FREQ_HZ = 44704409.98524446          # the reference deck's cavity frequency
RF_V_NORM = 9.688990660720476e-06       # its voltage, in units of m c^2


def _booster_as_reference(**options):
    """The deck as the reference has it: its reference particle (CODATA 2022),
    the cavities at its frequency and voltage."""
    lat, meta = parse_madx(str(DECK))
    for e in lat.elements:
        if isinstance(e, RFGap):
            e.harmonic = None
            e.frequency = RF_FREQ_HZ * 1e-6
            e.voltage = RF_V_NORM * MASS
    ref = RingReference.from_kinetic("proton", 800.0, constants="codata2022")
    assert ref.mass == MASS
    return compile_ring(lat, ref, lattice_energy=meta["reference"].w_kin,
                        **(options or REFERENCE_SETTINGS))


@pytest.fixture(scope="module")
def ring():
    return _booster_as_reference()


def test_reference_belongs_to_its_manifest():
    assert MAN["output_sha256"] == hashlib.sha256(REF.read_bytes()).hexdigest()
    assert MAN["n_parity"] == DATA["parity_in"].shape[1] == 1000
    # the parity run changed only the sextupoles' integration, which moves the
    # published deck's particles by this much after two turns
    assert MAN["published_vs_parity_max_abs_diff_turn2"] < 1e-11


def _track(rl, P, turns):
    X = C.from_time_lag(*P, beta0=rl.reference.beta0)
    n = X.shape[1]
    ptau, rpp, rvv = (np.ascontiguousarray(a) for a in _energy(X[5], rl.reference.beta0))
    st = np.ones(n, np.int8)
    lt = np.full(n, -1, np.int32)
    ln = np.full(n, -1, np.int32)
    lc = np.zeros(n, np.int8)
    from linac_gen.ring.kernels.cpu import backend
    out = []
    for turn in range(turns):
        backend().track_turns(X, ptau, rpp, rvv, st, lt, ln, lc, *rl.node_table(), 0,
                              rl.n_nodes, 1, turn, 0.0)
        out.append(np.vstack(C.to_time_lag(X, beta0=rl.reference.beta0)))
    assert np.all(st == 1)
    return out


#: two-turn bounds (x, y, t, px, py, pt), set by the reference's own rounding (see
#: the module docstring; measured 4.6e-12, 8.6e-15, 8.6e-13, 1.5e-13, 2.2e-15, 1.4e-16)
TOL = np.array([2e-11, 5e-14, 5e-12, 1e-12, 2e-14, 1e-15])[:, None]


def test_two_turns_particle_by_particle(ring):
    got = _track(ring, DATA["parity_in"], 2)
    for turn, g in zip((1, 2), got, strict=True):
        want = DATA[f"parity_turn{turn}"]
        d = np.abs(g - want)
        k, i = np.unravel_index(np.argmax(d / TOL), d.shape)
        assert np.all(d <= TOL), (f"turn {turn}, coordinate {['x', 'y', 't', 'px', 'py', 'pt'][k]} of "
                                  f"particle {i}: HELIX {g[k, i]!r} reference {want[k, i]!r}")


def test_the_residual_is_reference_rounding(ring):
    # one turn of the two particles where the codes differ most, in 40 digits:
    # HELIX's float64 is exact to rounding, the reference's is not
    cart, _curv, nd = mp_kernels.load(40)
    got = _track(ring, DATA["parity_in"], 1)[0]
    want = DATA["parity_turn1"]
    worst = np.argsort(np.abs(got[0] - want[0]))[-2:]
    ends = list(ring.offsets[1:]) + [ring.params.size]
    blocks = [(int(ring.kinds[j]), [mp.mpf(float(v)) for v in ring.params[ring.offsets[j]:ends[j]]])
              for j in range(ring.n_nodes)]
    b0 = mp.mpf(float(ring.reference.beta0))
    p0c = mp.mpf(float(ring.reference.p0c))
    bg2 = (p0c / mp.mpf(MASS)) ** 2
    X0 = C.from_time_lag(*DATA["parity_in"], beta0=ring.reference.beta0)
    for i in worst:
        x, px, y, py, zeta, delta = [mp.mpf(float(v)) for v in X0[:, i]]
        ptau, rpp, rvv = cart.energy_from_delta(delta, b0)
        for kind, p in blocks:
            (x, px, y, py, zeta, delta, ptau, rpp, rvv, _l) = nd.track_node(
                kind, p, 0, b0, p0c, bg2, x, px, y, py, zeta, delta, ptau, rpp, rvv)
        exact_x = float(x)
        assert abs(got[0, i] - exact_x) < 1e-14
        assert 1e-13 < abs(want[0, i] - exact_x) < 1e-11


def test_the_residual_has_no_structure(ring):
    # a difference of model (a missing or different term) would be a smooth
    # function of the particles' coordinates; the reference's rounding is not: a
    # full cubic polynomial in the six initial coordinates (84 terms, 1000
    # particles) explains about as much of the turn-1 residual in x as it
    # would of pure noise (84/1000)
    got = _track(ring, DATA["parity_in"], 1)[0]
    r = got[0] - DATA["parity_turn1"][0]
    P = DATA["parity_in"]
    z = (P - P.mean(axis=1, keepdims=True)) / P.std(axis=1, keepdims=True)
    cols = [np.ones(z.shape[1])]
    n = z.shape[0]
    for a in range(n):
        cols.append(z[a])
        for b in range(a, n):
            cols.append(z[a] * z[b])
            for c in range(b, n):
                cols.append(z[a] * z[b] * z[c])
    A = np.vstack(cols).T
    assert A.shape == (1000, 84)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    r2 = 1.0 - np.sum((r - A @ coef) ** 2) / np.sum((r - r.mean()) ** 2)
    assert r2 < 0.2, r2


INT = {2: 4, 4: 5, 6: 6}


def _element_nodes(case):
    k, p = case["class"], case["params"]
    if k == "ExactDrift":
        return [(N.N_DRIFT, B.drift(p["ds"], True))]
    if k == "ExactQuad":
        return [(N.N_CURVILINEAR, B.curvilinear_quad(p["ds"], p["k"],
                                                     integrator=INT[p["int_order"]],
                                                     nsteps=p["mapsteps"]))]
    if k == "ExactMultipole":
        return [(N.N_CURVILINEAR, B.curvilinear_multipole(p["ds"], p["k_normal"], p["k_skew"],
                                                          integrator=INT[p["int_order"]],
                                                          nsteps=p["mapsteps"]))]
    if k == "ExactCFbend":
        kn = list(p["k_normal"])
        rc = 1.0 / kn[0]
        kn[0] = 0.0
        return [(N.N_CURVILINEAR, B.curvilinear_cfbend(p["ds"], rc, kn, p["k_skew"],
                                                       integrator=INT[p["int_order"]],
                                                       nsteps=p["mapsteps"]))]
    if k == "DipEdge":
        return [(N.N_EDGE_LINEAR, B.edge_linear(*curv_kernels.dipedge_linear_coefficients(
            p["psi"], p["rc"], p["g"], p["K2"])))]
    raise AssertionError(k)


CASES = {c["id"]: c for c in json.loads(str(DATA["element_cases_json"]))}

#: where the two codes round differently by more than the last bits
#: (measured; each against a 40-digit evaluation of the same map): the exact
#: drift's c t, 1.1e-15 m (both codes within 8e-16 m of it); the reference's
#: sector bend with a sextupole at negative curvature, 2.5e-14 m in x, and with
#: an octupole and decapole, 2.9e-10 m (HELIX within 2e-18 m in both,
#: test_sector_bend_differences_are_reference_rounding)
ELEMENT_ATOL = {"exact_drift": 5e-15, "exact_cfbend_s4_negative": 1e-13,
                "exact_cfbend_s6": 1e-9}


def _element_out(cid, X):
    return np.vstack(C.to_time_lag(run(_element_nodes(CASES[cid]), X, beta0=BETA, p0c=P0C,
                                       mass=MASS), beta0=BETA))


@pytest.mark.parametrize("cid", sorted(CASES))
def test_single_elements(cid):
    # (the other cases agree to the last bits: 3e-17 m in c t, 1e-18 m or
    # exactly in the transverse coordinates)
    X = C.from_time_lag(*DATA["element_in"], beta0=BETA)
    got = _element_out(cid, X)
    want = DATA[f"element_out_{cid}"]
    tol = 1e-15 * np.abs(want) + 1e-18 + ELEMENT_ATOL.get(cid, 0.0)
    d = np.abs(got - want)
    assert np.all(d <= tol), f"{cid}: max |HELIX - reference| {d.max():.3e}"


@pytest.mark.parametrize("cid", ["exact_cfbend_s4_negative", "exact_cfbend_s6"])
def test_sector_bend_differences_are_reference_rounding(cid):
    # the bend cases where the codes differ beyond the last bits: HELIX is
    # the 40-digit result of the same map, the reference is not
    cart, _curv, nd = mp_kernels.load(40)
    X = C.from_time_lag(*DATA["element_in"], beta0=BETA)
    got = _element_out(cid, X)
    want = DATA[f"element_out_{cid}"]
    b0 = mp.mpf(float(BETA))
    p0c = mp.mpf(float(P0C))
    bg2 = (p0c / mp.mpf(float(MASS))) ** 2
    nodes = [(k, [mp.mpf(float(v)) for v in blk]) for k, blk in _element_nodes(CASES[cid])]
    exact = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        x, px, y, py, zeta, delta = [mp.mpf(float(v)) for v in X[:, i]]
        ptau, rpp, rvv = cart.energy_from_delta(delta, b0)
        for kind, p in nodes:
            (x, px, y, py, zeta, delta, ptau, rpp, rvv, _l) = nd.track_node(
                kind, p, 0, b0, p0c, bg2, x, px, y, py, zeta, delta, ptau, rpp, rvv)
        exact[i] = float(x)
    assert np.abs(got[0] - exact).max() < 1e-17
    assert np.abs(want[0] - exact).max() > 1e-14


def test_curvilinear_optics_match_the_reference(ring):
    # the reference's tunes and chromaticity, from its own tracking, vs the
    # same model's optics here
    qx, qy, dqx, dqy = MAN["optics_qx_qy_dqx_dqy_per_delta"]
    o = twiss(ring.with_rf(False))
    assert o.qx % 1.0 == pytest.approx(qx, abs=3e-8)
    assert o.qy % 1.0 == pytest.approx(qy, abs=3e-8)
    assert o.dqx == pytest.approx(dqx, abs=2e-3)
    assert o.dqy == pytest.approx(dqy, abs=2e-3)


def test_the_convention_difference_is_a_sextupole_in_the_bends():
    # the curvilinear quadrupole field differs from the Cartesian (MAD-X)
    # expansion inside a bend, at cubic order in the Hamiltonian, by -h k1
    # (x^3 - 3 x y^2)/6: a normal sextupole of strength -h k1 (higher orders
    # differ by less: about 1e-5 in Q').  Cartesian bends (with
    # linear pole faces) given that extra sextupole have the curvilinear
    # model's chromaticity.
    lat, meta = parse_madx(str(DECK))
    w = meta["reference"].w_kin
    ref = RingReference.from_kinetic("proton", w)
    curv = twiss(compile_ring(lat, ref, lattice_energy=w, model="curvilinear"))
    cart = twiss(compile_ring(lat, ref, lattice_energy=w, edge_model="linear"))
    for e in lat.elements:
        if isinstance(e, Dipole) and e.angle:
            h = math.radians(e.angle) / (e.length * 1e-3)
            k1 = -e.field_index / (abs(e.rho) * 1e-3) ** 2
            e.k2 = float(getattr(e, "k2", 0.0) or 0.0) - h * k1
    plus = twiss(compile_ring(lat, ref, lattice_energy=w, edge_model="linear"))
    assert abs(curv.dqx - cart.dqx) > 0.9 and abs(curv.dqy - cart.dqy) > 0.15
    assert plus.dqx == pytest.approx(curv.dqx, abs=0.01)
    assert plus.dqy == pytest.approx(curv.dqy, abs=1e-3)
    # and the linear optics are the same in both conventions
    assert curv.qx == pytest.approx(cart.qx, abs=1e-8)
    assert curv.qy == pytest.approx(cart.qy, abs=1e-8)
