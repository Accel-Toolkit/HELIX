"""Ring kernels against reference code A, element by element (L1).

``tests/ring/fixtures/refs/cartesian_kernels_ref.npz`` holds particles
tracked by the reference code through single elements, written by a dev-only
generator that never imports HELIX.  Each element is rebuilt here from the
reference's own stored parameters (with its defaults for those it leaves
out) as HELIX ring nodes, and the same particles are tracked by the
plain-Python kernels and by the compiled ones.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.ring.compile import blocks as B
from linac_gen.ring.kernels import cartesian as cart
from linac_gen.ring.kernels import nodes as nd
from tests.ring.kernel_harness import run

REF = Path(__file__).resolve().parent / "fixtures" / "refs" / "cartesian_kernels_ref.npz"
DATA = np.load(REF)
SPECS = {s["id"]: s for s in json.loads(str(DATA["specs_json"]))}
P0C, MASS, BETA0, _GAMMA0 = (float(v) for v in DATA["reference"])
P0C_MEV, MASS_MEV = P0C * 1e-6, MASS * 1e-6

#: the reference's defaults for parameters its stored elements leave out
_DEFAULT_MODEL = {"Bend": "rot-kick-rot", "Quadrupole": "mat-kick-mat",
                  "Sextupole": "drift-kick-drift-expanded",
                  "Octupole": "drift-kick-drift-expanded"}
_DEFAULT_INTEGRATOR = {"Bend": "yoshida4", "Quadrupole": "uniform", "Sextupole": "uniform",
                       "Octupole": "uniform"}


def _edge(p, side, k0, k1, k2, length):
    if not p.get(f"edge_{side}_active", 1):
        return []
    angle = p.get(f"edge_{side}_angle", 0.0)
    fint = p.get(f"edge_{side}_fint", 0.0)
    hgap = p.get(f"edge_{side}_hgap", 0.0)
    model = p.get(f"edge_{side}_model", "linear")
    if model == "linear":
        return [(nd.N_EDGE_LINEAR, B.edge_linear(*cart.edge_linear_coefficients(k0, angle, 0.0, hgap,
                                                                              fint)))]
    assert model == "full"
    return [(nd.N_EDGE_EXACT, B.edge_exact(
        is_exit=side == "exit", angle=angle, k0_fringe=k0, k0_wedge=k0, k1=k1, fint=fint,
        hgap=hgap, quad_wedge=1, mult_fringe=True, kn=(k0, k1, k2, 0.0), length=length))]


def build(spec):
    """HELIX ring nodes for one stored reference element."""
    c, p = spec["class"], spec["params"]
    if c == "Drift":
        return [(nd.N_DRIFT, B.drift(p["length"], p.get("model") == "exact"))]
    if c in ("Quadrupole", "Sextupole", "Octupole", "Bend"):
        L = p["length"]
        model = p.get("model", _DEFAULT_MODEL[c])
        integ = p.get("integrator", _DEFAULT_INTEGRATOR[c])
        if model == "adaptive":
            model = _DEFAULT_MODEL[c]
        if integ == "adaptive":
            integ = _DEFAULT_INTEGRATOR[c]
        nk = p.get("num_multipole_kicks", 0) or 1        # (straight magnets: one kick)
        h = p.get("angle", 0.0) / L
        k0 = p.get("k0", h)
        k1 = p.get("k1", 0.0)
        k2 = p.get("k2", 0.0)
        k3 = p.get("k3", 0.0)
        cfg = B.magnet_config(model, k0, k1, h)
        body = (nd.N_MAGNET, B.magnet(L, integrator=B.INTEGRATORS[integ], n_kicks=nk, k2=k2, k3=k3,
                                      knl=p.get("knl", []), ksl=p.get("ksl", []), **cfg))
        if c != "Bend":
            return [body]
        return _edge(p, "entry", k0, k1, k2, L) + [body] + _edge(p, "exit", k0, k1, k2, L)
    if c == "Multipole":
        return [(nd.N_MULTIPOLE, B.multipole(p["knl"], p["ksl"]))]
    if c == "Cavity":
        return [(nd.N_CAVITY, B.cavity(p["voltage"] * 1e-6, p["frequency"], p["lag"]))]
    if c == "SRotation":
        return [(nd.N_SROTATION, [math.sin(math.radians(p["angle"])), p["cos_z"]])]
    if c == "XYShift":
        return [(nd.N_XYSHIFT, B.xyshift(p["dx"], p["dy"]))]
    if c == "DipoleEdge":
        return [(nd.N_EDGE_LINEAR, B.edge_linear(*cart.edge_linear_coefficients(
            p["k"], p["e1"], 0.0, p["hgap"], p["fint"])))]
    raise AssertionError(f"no builder for {c}")


def _scalar(spec, X):
    """The cases without a node kind of their own: the scalar maps directly."""
    c, p = spec["class"], spec["params"]
    Y = np.array(X, copy=True)
    for i in range(X.shape[1]):
        x, px, y, py, zeta, delta = X[:, i]
        ptau, rpp, rvv = cart.energy_from_delta(delta, BETA0)
        if c == "YRotation":
            a = math.radians(p["angle"])
            x, px, y, zeta = cart.yrotation(x, px, y, py, zeta, ptau, BETA0, math.sin(a), math.cos(a),
                                          math.tan(a))
        elif c == "Wedge":
            x, px, y, zeta = cart.wedge(x, px, y, py, zeta, delta, ptau, rvv, BETA0, p["angle"], p["k"])
            px, py = cart.quad_wedge(x, px, y, py, p["angle"], p["k1"])
        Y[:, i] = (x, px, y, py, zeta, delta)
    return Y


#: (rtol, atol) per case: round-off only, except "bend-kick-bend", whose exact
#: curved-bend formula cancels terms of order 1/(h k0) (~1600 m here): HELIX
#: ports it unchanged, both codes round there (each ~1.5e-13 m from a 40-digit
#: evaluation) and agree to 1.4e-13 m.  zeta
#: is a sum of differences of terms the size of the element length, so its
#: absolute round-off is ~1e-16 × the length per step (ATOL_ZETA).
#: Two more cases are set by the reference's own rounding: its straight exact
#: bend divides sqrt(...) - pz by k0 (×1/k0 = 83 here; its compiled C also
#: fuses multiply-adds), and after a cavity it takes delta = sqrt(...) - 1
#: (1e-16 absolute) where HELIX uses a cancellation-free form.
_TOL = {"bend_bend-kick-bend_yoshida4": (1e-10, 5e-13),
        "straight_bend": (1e-11, 1e-16),
        "cavity": (1e-13, 5e-16)}
_DEFAULT_TOL = (1e-13, 1e-16)
ATOL_ZETA = 1e-14


def test_reference_file_is_the_generated_one():
    manifest = json.loads(REF.with_suffix(".manifest.json").read_text())
    assert manifest["output_sha256"] == hashlib.sha256(REF.read_bytes()).hexdigest()
    assert sorted(manifest["cases"]) == sorted(SPECS)


@pytest.mark.parametrize("case", sorted(SPECS))
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_kernel_matches_the_reference(case, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    spec = SPECS[case]
    X = DATA["x_in"]
    want = DATA[f"x_out_{case}"]
    if spec["class"] in ("YRotation", "Wedge"):
        if backend == "numba":
            pytest.skip("scalar map (no node kind of its own): the plain-Python case covers it")
        got = _scalar(spec, X)
    else:
        got = run(build(spec), X, beta0=BETA0, p0c=P0C_MEV, mass=MASS_MEV, backend=backend)
    rtol, atol = _TOL.get(case, _DEFAULT_TOL)
    atol = np.array([atol, atol, atol, atol, max(atol, ATOL_ZETA), atol])[:, None]
    err = np.abs(got - want) - (rtol * np.abs(want) + atol)
    worst = np.unravel_index(np.argmax(err), err.shape)
    assert err.max() <= 0.0, (f"{case}: coordinate {worst[0]} of particle {worst[1]}: HELIX "
                              f"{got[worst]!r} vs reference {want[worst]!r}")


def test_python_and_compiled_kernels_are_bit_identical():
    pytest.importorskip("numba")
    X = DATA["x_in"]
    for case, spec in SPECS.items():
        if spec["class"] in ("YRotation", "Wedge"):
            continue
        a = run(build(spec), X, beta0=BETA0, p0c=P0C_MEV, mass=MASS_MEV, backend="python")
        b = run(build(spec), X, beta0=BETA0, p0c=P0C_MEV, mass=MASS_MEV, backend="numba")
        assert np.array_equal(a, b), case
