"""Envelope I=0 full-matrix surrogate seam (``envelope._full_matrix_at``).

Contract under test (one regime per test):

* **Regime 1** — I = 0, no per-sub-step recording, no interior markers:
  the pure-linear path requests each surrogated FieldMap's FULL-element
  6x6 from the registered surrogate exactly once per traversal, and the
  returned matrix is actually USED in the Σ propagation.
* **Regime 1b** — surrogate disabled (``_enabled = False``) or OOD:
  the native RK4 matrix is used and the run is bit-identical to the
  unregistered run.
* **Regime 2 / 2b** — I > 0 SC bundles, and I = 0 with
  ``record_substeps=True`` or interior markers: the bundle walks stay
  RK4 (``SurrogateFieldMap.fitted_matrix_slice`` delegates partial
  slices); zero NN queries; bit-identical to the unregistered run.
* **Regime 3** — empty registry: the module-global
  ``get_element_matrix`` dispatch is untouched (same spy pattern as
  tests/tracking/test_envelope_lazy_matrix.py).

The surrogate here is a REAL :class:`SurrogateFieldMap` whose MLP
forward pass runs, but whose ``output_norm`` std = 0 pins the
de-normalised output bitwise to a known matrix (``y = 0*y_norm + mean``).

Note: ``SurrogateFieldMap.nn_calls`` is CUMULATIVE across modes for the
lifetime of the surrogate object (the MP fast path's ``_init_fast_path``
also queries ``fitted_matrix``); per-run counts are deltas.
"""
import numpy as np
import pytest

import linac_gen.tracking.envelope as envmod
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import (MlpHead, Scope, SurrogateFieldMap,
                                       SurrogateMetadata)
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import get_element_matrix

INIT = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0, alpha_y=0.0, beta_y=2.0,
            emit_y=1.0, alpha_z=0.0, beta_z=10.0, emit_z=0.3)


def _rf_cavity(name="CAV3D", L=200.0, n_steps=20):
    """Synthetic FieldMap3D RF cavity: uniform Ez = 1 MV/m, 352.21 MHz."""
    nz, n = 11, 3
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L, nz)
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=y, z=z, Fx=np.zeros((n, n, nz)),
        Fy=np.zeros((n, n, nz)), Fz=np.full((n, n, nz), 1.0))
    return FieldMap3D(name=name, length=L, field_data=fd, scale=1.0,
                      phase=-30.0, frequency=352.21, aperture=30.0,
                      n_steps=n_steps)


def _lattice():
    lat = Lattice()
    lat.add(Drift("D1", length=50.0, aperture=30.0))
    lat.add(_rf_cavity())
    lat.add(Drift("D2", length=50.0, aperture=30.0))
    return lat


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)


def _pinned_surrogate(cav, M_target, *, in_scope=True):
    """REAL SurrogateFieldMap whose MLP forward pass runs but whose
    de-normalisation pins the output exactly to ``M_target``
    (output std = 0 -> y = 0*y_norm + mean = mean, bitwise)."""
    mlp = MlpHead(input_dim=3, output_dim=36, hidden_dims=(4,))
    lo = np.array([2.0, 0.0, 1.0]) if in_scope else np.array([99.0, 0.0, 1.0])
    meta = SurrogateMetadata(
        element_key=cav.name, element_class="FieldMap3D",
        architecture={"input_dim": 3, "output_dim": 36, "hidden_dims": [4],
                      "activation": "silu", "param_names": []},
        scope=Scope(input_names=["w_kin", "beta", "gamma"], input_lo=lo,
                    input_hi=np.array([10.0, 1.0, 100.0])),
        input_norm={"mean": [0.0] * 3, "std": [1.0] * 3},
        output_norm={"mean": [float(v) for v in M_target.ravel()],
                     "std": [0.0] * 36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="")
    return SurrogateFieldMap(cav, mlp, meta)


def _spy_nn(monkeypatch):
    calls = []
    orig = SurrogateFieldMap.fitted_matrix

    def spy(self, ref):
        calls.append(self.name)
        return orig(self, ref)

    monkeypatch.setattr(SurrogateFieldMap, "fitted_matrix", spy)
    return calls


def _run(lat, current):
    return EnvelopeSolver(lat, _ref(), dict(INIT), current=current).run()


def _sigma_end(res):
    return (np.asarray(res.sigma_x), np.asarray(res.sigma_y),
            np.asarray(res.sigma_phi), np.asarray(res.sigma_w))


def _S0():
    from linac_gen.tracking.envelope import _build_sigma_matrix
    return _build_sigma_matrix(**INIT)


def _run_S0(lat, current, **kw):
    return EnvelopeSolver(lat, _ref(), dict(INIT), current=current,
                          initial_sigma=_S0(), **kw).run()


def test_i0_envelope_uses_registered_surrogate_full_matrix(monkeypatch):
    lat = _lattice()
    cav = lat.elements[1]
    base = _run_S0(lat, 0.0)
    M_rk4 = cav.fitted_matrix(_ref())
    M_pin = M_rk4.copy()
    M_pin[0, 0] *= 1.02     # scale a genuinely non-zero entry so the
                            # pinned matrix differs from RK4 BY DESIGN
                            # (M_rk4[1, 0] is exactly 0.0 for this
                            # uniform-Ez cavity, so scaling it would
                            # be a bitwise no-op)
    calls = _spy_nn(monkeypatch)
    surr = _pinned_surrogate(cav, M_pin)
    registry.register(surr)
    res = _run_S0(lat, 0.0)
    assert calls == ["CAV3D"], \
        f"NN called {len(calls)}x, expected exactly once at I=0"
    assert surr.nn_calls == 1
    bx = np.asarray(base.sigma_x)
    sx = np.asarray(res.sigma_x)
    assert not np.array_equal(bx, sx), \
        "surrogate matrix was not USED (sigma_x identical to RK4 run)"
    # Baseline comparison, not a call count: sigma_end must equal the
    # hand-propagated D2 . M_pin . D1 . S0 with the plain-path drift maps.
    d1 = get_element_matrix(lat.elements[0], _ref())
    ref_after = _ref()
    ref_after.s += 50.0          # D1: same recipe as
    ref_after.phi_s += 360.0 * 50.0 / (ref_after.beta * ref_after.wavelength)  # EnvelopeSolver._advance_ref
    cav.reset_run_state()
    cav.advance_ref(ref_after)
    d2 = get_element_matrix(lat.elements[2], ref_after)
    Mt = d2 @ M_pin @ d1
    S_end = Mt @ _S0() @ Mt.T
    np.testing.assert_allclose(sx[-1], np.sqrt(S_end[0, 0]), rtol=1e-12)
    np.testing.assert_allclose(res.sigma_y[-1], np.sqrt(S_end[2, 2]),
                               rtol=1e-12)
    np.testing.assert_allclose(res.sigma_phi[-1], np.sqrt(S_end[4, 4]),
                               rtol=1e-12)


def test_i0_record_substeps_keeps_rk4_slice_walk(monkeypatch):
    """Regime 2b: 'Record per-sub-step' forces the bundle walk; the
    surrogate is NOT consulted and sigma is bit-identical."""
    lat = _lattice()
    cav = lat.elements[1]
    base = _run_S0(lat, 0.0, record_substeps=True)
    M_pin = cav.fitted_matrix(_ref())
    M_pin[0, 0] *= 1.02
    calls = _spy_nn(monkeypatch)
    registry.register(_pinned_surrogate(cav, M_pin))
    res = _run_S0(lat, 0.0, record_substeps=True)
    assert calls == []
    for a, b in zip(_sigma_end(base), _sigma_end(res)):
        assert np.array_equal(a, b)


def test_i0_interior_marker_keeps_rk4_slice_walk(monkeypatch):
    """Regime 2b (marker flavour): a SHIFT_IN_FIELD_MAP-style interior
    marker forces the sub-stepped walk even with record_substeps off;
    the surrogate is NOT consulted and sigma is bit-identical."""
    from types import SimpleNamespace
    lat = _lattice()
    cav = lat.elements[1]
    cav.interior_markers = [(100.0, SimpleNamespace(name="MK1"))]
    base = _run_S0(lat, 0.0)
    M_pin = cav.fitted_matrix(_ref())
    M_pin[0, 0] *= 1.02
    calls = _spy_nn(monkeypatch)
    registry.register(_pinned_surrogate(cav, M_pin))
    res = _run_S0(lat, 0.0)
    assert calls == []
    for a, b in zip(_sigma_end(base), _sigma_end(res)):
        assert np.array_equal(a, b)


def test_sc_envelope_keeps_rk4_slice_walk_with_surrogate_registered(
        monkeypatch):
    lat = _lattice()
    cav = lat.elements[1]
    base = _run(lat, 5.0)
    M_pin = cav.fitted_matrix(_ref())
    M_pin[0, 0] *= 1.02
    calls = _spy_nn(monkeypatch)
    registry.register(_pinned_surrogate(cav, M_pin))
    res = _run(lat, 5.0)
    assert calls == [], \
        "I>0 SC bundles must slice-walk RK4; NN must not be queried"
    for a, b in zip(_sigma_end(base), _sigma_end(res)):
        assert np.array_equal(a, b)


def test_i0_ood_surrogate_falls_back_bit_identical(monkeypatch):
    lat = _lattice()
    cav = lat.elements[1]
    base = _run(lat, 0.0)
    M_pin = cav.fitted_matrix(_ref())
    M_pin[0, 0] *= 1.02
    registry.register(_pinned_surrogate(cav, M_pin, in_scope=False))
    res = _run(lat, 0.0)
    for a, b in zip(_sigma_end(base), _sigma_end(res)):
        assert np.array_equal(a, b)


def test_i0_disabled_surrogate_falls_back_bit_identical(monkeypatch):
    """Regime 1b (disabled flavour): ``_enabled = False`` routes through
    the wrapped element; results bit-identical, NN counter untouched."""
    lat = _lattice()
    cav = lat.elements[1]
    base = _run(lat, 0.0)
    M_pin = cav.fitted_matrix(_ref())
    M_pin[0, 0] *= 1.02
    surr = _pinned_surrogate(cav, M_pin)
    surr._enabled = False
    registry.register(surr)
    res = _run(lat, 0.0)
    assert surr.nn_calls == 0, \
        "disabled surrogate must never count an NN prediction"
    for a, b in zip(_sigma_end(base), _sigma_end(res)):
        assert np.array_equal(a, b)


def test_empty_registry_never_touches_surrogate_code(monkeypatch):
    lat = _lattice()
    calls = _spy_nn(monkeypatch)
    gem = []
    orig = envmod.get_element_matrix
    monkeypatch.setattr(envmod, "get_element_matrix",
                        lambda e, r: (gem.append(e.name), orig(e, r))[1])
    _run(lat, 0.0)
    assert calls == [] and gem == ["D1", "CAV3D", "D2"]
