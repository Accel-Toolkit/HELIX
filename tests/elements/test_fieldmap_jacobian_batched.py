"""Batched Jacobian probes are bit-identical to single-particle probes.

Since 2026-09-08 ``fitted_matrix`` / ``fitted_matrix_slice`` of the three
field-map elements track all their central-difference probes in ONE
``Beam`` (``rk4.numerical_jacobian_batched``).  These tests reproduce the
original one-probe-per-Beam construction here — with the element's OWN
step count, measured by counting ``track_rk4`` calls during the batched
call rather than re-deriving the refinement rule — and assert
``np.array_equal`` on the matrices and on the walk state each call leaves
behind.  Unlike ``test_fieldmap_matrix_baseline.py`` (a fixture generated on
one machine) this runs the two constructions on the SAME platform, so it is
what carries the guarantee to Linux and Windows CI, for both integrators and
both samplers.  It also pins the call counts, so a future edit cannot
silently un-batch.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements import field_map_3d as fm3d_mod
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.core.particle import H_MINUS
from tests.elements.regen_fieldmap_matrix_baseline import REFS as _REFS, elements

#: The fixture's references sit at the maps' own 352.21 MHz, so the step-0
#: FREQ rescale of the probe beam (field_map_3d.py:565, field_map.py:563,
#: superposed_field_map.py:222) never fires there; a 162.5 MHz reference
#: (ratio 2.167) makes it fire once per batch, where it used to fire once
#: per probe.
REFS = dict(_REFS, h10_162=dict(species=H_MINUS, w_kin=10.0, frequency=162.5))

DELTAS_3D = [1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-4]     # field_map_3d.fitted_matrix's own
EPS_DEFAULT = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.001])   # rk4.numerical_jacobian's


def _probes_default():
    P = []
    for j in range(6):
        p = np.zeros(6); m = np.zeros(6)
        p[j] += EPS_DEFAULT[j]; m[j] -= EPS_DEFAULT[j]
        P += [p, m]
    return P


def _state(el):
    vals = [float(el._step_idx)]
    if isinstance(el, SuperposedFieldMap):
        vals += [float(el._z_cursor), float(len(el._z_history))]
        for _z, c in el.children:
            so = getattr(c, "_sync_offset_deg", None)
            vals += [float("nan") if so is None else float(so),
                     float(getattr(c, "_phi_s_at_entrance", 0.0) or 0.0)]
        re_ = getattr(el, "_ref_entry", None)
        vals += [float("nan") if re_ is None else float(re_.w_kin)]
    else:
        so = getattr(el, "_sync_offset_deg", None)
        vals += [float("nan") if so is None else float(so),
                 float(getattr(el, "_phi_s_at_entrance", 0.0) or 0.0)]
    return np.asarray(vals, float)


class _Counting:
    """Count track_rk4 calls and remember the beams handed to it."""
    def __init__(self, el):
        self.el, self.calls, self.beams = el, 0, []
        self._orig = el.track_rk4
    def __enter__(self):
        def wrapped(beam, ds):
            self.calls += 1
            if beam not in self.beams:
                self.beams.append(beam)
            return self._orig(beam, ds)
        self.el.track_rk4 = wrapped
        return self
    def __exit__(self, *a):
        del self.el.track_rk4


def _single_full(el, ref, n, ds, probes, per_probe_setup, per_probe_teardown=None):
    """The pre-2026-09-08 fitted_matrix construction, one Beam per probe."""
    ys = []
    for p in probes:
        b = Beam(ref=ref.copy(), n_particles=1, current=0.0)
        b.particles[0, :] = p
        per_probe_setup(el)
        for _ in range(n):
            el.track_rk4(b, ds)
        if per_probe_teardown is not None:
            per_probe_teardown(el)
        ys.append(b.particles[0].copy())
    return el, np.array(ys)


def _matrix_from_rows(Y, probes_eps, first_row):
    M = np.zeros((6, 6)); r = first_row
    for j, dj in enumerate(probes_eps):
        M[:, j] = (Y[r] - Y[r + 1]) / (2.0 * dj); r += 2
    return M


def _cases():
    for ename, make in elements().items():
        for rn in REFS:
            if ename.startswith("fm3d"):
                for integ, fused in (("kd", True), ("kd", False), ("dkd", True), ("dkd", False)):
                    yield pytest.param(ename, rn, integ, fused, id=f"{ename}-{rn}-{integ}-{'fused' if fused else 'scipy'}")
            else:
                yield pytest.param(ename, rn, None, None, id=f"{ename}-{rn}")


@pytest.fixture
def _numerics(monkeypatch):
    saved = fm3d_mod.fused_kernel_enabled()
    def setup(integ, fused):
        if integ is not None:
            monkeypatch.setattr(FieldMap3D, "integrator_kind", integ)
        if fused is not None:
            if fused and not fm3d_mod.kernel_available():
                pytest.skip("fused kernel not built here")
            fm3d_mod.use_fused_kernel(fused)
    yield setup
    fm3d_mod.use_fused_kernel(saved)


@pytest.mark.parametrize("ename,rn,integ,fused", list(_cases()))
def test_fitted_matrix_batched_equals_single(ename, rn, integ, fused, _numerics):
    _numerics(integ, fused)
    make = elements()[ename]
    ref = ReferenceParticle(**REFS[rn])

    el = make(); el.reset_run_state()
    with _Counting(el) as c:
        M_batched = el.fitted_matrix(ref.copy())
    n = c.calls                                    # the element's own step count
    assert len(c.beams) == 1, "the probes must share one Beam"
    beam = c.beams[0]
    assert not beam.lost.any()
    is_3d = isinstance(el, FieldMap3D)
    assert beam.particles.shape[0] == (13 if is_3d else 12)
    state_batched = _state(el)

    ds = el.length / n
    el_s = make(); el_s.reset_run_state()
    if is_3d:
        # original: 13 single tracks (origin first, unused), _step_idx = 0
        # before each and again after the loop
        probes = [np.zeros(6)]
        for j, dj in enumerate(DELTAS_3D):
            p = np.zeros(6); m = np.zeros(6); p[j] = +dj; m[j] = -dj
            probes += [p, m]
        setup = lambda e: setattr(e, "_step_idx", 0)
        el_s, Y = _single_full(el_s, ref, n, ds, probes, setup)
        el_s._step_idx = 0
        M_single = np.eye(6)
        M_single[:, :] = _matrix_from_rows(Y, DELTAS_3D, 1)
    elif isinstance(el, SuperposedFieldMap):
        # original: walk state saved BEFORE the probes, cursor reset per
        # probe, saved state restored in a finally after all of them
        saved = el_s._save_walk_state()
        def setup(e):
            e._step_idx = 0; e._z_cursor = 0.0
        el_s, Y = _single_full(el_s, ref, n, ds, _probes_default(), setup)
        el_s._restore_walk_state(saved)
        M_single = _matrix_from_rows(Y, EPS_DEFAULT, 0)
    else:
        # original: _step_idx saved and restored around EACH probe
        def setup(e):
            e._saved = e._step_idx; e._step_idx = 0
        def teardown(e):
            e._step_idx = e._saved
        el_s, Y = _single_full(el_s, ref, n, ds, _probes_default(), setup, teardown)
        M_single = _matrix_from_rows(Y, EPS_DEFAULT, 0)

    np.testing.assert_array_equal(M_batched, M_single)
    np.testing.assert_array_equal(state_batched, _state(el_s))


@pytest.mark.parametrize("ename,rn,integ,fused", list(_cases()))
@pytest.mark.parametrize("frac", [None, 0.5, 1.0], ids=["native", "half", "full"])
def test_fitted_matrix_slice_batched_equals_single(ename, rn, integ, fused, frac, _numerics):
    _numerics(integ, fused)
    make = elements()[ename]
    ref = ReferenceParticle(**REFS[rn])
    el = make(); el.reset_run_state()
    ds = el.length / max(el.n_steps, 1) if frac is None else el.length * frac
    with _Counting(el) as c:
        M_batched = el.fitted_matrix_slice(ref.copy(), ds)
    n_sub = c.calls
    assert len(c.beams) == 1 and c.beams[0].particles.shape[0] == 12
    assert not c.beams[0].lost.any()
    state_batched = _state(el)

    # the pre-change construction: one Beam per probe, cursor reset per probe
    el_s = make(); el_s.reset_run_state()
    sub_ds = ds / n_sub
    if isinstance(el_s, SuperposedFieldMap):
        saved = el_s._save_walk_state(); z_from = el_s._z_cursor
        def setup(e):
            e._restore_walk_state(saved); e._z_cursor = z_from; e._step_idx = saved[0]
    else:
        saved_idx = el_s._step_idx
        def setup(e):
            e._step_idx = saved_idx
    ys = []
    for p in _probes_default():
        b = Beam(ref=ref.copy(), n_particles=1, current=0.0)
        b.particles[0, :] = p
        setup(el_s)
        for _ in range(n_sub):
            el_s.track_rk4(b, sub_ds)
        ys.append(b.particles[0].copy())
    if isinstance(el_s, SuperposedFieldMap):
        el_s._restore_walk_state(saved)
        el_s._step_idx = saved[0] + n_sub; el_s._z_cursor = z_from + ds
    else:
        el_s._step_idx = saved_idx + n_sub
    M_single = _matrix_from_rows(np.array(ys), EPS_DEFAULT, 0)

    np.testing.assert_array_equal(M_batched, M_single)
    np.testing.assert_array_equal(state_batched, _state(el_s))
    assert el._step_idx == el_s._step_idx


def test_call_counts_are_batched_and_calibration_runs_once(monkeypatch):
    """A future edit cannot silently un-batch: n_sub calls per slice (was
    12 x n_sub) and the sync-phase calibration body runs exactly once."""
    make = elements()["fm3d_rf_p1"]
    ref = ReferenceParticle(**REFS["p3"])
    el = make(); el.reset_run_state()
    calib = {"n": 0}
    orig = el._calibrate_sync_phase
    def counting(r, *a, **k):
        calib["n"] += 1
        return orig(r, *a, **k)
    monkeypatch.setattr(el, "_calibrate_sync_phase", counting)
    native = el.length / el.n_steps
    with _Counting(el) as c:
        el.fitted_matrix_slice(ref.copy(), 4 * native)
    assert c.calls == 4                      # n_sub, not 12 * n_sub
    assert calib["n"] == 1                   # once per batch, not once per probe
    assert el._sync_offset_deg is not None
    # the body ran once: a second call with the offset already set returns early
    before = el._sync_offset_deg
    with _Counting(el) as c2:
        el.fitted_matrix_slice(ref.copy(), 4 * native)
    assert c2.calls == 4 and el._sync_offset_deg == before and calib["n"] == 1
