# tests/tracking/test_backtrack_fieldmap_exact.py
"""Exact (closed-form) backward tracking through field maps (2026-07-11).

``field_map_mode="rk4"`` (the new default) undoes every forward
``track_rk4`` call algebraically — un-drift, un-slip, re-sample fields
at the recovered state, one 2×2 solve for the Lorentz⊕damping couple,
un-kick — driven by a zero-particle replay of the LITERAL tracker
schedule (bit-identical boundary refs + SET_SYNC_PHASE calibration).

Measured closures (2026-07-11, 500 particles, seed 3):
single buncher ~6e-15 mm, FULL MEBT (survivors) ~1e-13 mm — i.e. 11-12
orders below the frozen v1 linear floor.  Tolerances here are ~100×
the measured values.
"""
import os as _os_guard

import pytest as _pytest_guard

if not _os_guard.path.isdir("Fields"):
    _pytest_guard.skip("Fields/ field-map data is not distributed with "
                       "the repository (third-party ANL/CEA data) — see "
                       "examples/FIELD_MAPS.md", allow_module_level=True)

import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.surrogates.base import OutOfScopeError
from linac_gen.tracking.tracker import Tracker
from linac_gen.tracking.backtrack import (
    backtrack_distribution, build_replay_table,
)

REPO = Path(__file__).resolve().parents[2]
MEBT_DAT = REPO / "examples" / "pipii" / "mebt" / "mebt.dat"
MEBT_HWR_DAT = REPO / "examples" / "mebt_plus_hwr.dat"
# The first buncher FIELD_MAP in mebt.dat.  Was a hardcoded 36; the FREQ
# card now materializes as a Freq command element (index 0), shifting every
# element index by +1 — resolve by type instead so the tests stay pinned to
# the physical element.
def _first_field_map_index(dat_path):
    from linac_gen.elements.base import FieldMapElement
    lat, _ = parse_tracewin(str(dat_path))
    return next(i for i, e in enumerate(lat.elements)
                if isinstance(e, FieldMapElement))
BUNCHER = _first_field_map_index(MEBT_DAT)

# Frozen exact-mode ceilings (~100× measured; see module docstring).
EXACT_SINGLE = np.array([1e-12, 1e-11, 1e-12, 1e-11, 1e-12, 1e-14])
EXACT_FULL = np.array([1e-11, 1e-10, 1e-11, 1e-10, 1e-11, 1e-13])

# v1 linear floor (from test_backtrack_fieldmap.FLOOR) — used for the
# ≥1e4× improvement assertion.
V1_FLOOR_X = 6e-3


@pytest.fixture(scope="module")
def mebt():
    lattice, _ = parse_tracewin(str(MEBT_DAT))
    return lattice


def _make_ref():
    return ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)


def _fresh_beam(n=500, seed=3, ref=None):
    beam = Beam(ref=ref or _make_ref(), n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    for j, s in enumerate([0.8, 0.25, 0.8, 0.25, 4.0, 0.003]):
        beam.particles[:, j] = rng.normal(0, s, n)
    return beam


def _roundtrip(lattice, start, end, *, ref_factory=_make_ref, beam=None,
               mode="rk4"):
    """Forward [0..end], snapshot at entrance of ``start``, backtrack
    [start..end]; returns (per-column max |closure| over survivors,
    beam)."""
    beam = beam if beam is not None else _fresh_beam(ref=ref_factory())
    tracker = Tracker(lattice, beam)
    snap = None
    for i in range(end + 1):
        if i == start:
            snap = beam.particles.copy()
        tracker._track_element(lattice.elements[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backtrack_distribution(lattice, beam, ref_factory(),
                               start=start, end=end, field_map_mode=mode)
    alive = beam.alive_mask
    err = np.abs(beam.particles[alive] - snap[alive]).max(axis=0)
    return err, beam


# ---------------------------------------------------------------------------
# closures
# ---------------------------------------------------------------------------
def test_buncher_roundtrip_exact(mebt):
    """One QWR buncher (3-D map, SET_SYNC_PHASE): machine-precision
    closure — ≥1e4× better than the v1 linear floor."""
    err, _ = _roundtrip(mebt, BUNCHER, BUNCHER)
    assert (err < EXACT_SINGLE).all(), err
    assert err[0] < V1_FLOOR_X / 1e4


def test_full_mebt_roundtrip_exact(mebt):
    """The ENTIRE MEBT (quads, steerers, 4 bunchers, scrapers): closure
    at float round-off for survivors.  This range closed at ~2 %
    longitudinal with the v1 linear inverse."""
    err, beam = _roundtrip(mebt, 0, len(mebt.elements) - 1)
    assert (err < EXACT_FULL).all(), err
    # σ closure on the survivors (the physics observable)
    assert beam.n_alive > 400


def test_solenoid_1d_map_roundtrip_exact():
    """1-D magnetic-only solenoid map (5000 steps/m refinement floor)."""
    if not MEBT_HWR_DAT.exists():
        pytest.skip("mebt_plus_hwr example missing")
    lat, _ = parse_tracewin(str(MEBT_HWR_DAT))
    from linac_gen.elements.field_map import FieldMap
    idx = next(i for i, e in enumerate(lat.elements)
               if isinstance(e, FieldMap))
    err, _ = _roundtrip(lat, idx, idx)
    assert (err < EXACT_SINGLE).all(), err


def test_freq_jump_roundtrip_exact():
    """A range crossing the 162.5 → 325 MHz jump (first SSR1 cavity):
    the step-0 dphi rescale must be undone exactly (ratio 2.0 is a
    power of two → bit-exact division)."""
    ssr1 = REPO / "examples" / "mebt_plus_hwr_plus_ssr1.dat"
    if not ssr1.exists():
        pytest.skip("mebt_plus_hwr_plus_ssr1 example missing")
    lat, _ = parse_tracewin(str(ssr1))
    # find the first element whose effective frequency is 325 MHz
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D
    jump = next(i for i, e in enumerate(lat.elements)
                if isinstance(e, (FieldMap, FieldMap3D))
                and getattr(e, "effective_frequency", 0.0) == 325.0)
    err, _ = _roundtrip(lat, jump - 2, jump + 1)
    assert (err < EXACT_FULL).all(), err


def test_dkd_integrator_roundtrip_exact(mebt):
    """FieldMap3D with the second-order DKD integrator closes exactly
    too (half-slip/half-drift mirror)."""
    elem = mebt.elements[BUNCHER]
    from linac_gen.elements.field_map_3d import FieldMap3D
    if not isinstance(elem, FieldMap3D):
        pytest.skip("buncher fixture is not a FieldMap3D")
    elem.integrator_kind = "dkd"
    try:
        err, _ = _roundtrip(mebt, BUNCHER, BUNCHER)
    finally:
        elem.integrator_kind = "kd"
    assert (err < EXACT_SINGLE).all(), err


# ---------------------------------------------------------------------------
# structural guarantees
# ---------------------------------------------------------------------------
def test_backward_is_deterministic_rk4(mebt):
    """Two independent exact backward walks agree bit-for-bit
    (calibration replay is deterministic)."""
    outs = []
    for _ in range(2):
        _, beam = _roundtrip(mebt, BUNCHER, BUNCHER)
        outs.append(beam.particles.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


def test_forward_untouched_by_backward(mebt):
    """fwd → bwd → fresh fwd must be bit-identical to the first fwd —
    the backward walk leaves no state on the elements that changes a
    subsequent forward run."""
    def forward():
        beam = _fresh_beam()
        t = Tracker(mebt, beam)
        for i in range(BUNCHER + 1):
            t._track_element(mebt.elements[i])
        return beam
    b1 = forward()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backtrack_distribution(mebt, b1, _make_ref(),
                               start=BUNCHER, end=BUNCHER)
    b2 = forward()
    b3 = forward()
    np.testing.assert_array_equal(b2.particles, b3.particles)


def test_replay_table_tracker_mode_matches_forward_exactly(mebt):
    """field_map_ref_mode='tracker': the energy (which only field maps
    change) equals the forward tracker's BIT-FOR-BIT — the advance_ref
    mode differs by ~2e-6 relative here, pinned separately.  phi_s
    accumulates ~1e-12 relative float-summation noise from the DRIFT
    sub-stepping (many small increments in the tracker vs one per
    element in the table), so it is pinned at 1e-11 — still 6 orders
    tighter than the advance_ref pin."""
    beam = _fresh_beam(n=1)
    tracker = Tracker(mebt, beam)
    for i in range(BUNCHER + 1):
        tracker._track_element(mebt.elements[i])
    table = build_replay_table(mebt, _make_ref(), end=BUNCHER,
                               field_map_ref_mode="tracker")
    assert table[-1].w_kin == beam.ref.w_kin          # exact
    assert table[-1].phi_s == pytest.approx(beam.ref.phi_s, rel=1e-11)
    assert table[-1].s == pytest.approx(beam.ref.s, rel=1e-12)
    assert table[-1].frequency == beam.ref.frequency


def test_sc_cadence_identical_rk4_vs_linear(mebt, monkeypatch):
    """The SC-undo call sequence (count + bundle lengths) is IDENTICAL
    between the exact and linear modes — only the half-bundle content
    changed."""
    from linac_gen.tracking import backtrack as bt

    calls: dict[str, list[float]] = {"rk4": [], "linear": []}

    orig = bt._Backtracker._apply_sc_kick_negated

    for mode in ("rk4", "linear"):
        def spy(self, ds_mm, _mode=mode):
            calls[_mode].append(round(float(ds_mm), 9))
            return orig(self, ds_mm)
        monkeypatch.setattr(bt._Backtracker, "_apply_sc_kick_negated", spy)
        _roundtrip(mebt, BUNCHER - 2, BUNCHER, mode=mode)
    assert calls["rk4"] == calls["linear"]
    assert len(calls["rk4"]) > 0


# ---------------------------------------------------------------------------
# quadrupole g3–g6 exact undo
# ---------------------------------------------------------------------------
def _g_quad_lattice(**gkw):
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    lat.add(Drift("D0", 100.0, aperture=20.0))
    lat.add(Quadrupole("QG", 150.0, gradient=8.0, aperture=20.0,
                       n_steps=5, **gkw))
    lat.add(Drift("D1", 100.0, aperture=20.0))
    return lat


def _proton_ref():
    from linac_gen.core.particle import PROTON
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def test_quad_g_multipole_roundtrip_exact():
    """g3–g6 thin kicks are now undone EXACTLY (v1 warned + dropped
    them).  No BacktrackWarning may fire."""
    lat = _g_quad_lattice(g3=40.0, g4=300.0, g5=1e4, g6=2e5)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        err, _ = _roundtrip(lat, 0, 2, ref_factory=_proton_ref,
                            beam=_fresh_beam(n=300, seed=11,
                                             ref=_proton_ref()))
    assert (err < 1e-11).all(), err
    assert not [w for w in rec if "g3" in str(w.message)]


class _AlwaysOOSSurrogate:
    """A registered surrogate whose matrix query is always out-of-scope.

    Its *presence* (``get_by_element_name`` returns non-None while MP mode
    is engaged) is what forces the rk4 walk to warn and drop to the linear
    inverse.  Raising :class:`OutOfScopeError` from the matrix query then
    makes the linear path fall through to the REAL element fields
    (``envelope._fitted_matrix_slice_at`` catches OOS), so the numbers
    stay pure-physics — matching an explicit linear run to within the
    tracker-vs-advance_ref replay-mode gap.  This keeps the assertion
    about the *control-flow* (rk4 → linear) without coupling it to any
    surrogate's numeric output.
    """

    def __init__(self, wrapped=None):
        # get_for_element (2026-07 guard) verifies the registered
        # surrogate wraps the tracked element — a realistic engaged
        # surrogate always does, so the stub must too.
        self._wrapped = wrapped

    def fitted_matrix_slice(self, ref, ds_mm):
        raise OutOfScopeError("stub surrogate: always out of scope")


def test_surrogate_registered_element_falls_back_to_linear(mebt):
    """rk4 mode + an MP-engaged surrogate on the element: a neural
    surrogate has no algebraic inverse, so the walk must WARN and drop to
    the LINEAR inverse for that element.  We prove the control flow two
    ways: (1) the ``BacktrackWarning`` is emitted (it is raised on the
    exact line that sets ``use_rk4 = False``); (2) the reconstruction
    lands on the LINEAR result and is far from the exact-rk4 result.

    The fallback is *not* bit-identical to an explicit-linear run: rk4
    mode builds the replay table in ``tracker`` ref mode while explicit
    linear uses ``advance_ref``, and that ~1e-3 ref gap propagates
    through the field-map phase column.  So we assert *proximity* to
    linear and *distance* from exact-rk4, not equality.  (The forward run
    is surrogate-free, so the exit state is the true-physics one; the
    out-of-scope stub means the linear path uses the real element fields,
    isolating the claim to control flow, not surrogate numerics.)"""
    from linac_gen.surrogates import registry as surr
    from linac_gen.tracking.backtrack import BacktrackWarning

    name = mebt.elements[BUNCHER].name

    def _fwd():
        b = _fresh_beam()
        t = Tracker(mebt, b)
        for i in range(BUNCHER + 1):
            t._track_element(mebt.elements[i])
        return b

    def _back(beam, mode):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backtrack_distribution(mebt, beam, _make_ref(),
                                   start=BUNCHER, end=BUNCHER,
                                   field_map_mode=mode)

    # A: rk4 requested but a surrogate is engaged → warn + fall to linear.
    beam_a = _fwd()
    surr.register(_AlwaysOOSSurrogate(mebt.elements[BUNCHER]),
                  lattice_hash="test-fallback",
                  element_key=name)
    surr.set_mp_enabled(True)
    try:
        with pytest.warns(BacktrackWarning, match="surrogate"):
            backtrack_distribution(mebt, beam_a, _make_ref(),
                                   start=BUNCHER, end=BUNCHER,
                                   field_map_mode="rk4")
    finally:
        surr.set_mp_enabled(False)
        surr.clear()

    beam_b = _fwd()          # explicit linear (no surrogate)
    _back(beam_b, "linear")
    beam_c = _fwd()          # exact rk4 (no surrogate)
    _back(beam_c, "rk4")

    d_lin = np.abs(beam_a.particles[:, :6] - beam_b.particles[:, :6]).max()
    d_rk4 = np.abs(beam_a.particles[:, :6] - beam_c.particles[:, :6]).max()
    assert d_lin < 1e-2, f"fallback strayed from linear (d={d_lin:.2e})"
    assert d_rk4 > 1e-3, f"exact-rk4 too close to distinguish (d={d_rk4:.2e})"
    assert d_lin < 0.1 * d_rk4, (
        f"fallback not decisively linear: d_lin={d_lin:.2e} "
        f"vs d_rk4={d_rk4:.2e}")


def test_quad_g_multipole_skew_misaligned_roundtrip():
    """g kicks + skew rotation + transverse misalignment compose with
    the misalignment wrap and still close exactly."""
    lat = _g_quad_lattice(g3=40.0, g6=1e5)
    q = lat.elements[1]
    q.skew_angle = 15.0
    q.dx, q.dy, q.tilt_deg = 0.5, -0.3, 2.0
    err, _ = _roundtrip(lat, 0, 2, ref_factory=_proton_ref,
                        beam=_fresh_beam(n=300, seed=12,
                                         ref=_proton_ref()))
    assert (err < 1e-11).all(), err
