"""Tests for the opt-in per-element transfer-matrix cache.

The cache (``cache=`` kwarg on ``get_element_matrix`` /
``compute_transfer_matrix``) is a pure memoisation layer.  When
``cache is None`` (the default for every existing caller) the behaviour
must be byte-identical to no cache.  When a dict is passed, repeated
calls with the same (element, ref) state must hit the cache and return
the exact cached matrix; mutating an element parameter must invalidate
that entry via the param fingerprint.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.matrix_tracking import (
    get_element_matrix, compute_transfer_matrix,
    _element_fingerprint, _ref_fingerprint,
)


# ---------------------------------------------------------------------------
def _build_ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


def _build_lattice() -> Lattice:
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=10.0))
    lat.add(Drift(name="D2", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=10.0))
    lat.add(Solenoid(name="SOL", length=80.0, field=0.5, aperture=10.0))
    return lat


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------
def test_element_fingerprint_returns_none_for_unmarked_element():
    """Classes without ``_cache_keys`` opt out — fingerprint is ``None``."""
    class _NoKeys:
        length = 1.0
    fp = _element_fingerprint(_NoKeys())
    assert fp is None


def test_element_fingerprint_captures_param_mutation():
    q = Quadrupole(name="Q", length=50.0, gradient=10.0)
    fp_before = _element_fingerprint(q)
    q.gradient = 11.0
    fp_after = _element_fingerprint(q)
    assert fp_before != fp_after


def test_element_fingerprint_rounds_floats():
    """FP fuzz inside 1e-12 must not produce spurious cache misses."""
    q1 = Quadrupole(name="Q", length=50.0, gradient=10.0)
    q2 = Quadrupole(name="Q", length=50.0 + 1e-14, gradient=10.0 + 1e-14)
    # Same id() can't be tested here since they're different objects,
    # but the fingerprints alone should compare equal (FP fuzz suppressed).
    assert _element_fingerprint(q1) == _element_fingerprint(q2)


def test_ref_fingerprint_depends_on_w_kin():
    ref_a = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    ref_b = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)
    assert _ref_fingerprint(ref_a) != _ref_fingerprint(ref_b)


# ---------------------------------------------------------------------------
# Behaviour when cache=None (default)
# ---------------------------------------------------------------------------
def test_default_cache_is_none_byte_identical():
    """``cache=None`` (default) must reproduce the pre-cache result exactly."""
    lat = _build_lattice()
    ref = _build_ref()
    M_no_kwarg = compute_transfer_matrix(lat, ref)
    M_explicit_none = compute_transfer_matrix(lat, ref, cache=None)
    np.testing.assert_array_equal(M_no_kwarg, M_explicit_none)


def test_get_element_matrix_default_path_unchanged():
    ref = _build_ref()
    d = Drift(name="D", length=120.0)
    M_a = get_element_matrix(d, ref)
    M_b = get_element_matrix(d, ref, cache=None)
    np.testing.assert_array_equal(M_a, M_b)


# ---------------------------------------------------------------------------
# Behaviour with an opt-in cache
# ---------------------------------------------------------------------------
def test_cache_hit_returns_bit_identical_matrix():
    """A second call with the same (element, ref) hits the cache; the
    object returned is *the same numpy array*, bit-identical with the
    first call's result."""
    ref = _build_ref()
    q = Quadrupole(name="Q", length=50.0, gradient=12.0)
    cache: dict = {}
    M1 = get_element_matrix(q, ref, cache=cache)
    M2 = get_element_matrix(q, ref, cache=cache)
    assert len(cache) == 1
    # Same numpy array object — proves the cache returned the cached one.
    assert M2 is M1


def test_cache_value_matches_uncached_value():
    """Cached value must equal the uncached recompute value exactly."""
    ref = _build_ref()
    sol = Solenoid(name="SOL", length=80.0, field=0.5)
    M_uncached = get_element_matrix(sol, ref)
    cache: dict = {}
    M_cached = get_element_matrix(sol, ref, cache=cache)
    M_cached_again = get_element_matrix(sol, ref, cache=cache)
    np.testing.assert_array_equal(M_cached, M_uncached)
    np.testing.assert_array_equal(M_cached_again, M_uncached)


def test_cache_miss_after_param_mutation():
    """Mutating a matrix-affecting param must produce a new cache entry
    and a recomputed (correct) matrix — never a stale cached value."""
    ref = _build_ref()
    q = Quadrupole(name="Q", length=50.0, gradient=10.0)
    cache: dict = {}
    M_g10 = get_element_matrix(q, ref, cache=cache)

    q.gradient = 15.0
    M_g15 = get_element_matrix(q, ref, cache=cache)

    # New entry, distinct matrices, and the post-mutation matrix matches
    # an independent recompute.
    assert len(cache) == 2
    assert not np.array_equal(M_g10, M_g15)
    np.testing.assert_array_equal(
        M_g15,
        get_element_matrix(q, ref),  # uncached recompute
    )


def test_cache_miss_after_ref_change():
    """Same element, different ref energy → different cache entry."""
    q = Quadrupole(name="Q", length=50.0, gradient=10.0)
    cache: dict = {}
    ref_a = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    ref_b = ReferenceParticle(species=PROTON, w_kin=10.0, frequency=162.5)
    M_a = get_element_matrix(q, ref_a, cache=cache)
    M_b = get_element_matrix(q, ref_b, cache=cache)
    assert len(cache) == 2
    # Quadrupole's effective focusing scales with brho ∝ p; different
    # energy → different matrix.
    assert not np.array_equal(M_a, M_b)


def test_compute_transfer_matrix_cache_matches_uncached():
    """The whole-lattice product through ``compute_transfer_matrix`` must
    agree bit-for-bit between cache=None and cache={}."""
    lat = _build_lattice()
    ref = _build_ref()
    cache: dict = {}
    M_uncached = compute_transfer_matrix(lat, ref)
    M_cached_first = compute_transfer_matrix(lat, ref, cache=cache)
    M_cached_second = compute_transfer_matrix(lat, ref, cache=cache)
    np.testing.assert_array_equal(M_cached_first, M_uncached)
    np.testing.assert_array_equal(M_cached_second, M_uncached)
    # Second call should be all-hits (no new entries added beyond first).
    n_after_first = len(cache)
    compute_transfer_matrix(lat, ref, cache=cache)
    assert len(cache) == n_after_first


def test_cache_size_equals_distinct_elements_after_first_walk():
    """For a fixed (ref, lattice) the first walk creates one entry per
    matrix-yielding element; subsequent calls add nothing."""
    lat = _build_lattice()
    ref = _build_ref()
    cache: dict = {}
    _ = compute_transfer_matrix(lat, ref, cache=cache)
    n_first = len(cache)
    # 5 elements in the fixture, all matrix-yielding.
    assert n_first == 5
    _ = compute_transfer_matrix(lat, ref, cache=cache)
    assert len(cache) == n_first


# ---------------------------------------------------------------------------
# Phase-advance API also accepts cache=
# ---------------------------------------------------------------------------
def test_structure_phase_advance_accepts_cache():
    """``structure_phase_advance(cache=...)`` doesn't change the result."""
    from linac_gen.analysis.period_detect import PeriodicStructure
    from linac_gen.analysis.phase_advance import structure_phase_advance
    lat = _build_lattice()
    ref = _build_ref()
    period = PeriodicStructure(
        start=0, end=5, inner_period_length=5, inner_slice_end=5,
        n_repeats=1, label="full", source="test",
    )
    out_uncached = structure_phase_advance(lat, ref, period)
    out_cached = structure_phase_advance(lat, ref, period, cache={})
    # M_period must be byte-identical.
    np.testing.assert_array_equal(
        out_uncached["M_period"], out_cached["M_period"],
    )


def test_structure_phase_advance_along_s_accepts_cache():
    from linac_gen.analysis.period_detect import PeriodicStructure
    from linac_gen.analysis.phase_advance import structure_phase_advance_along_s
    lat = _build_lattice()
    ref = _build_ref()
    period = PeriodicStructure(
        start=0, end=5, inner_period_length=5, inner_slice_end=5,
        n_repeats=1, label="full", source="test",
    )
    out_uncached = structure_phase_advance_along_s(lat, ref, period)
    out_cached = structure_phase_advance_along_s(lat, ref, period, cache={})
    # Beta arrays must be bit-identical (NaN entries compare equal under
    # assert_array_equal only when both are NaN at the same index — use
    # equal_nan via assert_allclose with tiny atol).
    np.testing.assert_allclose(
        np.nan_to_num(out_cached["beta_x"]),
        np.nan_to_num(out_uncached["beta_x"]),
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        np.nan_to_num(out_cached["mu_x_deg"]),
        np.nan_to_num(out_uncached["mu_x_deg"]),
        rtol=0, atol=0,
    )
