# tests/elements/test_fieldmap3d_onaxis_prefill.py
"""FieldMap3D.prefill_onaxis: the reference walk's on-axis field in one call.

``tracker.field_map_reference_states`` walks the reference through a map on
the multiparticle schedule, 2·n_int half steps whose positions the cavity
calibration never probed.  ``_sample_onaxis`` interpolated those one point
per call (a cold first envelope run spent ~27 % of its time there), so the
walk now asks the map for all of them at once.  The values must be the bits
the one-point path gives, because multiparticle runs read the same cache.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking import tracker as T
from linac_gen.tracking.envelope import EnvelopeSolver

L = 240.0
INITIAL = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
           "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2}


def _channel(rng, x, z, ez):
    shape = (x.size, x.size, z.size)
    fz = ez * np.sin(np.pi * z / L)[None, None, :] + 0.3 * rng.normal(size=shape)
    return FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.3 * rng.normal(size=shape),
                        Fy=0.3 * rng.normal(size=shape), Fz=fz)


def _fm3d(kinds=(Channel.RF_E,), p_flag=1, kind="kd", seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-20.0, 20.0, 9)
    z = np.linspace(0.0, L, 61)
    fd = FieldMapData(z=z, frequency=162.5)
    for ch in kinds:
        fd.channels[ch] = _channel(rng, x, z, 3.0 if ch.is_electric else 0.2)
    el = FieldMap3D(name="FM3", length=L, field_data=fd, scale=1.0, phase=-30.0,
                    frequency=162.5, n_steps=200, p_flag=p_flag)
    el.integrator_kind = kind
    return el


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)


def _lattice(el, step=(100.0, 30.0)):
    lat = Lattice()
    lat.add(el)
    lat.step_config = StepConfig(*step)
    return lat


def _walk_positions(el, n_half):
    h = (L / (n_half // 2)) / 2
    return [el._z_map_start + k * h + h / 2.0 for k in range(n_half)]


def test_prefilled_values_are_the_one_point_values_bit_for_bit():
    """The batched call and one call per point give the same bits, at the
    walk's positions for several step counts, on grid nodes, between nodes
    and outside the map (fill value)."""
    a = _fm3d(kinds=(Channel.RF_E, Channel.STAT_E, Channel.RF_B))
    b = _fm3d(kinds=(Channel.RF_E, Channel.STAT_E, Channel.RF_B))
    rng = np.random.default_rng(7)
    zs = (_walk_positions(a, 400) + _walk_positions(a, 266) + _walk_positions(a, 2 * 33)
          + list(np.linspace(0.0, L, 61)) + list(rng.uniform(-5.0, L + 5.0, 300))
          + [-1e-9, L + 1e-9])
    a.prefill_onaxis(zs)
    electric = [ch for ch in a._interpolators if ch.is_electric]
    assert len(electric) == 2
    for ch in electric:
        for z in zs:
            want = b._sample_onaxis(ch, b._interpolators[ch], z)
            got = a._onaxis_cache[(ch, z)]
            assert got == want or (np.isnan(got) and np.isnan(want)), (ch, z, got, want)
    assert not any(ch.is_magnetic for ch, _ in a._onaxis_cache)   # B channels untouched


def test_prefill_keeps_values_already_cached():
    el = _fm3d()
    ch = Channel.RF_E
    el._onaxis_cache[(ch, 12.5)] = 123.0                    # a sentinel the prefill must not overwrite
    el.prefill_onaxis([12.5, 13.5, 13.5])
    assert el._onaxis_cache[(ch, 12.5)] == 123.0
    assert (ch, 13.5) in el._onaxis_cache


@pytest.mark.parametrize("step", [(100.0, 50.0), (100.0, 30.0), (150.0, 50.0)])
@pytest.mark.parametrize("kind", ["kd", "dkd"])
def test_the_walk_finds_every_position_prefilled(monkeypatch, step, kind):
    """No one-point interpolation is left in the walk (p_flag=0: no
    calibration probes), with and without a short last bundle."""
    el = _fm3d(p_flag=0, kind=kind)
    lat = _lattice(el, step)
    misses = []
    orig = FieldMap3D._sample_onaxis

    def spy(self, ch, ci, z):
        if (ch, z) not in self.__dict__.get("_onaxis_cache", {}):
            misses.append(z)
        return orig(self, ch, ci, z)

    monkeypatch.setattr(FieldMap3D, "_sample_onaxis", spy)
    states = T.field_map_reference_states(lat, el, _ref())
    n_int = T.field_map_step_counts(lat, el)[0]
    assert len(states) == 2 * n_int
    assert misses == []


@pytest.mark.parametrize("kind", ["kd", "dkd"])
@pytest.mark.parametrize("p_flag", [0, 1])
def test_reference_states_are_unchanged_by_the_prefill(monkeypatch, kind, p_flag):
    lat_a = _lattice(_fm3d(p_flag=p_flag, kind=kind))
    ra = _ref()
    sa = T.field_map_reference_states(lat_a, lat_a.elements[0], ra)
    monkeypatch.setattr(FieldMap3D, "prefill_onaxis", lambda self, positions: None)
    lat_b = _lattice(_fm3d(p_flag=p_flag, kind=kind))
    rb = _ref()
    sb = T.field_map_reference_states(lat_b, lat_b.elements[0], rb)
    assert [(s.ref_before.w_kin, s.ref_before.phi_s, s.ref_before.s) for s in sa] == \
           [(s.ref_before.w_kin, s.ref_before.phi_s, s.ref_before.s) for s in sb]
    assert (ra.w_kin, ra.phi_s, ra.s) == (rb.w_kin, rb.phi_s, rb.s)


def _mp(lat):
    beam = Beam(ref=_ref(), n_particles=200, current=0.0)
    rng = np.random.default_rng(3)
    beam.particles[:] = rng.normal(size=(200, 6)) * [0.5, 0.5, 0.5, 0.5, 2.0, 1e-4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T.Tracker(lat, beam).run()
    return beam


@pytest.mark.parametrize("kind", ["kd", "dkd"])
def test_a_multiparticle_run_after_an_envelope_run_is_unchanged(kind):
    """The envelope's walk fills the cache a multiparticle run then reads:
    the run gives the bits of a run on a map that was never prefilled."""
    lat = _lattice(_fm3d(kind=kind))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EnvelopeSolver(lat, _ref(), INITIAL, current=0.0).run()
    assert lat.elements[0]._onaxis_cache                     # the walk did fill it
    after = _mp(lat)
    fresh = _mp(_lattice(_fm3d(kind=kind)))
    np.testing.assert_array_equal(after.particles, fresh.particles)
    assert (after.ref.w_kin, after.ref.phi_s) == (fresh.ref.w_kin, fresh.ref.phi_s)


def test_other_interpolation_methods_keep_the_one_point_path(monkeypatch):
    monkeypatch.setattr(FieldMap3D, "interp_kind", "cubic")
    el = _fm3d()
    el.prefill_onaxis(_walk_positions(el, 400))
    assert el._onaxis_cache == {}
