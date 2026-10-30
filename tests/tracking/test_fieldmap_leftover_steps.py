# tests/tracking/test_fieldmap_leftover_steps.py
"""Field maps: the steps after the last space-charge bundle sample the field where
the beam is.

``Tracker._track_field_map`` walks a map in bundles of ``sc_every`` half steps, the
space-charge kick, ``sc_every`` half steps, then the ``n_int mod sc_every`` leftover
steps at full length.  The leftover steps sampled the field at ``_step_idx * ds``
with ``_step_idx`` counting the half steps — one map length further on, outside the
map — so the last millimetres were tracked as a field-free drift (in a
superconducting linac at 100/50 steps per metre: every QWR buncher, HWR and SSR1
cavity, 6-10 mm each).  The exact backtracker repeated the same position, and the
envelope's slices had the same fault when a half bundle and the leftover slice used
sub-steps of different lengths (odd ``sc_every``).

The leftover steps are now a short last bundle of half steps with their own
space-charge kick (``field_map_bundle_sizes``, see test_fieldmap_leftover_sc.py), so
the multiparticle walk is half steps throughout; the envelope's half-bundle slices
still use sub-steps of different lengths.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.vane_rfq import CellSpan, VaneRFQ
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.io.tracewin_vane import VaneGeometry
from linac_gen.tracking.backtrack import _replay_field_map_ref_states, backtrack_distribution
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.tracker import Tracker, field_map_bundle_sizes, field_map_step_counts

E0 = 2.0                                       # MV/m, uniform


def _map3d(L, kinds=("STAT_E",), seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-20.0, 20.0, 9)
    y = np.linspace(-20.0, 20.0, 9)
    z = np.linspace(0.0, L, 61)
    shape = (x.size, y.size, z.size)
    fd = FieldMapData(z=z, frequency=162.5)
    comps = {
        "STAT_E": dict(Fx=None, Fy=None, Fz=np.full(shape, E0)),
        "RF_E": dict(Fx=0.3 * rng.normal(size=shape), Fy=0.3 * rng.normal(size=shape),
                     Fz=3.0 + 0.3 * rng.normal(size=shape)),
        "RF_B": dict(Fx=0.002 * rng.normal(size=shape), Fy=0.002 * rng.normal(size=shape),
                     Fz=None),
    }
    for k in kinds:
        fd.channels[getattr(Channel, k)] = FieldChannel(geometry=7, x=x, y=y, z=z, **comps[k])
    return fd


def _fm3d(L=240.0, kinds=("STAT_E",), integrator="kd"):
    el = FieldMap3D(name="FM", length=L, field_data=_map3d(L, kinds), scale=1.0, phase=-30.0,
                    frequency=162.5, n_steps=200)
    el.integrator_kind = integrator
    return el


def _beam(n=40, species=PROTON, w=2.1, seed=1):
    b = Beam(ref=ReferenceParticle(species=species, w_kin=w, frequency=162.5),
             n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    b.particles[:] = rng.normal(size=(n, 6)) * [1.0, 0.5, 1.0, 0.5, 5.0, 0.002]
    return b


def _lattice(el):
    lat = Lattice()
    lat.add(el)
    return lat


def _spy(monkeypatch, el, name):
    zs = []
    orig = getattr(el, name)

    def spy(*args):
        zs.append(float(args[-1]))
        return orig(*args)
    monkeypatch.setattr(el, name, spy)
    return zs


@pytest.mark.parametrize("integrator", ["kd", "dkd"])
def test_every_multiparticle_step_samples_the_field_where_the_beam_is(monkeypatch, integrator):
    el = _fm3d(integrator=integrator)
    lat = _lattice(el)
    n_int, _n_sc, ds, sc_every, nb = field_map_step_counts(lat, el)
    t = n_int - nb * sc_every
    assert (sc_every, nb, t) == (16, 12, 8)           # the QWR buncher's schedule
    zs = _spy(monkeypatch, el, "_sample_onaxis")
    Simulation(lat, _beam()).run()
    h = ds / 2
    assert len(zs) == 2 * n_int                       # the short last bundle: half steps too
    assert zs == [0.0 + k * h + h / 2.0 for k in range(2 * n_int)]   # the old formula, bit for bit
    assert max(zs) < el.length


def test_a_1d_map_samples_the_field_where_the_beam_is(monkeypatch):
    L = 240.0
    z = np.linspace(0.0, L, 241)
    fd = FieldMapData(z=z, frequency=162.5)
    fd.channels[Channel.STAT_E] = FieldChannel(geometry=1, z=z, Fz=np.full_like(z, E0))
    el = FieldMap("FM1", length=L, field_data=fd, phase=0.0, frequency=162.5, n_steps=200)
    lat = _lattice(el)
    n_int, _n_sc, ds, sc_every, nb = field_map_step_counts(lat, el)
    t = n_int - nb * sc_every
    assert t > 0
    zs = _spy(monkeypatch, el, "_sample_Fz_on_axis")
    Simulation(lat, _beam()).run()
    np.testing.assert_allclose(zs, [(k + 0.5) * ds / 2 for k in range(2 * n_int)], rtol=1e-13)
    assert max(zs) < L


def test_the_reference_gains_the_whole_field_integral():
    """A uniform static Ez over the map: the reference must gain q·E·L, the energy
    the grid-based ``advance_ref`` gives, to round-off.  With the leftover steps
    outside the map it gained t·ds/L = 4 % less."""
    el = _fm3d()
    lat = _lattice(el)
    beam = _beam(species=H_MINUS)
    w0 = beam.ref.w_kin
    Simulation(lat, beam).run()
    gained = beam.ref.w_kin - w0
    ref = ReferenceParticle(species=H_MINUS, w_kin=w0, frequency=162.5)
    _fm3d().advance_ref(ref)
    expected = ref.w_kin - w0
    assert abs(expected) == pytest.approx(E0 * el.length * 1e-3, rel=1e-12)
    assert gained == pytest.approx(expected, rel=1e-12)
    # the backtracker's replay of the tracker schedule ends at the same energy
    ref2 = ReferenceParticle(species=H_MINUS, w_kin=w0, frequency=162.5)
    _replay_field_map_ref_states(lat, _fm3d(), ref2)
    assert ref2.w_kin - w0 == pytest.approx(expected, rel=1e-12)


def test_vane_rfq_leftover_steps_stay_in_the_cells(monkeypatch):
    L_cell = 20.0
    z = np.linspace(0.0, 2 * L_cell * 1e-3, 200)
    a = np.full_like(z, 5.0e-3)
    tc = np.full_like(z, 4.183e-3)
    vp, vn, flag = np.full_like(z, 3e4), np.full_like(z, -3e4), np.zeros_like(z)
    vane = VaneGeometry(z=z, aperture_v1=a, Tc_v1=tc, voltage_v1=vp, flag_v1=flag,
                        aperture_v2=a, Tc_v2=tc, voltage_v2=vn, flag_v2=flag,
                        aperture_v3=a, Tc_v3=tc, voltage_v3=vp, flag_v3=flag,
                        aperture_v4=a, Tc_v4=tc, voltage_v4=vn, flag_v4=flag)
    cells = [CellSpan(z_start_mm=k * L_cell, z_end_mm=(k + 1) * L_cell, voltage_V=6e4, A10=0.1,
                      modulation=1.5, length_mm=L_cell, phi_s_deg=-60.0, cell_type=2,
                      type_prev=2, type_next=2) for k in range(2)]
    el = VaneRFQ("RFQ", vane=vane, cells=cells, n_steps=23)
    lat = _lattice(el)
    n_int, _n_sc, ds, sc_every, nb = field_map_step_counts(lat, el)
    t = n_int - nb * sc_every
    assert t > 0
    zs = _spy(monkeypatch, el, "_cell_index")
    Simulation(lat, _beam(species=H_MINUS, w=0.05)).run()
    np.testing.assert_allclose(zs, [(k + 0.5) * ds / 2 for k in range(2 * n_int)], rtol=1e-13)
    assert max(zs) < el.length


@pytest.mark.parametrize("L", [250.0, 300.0])      # odd sc_every: 15 (HWR), 13 (SSR1)
def test_envelope_slices_sample_the_field_where_the_beam_is(monkeypatch, L):
    el = _fm3d(L=L)
    lat = _lattice(el)
    zs, inside = [], [False]
    orig_onaxis, orig_slice = el._sample_onaxis, el.fitted_matrix_slice

    def onaxis(*args):
        if inside[0]:
            zs.append(float(args[-1]))
        return orig_onaxis(*args)

    def slice_(ref, ds_mm):                         # no _z_from_mm, as FieldMap3D
        inside[0] = True
        try:
            return orig_slice(ref, ds_mm)
        finally:
            inside[0] = False
    monkeypatch.setattr(el, "_sample_onaxis", onaxis)
    monkeypatch.setattr(el, "fitted_matrix_slice", slice_)
    initial = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
               "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2}
    EnvelopeSolver(lat, ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
                   initial, current=5.0).run()
    n_int, _n_sc, ds, sc_every, nb = field_map_step_counts(lat, el)
    t = n_int - nb * sc_every
    assert sc_every % 2 == 1 and t > 0
    native = el.length / el.n_steps

    def sub_step(slice_mm):                         # as fitted_matrix_slice divides a slice
        return slice_mm / max(1, int(round(slice_mm / native)))
    last = field_map_bundle_sizes(n_int, sc_every, nb)[-1]
    sub = sub_step(last * ds / 2)                   # the last bundle's second half
    assert sub != sub_step(sc_every * ds / 2)       # the full half bundles use another length
    assert max(zs) < L
    assert max(zs) == pytest.approx(L - sub / 2, rel=1e-12)   # the last step's middle


def test_backtrack_recovers_the_beam_and_replays_the_true_positions():
    el = _fm3d(kinds=("RF_E", "RF_B"))
    lat = _lattice(el)
    n_int, _n_sc, ds, sc_every, nb = field_map_step_counts(lat, el)
    steps = _replay_field_map_ref_states(
        lat, _fm3d(kinds=("RF_E", "RF_B")),
        ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5))
    z = [st.z_entry for st in steps]
    assert n_int > nb * sc_every
    assert z == [k * (ds / 2) for k in range(2 * n_int)]
    beam = _beam(n=200, seed=4)
    start = beam.particles.copy()
    Tracker(lat, beam)._track_element(el)
    assert np.abs(beam.particles - start).max() > 1e-3     # the map did something
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backtrack_distribution(lat, beam, ReferenceParticle(species=PROTON, w_kin=2.1,
                                                            frequency=162.5),
                               start=0, end=0, field_map_mode="rk4")
    err = np.abs(beam.particles - start).max(axis=0)
    assert (err < [1e-12, 1e-11, 1e-12, 1e-11, 1e-11, 1e-13]).all(), err


def test_untrack_without_a_recorded_position_uses_the_step_index():
    """Callers that do not pass ``z_entry`` keep the old position, step_idx · ds."""
    el = _fm3d(kinds=("RF_E", "RF_B"))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    el._step_idx = 0
    b0 = _beam()
    el.track_rk4(b0, 1.2)                      # calibration / entrance state in place
    a, b = _beam(seed=7), _beam(seed=7)
    el.untrack_rk4(a, 1.2, ref, 3)
    el.untrack_rk4(b, 1.2, ref, 3, z_entry=3 * 1.2)
    np.testing.assert_array_equal(a.particles, b.particles)
