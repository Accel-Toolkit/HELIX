# tests/elements/test_fieldmap_z_cursor.py
"""The position cursor of a field-map walk (``FieldMapElement._advance_z_cursor``).

The tracker walks a field map in half steps inside its space-charge bundles and in
full steps after the last bundle, and the envelope slices it with sub-steps of
different lengths, so a step's position cannot be ``_step_idx * ds``.  It was: the
leftover full steps sampled the field one map length further on, outside the map.
The cursor must give exactly the old positions while the step length stays the same
(constant-step walks keep their bits), the true positions after it changes, and
restart cleanly when ``_step_idx`` is set from outside.
"""
import copy

import numpy as np

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel

L = 240.0
DS = 1.2                     # 200 steps over 240 mm, as a QWR buncher map
H = DS / 2


def _element(start=0.0):
    x = np.linspace(-20.0, 20.0, 5)
    y = np.linspace(-20.0, 20.0, 5)
    z = np.linspace(0.0, L, 41)
    fd = FieldMapData(z=z, frequency=162.5)
    fd.channels[Channel.STAT_E] = FieldChannel(
        geometry=7, x=x, y=y, z=z, Fx=None, Fy=None, Fz=np.ones((5, 5, 41)))
    el = FieldMap3D(name="FM", length=L, field_data=fd, scale=1.0, phase=0.0,
                    frequency=162.5, n_steps=200)
    el._z_map_start = start
    return el


def _walk(el, lengths):
    return [el._advance_z_cursor(ds) for ds in lengths]


def test_constant_step_gives_the_old_positions_bit_for_bit():
    for start in (0.0, 17.3, -4.25):
        for ds in (DS, H, 0.2, 2.3835):
            el = _element(start)
            el._step_idx = 0
            got = [el._z_map_start + e + ds / 2.0 for e in _walk(el, [ds] * 400)]
            old = [start + k * ds + ds / 2.0 for k in range(400)]
            assert got == old                      # exactly, not approximately


def test_half_steps_then_full_steps_give_the_true_positions():
    nb, sc_every, t = 12, 16, 8                    # the QWR's schedule (n_int 200, step2 50/m)
    el = _element()
    el._step_idx = 0
    entries = _walk(el, [H] * (2 * nb * sc_every) + [DS] * t)
    assert entries[:2 * nb * sc_every] == [k * H for k in range(2 * nb * sc_every)]
    np.testing.assert_allclose(entries[2 * nb * sc_every:],
                               [(nb * sc_every + j) * DS for j in range(t)], rtol=1e-13)
    assert entries[-1] + DS / 2 < L               # the last step samples inside the map
    assert el._step_idx == 2 * nb * sc_every + t  # still the call count


def test_a_one_ulp_change_of_step_does_not_move_the_origin():
    el = _element()
    el._step_idx = 0
    _walk(el, [DS] * 10)
    ds2 = 1.2000000000000028                       # what a slice remainder computes for 1.2
    assert el._advance_z_cursor(ds2) == 10 * ds2   # the old formula, no rebase
    assert el._z_base == 0.0


def test_a_step_index_set_from_outside_restarts_the_cursor():
    el = _element()
    el._step_idx = 0
    _walk(el, [H] * 50 + [DS])                     # moved its origin
    el._step_idx = 5
    assert el._advance_z_cursor(DS) == 5 * DS
    el._step_idx = 0
    assert el._advance_z_cursor(DS) == 0.0


def test_a_saved_state_replays_the_same_positions():
    el = _element()
    el._step_idx = 0
    _walk(el, [H] * 30)
    state = el._z_cursor_state()
    first = _walk(el, [H, DS, DS])
    el._restore_z_cursor_state(state)
    assert _walk(el, [H, DS, DS]) == first


def test_elements_never_walked_carry_no_cursor_attributes():
    """Class-level defaults: lattices dumped attribute by attribute, deep copies and
    pickles of elements that have never been walked look as before."""
    el = _element()
    keys = ("_z_sync_idx", "_z_base", "_z_unit", "_z_n", "_last_z_entry")
    assert not any(k in vars(el) for k in keys)
    assert not any(k in vars(copy.deepcopy(el)) for k in keys)


def test_envelope_slices_of_unequal_sub_steps_continue_at_the_true_position():
    """A slice of 9 mm (8 sub-steps of 1.125 mm) then one of 6 mm (5 of 1.2 mm):
    the second slice starts at 9 mm, not at 8 × 1.2 = 9.6 mm."""
    el = _element()
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    el._step_idx = 0
    el.fitted_matrix_slice(ref, 9.0)
    seen = []
    orig = el.track_rk4

    def spy(beam, ds):
        orig(beam, ds)
        seen.append(el._last_z_entry)
    el.track_rk4 = spy
    el.fitted_matrix_slice(ref, 6.0)
    np.testing.assert_allclose(sorted(set(seen)), [9.0 + k * DS for k in range(5)], rtol=1e-13)
    assert el._step_idx == 8 + 5
