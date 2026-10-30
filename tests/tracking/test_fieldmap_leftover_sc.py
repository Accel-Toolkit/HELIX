# tests/tracking/test_fieldmap_leftover_sc.py
"""Space charge over a field map's leftover steps.

When a map's integration steps are not a whole number of space-charge bundles,
the leftover steps form a short last bundle with its own kick over its own length
(``field_map_bundle_sizes``).  They used to run after the last bundle without any
kick: in a superconducting linac at 100/50 steps per metre, 4 % of each QWR buncher and
2.5 % of each HWR and SSR1 cavity had no space charge.  The kicks must now cover
the whole map in every mode that walks it: the tracker, the envelope solver and
both backtrackers.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking import backtrack as bt
from linac_gen.tracking import tracker as T
from linac_gen.tracking.envelope import EnvelopeSolver


def _element(L=240.0):
    rng = np.random.default_rng(0)
    x = np.linspace(-20.0, 20.0, 9)
    y = np.linspace(-20.0, 20.0, 9)
    z = np.linspace(0.0, L, 61)
    shape = (x.size, y.size, z.size)
    fd = FieldMapData(z=z, frequency=162.5)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=y, z=z, Fx=0.3 * rng.normal(size=shape),
        Fy=0.3 * rng.normal(size=shape), Fz=3.0 + 0.3 * rng.normal(size=shape))
    return FieldMap3D(name="FM", length=L, field_data=fd, scale=1.0, phase=-30.0,
                      frequency=162.5, n_steps=200)


def _lattice(el):
    lat = Lattice()
    lat.add(el)
    return lat


def _beam(n=300, current=5.0, seed=2):
    b = Beam(ref=ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
             n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    b.particles[:] = rng.normal(size=(n, 6)) * [1.0, 0.5, 1.0, 0.5, 5.0, 0.002]
    return b


def test_the_leftover_steps_are_a_short_last_bundle():
    assert T.field_map_bundle_sizes(200, 16, 12) == [16] * 12 + [8]
    assert T.field_map_bundle_sizes(200, 8, 25) == [8] * 25      # a whole number of bundles


@pytest.mark.parametrize("L", [240.0, 250.0, 300.0])     # QWR, HWR, SSR1 schedules
def test_multiparticle_kicks_cover_the_whole_map(monkeypatch, L):
    el = _element(L)
    lat = _lattice(el)
    n_int, _n_sc, ds, sc_every, nb = T.field_map_step_counts(lat, el)
    assert n_int > nb * sc_every
    lengths = []
    orig = T.Tracker._apply_sc_kick

    def spy(self, ds_mm, *a, **k):
        lengths.append(ds_mm)
        return orig(self, ds_mm, *a, **k)
    monkeypatch.setattr(T.Tracker, "_apply_sc_kick", spy)
    Simulation(lat, _beam(), space_charge=SpaceChargeConfig()).run()
    want = [sc_every * ds] * nb + [(n_int - nb * sc_every) * ds]
    assert lengths == want
    assert sum(lengths) == pytest.approx(L, rel=1e-12)


@pytest.mark.parametrize("L", [240.0, 250.0, 300.0])
def test_envelope_kicks_cover_the_whole_map(monkeypatch, L):
    el = _element(L)
    lat = _lattice(el)
    lengths = []
    orig = EnvelopeSolver._sc_matrix

    def spy(self, sigma, *a, **k):
        lengths.append(k.get("ds_mm", a[0] if a else None))
        return orig(self, sigma, *a, **k)
    monkeypatch.setattr(EnvelopeSolver, "_sc_matrix", spy)
    initial = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
               "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2}
    EnvelopeSolver(lat, ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
                   initial, current=5.0).run()
    assert sum(lengths) == pytest.approx(L, rel=1e-12)


@pytest.mark.parametrize("mode", ["rk4", "linear"])
def test_backtrack_undoes_every_kick(monkeypatch, mode):
    el = _element()
    lat = _lattice(el)
    lengths = []
    orig = bt._Backtracker._apply_sc_kick_negated

    def spy(self, ds_mm):
        lengths.append(ds_mm)
        return orig(self, ds_mm)
    monkeypatch.setattr(bt._Backtracker, "_apply_sc_kick_negated", spy)
    beam = _beam(current=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.backtrack_distribution(lat, beam, ReferenceParticle(species=PROTON, w_kin=2.1,
                                                               frequency=162.5),
                                  start=0, end=0, field_map_mode=mode)
    n_int, _n_sc, ds, sc_every, nb = T.field_map_step_counts(lat, el)
    assert sorted(lengths) == sorted([sc_every * ds] * nb + [(n_int - nb * sc_every) * ds])


def test_envelope_substep_rows_do_not_depend_on_the_current():
    """With sub-step rows on, a map with leftover steps gets the same rows with and
    without space charge: one per full bundle, then the exit."""
    initial = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
               "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2}
    rows = []
    for current in (0.0, 5.0):
        lat = _lattice(_element(240.0))
        res = EnvelopeSolver(lat, ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
                             initial, current=current, record_substeps=True).run()
        rows.append(np.asarray(res.s, dtype=float))
    n_int, _n_sc, ds, sc_every, nb = T.field_map_step_counts(lat, lat.elements[0])
    assert n_int > nb * sc_every
    np.testing.assert_allclose(rows[1], rows[0], rtol=0, atol=1e-9)
    inside = rows[1][(rows[1] > 0) & (rows[1] < 240.0 - 1e-9)]
    np.testing.assert_allclose(inside, [(b + 1) * sc_every * ds for b in range(nb)], rtol=1e-12)
