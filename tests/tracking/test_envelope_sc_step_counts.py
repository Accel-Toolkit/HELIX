# tests/tracking/test_envelope_sc_step_counts.py
"""The envelope's space-charge walk of a field map uses the tracker's cadence.

``field_map_step_counts`` is the one source of the sub-step counts of a field
map.  The envelope's space-charge path used to compute its own, without the
RfqCell / VaneRFQ / NCells step floors, so through an RFQ its kicks fell
elsewhere than the multiparticle run's (a 44 mm cell: five kicks against three
and a 0.2 mm one).  Each test walks the same lattice in both modes and compares
the kick lengths, in order.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.ncells import NCells
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.tracking import tracker as T
from linac_gen.tracking.envelope import EnvelopeSolver
from tests.dataguard import needs

RFQ_DECK = "examples/lebt_plus_rfq/rfq_only_matched.dat"
NCELLS_PROJECT = "examples/piplattice/fnalscl.lgproj"
INITIAL = {"alpha_x": 0.0, "beta_x": 0.1, "alpha_y": 0.0, "beta_y": 0.1,
           "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.05}


def _expected(lat):
    out = []
    for el in lat.elements:
        n_int, _n_sc, ds, sc_every, nb = T.field_map_step_counts(lat, el)
        out += [k * ds for k in T.field_map_bundle_sizes(n_int, sc_every, nb)]
    return out


def _envelope_kicks(monkeypatch, lat, ref):
    lengths = []
    orig = EnvelopeSolver._sc_matrix

    def spy(self, sigma, *a, **k):
        lengths.append(k.get("ds_mm", a[0] if a else None))
        return orig(self, sigma, *a, **k)
    monkeypatch.setattr(EnvelopeSolver, "_sc_matrix", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EnvelopeSolver(lat, ref.copy(), dict(INITIAL), current=5.0).run()
    monkeypatch.undo()
    return lengths


def _tracker_kicks(monkeypatch, lat, ref):
    lengths = []
    orig = T.Tracker._apply_sc_kick

    def spy(self, ds_mm, *a, **k):
        lengths.append(ds_mm)
        return orig(self, ds_mm, *a, **k)
    monkeypatch.setattr(T.Tracker, "_apply_sc_kick", spy)
    beam = Beam(ref=ref.copy(), n_particles=200, current=5.0)
    rng = np.random.default_rng(1)
    beam.particles[:] = rng.normal(size=(200, 6)) * [0.3, 3.0, 0.3, 3.0, 2.0, 1e-4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T.Tracker(lat, beam, pic_solver=PicSolver(SpaceChargeConfig(
            nx=8, ny=8, nz=8, grid_mode="adaptive", use_gpu="cpu"))).run()
    monkeypatch.undo()
    return lengths


def _sub_lattice(elements, step):
    lat = Lattice()
    for el in elements:
        lat.add(el)
    lat.step_config = step
    return lat


@needs(RFQ_DECK)
def test_rfq_cells_envelope_kicks_match_the_tracker(monkeypatch):
    full, _ = parse_tracewin(RFQ_DECK)
    cells = [e for e in full.elements if isinstance(e, RfqCell)]
    pick = cells[:6]                               # incl. the 44 mm first cell
    step = StepConfig(100.0, 50.0, rfq_solver="cells")
    lat = _sub_lattice(pick, step)
    want = _expected(lat)
    floor_binds = [e.n_steps > step.integration_steps_for_length_mm(e.length) for e in pick]
    assert all(floor_binds)                        # the floor is what used to differ
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.03, frequency=162.5)
    env = _envelope_kicks(monkeypatch, _sub_lattice(pick, step), ref)
    mp = _tracker_kicks(monkeypatch, _sub_lattice(pick, step), ref)
    assert mp == want
    np.testing.assert_allclose(env, want, rtol=1e-12, atol=0)


@needs(NCELLS_PROJECT)
def test_ncells_envelope_kicks_match_the_tracker_when_its_floor_binds(monkeypatch):
    from linac_gen.cli.common import load_input
    full, cfg, _conv = load_input(NCELLS_PROJECT)
    nc = next(e for e in full.elements if isinstance(e, NCells))
    step = StepConfig(2.0, 1.0)                    # coarse: the NCells floor binds
    assert nc.n_steps > step.integration_steps_for_length_mm(nc.length)
    lat = _sub_lattice([nc], step)
    want = _expected(lat)
    ref = ReferenceParticle(species=H_MINUS, w_kin=cfg.energy, frequency=cfg.frequency)
    env = _envelope_kicks(monkeypatch, _sub_lattice([nc], step), ref)
    mp = _tracker_kicks(monkeypatch, _sub_lattice([nc], step), ref)
    assert mp == want
    np.testing.assert_allclose(env, want, rtol=1e-12, atol=0)


def test_field_map_envelope_kicks_unchanged(monkeypatch):
    """A cavity map had the same counts before (no floor of its own was
    missing): the shared counts leave its kicks as they were."""
    rng = np.random.default_rng(0)
    x = np.linspace(-20.0, 20.0, 9)
    z = np.linspace(0.0, 250.0, 61)
    shape = (x.size, x.size, z.size)
    fd = FieldMapData(z=z, frequency=162.5)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=x, z=z, Fx=0.3 * rng.normal(size=shape),
        Fy=0.3 * rng.normal(size=shape), Fz=3.0 + 0.3 * rng.normal(size=shape))

    def element():
        return FieldMap3D(name="FM", length=250.0, field_data=fd, scale=1.0,
                          phase=-30.0, frequency=162.5, n_steps=200)
    step = StepConfig(100.0, 50.0)
    lat = _sub_lattice([element()], step)
    want = _expected(lat)
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    env = _envelope_kicks(monkeypatch, _sub_lattice([element()], step), ref)
    mp = _tracker_kicks(monkeypatch, _sub_lattice([element()], step), ref)
    assert mp == want
    np.testing.assert_allclose(env, want, rtol=1e-12, atol=0)
    assert sum(env) == pytest.approx(250.0, rel=1e-12)
