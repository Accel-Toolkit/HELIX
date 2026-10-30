"""A field map's aperture map (.ouv, Ka=1) is checked after every integration
call: a ring of the map shorter than a space-charge step stops the particles
it should, whatever the space-charge step.  Checked once per space-charge
bundle (20 mm in the PXIE solenoids), particles passed through rings that
TraceWin stops (0.35 % of the PXIE LEBT beam).  A field map without an
aperture map keeps its scalar check once per bundle, so those runs are
unchanged."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.io.test_tracewin_fieldmap_reader import _w_3d_cart

L_MM = 100.0
#: z (m), radius (m): a 30 mm pipe with an 8 mm ring from 40.1 to 45 mm
OUV = ((0.0, 0.030), (0.0400, 0.030), (0.0401, 0.008), (0.0450, 0.008),
       (0.0451, 0.030), (0.1, 0.030))


def _lattice(tmp_path, ka, sc_per_m):
    from linac_gen.core.step_config import StepConfig
    from linac_gen.io.tracewin_parser import parse_tracewin
    nz = 10
    zeros = np.zeros((nz + 1, 3, 3))
    for suf in (".bsx", ".bsy", ".bsz"):          # field-free 3-D map
        _w_3d_cart(str(tmp_path / f"sol{suf}"), nz, L_MM * 1e-3,
                   2, -0.04, 0.04, 2, -0.04, 0.04, norm=1.0, values=zeros)
    (tmp_path / "sol.ouv").write_text(
        "".join(f"{z} {r}\n" for z, r in OUV))
    deck = tmp_path / "ring.dat"
    deck.write_text("FIELD_MAP_PATH .\n"
                    f"FIELD_MAP 70 {L_MM:g} 0 40 0 0 0 {ka} sol\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    lat.step_config = StepConfig(integration_steps_per_metre=5000.0,
                                 sc_steps_per_metre=sc_per_m)
    return lat


def _beam():
    """Straight particles at x = 15 mm (outside the ring, inside the pipe)
    and at x = 5 mm (through the ring)."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=0.03,
                                   frequency=162.5),
             n_particles=6, current=0.0)
    b.particles[:] = 0.0
    b.particles[:3, 0] = 15.0
    b.particles[3:, 0] = 5.0
    b.particles[1, 2] = b.particles[4, 2] = 2.0          # some y too
    return b


@pytest.mark.parametrize("sc_per_m", [12.5, 25.0, 50.0, 100.0])
def test_a_short_ring_stops_the_particles_it_should(tmp_path, sc_per_m):
    from linac_gen.core.simulation import Simulation
    lat = _lattice(tmp_path, ka=1, sc_per_m=sc_per_m)
    b = _beam()
    Simulation(lat, b, space_charge="off", record_substeps=False).run()
    assert b.lost[:3].all() and not b.lost[3:].any()
    lt = b.loss_table
    assert len(lt) == 3
    assert np.all((lt["s"] > 40.0) & (lt["s"] < 40.25))   # at the ring's start


@pytest.mark.parametrize("ka", [0, 1])
def test_check_cadence(tmp_path, monkeypatch, ka):
    """Ka=1: a check after each of the 2 n_int half-steps (plus the exit
    check); Ka=0: once per bundle, as before (plus the exit check)."""
    from linac_gen.core.simulation import Simulation
    from linac_gen.tracking import tracker as T
    lat = _lattice(tmp_path, ka=ka, sc_per_m=50.0)
    el = lat.elements[0]
    n_int, _n_sc, _ds, sc_every, n_bundles = T.field_map_step_counts(lat, el)
    calls = []
    orig = T.Tracker._check_aperture

    def counting(self, element):
        if element is el:
            calls.append(element)
        return orig(self, element)

    monkeypatch.setattr(T.Tracker, "_check_aperture", counting)
    b = _beam()
    b.particles[:3, 0] = 5.0                               # nobody lost
    Simulation(lat, b, space_charge="off", record_substeps=False).run()
    trailing = n_int - n_bundles * sc_every
    want = (2 * n_int if ka == 1                            # every half-step
            else n_bundles + (1 if trailing else 0))        # every bundle
    assert len(calls) == want + 1                          # + the exit check
