"""The synthetic RFQ demo deck — the one RFQ case that RUNS anywhere.

Every other test in tests/rfq/ benchmarks against the PXIE/PIP-II
TraceWin project and skips cleanly when that data is absent, which is
the normal state on any machine but the development one and in every
public checkout.  Without this module a public clone would ship the
RFQ model with zero executable evidence that it works.

`examples/rfq_demo/` is generated from textbook two-term design
relations at parameters unlike any real machine, so it ships publicly
and these tests always run.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest

DEMO = (pathlib.Path(__file__).resolve().parents[2]
        / "examples" / "rfq_demo")


def _run(current, space_charge, n=1200, periodic=True):
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, meta = parse_tracewin(str(DEMO / "rfq_demo.dat"))
    cfg = BeamConfig(
        species="proton", energy=0.075, frequency=352.21, current=current,
        n_particles=n, distribution="gaussian", cutoff=4.0,
        emit_nx=0.20, alpha_x=0.0, beta_x=0.01,
        emit_ny=0.20, alpha_y=0.0, beta_y=0.01,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0,
        continuous=True, dc_energy_spread_keV=0.0,
        periodic_phase=periodic)
    beam = create_beam(cfg, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Simulation(lat, beam, space_charge=space_charge,
                         record_substeps=False).run()
    return lat, meta, beam, res


def test_deck_parses_clean():
    lat, meta, _, _ = _run(0.0, "off", n=100)
    assert not meta.get("warnings"), meta["warnings"]
    from linac_gen.elements.rfq_cell import RfqCell
    cells = [e for e in lat.elements if isinstance(e, RfqCell)]
    assert len(cells) == 199
    # r0 and voltage are the DEMO design, not any real machine's
    assert cells[0].r0_mm == pytest.approx(3.40)
    assert cells[0].voltage_V == pytest.approx(85_000.0)


def test_it_bunches_and_accelerates():
    """The whole point of an RFQ: DC in, bunched and accelerated out."""
    _, _, beam, _ = _run(0.0, "off")
    alive = beam.alive_mask
    assert alive.sum() > 0
    # accelerated ~25x from 75 keV
    assert beam.ref.w_kin == pytest.approx(1.898, abs=0.05)
    W = beam.ref.w_kin + beam.particles[alive, 5]
    # every survivor is accelerated — no un-captured low-energy junk
    assert (W > 0.9 * beam.ref.w_kin).sum() == alive.sum()
    # and it is genuinely BUNCHED: a DC beam enters uniform over the full
    # RF period (sigma_phi ~ 104 deg); the exit bunch is a few degrees.
    dphi = beam.particles[alive, 4]
    assert dphi.std() < 30.0


def test_transmission_is_stable():
    """Pins the demo's headline number so a physics change to the RFQ
    model cannot silently move it.  Wide band — this is a regression
    tripwire, not a validation of the design."""
    _, _, beam, _ = _run(0.0, "off")
    t = 100.0 * beam.alive_mask.sum() / beam.n_particles
    assert 60.0 < t < 85.0, t


def test_space_charge_runs_through_it():
    """15 mA through a 32^3 adaptive PIC — the SC path must survive an
    RFQ, which is where it is hardest (tight bore, strong bunching)."""
    from linac_gen.core.config import SpaceChargeConfig
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=6.0,
                           grid_mode="adaptive")
    _, _, beam, _ = _run(15.0, sc, n=800)
    alive = beam.alive_mask
    assert alive.sum() > 0
    assert beam.ref.w_kin == pytest.approx(1.898, abs=0.05)
    t = 100.0 * alive.sum() / beam.n_particles
    assert 50.0 < t < 80.0, t


def test_periodic_phase_bounds_the_phase_here_too():
    """The example ships with periodic_phase on; check it does what the
    RFQ manual says on a deck that is not PXIE."""
    _, _, on, _ = _run(0.0, "off", periodic=True)
    _, _, off, _ = _run(0.0, "off", periodic=False)
    assert on.bunch_train is True
    assert on.bunch_train_frequency == pytest.approx(352.21)
    assert np.abs(on.particles[on.alive_mask, 4]).max() <= 180.0
    # identical physics: same survivors either way
    np.testing.assert_array_equal(on.lost, off.lost)


def test_project_file_loads_and_matches_the_deck():
    from linac_gen.io.project import load_project
    proj = load_project(str(DEMO / "rfq_demo.lgproj"))
    assert proj.beam.species == "proton"
    assert proj.beam.continuous is True
    assert proj.beam.periodic_phase is True
    assert proj.beam.frequency == pytest.approx(352.21)
