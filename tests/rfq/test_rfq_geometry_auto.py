"""TW-faithful RFQ_GEOM auto-detection (Simulation.rfq_geometry).

TraceWin's Partran mode hands the RFQ to Toutatis (vane geometry) when
RFQ_GEOM is present, while its envelope mode stays on the cards.  HELIX
mirrors that: multiparticle Simulation.run() auto-uses the vane profile
when the RFQ_GEOM file exists; run_envelope() never does.
"""
import logging

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.io.rfq_geometry_helper import apply_rfq_geometry
from linac_gen.io.tracewin_parser import parse_tracewin

R0 = 5.0
V = 60000.0


def _write_vane(path):
    """Synthetic 17-column vane table: 40 mm flare + ideal-quad plateau."""
    z = np.arange(0.0, 295.0 + 1e-9, 0.25)
    a = np.full(z.shape, R0)
    fl = z < 40.0
    a[fl] = R0 + 10.0 * (1.0 - z[fl] / 40.0) ** 2
    a_m = a * 1e-3
    tc = np.full(z.shape, 0.75 * R0 * 1e-3)
    vp = np.full(z.shape, +V / 2)
    vm = np.full(z.shape, -V / 2)
    f0 = np.zeros_like(z)
    cols = [z * 1e-3,
            a_m, tc, vp, f0,     # vane 1 (+x)
            a_m, tc, vm, f0,     # vane 2 (+y)
            a_m, tc, vp, f0,     # vane 3 (−x)
            a_m, tc, vm, f0]     # vane 4 (−y)
    np.savetxt(path, np.column_stack(cols), fmt="%.9e")


def _write_deck(path, vane_name="test.vane", with_geom=True):
    lines = []
    if with_geom:
        lines.append(f"RFQ_GEOM 1 {vane_name}")
    lines.append("RFQ_CELL 60000 5.0 0 1 40 -90 3 3.75 1 0 0 0 0 0")
    for _ in range(34):
        lines.append("RFQ_CELL 60000 5.0 0.001 1.0004 7.5 -90 2 "
                     "3.75 -0.5 0 0 0 0 0")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def deck(tmp_path):
    _write_vane(tmp_path / "test.vane")
    _write_deck(tmp_path / "mini_rfq.dat")
    return tmp_path / "mini_rfq.dat"


def _beam():
    cfg = BeamConfig(
        species="H-", energy=0.03, frequency=162.5,
        current=0.0, duty_cycle=100.0, n_particles=200,
        distribution="gaussian", cutoff=3.0,
        emit_nx=0.11, alpha_x=0.0, beta_x=0.05,
        emit_ny=0.11, alpha_y=0.0, beta_y=0.05,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0,
        continuous=True, dc_energy_spread_keV=0.0, periodic_phase=True)
    return create_beam(cfg, seed=7)


_BUILD = dict(nx=21, z_subsample=6, solver="spsolve", use_cache=False)


class TestParserRecordsVane:
    def test_recorded_and_deck_relative(self, deck):
        lat, _ = parse_tracewin(str(deck))
        assert lat.rfq_geom_file is not None
        assert lat.rfq_geom_file.endswith("test.vane")
        assert str(deck.parent) in lat.rfq_geom_file

    def test_absent_card_is_none(self, tmp_path):
        _write_deck(tmp_path / "plain.dat", with_geom=False)
        lat, _ = parse_tracewin(str(tmp_path / "plain.dat"))
        assert lat.rfq_geom_file is None


class TestAutoArm:
    def test_mp_run_uses_profile_and_disarms_after(self, deck, caplog):
        lat, _ = parse_tracewin(str(deck))
        with caplog.at_level(logging.INFO):
            Simulation(lat, _beam(), space_charge="off",
                       rfq_geometry="auto").run()
        assert any("vane-geometry profile" in r.message
                   for r in caplog.records)
        # scoped: disarmed after the run -> later envelope = cards
        assert all(c._geom_z is None for c in lat.elements
                   if hasattr(c, "_geom_z"))

    def test_off_switch(self, deck, caplog):
        lat, _ = parse_tracewin(str(deck))
        with caplog.at_level(logging.INFO):
            Simulation(lat, _beam(), space_charge="off",
                       rfq_geometry="off").run()
        assert not any("vane-geometry profile" in r.message
                       for r in caplog.records)

    def test_missing_vane_falls_back_to_cards(self, deck, caplog):
        (deck.parent / "test.vane").unlink()
        lat, _ = parse_tracewin(str(deck))
        with caplog.at_level(logging.INFO):
            Simulation(lat, _beam(), space_charge="off").run()
        assert any("not found" in r.message for r in caplog.records)

    def test_forced_mode_missing_vane_raises(self, deck):
        (deck.parent / "test.vane").unlink()
        lat, _ = parse_tracewin(str(deck))
        with pytest.raises(FileNotFoundError):
            Simulation(lat, _beam(), space_charge="off",
                       rfq_geometry="antisym").run()

    def test_profile_changes_the_physics(self, deck):
        lat, _ = parse_tracewin(str(deck))
        b1 = _beam()
        Simulation(lat, b1, space_charge="off",
                   rfq_geometry="off").run()
        lat2, _ = parse_tracewin(str(deck))
        b2 = _beam()
        Simulation(lat2, b2, space_charge="off",
                   rfq_geometry="auto").run()
        alive1 = ~b1.lost
        alive2 = ~b2.lost
        s1 = b1.particles[alive1, 0].std()
        s2 = b2.particles[alive2, 0].std()
        assert not np.isclose(s1, s2, rtol=1e-6)

    def test_explicit_arming_survives_the_run(self, deck):
        lat, _ = parse_tracewin(str(deck))
        n = apply_rfq_geometry(lat, str(deck.parent / "test.vane"),
                               **_BUILD)
        assert n == 35
        Simulation(lat, _beam(), space_charge="off").run()
        armed = [c for c in lat.elements
                 if getattr(c, "_geom_z", None) is not None]
        assert len(armed) == 35

    def test_envelope_never_auto_arms(self, deck, caplog):
        lat, _ = parse_tracewin(str(deck))
        with caplog.at_level(logging.INFO):
            Simulation(lat, _beam(), space_charge="off").run_envelope()
        assert not any("vane-geometry profile" in r.message
                       for r in caplog.records)
        assert all(getattr(c, "_geom_z", None) is None
                   for c in lat.elements)

    def test_bad_mode_string_raises(self, deck):
        lat, _ = parse_tracewin(str(deck))
        with pytest.raises(ValueError):
            Simulation(lat, _beam(), space_charge="off",
                       rfq_geometry="toutatis")
