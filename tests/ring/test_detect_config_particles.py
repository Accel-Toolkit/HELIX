"""Ring detection (with negative controls), configuration validation and the
particle container's exact bookkeeping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.ring.config import RingConfig
from linac_gen.ring.detect import detect_ring
from linac_gen.ring.particles import CIRCULATING, LOST, WASTE, RingParticles
from linac_gen.ring.reference import RingReference

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"


def _ring(n_cells=8, frac=1.0, drift_mm=2000.0, rho_mm=10_000.0):
    """n_cells × (drift, bend of 360°/n_cells·frac)."""
    lat = Lattice()
    for i in range(n_cells):
        lat.add(Drift(name=f"d{i}", length=drift_mm))
        lat.add(Dipole(name=f"b{i}", angle=360.0 / n_cells * frac, rho=rho_mm))
    return lat


def test_closed_ring_is_detected():
    v = detect_ring(_ring())
    assert v.is_ring and not v.declared
    assert v.total_angle_deg == pytest.approx(360.0, abs=1e-9)
    assert v.closure_mm < 1e-6


def test_half_ring_and_open_spiral_are_not_rings():
    assert not detect_ring(_ring(frac=0.5)).is_ring
    # 360 degrees of bending but cells of unequal size: the survey does not close
    lat = Lattice()
    for i in range(8):
        lat.add(Drift(name=f"d{i}", length=1000.0 * (1 + i)))
        lat.add(Dipole(name=f"b{i}", angle=45.0, rho=10_000.0))
    v = detect_ring(lat)
    assert v.total_angle_deg == pytest.approx(360.0, abs=1e-9)
    assert not v.is_ring and "not closed" in v.reason


def test_straight_linac_is_a_line():
    lat = Lattice()
    lat.add(Drift(name="d", length=1000.0))
    v = detect_ring(lat)
    assert not v.is_ring and v.reason.startswith("a line")


def test_declaration_wins_and_harmonic_is_found():
    lat = _ring(frac=0.5)
    lat.lg_options = {"ring": True, "harmonic": 84}
    v = detect_ring(lat)
    assert v.is_ring and v.declared and v.harmonic == 84
    lat2 = _ring()
    g = RFGap(name="rf", voltage=0.0, phase=-90.0, frequency=44.7)
    g.harmonic = 84
    lat2.add(g)
    assert detect_ring(lat2).harmonic == 84


def test_imported_booster_is_detected_as_a_ring():
    from linac_gen.io.madx_parser import parse_madx
    lat, meta = parse_madx(str(FIXTURE))
    v = detect_ring(lat, meta)
    assert v.is_ring and v.harmonic == 84
    assert v.length_m == pytest.approx(474.202752, abs=1e-9)


def test_config_validation_and_round_trip():
    c = RingConfig(n_turns=10, harmonic=84, constants="codata2010", snapshot_turns=[0, 5])
    assert RingConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="model"):
        RingConfig(model="other")
    with pytest.raises(ValueError, match="n_turns"):
        RingConfig(n_turns=0)
    with pytest.raises(ValueError, match="unknown key"):
        RingConfig.from_dict({"n_turns": 1, "painting": {}})    # not a setting: refused
    with pytest.raises(ValueError, match="not a table of settings"):
        RingConfig.from_dict({"n_turns": 1, "space_charge": "2.5d"})
    with pytest.raises(ValueError, match="current_ma"):
        RingConfig.from_dict({"n_turns": 1, "injection": {}})


def test_snapshot_turns_statistics_interval_and_survival_text():
    from linac_gen.ring.config import auto_stats_every, parse_snapshot_turns, snapshot_turn_set
    from linac_gen.ring.results import survival_text
    assert parse_snapshot_turns("0, 100;last  7") == [0, 100, "last", 7]
    assert parse_snapshot_turns("") == [] and parse_snapshot_turns("5, 5") == [5]
    for bad in ("-1", "1.5", "first", "²"):
        with pytest.raises(ValueError, match="whole number"):
            parse_snapshot_turns(bad)
    c = RingConfig(snapshot_turns=[0, "LAST", 7.0])
    assert c.snapshot_turns == [0, "last", 7]
    assert RingConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="whole number"):
        RingConfig(snapshot_turns=[True])
    assert snapshot_turn_set([0, "last", 4], 12) == {0, 4, 12}
    # every turn up to 100 000 rows, then every k turns
    assert [auto_stats_every(n) for n in (1, 100_000, 100_001, 10_000_000)] == [1, 1, 2, 100]
    assert RingConfig().source == "matched"
    # a loss never reads 100 %
    assert survival_text(100_000, 99_996) == "99.996 % (4 of 100000 lost)"
    assert survival_text(10, 10) == "100 % (none lost)"
    assert survival_text(10, 0) == "0.0 % (10 of 10 lost)"
    assert survival_text(10**9, 10**9 - 1).startswith("99.9999999 %")


def test_particles_capacity_ledger_and_losses():
    ref = RingReference.from_kinetic("proton", 800.0)
    p = RingParticles.allocate(5)
    sl = p.add(np.zeros((6, 3)), 2.0, ref=ref, turn=0)
    assert sl == slice(0, 3) and p.n_used == 3
    with pytest.raises(ValueError, match="capacity 5 exceeded"):
        p.add(np.zeros((6, 3)), 1.0, ref=ref)
    p.add(np.zeros((6, 2)), 1.0, ref=ref, turn=4, state=WASTE)
    n = p.mark_lost(np.array([True, False, False, True, True]), turn=7, node=3, s=12.5, cause=1)
    assert n == 1                                    # only circulating particles can be lost
    L = p.ledger()
    assert (L["circulating"], L["lost"], L["waste"]) == (2, 1, 2)
    assert L["injected_charge"] == pytest.approx(L["circulating_charge"] + L["lost_charge"] + L["waste_charge"])
    assert p.state[0] == LOST and p.lost_turn[0] == 7 and p.state[1] == CIRCULATING


def test_reference_change_keeps_energy_helpers_consistent():
    old = RingReference.from_kinetic("proton", 800.0)
    new = RingReference.from_kinetic("proton", 810.0)
    p = RingParticles.allocate(3)
    X = np.zeros((6, 3))
    X[5] = [-1e-3, 0.0, 1e-3]
    p.add(X, 1.0, ref=old)
    p.change_reference(old, new)
    d = p.coords[5, :3]
    np.testing.assert_allclose(p.rpp[:3], 1.0 / (1.0 + d), rtol=1e-15)
    # the absolute momentum is unchanged by a change of reference
    np.testing.assert_allclose(new.p0c * (1 + d), old.p0c * (1 + X[5]), rtol=1e-15)
