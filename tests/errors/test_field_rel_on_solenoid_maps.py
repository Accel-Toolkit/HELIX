"""A relative field error on a static magnet field map lands in its ``voltage_rel``.

The Error Study tab's solenoid target draws ``field_rel``; a field map with static magnetic fields
only (a solenoid's, a quadrupole's, a dipole's) has neither ``field`` nor ``field_rel``.  Its field
is ``kb · (1 + voltage_rel)``, so the draw is added there.  Each matched element is drawn before
it is applied, so where a draw lands takes no other draw from the stream.  A map with an RF or
electric channel (a cavity, an unpowered spare) is drawn too, but the draw is not applied (a
warning, once per pattern, parameter and element type); the audit says what lands, and two defs
that write the same slot are overlapping selectors.
"""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.drift import Drift
from linac_gen.errors.error_model import ErrorStudy
from linac_gen.io.tracewin_geom import Channel
from tests.surrogates.accuracy_helpers import W, native
from tests.surrogates.test_frequency_jump import F_NEW, _cavity, _lattice, _magnet, _map


def _mixed(first, second):
    """A map with two channels: ``first`` and a copy of it as ``second``."""
    el = _map(first, F_NEW)
    el.field_data.channels[second] = copy.deepcopy(el.field_data.channels[first])
    return el


def _line():
    sol = _magnet()
    sol.name, sol.ke = "SOL_MAP", 0.0                 # as a solenoid card sets it
    cav = _cavity()
    cav.name = "CAV_MAP"
    return _lattice(Drift("D0", 100.0), sol, Drift("D1", 100.0), cav, Drift("D2", 100.0))


def _study(lat, *errors):
    study = ErrorStudy(copy.deepcopy(lat), BeamConfig(species="proton", energy=W,
                                                      frequency=F_NEW), n_seeds=2)
    for pattern, parameter in errors:
        study.add_error(pattern, parameter, sigma=0.01, cutoff=3.0)
    return study


def test_a_field_error_on_a_solenoid_map_scales_its_field():
    lat = _line()
    study = _study(lat, ("*_MAP", "field_rel"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        seeded = study._apply_errors(0)
    log = {e["element"]: e for e in study.applied_log[0]}
    sol = next(e for e in seeded.elements if e.name == "SOL_MAP")
    assert log["SOL_MAP"]["applied"] and log["SOL_MAP"]["mode"] == "voltage_rel"
    assert sol.voltage_rel == pytest.approx(log["SOL_MAP"]["value"], rel=0, abs=0)
    # the map is the native map with kb scaled by (1 + draw)
    scaled = copy.deepcopy(next(e for e in lat.elements if e.name == "SOL_MAP"))
    scaled.kb *= 1.0 + log["SOL_MAP"]["value"]
    np.testing.assert_allclose(native(sol), native(scaled), rtol=1e-12, atol=1e-14)
    # a cavity map has an electric field: the error is not its to take, and says so
    assert not log["CAV_MAP"]["applied"] and log["CAV_MAP"]["mode"] == "no-slot"
    assert [str(w.message) for w in seen if "CAV_MAP" in str(w.message)]
    assert next(e for e in seeded.elements if e.name == "CAV_MAP").voltage_rel == 0.0


def test_the_draw_adds_to_the_voltage_rel_the_map_has():
    lat = _line()
    next(e for e in lat.elements if e.name == "SOL_MAP").voltage_rel = 0.02
    study = _study(lat, ("SOL_MAP", "field_rel"))
    seeded = study._apply_errors(0)
    draw = study.applied_log[0][0]["value"]
    assert next(e for e in seeded.elements if e.name == "SOL_MAP").voltage_rel == 0.02 + draw


@pytest.mark.parametrize("make, ke, kb, lands", [
    (_magnet, 1.0, 1.0, True),                            # a magnet, ke left at its default
    (_cavity, 0.0, 0.0, False),                           # set as an unpowered spare cavity is
    (lambda: _map(Channel.STAT_E, F_NEW), 1.0, 0.0, False),   # an electrostatic map
    (lambda: _map(Channel.RF_B, F_NEW), 0.0, 1.0, False),     # an RF magnetic field alone
    (lambda: _mixed(Channel.STAT_B, Channel.RF_E), 1.0, 1.0, False),    # a magnet with an RF field
    (lambda: _mixed(Channel.STAT_B, Channel.STAT_E), 1.0, 1.0, False),  # one with a static E field
], ids=["magnet-ke-1", "spare-cavity", "electrostatic", "rf-magnetic", "magnet+rf-electric",
        "magnet+electrostatic"])
def test_the_channels_decide_not_ke(make, ke, kb, lands):
    """A map's own fields say whether it is a static magnet: ``ke`` does not (a magnet map built
    with the class default ``ke = 1``; an unpowered spare cavity, ``ke = kb = 0``, RF channels)."""
    el = make()
    el.name, el.ke, el.kb = "THE_MAP", ke, kb
    study = _study(_lattice(Drift("D0", 100.0), el), ("THE_MAP", "field_rel"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        seeded = study._apply_errors(0)
    row = study.applied_log[0][0]
    vrel = next(e for e in seeded.elements if e.name == "THE_MAP").voltage_rel
    assert row["applied"] is lands and (vrel == row["value"] if lands else vrel == 0.0)
    assert bool([w for w in seen if "THE_MAP" in str(w.message)]) is not lands
    assert study.audit_errors()["defs"][0]["applicable"] == (["THE_MAP"] if lands else [])


def _cluster(*children):
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    return SuperposedFieldMap("SUP_MAP", [(0.0, child) for child in children])


def _solenoid_and_corrector():
    sol = _magnet()
    sol.name, sol.ke = "SOL", 0.0
    cor = _map(Channel.STAT_B, F_NEW)
    cor.name, cor.ke, cor.kb = "COR", 0.0, 0.0                 # an unpowered corrector
    return sol, cor


def test_a_superposed_cluster_of_static_magnets_takes_it_in_its_own_voltage_rel():
    """A SUPERPOSE_MAP cluster of a solenoid map and an unpowered corrector: the draw lands in
    the cluster's voltage_rel, which scales every child — here as scaling the solenoid alone."""
    lat = _lattice(Drift("D0", 100.0), _cluster(*_solenoid_and_corrector()), Drift("D1", 100.0))
    study = _study(lat, ("SUP_MAP", "field_rel"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        seeded = study._apply_errors(0)
    row = study.applied_log[0][0]
    sup = next(e for e in seeded.elements if e.name == "SUP_MAP")
    assert row["applied"] and row["mode"] == "voltage_rel" and sup.voltage_rel == row["value"]
    assert not [w for w in seen if "SUP_MAP" in str(w.message)]
    scaled = copy.deepcopy(next(e for e in lat.elements if e.name == "SUP_MAP"))
    scaled.children[0][1].kb *= 1.0 + row["value"]
    np.testing.assert_allclose(native(sup), native(scaled), rtol=1e-12, atol=1e-14)
    assert study.audit_errors()["defs"][0]["applicable"] == ["SUP_MAP"]


def test_a_cluster_with_an_rf_map_skips_it():
    sol, _cor = _solenoid_and_corrector()
    cav = _cavity()
    cav.name = "CAV"
    lat = _lattice(Drift("D0", 100.0), _cluster(sol, cav), Drift("D1", 100.0))
    study = _study(lat, ("SUP_MAP", "field_rel"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        seeded = study._apply_errors(0)
    assert not study.applied_log[0][0]["applied"]
    assert next(e for e in seeded.elements if e.name == "SUP_MAP").voltage_rel == 0.0
    assert [w for w in seen if "SUP_MAP" in str(w.message)]


def test_a_cluster_with_a_map_that_also_has_an_electric_field_skips_it():
    """Every child must be a static magnet map: a child with a static magnetic and a static
    electric channel is not, so the cluster keeps the warning."""
    sol, _cor = _solenoid_and_corrector()
    both = _mixed(Channel.STAT_B, Channel.STAT_E)
    both.name = "BOTH"
    lat = _lattice(Drift("D0", 100.0), _cluster(sol, both), Drift("D1", 100.0))
    study = _study(lat, ("SUP_MAP", "field_rel"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        seeded = study._apply_errors(0)
    assert not study.applied_log[0][0]["applied"]
    assert next(e for e in seeded.elements if e.name == "SUP_MAP").voltage_rel == 0.0
    assert [w for w in seen if "SUP_MAP" in str(w.message)]
    assert study.audit_errors()["defs"][0]["applicable"] == []


def test_field_rel_and_voltage_rel_on_one_map_overlap():
    """Both write the map's voltage_rel: two defs drawing them are overlapping selectors, as two
    voltage_rel defs are."""
    lat = _line()
    audit = _study(lat, ("SOL_MAP", "voltage_rel"), ("SOL_MAP", "field_rel")).audit_errors()
    assert audit["multi"] == {("SOL_MAP", "voltage_rel"): [0, 1]}


def test_the_draws_do_not_depend_on_where_they_land(monkeypatch):
    """Each matched element is drawn before its value is applied: the solenoid map's draw
    landing takes no other draw (this one's, or a later error's) from the stream."""
    lat = _line()
    landed = _study(lat, ("*_MAP", "field_rel"), ("*_MAP", "dx"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        landed._apply_errors(1)
    dropped = _study(lat, ("*_MAP", "field_rel"), ("*_MAP", "dx"))
    monkeypatch.setattr(ErrorStudy, "_apply_draw",
                        lambda self, elem, err, val, log: log(elem, err, val, False, "dropped"))
    dropped._apply_errors(1)

    def draws(study):
        return [(e["element"], e["parameter"], e["value"]) for e in study.applied_log[1]]
    assert draws(landed) == draws(dropped) and len(draws(landed)) == 4


def test_the_audit_says_what_lands():
    lat = _line()
    audit = _study(lat, ("*_MAP", "field_rel")).audit_errors()
    row = audit["defs"][0]
    assert row["matched"] == ["SOL_MAP", "CAV_MAP"]
    assert row["applicable"] == ["SOL_MAP"]
    assert row["skipped"] == [("CAV_MAP", "FieldMap3D")]


def test_a_curved_cluster_skips_it():
    """A curved cluster (SUPERPOSE_MAP_OUT) traces its reference path through its fields: a
    field error there is not a scaled straight map, and it is left alone, with a warning."""
    from linac_gen.elements.curved_field_map import CurvedFieldMapCluster
    sol, cor = _solenoid_and_corrector()
    lat = _lattice(Drift("D0", 100.0),
                   CurvedFieldMapCluster("SUP_MAP", [(0.0, sol), (0.0, cor)]), Drift("D1", 100.0))
    study = _study(lat, ("SUP_MAP", "field_rel"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        study._apply_errors(0)
    assert not study.applied_log[0][0]["applied"]
    assert [w for w in seen if "SUP_MAP" in str(w.message)]
