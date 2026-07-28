"""Itemized unsaved-changes details for the discard prompt.

Two independently testable halves:
* ``CommandBus.describe_changes_since_clean`` — the lattice side,
  including the save-point marker, undo-past-save divergence, and the
  forced-dirty (wholesale replacement) line;
* ``app._project_dict_diff`` — the project side, a pure dict diff.
"""
from __future__ import annotations

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen_gui.interphase.app import _project_dict_diff
from linac_gen_gui.interphase.commands import (
    CommandBus, DeleteCommand, InsertCommand, ParamChangeCommand,
)


def _bus_with_lattice():
    lat = Lattice()
    lat.add(Drift("D1", 100.0, aperture=20.0))
    lat.add(Quadrupole("Q1", 50.0, gradient=5.0, aperture=20.0, n_steps=5))
    return CommandBus(lambda: lat), lat


def test_fresh_edits_are_itemized(qapp):
    bus, lat = _bus_with_lattice()
    q = lat.elements[1]
    bus.do(ParamChangeCommand(q, "gradient", 5.0, 7.25))
    bus.do(DeleteCommand(lat.elements[0]))
    lines = bus.describe_changes_since_clean()
    assert len(lines) == 2
    assert "Q1" in lines[0] and "5" in lines[0] and "7.25" in lines[0]
    assert "deleted" in lines[1] and "D1" in lines[1]


def test_mark_clean_resets_itemization(qapp):
    bus, lat = _bus_with_lattice()
    q = lat.elements[1]
    bus.do(ParamChangeCommand(q, "gradient", 5.0, 7.0))
    bus.mark_clean()
    assert bus.describe_changes_since_clean() == []
    bus.do(ParamChangeCommand(q, "gradient", 7.0, 8.0))
    lines = bus.describe_changes_since_clean()
    assert len(lines) == 1 and "8" in lines[0]


def test_undo_past_save_point_reports_divergence(qapp):
    bus, lat = _bus_with_lattice()
    q = lat.elements[1]
    bus.do(ParamChangeCommand(q, "gradient", 5.0, 7.0))
    bus.mark_clean()
    bus.undo()                       # now BELOW the save point
    assert bus.dirty
    lines = bus.describe_changes_since_clean()
    assert any("PAST the last save point" in ln for ln in lines)


def test_forced_dirty_gets_wholesale_line(qapp):
    bus, _ = _bus_with_lattice()
    bus.mark_dirty()                 # e.g. matcher applied a result
    lines = bus.describe_changes_since_clean()
    assert any("wholesale" in ln for ln in lines)


def test_insert_describe_and_limit(qapp):
    bus, lat = _bus_with_lattice()
    for i in range(35):
        bus.do(InsertCommand(0, Drift(f"N{i}", 10.0, aperture=20.0)))
    lines = bus.describe_changes_since_clean(limit=30)
    assert len(lines) == 31
    assert "and 5 more" in lines[-1]
    assert "inserted Drift 'N0'" in lines[0]


def test_project_dict_diff_reports_changed_fields():
    saved = {"__kind__": "linac_gen_project", "lattice_path": "a.dat",
             "beam": {"current": 5.0, "energy": 2.1226695,
                      "species": "H-"},
             "convergence": {"grid_nx": 48, "grid_extent_sigma": 7.0}}
    current = {"__kind__": "linac_gen_project", "lattice_path": "a.dat",
               "beam": {"current": 4.8, "energy": 2.1226695,
                        "species": "H-"},
               "convergence": {"grid_nx": 64, "grid_extent_sigma": 7.0}}
    lines = _project_dict_diff(current, saved)
    assert lines == ["beam.current: 5 → 4.8",
                     "convergence.grid_nx: 48 → 64"]


def test_project_dict_diff_float_noise_and_absent_keys():
    saved = {"beam": {"current": 5.0}}
    current = {"beam": {"current": 5.0 + 1e-15}, "new_section": {"k": 1}}
    lines = _project_dict_diff(current, saved)
    assert lines == ["(1 newer-schema field not in the saved file — "
                     "added with the current values on save)"]


def test_project_dict_diff_collapses_schema_noise_to_one_line():
    """One genuine edit must not be buried under schema-default noise
    (user report: 22 '<absent>' lines hid the single real change)."""
    saved = {"beam": {"n_particles": 100000}}
    current = {"beam": {"n_particles": 5000, "disp_x": 0.0, "halo": 0.05},
               "correction": {"enabled": False}}
    lines = _project_dict_diff(current, saved)
    assert lines == [
        "beam.n_particles: 100000 → 5000",
        "(3 newer-schema fields not in the saved file — "
        "added with the current values on save)"]


def test_project_dict_diff_hides_subdisplay_roundtrip_noise():
    """Widget round-trips change floats below display precision; a
    '7.01947 → 7.01947' line informs nothing and must be dropped —
    while a VISIBLE quantization (4.84235 → 4.842) must stay."""
    saved = {"beam": {"alpha_z": 7.0194712345, "current": 4.84235}}
    current = {"beam": {"alpha_z": 7.01947125, "current": 4.842}}
    lines = _project_dict_diff(current, saved)
    assert lines == ["beam.current: 4.84235 → 4.842"]


def test_project_dict_diff_limit():
    saved = {"s": {f"k{i:03d}": 0 for i in range(50)}}
    current = {"s": {f"k{i:03d}": 1 for i in range(50)}}
    lines = _project_dict_diff(current, saved, limit=10)
    assert len(lines) == 11 and "and 40 more" in lines[-1]
