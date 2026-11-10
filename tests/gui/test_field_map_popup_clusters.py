"""The Results-tab field-map and TTF popups list the maps inside a cluster.

Both popups took only plain FieldMap / FieldMap3D elements, so a deck whose maps
all sit in SUPERPOSE_MAP clusters (a solenoid with its two correctors, a curved
dipole) showed "(no field-map elements)".  Every map of a cluster is listed now,
unpowered correctors too, and each one draws.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

_noop = lambda *a, **k: None                                  # noqa: E731
#: not distributed: three solenoids with their correctors and a curved dipole
_REAL_DECKS = sorted((Path(__file__).resolve().parents[2]).glob(
    "examples/lebt_plus_rfq/*/*_helix.dat"))


def _write_1d(path, vals, zmax_m):
    path.write_text(f"{len(vals) - 1} {zmax_m!r}\n1\n"
                    + "\n".join(repr(float(v)) for v in vals) + "\n",
                    encoding="utf-8")


def _write_3d(prefix, bz_axis, zmax_m, n=4, half=0.016, norm=1):
    """3-D static-B map: Bz(z) uniform across (x, y); Bx = 0.25 Bz; By = 0."""
    nz = len(bz_axis) - 1
    hdr = (f"{nz} {zmax_m!r}\n{n} {-half!r} {half!r}\n{n} {-half!r} {half!r}\n"
           f"{norm}\n")
    for comp, f_ in (("bsx", 0.25), ("bsy", 0.0), ("bsz", 1.0)):
        with open(f"{prefix}.{comp}", "w", encoding="utf-8") as f:
            f.write(hdr)
            for k in range(nz + 1):
                f.write(f"{float(f_ * bz_axis[k])!r}\n" * ((n + 1) * (n + 1)))


def _deck(tmp_path, body):
    from linac_gen.io.tracewin_parser import parse_tracewin
    z = np.linspace(0.0, 0.3, 61)
    bump = 1.0 / (1.0 + ((z - 0.15) / 0.05) ** 4)
    _write_3d(str(tmp_path / "sol3"), bump, 0.3)
    _write_3d(str(tmp_path / "cor3"), 0.02 * np.ones(61), 0.3)
    _write_3d(str(tmp_path / "sol3n"), bump, 0.3, norm=2)
    _write_1d(tmp_path / "cav.edz", np.sin(np.pi * z / 0.3), 0.3)
    p = tmp_path / "d.dat"
    p.write_text("FREQ 162.5\n" + body + "END\n", encoding="utf-8")
    lat, _ = parse_tracewin(str(p))
    return lat


_CLUSTER = ("SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 1.5 0 0 0 sol3\n"
            "SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 0 0 0 0 cor3\n"
            "SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 0 0 0 0 cor3\n"
            "DRIFT 50 16 0\n")


def _popup(lat, key="field_map"):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    state = AppState()
    state.set_lattice(lat, path=None)
    tab = ResultsTab(state, _noop, _noop, _noop)
    tab._open_popup(key)
    return tab._popups[key], tab


def _rows(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _peak_image(pop):
    img = pop._image.image
    return 0.0 if img is None else float(np.nanmax(np.abs(img)))


def test_every_map_of_a_cluster_is_listed_with_file_and_strength(tmp_path, qapp):
    pop, tab = _popup(_deck(tmp_path, _CLUSTER))
    rows = _rows(pop._el_combo)
    assert len(rows) == 3, rows
    assert all(r.startswith("[1] ") and "›" in r for r in rows), rows
    assert "sol3" in rows[0] and "Kb 1.5" in rows[0]
    # the unpowered correctors stay in the list
    assert "cor3" in rows[1] and "Kb 0" in rows[1]
    assert "cor3" in rows[2] and "Kb 0" in rows[2]
    # a static-B map inside a cluster may be a corrector or a dipole: not "solenoid"
    assert not any("solenoid" in r for r in rows), rows
    pop.close()


def test_a_map_inside_a_cluster_draws_the_file_values(tmp_path, qapp):
    """Selecting each child fills the component list and the image; the plot
    holds the file's values, and the info line gives the factor of a run."""
    pop, tab = _popup(_deck(tmp_path, _CLUSTER))
    assert set(_rows(pop._cp_combo)) >= {"Fx", "Fy", "Fz"}
    pop._cp_combo.setCurrentIndex(pop._cp_combo.findText("Fz"))
    assert _peak_image(pop) == pytest.approx(1.0, rel=1e-6)    # not × Kb = 1.5
    assert "the run scales these ×1.5" in pop._info.text()
    pop._el_combo.setCurrentIndex(1)                # an unpowered corrector
    pop._cp_combo.setCurrentIndex(pop._cp_combo.findText("Fz"))
    assert _peak_image(pop) == pytest.approx(0.02, rel=1e-6)   # the map, not 0
    assert "the run scales these ×0" in pop._info.text()
    pop.close()


def test_the_info_line_gives_the_factor_with_the_map_norm(tmp_path, qapp):
    """A run applies Kb / norm: plot × Kb alone would be 2× too large here."""
    lat = _deck(tmp_path, "SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 1.5 0 0 0 sol3n\n"
                          "SUPERPOSE_MAP 0\nFIELD_MAP 70 300 0 16 0 0 0 0 cor3\n"
                          "DRIFT 50 16 0\n")
    pop, tab = _popup(lat)
    pop._cp_combo.setCurrentIndex(pop._cp_combo.findText("Fz"))
    assert _peak_image(pop) == pytest.approx(1.0, rel=1e-6)
    assert "the run scales these ×0.75" in pop._info.text()
    pop.close()


def test_the_info_line_does_not_widen_the_window(tmp_path, qapp):
    """A long info line wraps: the window keeps its default width."""
    pop, tab = _popup(_deck(tmp_path, _CLUSTER))
    pop._info.setText("Fz  ·  shape  ·  geom  ·  stats  " * 40)
    assert pop._info.minimumSizeHint().width() < 600
    pop.close()


def test_a_curved_cluster_lists_its_map(qapp):
    from linac_gen.core.lattice import Lattice
    from tests.elements.test_curved_field_map import _sector
    lat = Lattice()
    lat.add(_sector(30.0))
    pop, tab = _popup(lat)
    rows = _rows(pop._el_combo)
    assert len(rows) == 1 and "›" in rows[0], rows
    pop._cp_combo.setCurrentIndex(pop._cp_combo.findText("Fy"))
    assert _peak_image(pop) > 0.0                   # the bend field, drawn
    pop.close()


def test_a_plain_map_keeps_its_label(tmp_path, qapp):
    lat = _deck(tmp_path, "FIELD_MAP 70 300 0 16 1.5 0 0 0 sol3\nDRIFT 50 16 0\n")
    pop, tab = _popup(lat)
    idx = next(i for i, e in enumerate(lat.elements)
               if type(e).__name__ == "FieldMap3D")
    name = lat.elements[idx].name
    assert _rows(pop._el_combo) == [f"[{idx}] {name}  (FieldMap3D) — solenoid"]
    assert _peak_image(pop) > 0.0
    pop.close()


def test_a_lattice_with_no_map_keeps_the_empty_message(tmp_path, qapp):
    pop, tab = _popup(_deck(tmp_path, "DRIFT 50 16 0\n"))
    assert _rows(pop._el_combo) == ["(no field-map elements)"]
    assert "FIELD_MAP" in pop._info.text()
    pop.close()


def test_refresh_rebuilds_the_list(tmp_path, qapp):
    pop, tab = _popup(_deck(tmp_path, _CLUSTER))
    assert len(_rows(pop._el_combo)) == 3
    other = tmp_path / "other"
    other.mkdir()
    pop._state.set_lattice(_deck(other, "FIELD_MAP 70 300 0 16 1.5 0 0 0 sol3\n"),
                           path=None)
    pop.refresh(None)
    assert len(_rows(pop._el_combo)) == 1
    pop.close()


def test_ttf_lists_a_cavity_inside_a_cluster(tmp_path, qapp):
    body = ("SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
            "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
            "DRIFT 50 16 0\n"
            "FIELD_MAP 100 300 0 16 0 1.0 0 0 cav\n")
    lat = _deck(tmp_path, body)
    pop, tab = _popup(lat, "ttf")
    rows = _rows(pop._el_combo)
    assert len(rows) == 3, rows
    assert "›" in rows[0] and "›" in rows[1] and "›" not in rows[2], rows
    # T(β) is normalised: half the amplitude in a cluster, the same curve
    curves = []
    for i in range(3):
        pop._el_combo.setCurrentIndex(i)
        assert pop._info.text().startswith("β_opt")
        curves.append(np.asarray(pop._curve.getData()[1]))
    assert curves[0].size == 600
    assert np.array_equal(curves[0], curves[2])
    pop.close()


def test_ttf_marks_the_beam_at_a_cavity_inside_a_cluster(tmp_path, qapp):
    """A run records the cluster's name, not the name of a map inside it."""
    from types import SimpleNamespace
    body = ("SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
            "SUPERPOSE_MAP 0\nFIELD_MAP 100 300 0 16 0 0.5 0 0 cav\n"
            "DRIFT 50 16 0\n")
    lat = _deck(tmp_path, body)
    pop, tab = _popup(lat, "ttf")
    assert "β_beam" not in pop._info.text()
    names = [e.name for e in lat.elements]
    pop._state._results = SimpleNamespace(element_names=names,
                                          ref_bg=[0.14] * len(names))
    pop._redraw()
    assert "β_beam=0.1386" in pop._info.text()
    pop.close()


@pytest.mark.skipif(not _REAL_DECKS, reason="undistributed reference deck absent")
def test_the_real_deck_lists_all_ten_maps(qapp):
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(_REAL_DECKS[0]))
    pop, tab = _popup(lat)
    rows = _rows(pop._el_combo)
    assert len(rows) == 10, rows
    assert sum("Kb 0" in r and "Kb 0." not in r for r in rows) == 6, rows
    assert any("DIPOLE" in r and "Kb -0.0685" in r for r in rows), rows
    for i in range(len(rows)):
        pop._el_combo.setCurrentIndex(i)
        peak = max(float(np.nanmax(np.abs(v))) for v in pop._comps.values()
                   if v is not None and v.size)
        assert peak > 0.0, rows[i]
    pop.close()
