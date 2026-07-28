# tests/io/test_diag_targets_file.py
"""External BPM-target file loader (diagnostic matching set-points)."""
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.io.diag_targets import (apply_diag_targets,
                                       clear_diag_targets,
                                       load_diag_targets)


def _lat(n_bpm=2):
    lat = Lattice()
    for i in range(n_bpm):
        lat.add(Drift(f"D{i}", 100.0))
        lat.add(Marker(f"B{i}", is_bpm=True,
                       diag_family=1, x_target_mm=9.0, y_target_mm=9.0))
    return lat


def test_load_two_and_three_columns(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("# measured orbit\n"
                 "0.1 -0.2\n"
                 "0.3  0.4  2.5   # weighted\n")
    rows = load_diag_targets(str(f))
    assert rows == [(0.1, -0.2, None), (0.3, 0.4, 2.5)]


def test_nan_disables_plane(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("nan 0.5\n0.1 nan\n")
    rows = load_diag_targets(str(f))
    assert rows[0] == (None, 0.5, None)
    assert rows[1] == (0.1, None, None)


def test_bad_column_count_raises(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("0.1\n")
    with pytest.raises(ValueError, match="x_mm y_mm"):
        load_diag_targets(str(f))


def test_count_mismatch_raises(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("0.1 0.2\n")
    with pytest.raises(ValueError, match="1 row.*2 BPM"):
        apply_diag_targets(_lat(2), load_diag_targets(str(f)))


def test_apply_sets_overrides_and_precedence(tmp_path):
    from linac_gen.matching.constraints import _effective_targets
    f = tmp_path / "t.txt"
    f.write_text("0.1 -0.2 3.0\n0.3 0.4\n")
    lat = _lat(2)
    n = apply_diag_targets(lat, load_diag_targets(str(f)))
    assert n == 2
    b0 = next(e for e in lat.elements if e.name == "B0")
    tx, ty, w = _effective_targets(b0)
    assert (tx, ty, w) == (0.1, -0.2, 3.0)   # override beats deck 9.0
    assert clear_diag_targets(lat) == 2
    tx, ty, _ = _effective_targets(b0)
    assert (tx, ty) == (9.0, 9.0)            # deck targets restored


def test_nan_nan_row_frees_both_planes_not_deck_fallback(tmp_path):
    """Override PRESENCE wins: 'nan nan' must free both planes, never
    silently steer back onto the deck targets it was meant to disable."""
    from linac_gen.errors.correction import _resolve_targets
    from linac_gen.matching.constraints import _effective_targets
    f = tmp_path / "t.txt"
    f.write_text("nan nan\n0.3 0.4\n")
    lat = _lat(2)
    apply_diag_targets(lat, load_diag_targets(str(f)))
    b0 = next(e for e in lat.elements if e.name == "B0")
    assert _effective_targets(b0)[:2] == (None, None)
    bpms = [e for e in lat.elements if getattr(e, "is_bpm", False)]
    resolved = _resolve_targets(bpms, "deck")
    assert resolved["B0"] == (None, None)     # both planes excluded
    assert resolved["B1"] == (0.3, 0.4)


def test_overrides_not_serialized_by_writer(tmp_path):
    from linac_gen.io.tracewin_writer import write_tracewin
    lat = _lat(1)
    apply_diag_targets(lat, [(1.234, -5.678, None)])
    out = tmp_path / "o.dat"
    write_tracewin(lat, str(out))
    txt = out.read_text()
    assert "1.234" not in txt and "5.678" not in txt
    assert "9" in txt                         # deck targets still emitted
