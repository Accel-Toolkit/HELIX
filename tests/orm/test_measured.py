"""FORMA folder and HELIX CSV readers."""
from __future__ import annotations
import json
import numpy as np
import pytest

from linac_gen.orm import MeasuredOrm, load_measured, read_forma_folder, read_orm_csv, write_orm_csv
from tests.orm.conftest import make_forma_folder

TRIMS = ["L:D01TMH", "L:D02TMH", "L:D03TMH"]
BPMS = {"x": ["L:BPH5OT", "L:D02BPH", "L:D03BPH"], "y": ["L:BPV5OT", "L:D02BPV", "L:D03BPV"]}


def _mats(seed=1):
    rng = np.random.default_rng(seed)
    return {p: rng.normal(size=(3, 3)) for p in "xy"}, {p: np.abs(rng.normal(size=(3, 3))) * 0.01 for p in "xy"}


def test_read_forma_folder(tmp_path):
    A, E = _mats()
    folder = make_forma_folder(tmp_path, stamp="20260826_134102", kick_plane="x", trims=TRIMS, bpms=BPMS, values=A, errors=E, dead=["L:D04BPH"])
    m = read_forma_folder(folder)
    assert m.kick_plane == "x" and m.trims == TRIMS and m.units == "mm/A"
    assert m.bpms["x"] == BPMS["x"] and m.bpms["y"] == BPMS["y"]
    np.testing.assert_array_equal(m.values["x"], A["x"]); np.testing.assert_array_equal(m.errors["y"], E["y"])
    assert m.meta["stamp"] == "20260826_134102" and m.meta["dead_devices"] == ["L:D04BPH"]
    assert set(m.meta["corrector_gains"]) == set(TRIMS) and m.has_errors("x")
    assert "kick plane x" in m.summary()


def test_missing_error_file_gives_nan_and_warning(tmp_path):
    A, _ = _mats()
    folder = make_forma_folder(tmp_path, stamp="s", kick_plane="x", trims=TRIMS, bpms=BPMS, values=A, errors=None)
    with pytest.warns(UserWarning, match="no horizontal error file"):
        m = read_forma_folder(folder)
    assert np.isnan(m.errors["x"]).all() and not m.has_errors("x")


def test_error_file_row_mismatch_is_refused(tmp_path):
    A, E = _mats()
    folder = make_forma_folder(tmp_path, stamp="s", kick_plane="x", trims=TRIMS, bpms=BPMS, values=A, errors=E)
    f = folder / "response_matrix_s_horizontal_error.csv"
    lines = f.read_text().splitlines(); lines[1], lines[2] = lines[2], lines[1]      # swap two BPM rows
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="device order differs"):
        read_forma_folder(folder)


def test_mixed_planes_refused_and_plane_override(tmp_path):
    A, E = _mats()
    folder = make_forma_folder(tmp_path, stamp="s", kick_plane="x", trims=["L:D01TMH", "L:D02TMV", "L:D03TMH"], bpms=BPMS, values=A, errors=E)
    with pytest.raises(ValueError, match="driven plane"):
        read_forma_folder(folder)
    assert read_forma_folder(folder, plane="y").kick_plane == "y"


def test_bom_and_latin1_tolerated(tmp_path):
    A, E = _mats()
    folder = make_forma_folder(tmp_path, stamp="s", kick_plane="y", trims=[t.replace("H", "V") for t in TRIMS], bpms=BPMS, values=A, errors=E)
    f = folder / "response_matrix_s_vertical.csv"
    f.write_bytes(b"\xef\xbb\xbf" + f.read_bytes())            # UTF-8 BOM (Windows export)
    (folder / "scan_info.json").write_bytes(b"{\"note\": \"d\xe9viation\"}")   # stray latin-1 byte in metadata
    m = read_forma_folder(folder)
    assert m.kick_plane == "y" and m.trims[0] == "L:D01TMV"


def test_helix_csv_round_trip(tmp_path):
    A = np.array([[1.5, -2.25e-3], [np.nan, 4e5]]); E = np.array([[0.1, 0.2], [0.3, np.nan]])
    write_orm_csv(tmp_path / "orm_xx.csv", A, ["D01BPM", "D02BPM"], ["D01T", "D02T"], kick_plane="x", read_plane="x", units="mm/A", errors=E, header_lines=["source: test"])
    m = read_orm_csv(tmp_path / "orm_xx.csv")
    assert isinstance(m, MeasuredOrm) and m.kick_plane == "x" and m.read_planes() == ["x"] and m.units == "mm/A"
    np.testing.assert_array_equal(m.values["x"], A); np.testing.assert_array_equal(m.errors["x"], E)
    assert m.bpms["x"] == ["D01BPM", "D02BPM"] and m.trims == ["D01T", "D02T"]
    with pytest.raises(ValueError, match="not a HELIX ORM CSV"):
        (tmp_path / "bad.csv").write_text("bpm,D01T\nD01BPM,1\n"); read_orm_csv(tmp_path / "bad.csv")


def test_load_measured_merges_and_refuses_duplicates(tmp_path):
    A, E = _mats()
    fx = make_forma_folder(tmp_path, stamp="sx", kick_plane="x", trims=TRIMS, bpms=BPMS, values=A, errors=E)
    fy = make_forma_folder(tmp_path, stamp="sy", kick_plane="y", trims=[t.replace("H", "V") for t in TRIMS], bpms=BPMS, values=A, errors=E)
    meas = load_measured([str(fx), str(fy)])
    assert set(meas) == {"x", "y"} and meas["y"].trims[0] == "L:D01TMV"
    write_orm_csv(tmp_path / "a.csv", A["x"], BPMS["x"], TRIMS, kick_plane="x", read_plane="x")
    write_orm_csv(tmp_path / "b.csv", A["y"], BPMS["y"], TRIMS, kick_plane="x", read_plane="y")
    merged = load_measured([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])
    assert merged["x"].read_planes() == ["x", "y"] and merged["x"].meta["merged_from"]
    with pytest.raises(ValueError, match="already loaded"):
        load_measured([str(tmp_path / "a.csv"), str(tmp_path / "a.csv")])
    with pytest.raises(ValueError, match="already loaded"):
        load_measured([str(fx), str(tmp_path / "a.csv")])


def test_kick_plane_from_scan_info_when_names_carry_no_suffix(tmp_path):
    """Synthetic / non-Fermilab exports: trims without an H/V suffix, plane declared in scan_info.json."""
    folder = make_forma_folder(tmp_path, stamp="20260101_000000", kick_plane="y", trims=["D01T", "D02T"], bpms={"x": ["D011BPM", "D012BPM"], "y": ["D011BPM", "D012BPM"]},
                               values={"x": np.zeros((2, 2)), "y": np.ones((2, 2))})
    with pytest.raises(ValueError, match="cannot infer the driven plane"):
        read_forma_folder(folder)
    (folder / "scan_info.json").write_text(json.dumps({"forma_version": "synthetic", "kick_plane": "y"}), encoding="utf-8")
    m = read_forma_folder(folder)
    assert m.kick_plane == "y" and m.trims == ["D01T", "D02T"]
    assert read_forma_folder(folder, plane="x").kick_plane == "x"          # the explicit argument still wins
