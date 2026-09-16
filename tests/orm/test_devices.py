"""Device ↔ element mapping and alignment."""
from __future__ import annotations
import numpy as np
import pytest

from linac_gen.elements.marker import Marker
from linac_gen.orm import DeviceMap, align_measured, default_device_map, element_label, resolve_devices
from linac_gen.orm.devices import upstream_rows
from linac_gen.orm.measured import MeasuredOrm


def _measured(kick="x", trims=("L:D01TMH", "L:D02TMH", "L:D99TMH"), bpms=("L:BPH5OT", "L:BPH2OT", "L:D01BPH", "L:D02BPH"), dead=()):
    n_b, n_t = len(bpms), len(trims)
    vals = {"x": np.arange(n_b * n_t, dtype=float).reshape(n_b, n_t) + 1.0}
    return {kick: MeasuredOrm(kick_plane=kick, trims=list(trims), bpms={"x": list(bpms)}, values=vals,
                              errors={"x": np.full((n_b, n_t), 0.01)}, meta={"dead_devices": list(dead)}, source="t")}


def test_default_map_rules(fodo_lattice):
    meas = _measured()
    dm = default_device_map(fodo_lattice, meas)
    assert dm.trims == {"L:D01TMH": "D01T", "L:D02TMH": "D02T", "L:D99TMH": "D99T"}
    assert dm.bpms["L:BPH5OT"] == "D01BPM" and dm.bpms["L:D02BPH"] == "D02BPM" and "L:BPH2OT" not in dm.bpms
    d = DeviceMap.from_dict(dm.to_dict()); assert d.trims == dm.trims and d.bpms == dm.bpms


def test_resolve_lists_unmatched_and_aligns(fodo_lattice):
    # the fodo lattice has D01BPM? no: its BPMs are D011BPM/D012BPM… → map D01BPH explicitly to D011BPM
    meas = _measured(bpms=("L:BPH5OT", "L:BPH2OT", "L:D011BPH", "L:D02BPH"))
    dm = default_device_map(fodo_lattice, meas)
    dm.bpms["L:BPH5OT"] = "D011BPM"; dm.bpms["L:D011BPH"] = "D011BPM"; dm.bpms["L:D02BPH"] = "D012BPM"
    sel = resolve_devices(fodo_lattice, meas, dm)
    assert sel.trim_labels == ["D01T", "D02T"] and sel.unmatched_devices["trim"] == ["L:D99TMH"]
    assert sel.bpm_labels == ["D011BPM", "D012BPM"]
    # tank BPM has no rule; the second device reading D011BPM in the same plane is ignored with a note
    assert sel.unmatched_devices["bpm"] == ["L:BPH2OT", "L:D011BPH"]
    assert any("already read by L:BPH5OT" in n for n in sel.notes)
    assert sel.unmatched_elements["trim"] == ["D03T (STEER_003)", "D04T (STEER_004)"]
    assert sel.down.shape == (2, 2) and sel.down[0, 0] and not sel.down[0, 1]       # D011BPM after D01T, before D02T
    A, E = align_measured(meas, sel, "x", "x")
    assert A.shape == (2, 2) and np.isfinite(A).all() and (E == 0.01).all()
    assert upstream_rows(meas, sel, "x", "x").shape == (2, 3)             # BPH5OT (ignored) and BPH2OT rows


def test_exclusions_sign_dead_and_wrong_kind(fodo_lattice):
    meas = _measured(bpms=("L:D011BPH", "L:D012BPH"), trims=("L:D01TMH", "L:D02TMH"), dead=["L:D012BPH"])
    dm = DeviceMap(bpms={"L:D011BPH": "D011BPM", "L:D012BPH": "D012BPM"}, trims={"L:D01TMH": "D01T", "L:D02TMH": "Q021"},
                   bpm_sign={"L:D011BPH": -1}, exclude_trims={"L:D01TMH"})
    sel = resolve_devices(fodo_lattice, meas, dm)
    assert sel.trim_labels == ["D01T"] and sel.unmatched_devices["trim"] == ["L:D02TMH"]
    assert any("is a Quadrupole, not a steerer" in n for n in sel.notes)
    assert sel.trim_include["x"].tolist() == [False]                      # excluded trim
    assert sel.bpm_include["x"].tolist() == [True, False]                 # dead BPM excluded
    A, _ = align_measured(meas, sel, "x", "x")
    assert np.isnan(A).all()                                              # the only trim is excluded
    dm.exclude_trims = set(); sel = resolve_devices(fodo_lattice, meas, dm)
    A, _ = align_measured(meas, sel, "x", "x")
    assert A[0, 0] == -meas["x"].values["x"][0, 0] and np.isnan(A[1, 0])   # sign applied; dead row NaN


def test_duplicate_labels(fodo_lattice):
    # two steerers carrying the same label at the same s → first wins with a note; at different s → unresolved
    from linac_gen.elements.steerer import Steerer
    twin = Steerer("STEER_TWIN"); twin.label = "D01T"; fodo_lattice.insert(1, twin)
    meas = _measured(bpms=("L:D011BPH",), trims=("L:D01TMH",))
    dm = DeviceMap(bpms={"L:D011BPH": "D011BPM"}, trims={"L:D01TMH": "D01T"})
    sel = resolve_devices(fodo_lattice, meas, dm)
    assert sel.trim_labels == ["D01T"] and any("same s" in n for n in sel.notes)
    far = Steerer("STEER_FAR"); far.label = "D01T"; fodo_lattice.add(far)
    sel = resolve_devices(fodo_lattice, meas, dm)
    assert sel.n_trim == 0 and any("ambiguous" in n for n in sel.notes)
    dm.trims["L:D01TMH"] = "STEER_001"                                     # explicit element name resolves it
    assert resolve_devices(fodo_lattice, meas, dm).n_trim == 1


def test_unlabeled_lattice_uses_names(fodo_lattice):
    for e in fodo_lattice.elements:
        e.label = None
    meas = _measured(bpms=("D011BPM",), trims=("STEER_001",))
    dm = default_device_map(fodo_lattice, meas)
    assert dm.trims == {"STEER_001": "STEER_001"} and dm.bpms == {"D011BPM": "D011BPM"}
    sel = resolve_devices(fodo_lattice, meas, dm)
    assert sel.n_trim == 1 and sel.n_bpm == 1 and element_label(fodo_lattice.elements[0]) == "STEER_001"


def test_plain_marker_is_not_a_bpm(fodo_lattice):
    fodo_lattice.add(Marker("MK"))
    meas = _measured(bpms=("L:MK",), trims=("L:D01TMH",))
    dm = DeviceMap(bpms={"L:MK": "MK"}, trims={"L:D01TMH": "D01T"})
    sel = resolve_devices(fodo_lattice, meas, dm)
    assert sel.n_bpm == 0 and any("plain marker" in n for n in sel.notes)


@pytest.mark.slow
def test_fnalscl_real_deck_mapping():
    import contextlib, io
    from pathlib import Path
    from linac_gen.cli import common
    repo = Path(__file__).resolve().parents[2]
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, _ = common.load_input(str(repo / "examples/piplattice/fnalscl.lgproj"))
    trims = [f"L:D{k}TMH" for k in ("01", "02", "03", "04", "11", "21", "22", "31", "32", "41", "42", "51", "52", "61", "62", "71", "72", "73", "74")]
    bpms = ["L:BPH2OT", "L:BPH5OT"] + [f"L:D{k}BPH" for k in ("02", "03", "11", "12", "13", "21", "22", "23", "31", "32", "33", "34", "41", "42", "43", "44", "51", "52", "53", "54", "61", "62", "63", "64", "71", "72", "73", "74")]
    meas = _measured(trims=trims, bpms=bpms)
    sel = resolve_devices(lat, meas, default_device_map(lat, meas))
    assert sel.n_bpm == 29 and sel.n_trim == 18
    assert sel.unmatched_devices == {"bpm": ["L:BPH2OT"], "trim": ["L:D73TMH"]}
    assert sel.unmatched_elements["trim"] == ["D01T (STEER_002)"] and any("same s" in n for n in sel.notes)
