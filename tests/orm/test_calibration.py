"""Calibration JSON, apply/revert, and the recalibrated-deck export with re-parse verification."""
from __future__ import annotations
import contextlib, io, json, os, shutil
from pathlib import Path

import numpy as np
import pytest

from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.orm.calibration import (apply_calibration, apply_changes, calibration_from_fit, export_recalibrated_deck,
                                       load_calibration, revert_changes, save_calibration, verify_exported_deck)
from linac_gen.orm.fit import OrmFitOptions, fit_orm
from linac_gen.orm.model import OrmModel
from linac_gen.orm.devices import DeviceMap, resolve_devices
from tests.orm.test_fit import _setup

REPO = Path(__file__).resolve().parents[2]
DECK = (
    "; test deck\r\n"
    "FREQ 352.2\r\n"
    "D01T: THIN_STEERING 0 0 20 0\r\n"
    "DRIFT 300 20 0\r\n"
    "Q011: QUAD 100 12 20 ; d\xe9viation\r\n"
    "DRIFT 300 20 0\r\n"
    "D011BPM: DIAG_POSITION 1\r\n"
    "DRIFT 300 20 0\r\n"
    "Q012: QUAD 100 -12 20\r\n"
    "DRIFT 300 20 0\r\n"
    "D012BPM: DIAG_POSITION 2\r\n"
    "END\r\n"
)


def _fit_case():
    meas, model, truth = _setup(n_cells=3, g_true=1.0 + np.array([0.03, -0.02, 0.01, 0.0, 0.02, -0.03]), seed=11)
    fit = fit_orm(model, meas, OrmFitOptions(stage="trims+quads"))
    return meas, model, fit


def test_calibration_record_round_trip(tmp_path):
    meas, model, fit = _fit_case()
    cal = calibration_from_fit(fit, model, meas, lattice_path=None, beam_cfg=model.beam_cfg)
    assert cal["__kind__"] == "helix_orm_calibration" and len(cal["quads"]) == 6 and len(cal["trims"]) == 3 and len(cal["bpms"]) == 6
    assert cal["metrics"]["after"]["x"]["nrms_all"] <= cal["metrics"]["before"]["x"]["nrms_all"]
    save_calibration(tmp_path / "cal.json", cal)
    back = load_calibration(tmp_path / "cal.json")
    assert back == json.loads(json.dumps(cal))                   # JSON-safe (no NaN, no numpy)
    json.dumps(cal, allow_nan=False)
    with pytest.raises(ValueError, match="not a HELIX ORM calibration"):
        (tmp_path / "x.json").write_text("{}"); load_calibration(tmp_path / "x.json")


def test_apply_rel_and_bake_and_revert():
    meas, model, fit = _fit_case()
    cal = calibration_from_fit(fit, model, meas)
    quads = model.quads; g0 = [q.gradient for q in quads]
    ch = apply_calibration(model.lattice, cal, mode="rel")
    assert all(q.gradient_rel == 0.0 for q in quads)             # nothing mutated yet
    apply_changes(ch)
    np.testing.assert_allclose([q.gradient_rel for q in quads], fit.g - 1.0, atol=1e-15)
    assert [q.gradient for q in quads] == g0
    revert_changes(ch); assert all(q.gradient_rel == 0.0 for q in quads)
    ch = apply_calibration(model.lattice, cal, mode="bake"); apply_changes(ch)
    np.testing.assert_allclose([q.gradient for q in quads], np.array(g0) * fit.g)
    revert_changes(ch); assert [q.gradient for q in quads] == g0
    quads[0].gradient *= 1.01
    with pytest.raises(ValueError, match="design gradient"):
        apply_calibration(model.lattice, cal)
    quads[0].gradient = g0[0]
    cal["quads"][1]["label"] = "Q99"; cal["quads"][1]["name"] = "NOPE"; cal["quads"][1]["index"] = 0
    with pytest.raises(ValueError, match="matching quadrupole"):
        apply_calibration(model.lattice, cal)


def _cal_for_deck(tmp_path, scales):
    """A minimal calibration record for DECK's two quads."""
    return {"__kind__": "helix_orm_calibration", "__version__": 1, "measured": [], "fit": {"stage": "trims+quads"},
            "metrics": {"before": {"x": {"nrms_all": 0.5}}, "after": {"x": {"nrms_all": 0.05}}},
            "quads": [{"label": "Q011", "name": "QUAD_001", "index": 4, "gradient_design": 12.0, "scale": scales[0], "scale_err": 0.002, "fixed": False},
                      {"label": "Q012", "name": "QUAD_002", "index": 8, "gradient_design": -12.0, "scale": scales[1], "scale_err": None, "fixed": scales[1] == 1.0}]}


def test_export_text_surgery_preserves_everything_else(tmp_path):
    src = tmp_path / "src.dat"; src.write_bytes(DECK.encode("latin-1"))
    cal = _cal_for_deck(tmp_path, (1.05, 1.0))
    rep = export_recalibrated_deck(src, tmp_path / "out.dat", cal, stamp="2026-09-14")
    out = (tmp_path / "out.dat").read_bytes()
    assert b"\r\n" in out and b"d\xe9viation" in out                     # CRLF and the latin-1 byte survive
    body = out.split(b"\r\n")[rep["header_lines"]:]
    src_lines = DECK.encode("latin-1").split(b"\r\n")
    assert len(body) == len(src_lines)
    diff = [(a, b) for a, b in zip(src_lines, body) if a != b]
    assert len(diff) == 1 and diff[0][0].startswith(b"Q011: QUAD 100 12 20")
    assert diff[0][1].startswith(b"Q011: QUAD 100 12.6 20 ; d\xe9viation\t; ORM fit 2026-09-14: x1.05000")
    assert rep["n_quads_checked"] == 2 and rep["changed"][0][0] == "Q011"
    lat = parse_tracewin(str(tmp_path / "out.dat"))[0]
    assert [q.gradient for q in lat.elements if isinstance(q, Quadrupole)] == [12.6, -12.0]


def test_export_refusals_and_ordinal_fallback(tmp_path):
    src = tmp_path / "src.dat"; src.write_bytes(DECK.encode("latin-1"))
    cal = _cal_for_deck(tmp_path, (1.05, 0.98))
    drift = src.read_bytes().replace(b"Q011: QUAD 100 12 20", b"Q011: QUAD 100 12.3 20"); (tmp_path / "drift.dat").write_bytes(drift)
    with pytest.raises(ValueError, match="not the one that was fitted"):
        export_recalibrated_deck(tmp_path / "drift.dat", tmp_path / "o.dat", cal)
    dup = src.read_bytes().replace(b"Q012: QUAD", b"Q011: QUAD"); (tmp_path / "dup.dat").write_bytes(dup)
    with pytest.raises(ValueError, match="appears on 2 QUAD lines"):
        export_recalibrated_deck(tmp_path / "dup.dat", tmp_path / "o.dat", cal)
    # unlabeled deck: ordinal fallback
    bare = src.read_bytes().replace(b"Q011: ", b"").replace(b"Q012: ", b""); (tmp_path / "bare.dat").write_bytes(bare)
    cal2 = _cal_for_deck(tmp_path, (1.05, 0.98))
    for e in cal2["quads"]:
        e["label"] = None
    rep = export_recalibrated_deck(tmp_path / "bare.dat", tmp_path / "bare_out.dat", cal2, stamp="s")
    lat = parse_tracewin(str(tmp_path / "bare_out.dat"))[0]
    np.testing.assert_allclose([q.gradient for q in lat.elements if isinstance(q, Quadrupole)], [12.6, -11.76])
    assert rep["changed"][1][0] == "QUAD #2"


def test_export_fnalscl_only_fitted_lines_change(tmp_path):
    src = REPO / "examples/piplattice/fnalscl.dat"
    cwd = os.getcwd(); os.chdir(src.parent)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lat = parse_tracewin(str(src))[0]
    finally:
        os.chdir(cwd)
    quads = [q for q in lat.elements if isinstance(q, Quadrupole)]
    pick = [(q, s) for q, s in zip(quads[:8], (1.0, 0.96, 1.02, 1.0, 1.05, 1.0, 0.9, 1.0))]
    cal = {"__kind__": "helix_orm_calibration", "__version__": 1, "measured": [{"source": "t", "stamp": "0"}], "fit": {"stage": "trims+quads"},
           "metrics": {"before": {"x": {"nrms_all": 0.5}}, "after": {"x": {"nrms_all": 0.04}}},
           "quads": [{"label": q.label, "name": q.name, "index": lat.elements.index(q), "gradient_design": q.gradient, "scale": s, "scale_err": 0.01, "fixed": False} for q, s in pick]}
    shutil.copy(src, tmp_path / "fnalscl.dat"); shutil.copy(REPO / "examples/piplattice/fnalscl.lgproj", tmp_path / "fnalscl.lgproj")
    rep = export_recalibrated_deck(tmp_path / "fnalscl.dat", tmp_path / "fnalscl_fit.dat", cal, stamp="s", lgproj_src=tmp_path / "fnalscl.lgproj")
    src_lines = (tmp_path / "fnalscl.dat").read_bytes().split(b"\n"); out_lines = (tmp_path / "fnalscl_fit.dat").read_bytes().split(b"\n")[rep["header_lines"]:]
    assert len(src_lines) == len(out_lines)
    changed = [i for i, (a, b) in enumerate(zip(src_lines, out_lines)) if a != b]
    assert len(changed) == 4 and all(out_lines[i].startswith((b"Q02: QUAD", b"Q03: QUAD", b"Q11: QUAD", b"Q13: QUAD")) for i in changed)
    assert json.loads((tmp_path / "fnalscl_fit.lgproj").read_text())["lattice_path"] == "fnalscl_fit.dat"
    assert rep["n_quads_checked"] == 8 and rep["max_rel_dev"] < 1e-9     # tokens are written with 10 significant digits


def test_exported_deck_reproduces_the_fitted_orm(tmp_path):
    """Round trip: fit on a written deck, export, reload, model of the exported deck == G·k·M(g) of the fit."""
    src = tmp_path / "src.dat"; src.write_bytes(DECK.encode("latin-1"))
    with contextlib.redirect_stdout(io.StringIO()):
        lat = parse_tracewin(str(src))[0]
    from linac_gen.orm.measured import MeasuredOrm
    from tests.orm.test_model import _cfg
    devs_b, devs_t = ["L:D011BPH", "L:D012BPH"], ["L:D01TMH"]
    dm = DeviceMap(bpms={"L:D011BPH": "D011BPM", "L:D012BPH": "D012BPM"}, trims={"L:D01TMH": "D01T"})
    stub = {"x": MeasuredOrm(kick_plane="x", trims=devs_t, bpms={"x": devs_b}, values={"x": np.zeros((2, 1))}, errors={"x": np.zeros((2, 1))})}
    sel = resolve_devices(lat, stub, dm); model = OrmModel(lat, _cfg(), sel)
    R_true = model.response(np.array([1.04, 0.97])); A = -5e-4 * R_true.per_Tm[("x", "x")]
    meas = {"x": MeasuredOrm(kick_plane="x", trims=devs_t, bpms={"x": devs_b}, values={"x": A}, errors={"x": np.full_like(A, 1e-6)})}
    sel = resolve_devices(lat, meas, dm); model = OrmModel(lat, _cfg(), sel)
    fit = fit_orm(model, meas, OrmFitOptions(stage="trims+quads", prior_g=100.0, sys_floor=1e-4, rel_tol=1e-12, max_iter=40, min_bpms_per_quad=1))
    cal = calibration_from_fit(fit, model, meas, lattice_path=src)
    export_recalibrated_deck(src, tmp_path / "out.dat", cal)
    with contextlib.redirect_stdout(io.StringIO()):
        lat2 = parse_tracewin(str(tmp_path / "out.dat"))[0]
    sel2 = resolve_devices(lat2, meas, dm); R2 = OrmModel(lat2, _cfg(), sel2).response()
    np.testing.assert_allclose(fit.k["x"][None, :] * R2.per_Tm[("x", "x")], fit.fitted_block("x"), rtol=1e-9)
    np.testing.assert_allclose(fit.k["x"][None, :] * R2.per_Tm[("x", "x")], A, rtol=1e-5)


def test_export_accepts_relative_paths(tmp_path, monkeypatch):
    """The verifier chdirs into the deck folder (field maps resolve there): relative src/dst must still work."""
    meas, model, fit = _fit_case()
    (tmp_path / "in").mkdir(); (tmp_path / "out").mkdir()
    (tmp_path / "in" / "fodo.dat").write_bytes(DECK.encode("latin-1"))
    lat = parse_tracewin(str(tmp_path / "in" / "fodo.dat"))[0]
    cal = calibration_from_fit(fit, model, meas)
    # bind the calibration to this deck: the labels of the 3-cell fixture start with the deck's two quads
    cal["quads"] = [dict(q) for q in cal["quads"] if q["label"] in ("Q011", "Q012")]
    monkeypatch.chdir(tmp_path)
    rep = export_recalibrated_deck("in/fodo.dat", "out/fodo_fit.dat", cal, stamp="s")
    assert (tmp_path / "out" / "fodo_fit.dat").exists() and rep["n_quads_checked"] == 2
    assert verify_exported_deck("in/fodo.dat", "out/fodo_fit.dat", cal)["max_rel_dev"] < 1e-9


def test_apply_rel_is_idempotent_and_bake_after_rel_is_refused():
    """Review finding: applying twice compounded the scale factors (scale²−1); now the second apply is a no-op."""
    meas, model, fit = _fit_case()
    cal = calibration_from_fit(fit, model, meas)
    ch1 = apply_calibration(model.lattice, cal, mode="rel"); apply_changes(ch1)
    rel_once = [q.gradient_rel for q in model.quads]
    assert apply_calibration(model.lattice, cal, mode="rel") == []                  # already at the target
    assert [q.gradient_rel for q in model.quads] == rel_once
    with pytest.raises(ValueError, match="undo the applied calibration"):
        apply_calibration(model.lattice, cal, mode="bake")
    revert_changes(ch1)
    assert all(q.gradient_rel == 0.0 for q in model.quads)
    # a foreign gradient_rel (error study, other calibration) is refused atomically
    model.quads[1].gradient_rel = 0.05
    with pytest.raises(ValueError, match="neither the calibration's starting value"):
        apply_calibration(model.lattice, cal, mode="rel")
    model.quads[1].gradient_rel = 0.0
    # a calibration fitted on a lattice that already carried gradient_rel applies from that value only
    for q in model.quads:
        q.gradient_rel = 0.02
    model2 = OrmModel(model.lattice, model.beam_cfg, model.sel)
    fit2 = fit_orm(model2, meas, OrmFitOptions(stage="trims+quads"))
    cal2 = calibration_from_fit(fit2, model2, meas)
    ch = apply_calibration(model.lattice, cal2, mode="rel")
    for q, attr, old, new in ch:
        assert attr == "gradient_rel" and old == 0.02 and abs(new - ((1.02) * fit2.g[model2.quads.index(q)] - 1.0)) < 1e-12
    for q in model.quads:
        q.gradient_rel = 0.0


def test_load_calibration_validates_entries(tmp_path):
    bad = {"__kind__": "helix_orm_calibration", "__version__": 1, "quads": [{"label": "Q1", "scale": None, "gradient_design": 12.0, "gradient_rel": 0.0}]}
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="'scale' must be a finite number"):
        load_calibration(tmp_path / "bad.json")
    bad["quads"] = ["Q1"]
    (tmp_path / "bad2.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="not a calibration entry"):
        load_calibration(tmp_path / "bad2.json")
