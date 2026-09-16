"""``python -m linac_gen orm`` end to end on the synthetic FORMA export of the demo FODO."""
from __future__ import annotations
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linac_gen.__main__ import main

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "orm_demo"


def _synth_main():
    spec = importlib.util.spec_from_file_location("make_synthetic_forma", DEMO / "make_synthetic_forma.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.main


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orm_cli")
    assert _synth_main()(["--out", str(tmp / "synthetic"), "--g-sigma", "0.02", "--noise", "0.005", "--seed", "3"]) == 0
    return dict(tmp=tmp, proj=DEMO / "fodo_orm.lgproj", h=tmp / "synthetic/20260101_000000_H", v=tmp / "synthetic/20260101_000000_V")


def _deck_body(path: Path) -> list:
    """Deck lines without the provenance header and with the date stamp of the inline comments removed."""
    out = []
    for ln in path.read_bytes().split(b"\n"):
        if ln.startswith(b"; DERIVED DECK") or ln.startswith(b";   "):
            continue
        out.append(re.sub(rb"; ORM fit \S+:", b"; ORM fit X:", ln))
    return out


def test_compare_writes_outputs_and_plots(case, capsys):
    out = case["tmp"] / "cmp"
    rc = main(["orm", "compare", str(case["proj"]), "--measured", str(case["h"]), "--measured", str(case["v"]),
               "--out", str(out), "--plots", "--check-tracking", "--write-map", str(out / "map.json")])
    txt = capsys.readouterr().out
    assert rc == 0 and "plane x:" in txt and "plane y:" in txt and "kick-and-read cross-check" in txt
    for f in ("orm_compare.json", "orm_model_xx.csv", "orm_measured_yy.csv", "orm_measured_yy_error.csv", "orm_trim_calibration.csv",
              "orm_fig1_heatmaps.png", "orm_fig2_trends_H.png", "orm_fig2_trends_V.png", "orm_fig4_calibration.png", "map.json"):
        assert (out / f).exists(), f
    cmp = json.loads((out / "orm_compare.json").read_text(encoding="utf-8"))     # strict JSON (no NaN literals)
    assert cmp["reason"] is None and len(cmp["planes"]["y"]["per_trim"]) == 6
    for p in ("x", "y"):                                   # random per-trim calibrations: judge the per-trim correlation
        rs = [abs(r["r"]) for r in cmp["planes"][p]["per_trim"] if r["r"] is not None]
        assert len(rs) >= 4 and np.median(rs) > 0.98
    assert json.loads((out / "map.json").read_text(encoding="utf-8"))["trims"]["D01T"] == "D01T"
    assert len((out / "orm_trim_calibration.csv").read_text(encoding="utf-8").splitlines()) == 1 + 12


def test_fit_export_and_reload(case, capsys):
    out = case["tmp"] / "fit"; deck_out = case["tmp"] / "fodo_fit.dat"
    rc = main(["orm", "fit", str(case["proj"]), "--measured", str(case["h"]), "--measured", str(case["v"]), "--out", str(out), "--plots",
               "--stage", "trims+quads", "--export-deck", str(deck_out), "--lgproj", "-q"])
    txt = capsys.readouterr().out
    assert rc == 0 and "residual" in txt and deck_out.exists() and deck_out.with_suffix(".lgproj").exists()
    cal = json.loads((out / "orm_calibration.json").read_text(encoding="utf-8"))
    truth = json.loads((case["tmp"] / "synthetic/truth.json").read_text(encoding="utf-8"))
    err = np.array([q["scale"] - truth["quads"][q["label"]] for q in cal["quads"] if not q["fixed"]])
    inj = np.array([truth["quads"][q["label"]] - 1.0 for q in cal["quads"] if not q["fixed"]])
    # plumbing test: the fit moves the free quads towards the injected values (the precision floors of the
    # solver on small FODOs are pinned in tests/orm/test_fit.py; the first cell's quads are weakly determined here)
    assert len(err) >= 8 and np.sqrt(np.mean(err ** 2)) < 0.6 * np.sqrt(np.mean(inj ** 2)) and np.median(np.abs(err)) < 0.01
    for p in ("x", "y"):
        assert cal["metrics"]["after"][p]["nrms_all"] < 0.5 * cal["metrics"]["before"][p]["nrms_all"]
    for f in ("fit_fig1_quads.png", "fit_fig2_residuals.png", "fit_fig3_trends_H.png", "fit_fig4_residual_maps.png", "orm_fit_summary.txt", "orm_compare.json"):
        assert (out / f).exists(), f
    # the exported project loads and reproduces the fitted gradients
    from linac_gen.cli.common import load_input
    lat, _cfg, _ = load_input(str(deck_out.with_suffix(".lgproj")))
    by_label = {getattr(e, "label", None): e for e in lat.elements}
    for q in cal["quads"]:
        assert abs(by_label[q["label"]].gradient - q["gradient_design"] * q["scale"]) <= 1e-9 * abs(q["gradient_design"])
    # export mode from the calibration file reproduces the same deck
    rc = main(["orm", "export", str(case["proj"]), "--calibration", str(out / "orm_calibration.json"), "--out-deck", str(case["tmp"] / "fodo_fit2.dat")])
    assert rc == 0 and _deck_body(deck_out) == _deck_body(case["tmp"] / "fodo_fit2.dat")


def test_validate_and_model_modes(case, capsys):
    rc = main(["orm", "validate", str(case["proj"]), "--measured", str(case["h"]), "--measured", str(case["v"]),
               "--seed", "2", "--g-sigma", "0.02", "--out", str(case["tmp"] / "val"), "-q"])
    txt = capsys.readouterr().out
    assert rc == 0 and "synthetic validation" in txt and (case["tmp"] / "val/orm_validation.json").exists()
    out = case["tmp"] / "model"
    assert main(["orm", "model", str(case["proj"]), "--out", str(out)]) == 0
    assert all((out / f"orm_model_{b}.csv").exists() for b in ("xx", "xy", "yx", "yy"))
    assert main(["orm", "model", str(case["proj"])]) == 2                      # --out required


def test_exit_codes(case, capsys):
    assert main(["orm", "compare", "nope.lgproj", "--measured", str(case["h"])]) == 2
    assert main(["orm", "compare", str(case["proj"])]) == 1                     # no --measured
    with pytest.raises(SystemExit):
        main(["orm"])
    capsys.readouterr()


def test_subprocess_cp1252(case):
    """Console encoding as on Windows (cp1252): the summary (arrows, sigma) must print without a crash."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'gui'}", "PYTHONIOENCODING": "cp1252"}
    r = subprocess.run([sys.executable, "-m", "linac_gen", "orm", "fit", str(case["proj"]), "--measured", str(case["h"]), "--measured", str(case["v"]), "-q"],
                       capture_output=True, text=True, encoding="cp1252", errors="replace", env=env, cwd=REPO, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    assert "residual" in r.stdout
