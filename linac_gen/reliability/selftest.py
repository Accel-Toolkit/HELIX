"""The built-in self-test of the Reliability Study mode.

``run_selftest`` runs a quick campaign on the shipped demo deck
(``examples/reliability_demo/``) in a scratch folder and compares every
leg against ``truth.json`` — answers computed by the deck generator's own
kinematics with no HELIX import — plus closed forms and the calibration
points, labelled as such.  With ``regression=True`` it also re-runs the
public control cases of ``scripts/physics_fix_report.py`` and compares
them bit for bit with a snapshot of the reference tree (``before_dir``),
refusing when any control case is missing, skipped or errors.  The same
function backs ``python -m linac_gen reliability selftest`` and the
window's Self-test button.

Verdicts: ``PASS`` (every check passed, regression bit-identical when
requested), ``FAIL`` (a check failed or a control case changed),
``REFUSED`` (the comparison could not be completed: no BEFORE snapshot,
a control case skipped or erroring).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

__all__ = ["run_selftest", "format_result", "REPO", "DEMO", "PUBLIC_CONTROL_CASES",
           "BASELINE_PATH", "baseline_values"]

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"
BASELINE_PATH = REPO / "tests" / "reliability" / "fixtures" / "reliability_selftest_baseline.npz"

#: The physics_fix_report cases that run on a public checkout.
PUBLIC_CONTROL_CASES = ("fodo_cell", "halo_fodo", "bend_line", "csr_chicane", "chicane_batch",
                        "coupled_fodo", "dtl_section", "solenoid_channel", "matching_demo",
                        "correction_demo", "lebt_scc_demo", "hofmann_demo", "fodo_madx")

BUDGET = {"element": [{"pattern": "SOL_*", "parameter": "dx", "sigma": 0.2},
                      {"pattern": "GAP_*", "parameter": "voltage_rel", "sigma": 0.01}], "beam": []}


class _Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, name, ok, expected=None, got=None, tol=None, note=""):
        self.rows.append({"name": name, "status": "PASS" if ok else "FAIL", "expected": expected,
                          "got": got, "tol": tol, "note": note})
        return ok

    def rel(self, name, got, expected, tol, note=""):
        try:
            ok = (got is not None and expected is not None
                  and abs(float(got) - float(expected)) <= tol * max(abs(float(expected)), 1e-300))
        except (TypeError, ValueError):
            ok = False
        return self.add(name, ok, expected, got, f"rel {tol:g}", note)

    def exact(self, name, got, expected, note=""):
        return self.add(name, got == expected, expected, got, "exact", note)

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


def _campaign_spec(inp: Path):
    from linac_gen.reliability.spec import default_spec
    return default_spec(str(inp), preset="quick", name="selftest", circuits="circuits.json",
                        error_budget=BUDGET,
                        correction={"enabled": True, "n_iter": 5, "tol_mm": 0.01, "reading_backend": "envelope"},
                        seeds={"n": 2, "faults_on_seeds": {"top_n": 1, "n_seeds": 2}},
                        landmarks={"treaty_element": "BPM_003"},
                        availability={"n_trials": 30},
                        execution={"max_workers": 1, "serial": True})


def _demo_checks(ck: _Checks, work: Path, progress, should_stop) -> dict:
    """Run the quick campaign on a copy of the demo deck; return the values
    the baseline pins (a flat {key: array} dict)."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.reliability.summary import read_csv
    truth = json.loads((DEMO / "truth.json").read_text(encoding="utf-8"))
    tol = float(truth["tolerances"]["energy_rel"])
    deck_dir = work / "deck"
    deck_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json",
                 "make_reliability_demo.py", "truth.json"):
        shutil.copy2(DEMO / name, deck_dir / name)
    # -- the deck itself
    lat, meta = parse_tracewin(str(deck_dir / "reliability_demo.dat"))
    ck.exact("deck.parser_warnings", len(meta.get("warnings", [])), 0)
    gen_dir = work / "regen"
    gen_dir.mkdir(exist_ok=True)
    shutil.copy2(DEMO / "make_reliability_demo.py", gen_dir / "make_reliability_demo.py")
    r = subprocess.run([sys.executable, str(gen_dir / "make_reliability_demo.py")],
                       capture_output=True, text=True, timeout=120)
    same = r.returncode == 0 and all((gen_dir / n).read_bytes() == (DEMO / n).read_bytes()
                                     for n in ("reliability_demo.dat", "truth.json", "circuits.json"))
    ck.add("deck.generator_reproduces_files", same, "byte-identical", "identical" if same else r.stderr[-200:])
    # -- the campaign
    spec = _campaign_spec(deck_dir / "reliability_demo.lgproj")
    cdir = work / "campaign"
    c = ReliabilityCampaign.create(cdir, spec)
    out = c.run(serial=True, progress_cb=progress, should_stop=should_stop)
    ck.add("campaign.completed", out is not None, "summary.json", str(out))
    if out is None:
        return {}
    st = c.status()
    ck.add("campaign.no_failed_items", all(v["failed"] == 0 for v in st.values()), 0,
           sum(v["failed"] for v in st.values()))
    pins = json.loads((cdir / "pins.json").read_text(encoding="utf-8"))["env"]["pins"]
    for g, want in truth["nominal"]["clock_at_gap_entrance_deg"].items():
        ck.rel(f"legA.pin.clock.{g}", pins.get(g), want, 1e-12)
    rows = {r["case_id"] if r["case_id"] != "baseline" else f"baseline_{r['bracket']}": r
            for r in read_csv(cdir / "legs" / "faults" / "faults.csv")}
    base = rows.get("baseline_rephased", {})
    ck.rel("legA.rephased.exit_energy", float(base.get("ref_w_kin") or "nan"),
           truth["nominal"]["w_end_envelope_mev"], tol)
    fb = rows.get("baseline_frozen", {})
    ck.exact("legA.frozen_baseline_bit_identical", fb.get("ref_w_kin"), base.get("ref_w_kin"),
             "the pin IS the design pass")
    values: dict = {}
    for g in ("GAP_001", "GAP_002", "GAP_003", "GAP_004"):
        re_ = rows.get(f"S1_{g}_re", {})
        fr = rows.get(f"S10_{g}_fr", {})
        ck.rel(f"legA.rephased.deficit.{g}", float(re_.get("d_energy_mev") or "nan"),
               truth["cavity_off_rephased"][g]["d_energy_mev"], tol, "V cos phi_s")
        ck.rel(f"legA.frozen.deficit.{g}", float(fr.get("d_energy_mev") or "nan"),
               truth["cavity_off_frozen"][g]["d_energy_mev"], tol, "kinematic chain")
        for key, row in (("re", re_), ("fr", fr)):
            values[f"legA|{g}|{key}|metrics"] = np.array(
                [float(row.get(k) or "nan") for k in ("ref_w_kin", "emit_nx", "emit_ny", "emit_nz",
                                                       "d_phi_end_deg", "criticality")])
    for s in ("SOL_001", "SOL_002", "SOL_003"):
        ck.exact(f"legA.solenoid_off.d_energy.{s}", rows.get(f"S2_{s}_re", {}).get("d_energy_mev"), "0",
                 "a solenoid carries no RF")
    for s in ("STEER_001", "STEER_002", "STEER_003"):
        ck.exact(f"legA.steerer_off.noncritical.{s}", rows.get(f"S5_{s}_re", {}).get("critical"), "False")
    top_gap = truth["ranking"]["top_single_cavity_rephased"]
    s1 = sorted((r for k, r in rows.items() if k.startswith("S1_")),
                key=lambda r: -float(r.get("criticality") or 0.0))
    ck.exact("legA.ranking.top_cavity", s1[0]["case_id"] if s1 else None, f"S1_{top_gap}_re")
    comps = {r["case_id"]: r for r in read_csv(cdir / "legs" / "faults" / "compensation.csv")}
    c4 = comps.get(f"S1_{top_gap}_re", {})
    ck.exact("legA.compensation.k1_neighbours", c4.get("compensators"),
             ";".join(truth["ranking"]["k1_compensators_for_GAP_004"]))
    ck.exact("legA.compensation.recovered", c4.get("recovered"), "True",
             f"|dW| <= {spec.compensation.get('recover_tol_energy_mev', 0.05)} MeV")
    if c4.get("settings"):
        st4 = json.loads(c4["settings"])
        values["legA|compensate|GAP_004|settings"] = np.array([st4[k] for k in sorted(st4)])
    # -- leg C
    seeds = {r["id"]: r for r in read_csv(cdir / "legs" / "imperfections" / "seeds.csv")}
    s0 = next((r for r in seeds.values() if r["kind"] == "seed"), {})
    ck.exact("legC.seed0.correction_converged", s0.get("correction_status"), "converged")
    ck.exact("legC.seed0.n_draws", s0.get("n_draws"), "7", "3 dx + 4 voltage_rel")
    d0 = json.loads((cdir / "legs" / "imperfections" / "draws" / "seed_0000.json").read_text(encoding="utf-8"))
    values["legC|seed0|draws"] = np.array([float(v) for _s, v in d0["overrides"]])
    values["legC|seed0|kicks"] = np.array([float(v) for _s, v in d0["kick_overrides"]])
    ck.exact("legC.seed0.correction_pairs", (d0.get("correction") or {}).get("n_pairs"), 3)
    fos = read_csv(cdir / "legs" / "imperfections" / "faults_on_seeds.csv")
    ck.exact("legC.faults_on_seeds.rows", len(fos), 4, "1 case x 2 seeds x before/after")
    # -- leg D
    foil = {r["case_id"]: r for r in read_csv(cdir / "legs" / "foil" / "foil.csv")}
    nom = foil.get("nominal", {})
    ck.add("legD.foil.spot_present", float(nom.get("foil_sigma_x_mm") or 0.0) > 0.0, "> 0",
           nom.get("foil_sigma_x_mm"))
    values["legD|foil|sigma"] = np.array([float(nom.get("foil_sigma_x_mm") or "nan"),
                                          float(nom.get("foil_sigma_y_mm") or "nan")])
    from linac_gen.core.particle import H_MINUS, PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.foil import Foil
    fel = next(e for e in lat.elements if isinstance(e, Foil))
    ref = ReferenceParticle(species=PROTON, w_kin=truth["foil"]["w_at_foil_mev"], frequency=162.5)
    ck.rel("legD.foil.mip_mean_loss", fel._mean_energy_loss_MeV(ref), truth["foil"]["mean_loss_mev"], 1e-12)
    ck.rel("legD.foil.highland_theta", fel._highland_theta_rms(ref) * 1e3,
           truth["foil"]["highland_theta_rms_mrad"], 1e-9)
    hm = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    ck.rel("legD.strip.anchor_600", fel.stripping_fractions(hm, 600.0)[0], 0.99956, 1e-9,
           "CALIBRATION point, not truth")
    ck.rel("legD.strip.anchor_380", fel.stripping_fractions(hm, 380.0)[0], 0.991, 1e-9,
           "CALIBRATION point, not truth")
    # -- leg B
    from linac_gen.reliability.availability import Block, analytic_availability, simulate
    cf = truth["availability_closed_forms"]
    a1 = cf["single_block"]
    ck.rel("legB.closed.single_block", analytic_availability([Block("S", a1["mtbf_h"], a1["mttr_h"])]),
           a1["availability"], 1e-15)
    ck.rel("legB.closed.series", analytic_availability([Block("S", 100.0, 1.0), Block("T", 100.0, 1.0)]),
           cf["series_two_identical"], 1e-15)
    ck.rel("legB.closed.parallel", analytic_availability([Block("P", 100.0, 1.0, n_parallel=2)]),
           cf["parallel_two_identical"], 1e-15)
    ck.rel("legB.closed.two_of_three", analytic_availability([Block("K", 100.0, 1.0, n_parallel=3, k_required=2)]),
           cf["two_of_three_identical"], 1e-15)
    mc = simulate([Block("S", 100.0, 1.0)], hours_per_year=20000.0, n_trials=60, seed=7)
    se = float(mc.availability.std(ddof=1) / math.sqrt(mc.n_trials))
    ck.add("legB.mc.single_block_within_3.5sigma", abs(mc.mean - a1["availability"]) <= 3.5 * se + 2e-4,
           a1["availability"], mc.mean, f"3.5 sigma ({se:.2e})")
    values["legB|mc|availability_seed7"] = np.asarray(mc.availability)
    det = simulate([Block("RF", 10.0, 1.0, fault_class="auto_rephase")], hours_per_year=1000.0,
                   n_trials=10, seed=7)
    ck.add("legB.bins.deterministic_recovery_one_bin", bool(np.all(det.trip_histogram[1:] == 0.0)),
           "all trips in the first bin", det.trip_histogram.tolist())
    av = read_csv(cdir / "legs" / "availability" / "availability.csv")
    ck.exact("legB.campaign.variants", sorted(r["variant"] for r in av), ["srf_fdr", "srf_sns"])
    # -- report
    html = (cdir / "report.html").read_text(encoding="utf-8")
    ck.add("report.embedded_png", "data:image/png;base64," in html, "data URI PNG", "present" if
           "data:image/png;base64," in html else "absent")
    return values


def _baseline_compare(ck: _Checks, values: dict) -> None:
    if not BASELINE_PATH.exists():
        ck.rows.append({"name": "baseline.fixture", "status": "SKIP", "expected": str(BASELINE_PATH),
                        "got": "absent", "tol": None, "note": "regen_reliability_selftest_baseline.py"})
        return
    with np.load(BASELINE_PATH, allow_pickle=False) as z:
        pinned = {k: np.asarray(z[k]) for k in z.files}
    meta = {k: pinned.pop(k) for k in list(pinned) if k.startswith("__")}
    ck.exact("baseline.key_set", sorted(values), sorted(pinned), f"fixture {meta.get('__helix_version__')}")
    exact = os.environ.get("HELIX_BASELINE_EXACT") == "1"
    for k in sorted(pinned):
        if k not in values:
            continue
        a, b = np.asarray(values[k], dtype=float), np.asarray(pinned[k], dtype=float)
        if a.shape != b.shape:
            ck.add(f"baseline.{k}", False, b.shape, a.shape, "shape")
            continue
        if exact:
            ok = bool(np.array_equal(a, b, equal_nan=True))
            ck.add(f"baseline.{k}", ok, "bit-identical", "identical" if ok else "differs", "exact")
        else:
            atol = 1e-12 * max(1.0, float(np.nanmax(np.abs(b))) if b.size else 1.0)
            ok = bool(np.allclose(a, b, rtol=1e-14, atol=atol, equal_nan=True))
            ck.add(f"baseline.{k}", ok, "within one ulp", "ok" if ok else "differs", f"rtol 1e-14, atol {atol:.1e}")


def baseline_values(work_dir=None) -> dict:
    """The values the fixture pins (runs the quick campaign)."""
    ck = _Checks()
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="helix_reliability_selftest_"))
    return _demo_checks(ck, tmp, None, None)


def _load_harness():
    path = REPO / "scripts" / "physics_fix_report.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    spec = importlib.util.spec_from_file_location("physics_fix_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _regression(before_dir, out_dir: Path, progress, cases=None) -> dict:
    """Run the public control cases in-process (no chdir, no tree
    assertion) and compare with ``before_dir`` through the harness's own
    cmd_compare.  ``cases`` restricts to the named control cases."""
    from types import SimpleNamespace
    mod = _load_harness()
    after = out_dir / "regression_after"
    after.mkdir(parents=True, exist_ok=True)
    wanted = tuple(cases) if cases else PUBLIC_CONTROL_CASES
    bad = [c for c in wanted if c not in PUBLIC_CONTROL_CASES]
    if bad:
        raise ValueError(f"unknown control case(s) {bad}; public cases: {list(PUBLIC_CONTROL_CASES)}")
    cases = {n: s for n, s in mod.CASES.items() if n.split(":")[0] in wanted}
    problems = []
    n_run = 0
    for name, spec in cases.items():
        if progress is not None:
            progress(SimpleNamespace(leg="regression", phase=name, done=n_run, failed=len(problems),
                                     total=len(cases), mean_elapsed=None, eta_s=None))
        try:
            res = mod.run_case(REPO, name, spec)
        except Exception as exc:                          # noqa: BLE001
            res = {"error": f"{type(exc).__name__}: {exc}"}
        if "skipped" in res or "error" in res:
            problems.append(f"{name}: {res.get('skipped') or res.get('error')}")
            continue
        np.savez_compressed(after / (name.replace(":", "__") + ".npz"), **res)
        n_run += 1
    out: dict = {"n_cases": len(cases), "n_run": n_run, "problems": problems, "before": None,
                 "n_identical": 0, "n_changed": 0, "n_missing": 0, "table": ""}
    if before_dir is None:
        out["note"] = ("no BEFORE snapshot: take one from the reference tree with\n    "
                       f"PYTHONPATH=/path/to/before python3 scripts/physics_fix_report.py snapshot "
                       f"--tree /path/to/before --out /tmp/snap_before --cases "
                       + " ".join(f"'{c}:*'" for c in PUBLIC_CONTROL_CASES[:3]) + " …")
        return out
    out["before"] = str(before_dir)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = mod.cmd_compare(SimpleNamespace(before=str(before_dir), after=str(after), out=None))
    text = buf.getvalue()
    out["table"] = text
    for line in text.splitlines():
        if line.endswith("not compared") and "bit-identical" in line:
            parts = line.replace(",", "").split()
            out["n_identical"], out["n_changed"], out["n_missing"] = int(parts[0]), int(parts[2]), int(parts[4])
    out["rc"] = rc
    return out


def run_selftest(*, quick: bool = True, regression: bool = False, before_dir=None, out_dir=None,
                 progress=None, should_stop=None, regression_cases=None) -> dict:
    t0 = time.time()
    ck = _Checks()
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="helix_reliability_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    values = _demo_checks(ck, work, progress, should_stop)
    _baseline_compare(ck, values)
    reg = None
    verdict = "PASS" if not ck.failed else "FAIL"
    if regression:
        reg = _regression(before_dir, work, progress, regression_cases)
        if reg["problems"] or reg["before"] is None or reg["n_missing"]:
            verdict = "REFUSED"
        elif reg["n_changed"]:
            verdict = "FAIL"
    return {"verdict": verdict, "checks": ck.rows, "n_checks": len(ck.rows), "n_failed": len(ck.failed),
            "regression": reg, "elapsed_s": time.time() - t0, "work_dir": str(work),
            "machine": platform.machine(), "python": platform.python_version()}


def format_result(res: dict) -> str:
    lines = [f"{'check':<46} {'status':<6} {'expected':<24} {'got':<24} tol"]
    for r in res["checks"]:
        exp = str(r.get("expected"))[:24]
        got = str(r.get("got"))[:24]
        note = f"  ({r['note']})" if r.get("note") else ""
        lines.append(f"{r['name']:<46} {r['status']:<6} {exp:<24} {got:<24} {r.get('tol') or ''}{note}")
    reg = res.get("regression")
    if reg is not None:
        if reg.get("before") is None:
            lines.append(f"regression: {reg['n_run']}/{reg['n_cases']} control cases run; "
                         f"REFUSED — {reg.get('note')}")
        else:
            lines.append(f"regression: {reg['n_identical']} bit-identical, {reg['n_changed']} changed, "
                         f"{reg['n_missing']} not compared ({reg['n_run']}/{reg['n_cases']} control cases run)")
        for p in reg.get("problems") or []:
            lines.append(f"  control case problem: {p}")
    lines.append(f"VERDICT {res['verdict']}  ({res['n_checks']} checks, {res['n_failed']} failed, "
                 f"{res['elapsed_s']:.1f} s, work dir {res['work_dir']})")
    return "\n".join(lines)
