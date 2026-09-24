"""The campaign engine on the demo deck: create / plan / run / resume,
the fault leg's deltas and criticality, compensation, seeds and faults on
seeds, foil scenarios, availability, the summary files and report, the
refusals, and a job export → remote run → import round trip."""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest

from linac_gen.reliability.campaign import ReliabilityCampaign
from linac_gen.reliability.jobs import export_job, import_results
from linac_gen.reliability.spec import (PRESETS, ReliabilitySpec, default_spec, load_spec,
                                        save_spec, spec_sha256)
from linac_gen.reliability.summary import read_csv

import importlib.util as _ilu
_HAS_MPL = _ilu.find_spec("matplotlib") is not None      # figures are optional

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"

BUDGET = {"element": [{"pattern": "SOL_*", "parameter": "dx", "sigma": 0.2},
                      {"pattern": "GAP_*", "parameter": "voltage_rel", "sigma": 0.01}], "beam": []}


def _demo_copy(tmp_path: Path) -> Path:
    d = tmp_path / "deck"
    d.mkdir()
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(DEMO / name, d / name)
    return d / "reliability_demo.lgproj"


def _spec(inp: Path, **kw) -> ReliabilitySpec:
    base = dict(preset="quick", name="t", circuits="circuits.json", error_budget=BUDGET,
                correction={"enabled": True, "n_iter": 4, "tol_mm": 0.01, "reading_backend": "envelope"},
                seeds={"n": 2, "faults_on_seeds": {"top_n": 1, "n_seeds": 2}},
                landmarks={"treaty_element": "BPM_003"},
                availability={"n_trials": 20})
    base.update(kw)
    return default_spec(str(inp), **base)


def test_spec_presets_round_trip_and_validation(tmp_path):
    for preset in ("quick", "full"):
        s = default_spec("x.dat", preset=preset)
        p = tmp_path / f"{preset}.json"
        save_spec(s, p)
        s2 = load_spec(p)
        assert s2 == s and spec_sha256(s2) == spec_sha256(s)
        assert s.classes == PRESETS[preset]["classes"]
    doc = json.loads((tmp_path / "quick.json").read_text())
    doc["unknown_future_key"] = 1
    (tmp_path / "q2.json").write_text(json.dumps(doc))
    assert load_spec(tmp_path / "q2.json") == default_spec("x.dat")
    with pytest.raises(ValueError, match="scenario class"):
        default_spec("x.dat", classes=["S99"]).validate_shape()
    with pytest.raises(ValueError, match="preset"):
        ReliabilitySpec(input="x.dat", preset="nope").validate_shape()
    full = default_spec("x.dat", preset="full")
    assert full.seeds["n"] == 200 and full.forward["mp_verify"]["critical"] is True
    custom = default_spec("x.dat", preset="custom", classes=["S1"])
    assert custom.classes == ["S1"] and custom.compensation["top_n"] == 5


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("camp")
    inp = _demo_copy(tmp)
    c = ReliabilityCampaign.create(tmp / "run", _spec(inp))
    out = c.run(serial=True)
    return c, out, tmp


def test_create_plans_every_leg_and_run_completes(campaign):
    c, out, _tmp = campaign
    plan = c.plan()
    assert {k: len(v) for k, v in plan.items()} == {
        "faults": 2 + 14 + 5, "imperfections": 1 + 2 + 2 + 4, "foil": 7, "availability": 1}
    assert out == c.dir / "summary.json"
    st = c.status()
    assert all(v["done"] == v["total"] and v["failed"] == 0 for v in st.values()), st
    assert c.pending() == []
    for name in ("campaign.json", "circuits.json", "pins.json", "plan/manifest.json",
                 "plan/manifest_comp.json", "plan/manifest_fos.json", "summary.json", "report.html",
                 "legs/faults/faults.csv", "legs/faults/compensation.csv", "legs/faults/criticality_map.csv",
                 "legs/imperfections/seeds.csv", "legs/imperfections/faults_on_seeds.csv",
                 "legs/foil/foil.csv", "legs/availability/availability.csv",
                 "legs/availability/blocks.csv") + (
                     ("figures/criticality_by_case.png",) if _HAS_MPL else ()):
        assert (c.dir / name).exists(), name
    pins = json.loads((c.dir / "pins.json").read_text())
    assert set(pins["env"]["kinds"]) == {"GAP_001", "GAP_002", "GAP_003", "GAP_004"}


def test_fault_leg_deltas_criticality_and_brackets(campaign):
    c, _out, _tmp = campaign
    truth = json.loads((DEMO / "truth.json").read_text())
    rows = {r["case_id"]: r for r in read_csv(c.dir / "legs/faults/faults.csv")}
    for g in ("GAP_001", "GAP_002", "GAP_003", "GAP_004"):
        re_ = rows[f"S1_{g}_re"]; fr = rows[f"S10_{g}_fr"]
        assert float(re_["d_energy_mev"]) == pytest.approx(truth["cavity_off_rephased"][g]["d_energy_mev"], rel=1e-9)
        assert float(fr["d_energy_mev"]) == pytest.approx(truth["cavity_off_frozen"][g]["d_energy_mev"], rel=1e-9)
        assert re_["critical"] == "True" and re_["bracket"] == "rephased" and fr["bracket"] == "frozen"
    for s in ("STEER_001", "STEER_002", "STEER_003"):
        assert rows[f"S5_{s}_re"]["critical"] == "False" and rows[f"S5_{s}_re"]["recovered_by"] == "auto_rephase"
    assert rows["S2_SOL_001_re"]["d_energy_mev"] == "0"
    assert float(rows["baseline"]["ref_w_kin"]) == pytest.approx(truth["nominal"]["w_end_envelope_mev"], rel=1e-9)
    top = c.ranking("rephased")[0]["item"]["case_id"]
    assert top.startswith(("S1_", "S2_"))
    summ = json.loads((c.dir / "summary.json").read_text())
    assert summ["faults"]["n_critical"] >= 8 and summ["faults"]["by_class"]["S5"]["critical"] == 0
    comps = read_csv(c.dir / "legs/faults/compensation.csv")
    assert len(comps) == 5 and all(r["recovered"] == "True" for r in comps)
    # the rule verdict: a magnet compensation is only recovered when the
    # compensated beam is not critical by the study rule (energy alone is
    # trivially met by a solenoid)
    for r in comps:
        assert r["after_critical"] in ("True", "False")
        assert (r["recovered_by_rule"] == "True") == (r["recovered"] == "True" and r["after_critical"] == "False")
    gap_comp = [r for r in comps if r["case_id"].startswith("S1_GAP_004")]
    if gap_comp:
        assert gap_comp[0]["compensators"] == "GAP_003;GAP_002" and gap_comp[0]["recovered_by_rule"] == "True"
    assert summ["compensation"]["recovered"] + summ["compensation"]["energy_only"] == 5


def test_seeds_faults_on_seeds_foil_and_availability(campaign):
    c, _out, _tmp = campaign
    seeds = read_csv(c.dir / "legs/imperfections/seeds.csv")
    assert [r["kind"] for r in seeds].count("seed") == 2 and any(r["kind"] == "control" for r in seeds)
    assert all(r["correction_status"] == "converged" for r in seeds if r["kind"] == "seed")
    assert all(float(r["orbit_rms_after_mm"]) < float(r["orbit_rms_before_mm"]) and int(r["correction_passes"]) >= 1
               for r in seeds if r["kind"] == "seed")
    assert all(r["orbit_rms_before_mm"] == "" for r in seeds if r["kind"] == "control")
    assert all(int(r["n_draws"]) == 7 for r in seeds if r["kind"] == "seed")     # 3 dx + 4 voltage_rel
    fos = read_csv(c.dir / "legs/imperfections/faults_on_seeds.csv")
    assert len(fos) == 4 and {r["variant"] for r in fos} == {"before", "after"}
    rob = read_csv(c.dir / "legs/imperfections/robustness.csv")
    assert len(rob) == 2 and all(r["n"] == "2" for r in rob)
    foil = {r["case_id"]: r for r in read_csv(c.dir / "legs/foil/foil.csv")}
    assert set(foil) == {"nominal", "thick_0", "thick_435", "thick_800", "off_1_0", "off_0_1", "thinned"}
    assert float(foil["nominal"]["foil_sigma_x_mm"]) > 0.0
    assert float(foil["nominal"]["strip_eff_analytic"]) == 1.0          # proton beam: nothing to strip
    av = read_csv(c.dir / "legs/availability/availability.csv")
    assert {r["variant"] for r in av} == {"srf_fdr", "srf_sns"}
    assert all(0.0 < float(r["availability_mean"]) < 1.0 for r in av)
    assert (c.dir / "legs/availability/blocks.csv").read_text().startswith("# Reliability block diagram")
    html = (c.dir / "report.html").read_text(encoding="utf-8")
    assert "Leg B" in html and "<script" not in html
    if _HAS_MPL:                         # figures need the optional matplotlib
        assert "data:image/png;base64," in html
        img = html.split("data:image/png;base64,")[1].split('"')[0]
        assert base64.b64decode(img)[:8] == b"\x89PNG\r\n\x1a\n"


def test_resume_is_a_noop_and_retry_failed_requeues(campaign):
    c, _out, _tmp = campaign
    c2 = ReliabilityCampaign.load(c.dir)
    assert c2.pending() == []
    calls = []
    out = c2.run(serial=True, progress_cb=calls.append)
    assert out == c.dir / "summary.json" and calls == []
    it = c2.plan()["faults"][3]
    sp = c2._status_path(it)
    doc = json.loads(sp.read_text()); doc["status"] = "failed"; doc["error"] = "injected"
    sp.write_text(json.dumps(doc))
    assert c2.pending() == [] and [x["id"] for x in c2.pending(retry_failed=True)] == [it["id"]]
    c2.run(legs=["faults"], serial=True, retry_failed=True)
    assert json.loads(sp.read_text())["status"] == "ok"


def test_load_refuses_drift(tmp_path):
    inp = _demo_copy(tmp_path)
    c = ReliabilityCampaign.create(tmp_path / "run", _spec(inp, legs={"faults": True, "availability": False,
                                                                     "imperfections": False, "foil": False}))
    with pytest.raises(FileExistsError):
        ReliabilityCampaign.create(tmp_path / "run", _spec(inp))
    doc = json.loads((c.dir / "campaign.json").read_text())
    doc["seeds"]["n"] = 99
    (c.dir / "campaign.json").write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="spec hash"):
        ReliabilityCampaign.load(c.dir)
    save_spec(c.spec, c.dir / "campaign.json")
    ReliabilityCampaign.load(c.dir)
    deck = inp.parent / "reliability_demo.dat"
    deck.write_text(deck.read_text(encoding="utf-8") + "; touched\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since"):
        ReliabilityCampaign.load(c.dir)
    bad = _spec(inp, landmarks={"treaty_element": "NOPE"})
    with pytest.raises(ValueError, match="landmark"):
        ReliabilityCampaign.create(tmp_path / "run2", bad)
    bad2 = _spec(inp, error_budget={"element": [{"pattern": "NOPE_*", "parameter": "dx", "sigma": 0.1}]})
    with pytest.raises(ValueError, match="matches no element"):
        ReliabilityCampaign.create(tmp_path / "run3", bad2)


def test_export_remote_run_import_round_trip(tmp_path):
    inp = _demo_copy(tmp_path)
    spec = _spec(inp, legs={"faults": True, "availability": False, "imperfections": False, "foil": True})
    c = ReliabilityCampaign.create(tmp_path / "run", spec)
    c.run(legs=["faults"], serial=True)
    assert [it["leg"] for it in c.pending()] == ["foil"] * 7
    job = export_job(c, tmp_path / "job", legs=["foil"])
    assert (job / "job.json").exists() and (job / "lattice" / "reliability_demo.dat").exists()
    assert json.loads((job / "job.json").read_text())["expected"][0]["leg"] == "foil"
    remote = ReliabilityCampaign.load(job)                    # the job folder IS a campaign
    assert remote.input_path == (job / "lattice" / "reliability_demo.lgproj").resolve()
    remote.run(legs=["foil"], serial=True)
    with pytest.raises(RuntimeError, match="missing"):
        broken = ReliabilityCampaign.load(tmp_path / "run")
        shutil.move(remote.item_dir(remote.plan()["foil"][0]) / "status.json", tmp_path / "hidden.json")
        import_results(job, broken)
    shutil.move(tmp_path / "hidden.json", remote.item_dir(remote.plan()["foil"][0]) / "status.json")
    rec = import_results(job, ReliabilityCampaign.load(tmp_path / "run"))
    assert len(rec["imported"]) == 7 and rec["missing"] == []
    c3 = ReliabilityCampaign.load(tmp_path / "run")
    assert c3.pending() == []
    c3.summarize()
    assert len(read_csv(c3.dir / "legs/foil/foil.csv")) == 7
    rec2 = import_results(job, c3)
    assert rec2["imported"] == [] and len(rec2["skipped_ok"]) == 7    # never overwrites an ok item
    doc = json.loads((job / "job.json").read_text()); doc["lattice_sha256"] = "0" * 64
    (job / "job.json").write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="lattice sha"):
        import_results(job, c3)


def test_parallel_pool_matches_serial(tmp_path):
    inp = _demo_copy(tmp_path)
    kw = dict(legs={"faults": True, "availability": False, "imperfections": False, "foil": False},
              classes=["S1"], compensation={"top_n": 0})
    a = ReliabilityCampaign.create(tmp_path / "a", _spec(inp, name="a", **kw))
    b = ReliabilityCampaign.create(tmp_path / "b", _spec(inp, name="b", **kw))
    a.run(serial=True)
    b.run(max_workers=2)
    ra = read_csv(a.dir / "legs/faults/faults.csv"); rb = read_csv(b.dir / "legs/faults/faults.csv")
    assert [r["ref_w_kin"] for r in ra] == [r["ref_w_kin"] for r in rb]
    assert [r["criticality"] for r in ra] == [r["criticality"] for r in rb]


# --------------------------------------------------------------------------
# review fixes (2026-09-17 adversarial review)
# --------------------------------------------------------------------------
def test_recovered_by_split_reaches_the_blocks_by_scenario_class(campaign):
    c, _out, _tmp = campaign
    av = json.loads((c.dir / "legs/availability/availability.json").read_text())
    by_class = av["class_split_by_class"]
    assert "S1" in by_class and abs(sum(by_class["S1"].values()) - 1.0) < 1e-12
    applied = av["class_split_applied"]
    assert applied["SRF_CAVITY_TRIP"] == by_class["S1"]            # template row scenario_class=S1
    assert applied["MAGNET_PS"] == by_class["S2"]
    assert "CRYOPLANT" in av["blocks_without_split"]                # no scenario_class
    summ = json.loads((c.dir / "summary.json").read_text())
    assert summ["class_split_applied"] == applied
    # a critical case never attempted is unknown, not downtime: the split
    # counts only cases with a compensation record
    rows = read_csv(c.dir / "legs/faults/faults.csv")
    never = [r for r in rows if r["recovered_by"] == "not_compensated"]
    assert never                                                      # top_n 5 < 11 critical cases
    assert by_class["S1"].get("downtime", 0.0) == 0.0                  # every attempted S1 recovered


def test_status_counts_only_complete_items_and_verification_rows_are_labelled(campaign):
    c, _out, _tmp = campaign
    it = c.plan()["faults"][5]
    h5 = c.item_dir(it) / "results.h5"
    data = h5.read_bytes()
    h5.unlink()
    try:
        st = c.status()["faults"]
        assert st["done"] == st["total"] - 1 and [x["id"] for x in c.pending()] == [it["id"]]
    finally:
        h5.write_bytes(data)
    rows = read_csv(c.dir / "legs/faults/faults.csv")
    assert all(r["verification"] == "" for r in rows)                 # envelope campaign: no MP verification
    assert "verification" in rows[0]


def test_force_is_scoped_and_keeps_the_block_table(tmp_path):
    inp = _demo_copy(tmp_path)
    spec = _spec(inp, legs={"faults": True, "availability": True, "imperfections": False, "foil": False},
                 classes=["S5"], compensation={"top_n": 0})
    c = ReliabilityCampaign.create(tmp_path / "run", spec)
    c.run(serial=True)
    blocks = c.legs_dir / "availability" / "blocks.csv"
    blocks.write_text(blocks.read_text(encoding="utf-8") + "MY_BLOCK,X,100,1,exponential,1,1,downtime,\n",
                      encoding="utf-8")
    faults_before = sorted(p.name for p in (c.legs_dir / "faults" / "runs").iterdir())
    c.run(legs=["availability"], serial=True, force=True)
    assert "MY_BLOCK" in blocks.read_text(encoding="utf-8")
    assert sorted(p.name for p in (c.legs_dir / "faults" / "runs").iterdir()) == faults_before
    av = read_csv(c.dir / "legs/availability/availability.csv")
    assert av and (c.dir / "legs/availability/availability.json").exists()


def test_empty_dynamic_manifest_is_replanned_and_retry_invalidates_the_waves(tmp_path):
    inp = _demo_copy(tmp_path)
    spec = _spec(inp, legs={"faults": True, "availability": False, "imperfections": True, "foil": False},
                 classes=["S1"], compensation={"top_n": 1}, seeds={"n": 1, "faults_on_seeds": {"top_n": 1, "n_seeds": 1}})
    c = ReliabilityCampaign.create(tmp_path / "run", spec)
    c.run(legs=["imperfections"], serial=True)                     # no fault rows yet → no fos wave
    fos = c.plan_dir / "manifest_fos.json"
    assert not fos.exists() or json.loads(fos.read_text()) == []
    fos.write_text("[]")                                            # an empty wave left behind (a stopped run)
    c.run(serial=True)                                              # faults now ran: the empty wave is replanned
    assert len(json.loads(fos.read_text())) == 2 and c.pending() == []
    comp = json.loads((c.plan_dir / "manifest_comp.json").read_text())
    assert [x["case_id"] for x in comp] == ["S1_GAP_004_re"]
    # re-queue the top scenario: the waves planned from the ranking go away and come back
    it = next(x for x in c.plan()["faults"] if x["case_id"] == "S1_GAP_004_re")
    sp = c._status_path(it)
    doc = json.loads(sp.read_text()); doc["status"] = "failed"; sp.write_text(json.dumps(doc))
    c.run(legs=["faults"], serial=True, retry_failed=True)
    assert json.loads(sp.read_text())["status"] == "ok"
    assert [x["case_id"] for x in json.loads((c.plan_dir / "manifest_comp.json").read_text())] == ["S1_GAP_004_re"]


def test_a_default_added_by_a_later_version_does_not_invalidate_a_campaign(tmp_path, monkeypatch):
    from linac_gen.reliability import spec as S
    inp = _demo_copy(tmp_path)
    c = ReliabilityCampaign.create(tmp_path / "run", _spec(inp, legs={"faults": True, "availability": False,
                                                                     "imperfections": False, "foil": False}))
    monkeypatch.setitem(S.PRESETS["quick"]["compensation"], "brand_new_knob", 42)
    c2 = ReliabilityCampaign.load(c.dir)                              # filled from the preset, identity kept
    assert c2.spec.compensation["brand_new_knob"] == 42
    doc = json.loads((c.dir / "campaign.json").read_text()); doc["classes"] = ["S5"]
    (c.dir / "campaign.json").write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="spec hash"):               # an edit is still an edit
        ReliabilityCampaign.load(c.dir)


def test_load_refuses_an_edited_circuit_map_and_a_bad_name(tmp_path):
    inp = _demo_copy(tmp_path)
    c = ReliabilityCampaign.create(tmp_path / "run", _spec(inp, legs={"faults": True, "availability": False,
                                                                     "imperfections": False, "foil": False}))
    cj = c.dir / "circuits.json"
    doc = json.loads(cj.read_text()); doc["sections"] = [{"name": "ALL", "end": "FOIL_001"}]
    cj.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="circuits.json changed"):
        ReliabilityCampaign.load(c.dir)
    with pytest.raises(ValueError, match="plain folder name"):
        ReliabilityCampaign.create(tmp_path / "run2", _spec(inp, name="../escape"))


def test_export_carries_completed_items_and_the_remote_waves_come_back(tmp_path):
    inp = _demo_copy(tmp_path)
    spec = _spec(inp, legs={"faults": True, "availability": False, "imperfections": False, "foil": False},
                 classes=["S1"], compensation={"top_n": 1})
    c = ReliabilityCampaign.create(tmp_path / "run", spec)
    plan_items = c.plan()["faults"]
    # complete the baselines locally, export, and let the "remote" run the rest incl. the compensation wave
    c._run_points([it for it in plan_items if it["case_id"] == "baseline"], max_workers=1, serial=True,
                  progress_cb=None, should_stop=None, leg="faults", phase="x",
                  counters={"done": 0, "failed": 0, "total": 2})
    job = export_job(c, tmp_path / "job", legs=["faults"])
    jj = json.loads((job / "job.json").read_text())
    assert jj["legs"] == ["faults"] and "--legs faults" in (job / "README.txt").read_text()
    remote = ReliabilityCampaign.load(job)
    assert len(remote.pending()) == 4                               # the two baselines travelled with the job
    remote.run(legs=["faults"], serial=True)
    assert (job / "plan" / "manifest_comp.json").exists()
    rec = import_results(job, ReliabilityCampaign.load(tmp_path / "run"))
    assert "A_comp_S1_GAP_004_re_k_out_of_n" in rec["imported"] and len(rec["skipped_ok"]) == 2
    c2 = ReliabilityCampaign.load(tmp_path / "run")
    assert c2.pending() == [] and (c2.plan_dir / "manifest_comp.json").exists()


def test_nested_project_exports_and_reloads(tmp_path):
    d = tmp_path / "proj" / "sub"
    d.mkdir(parents=True)
    shutil.copy2(DEMO / "reliability_demo.dat", d / "reliability_demo.dat")
    shutil.copy2(DEMO / "circuits.json", tmp_path / "proj" / "circuits.json")
    doc = json.loads((DEMO / "reliability_demo.lgproj").read_text())
    doc["lattice_path"] = "sub/reliability_demo.dat"
    (tmp_path / "proj" / "nested.lgproj").write_text(json.dumps(doc, indent=2))
    spec = _spec(tmp_path / "proj" / "nested.lgproj", legs={"faults": True, "availability": False,
                                                            "imperfections": False, "foil": False},
                 classes=["S5"], compensation={"top_n": 0})
    c = ReliabilityCampaign.create(tmp_path / "run", spec)
    assert (c.dir / "lattice" / "reliability_demo.dat").exists()      # the pointed deck is snapshotted too
    job = export_job(c, tmp_path / "job")
    remote = ReliabilityCampaign.load(job)                            # digest unchanged: relative layout kept
    assert remote.input_path.name == "nested.lgproj"
