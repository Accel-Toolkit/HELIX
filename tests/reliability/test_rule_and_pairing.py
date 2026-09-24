"""Criticality-rule terms switch off with ``None``; orbit correction
without ``ADJUST_STEERER`` cards (``correction.pairing = "auto"``) and its
agreement with the card-driven driver; an error budget kept in its own
file next to the spec."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from linac_gen.reliability.observables import criticality_rule

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"

_BASE = {"emit_nx": 1.0, "emit_ny": 1.0, "emit_nz": 1.0, "transmission": 100.0, "ref_w_kin": 20.0}


def test_rule_terms_switch_off_with_none():
    rule = {"emit_growth_pct": 5.0, "loss_frac": 1e-4, "energy_pct": 0.5}
    energy_only = dict(_BASE, ref_w_kin=20.0 * 1.02)            # 2 % off, no growth, no loss
    crit, terms = criticality_rule(_BASE, energy_only, rule)
    assert crit and terms["energy_dev_pct"] == pytest.approx(2.0)
    crit, terms = criticality_rule(_BASE, energy_only, dict(rule, energy_pct=None))
    assert not crit and terms["energy_dev_pct"] == pytest.approx(2.0)   # reported, not judged
    growth_only = dict(_BASE, emit_nx=1.08)                       # 8 % growth
    assert criticality_rule(_BASE, growth_only, rule)[0]
    assert not criticality_rule(_BASE, growth_only, dict(rule, emit_growth_pct=10.0))[0]
    assert not criticality_rule(_BASE, growth_only, dict(rule, emit_growth_pct=None))[0]
    loss_only = dict(_BASE, transmission=99.0)                    # 1 % loss
    assert criticality_rule(_BASE, loss_only, rule)[0]
    assert not criticality_rule(_BASE, loss_only, dict(rule, loss_frac=None))[0]
    # the earlier paper's rule: 10 % growth in any plane or 1 % loss, no energy term
    paper = {"emit_growth_pct": 10.0, "loss_frac": 0.01, "energy_pct": None}
    assert not criticality_rule(_BASE, energy_only, paper)[0]
    assert not criticality_rule(_BASE, growth_only, paper)[0]
    assert criticality_rule(_BASE, dict(_BASE, emit_nz=1.11), paper)[0]
    assert criticality_rule(_BASE, dict(_BASE, transmission=98.9), paper)[0]
    # an unknown exit energy is critical whatever the rule
    assert criticality_rule(_BASE, dict(_BASE, ref_w_kin=None), paper)[0]


def _demo(tmp_path: Path, *, strip_cards: bool = False) -> Path:
    for name in ("reliability_demo.dat", "reliability_demo.lgproj"):
        shutil.copy2(DEMO / name, tmp_path / name)
    deck = tmp_path / "reliability_demo.dat"
    if strip_cards:
        lines = [ln for ln in deck.read_text(encoding="utf-8").splitlines()
                 if not ln.startswith("ADJUST_STEERER")]
        deck.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "reliability_demo.lgproj"


_BUDGET = {"element": [{"pattern": "SOL_*", "parameter": "dx", "sigma": 0.2, "kind": "Solenoid"},
                       {"pattern": "SOL_*", "parameter": "dy", "sigma": 0.2, "kind": "Solenoid"}],
           "beam": []}


def _draw(inp: Path, pairing, seed: int = 3) -> dict:
    from dataclasses import asdict
    from linac_gen.io.project import load_project
    from linac_gen.reliability.errors import draw_seed
    beam = asdict(load_project(inp).beam)
    corr = {"enabled": True, "method": None, "n_iter": 3, "tol_mm": 1e-3, "bpm_noise": 0.0,
            "reading_backend": "envelope", "pairing": pairing}
    return draw_seed({"lattice_path": str(inp), "beam_config": beam, "budget": _BUDGET,
                      "seed": seed, "base_seed": 0, "correction": corr})


def test_auto_pairing_matches_the_cards_on_the_demo_deck(tmp_path):
    inp = _demo(tmp_path)
    cards = _draw(inp, "cards")
    auto = _draw(inp, "auto")
    assert cards["error"] is None and auto["error"] is None
    assert cards["correction"]["method"] == "one_to_one" == auto["correction"]["method"]
    assert auto["correction"]["pairing"] == "auto" and auto["correction"]["n_pairs"] == 3
    assert cards["kick_overrides"] and cards["kick_overrides"] == auto["kick_overrides"]  # same pairs, same kicks
    assert cards["correction"]["status"] == auto["correction"]["status"]
    # the draws themselves are the same stream: the correction consumes no entropy
    assert cards["overrides"] == auto["overrides"]


def test_auto_pairing_corrects_a_deck_without_cards(tmp_path):
    inp = _demo(tmp_path, strip_cards=True)
    cards = _draw(inp, "cards")
    assert cards["error"] is None and cards["correction"]["method"] == "none" and cards["kick_overrides"] == []
    auto = _draw(inp, "auto")
    assert auto["error"] is None and auto["correction"]["method"] == "one_to_one"
    assert len(auto["kick_overrides"]) >= 3 and auto["correction"]["status"] in ("converged", "max_iter")
    hist = auto["correction"]["history"]                         # per-pass rms travels with the draw
    assert hist and hist[-1]["rms_orbit_mm"] < auto["correction"]["rms_before_mm"]
    json.dumps(auto)                                             # the row stays JSON-serialisable
    bogus = _draw(inp, "bogus")
    assert bogus["error"] and "pairing" in bogus["error"]


def test_spec_error_budget_file_reference(tmp_path):
    from linac_gen.reliability.spec import ReliabilitySpec, load_spec, save_spec
    inp = _demo(tmp_path)
    (tmp_path / "budget.json").write_text(json.dumps(dict(_BUDGET, source="test")), encoding="utf-8")
    doc = {"__kind__": "linac_gen_reliability", "__version__": 1, "name": "ref", "input": inp.name,
           "preset": "quick", "error_budget": "budget.json"}
    (tmp_path / "spec.json").write_text(json.dumps(doc), encoding="utf-8")
    spec = load_spec(tmp_path / "spec.json")
    assert spec.error_budget == _BUDGET                          # inlined, metadata dropped
    assert spec.correction["pairing"] == "cards"                 # the preset default
    out = tmp_path / "saved.json"
    save_spec(spec, out)
    assert load_spec(out).error_budget == _BUDGET                # self-contained once saved
    doc["error_budget"] = "missing.json"
    (tmp_path / "spec2.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="error_budget"):
        load_spec(tmp_path / "spec2.json")
    bad = ReliabilitySpec(input=str(inp), error_budget="x.json")
    with pytest.raises(ValueError, match="error_budget"):
        bad.validate_shape()
    bad2 = ReliabilitySpec(input=str(inp), correction={"pairing": "nope"})
    with pytest.raises(ValueError, match="pairing"):
        bad2.validate_shape()


def test_preset_fill_keeps_the_user_tables_atomic(tmp_path):
    """A spec that names ONE SRF-trip variant runs one — the preset's other
    variant must not creep back in through the deep fill; the knob
    dictionaries around it are still filled."""
    from linac_gen.reliability.spec import default_spec
    inp = _demo(tmp_path)
    spec = default_spec(str(inp), availability={"variants": {"only": 10.0}, "n_trials": 7},
                        criticality={"rule": {"energy_pct": None}})
    assert spec.availability["variants"] == {"only": 10.0}
    assert spec.availability["n_trials"] == 7 and spec.availability["hours_per_year"] == 5000.0
    assert spec.criticality["rule"]["energy_pct"] is None and spec.criticality["rule"]["emit_growth_pct"] == 5.0
    spec2 = default_spec(str(inp), availability={"fault_classes": {"downtime": {"dist": "fixed", "value_s": 5.0}}})
    assert spec2.availability["fault_classes"] == {"downtime": {"dist": "fixed", "value_s": 5.0}}
    assert set(spec2.availability["variants"]) == {"srf_fdr", "srf_sns"}


def _quick_spec(inp: Path, **kw):
    from linac_gen.reliability.spec import default_spec
    base = dict(legs={"faults": True, "availability": False, "imperfections": False, "foil": True},
                classes=["S5"], compensation={"top_n": 0}, execution={"max_workers": 1, "serial": True})
    base.update(kw)
    return default_spec(str(inp), name="run", **base)


def test_export_preserves_a_pre_diff_campaign_document(tmp_path):
    """A campaign created before a preset gained a default key must still
    export a job the remote accepts: the job's spec is the campaign's own
    document with the paths patched, not a re-serialisation of the filled
    spec (whose extra key would change the stored hash)."""
    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.reliability.jobs import export_job
    from linac_gen.reliability.spec import spec_sha256
    inp = _demo(tmp_path)
    ReliabilityCampaign.create(tmp_path / "run", _quick_spec(inp))
    doc_path = tmp_path / "run" / "campaign.json"
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    assert doc["correction"].pop("pairing") == "cards"          # make it a pre-`pairing` document
    doc["spec_sha256"] = spec_sha256({k: v for k, v in doc.items() if not k.startswith("__")})
    doc_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    c = ReliabilityCampaign.load(tmp_path / "run")             # self-consistent: loads
    assert c.spec.correction["pairing"] == "cards"              # the preset fills it in memory
    job = export_job(c, tmp_path / "job", legs=["foil"])
    jdoc = json.loads((job / "campaign.json").read_text(encoding="utf-8"))
    assert "pairing" not in jdoc["correction"] and jdoc["input"] == "lattice/reliability_demo.lgproj"
    assert jdoc["circuits"] is None and jdoc["spec_sha256"] == doc["spec_sha256"]
    remote = ReliabilityCampaign.load(job)                      # would raise "spec hash drift" before the fix
    assert remote.spec.spec_sha256 == c.spec.spec_sha256
    assert len([it for it in remote.pending() if it["leg"] == "foil"]) == 7


def test_block_table_is_pinned_into_the_campaign_and_the_job(tmp_path):
    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.reliability.jobs import export_job
    from linac_gen.reliability.spec import load_spec, save_spec
    inp = _demo(tmp_path)
    table = tmp_path / "tables" / "my_blocks.csv"
    table.parent.mkdir()
    table.write_text("name,parent,mtbf_h,mttr_h,mttr_dist,n_parallel,k_required,fault_class,scenario_class\n"
                     "A,X,100,1,exponential,1,1,downtime,\nB,X,200,2,fixed,1,1,downtime,\n", encoding="utf-8")
    # a relative path in a spec file resolves against the spec, not the cwd
    spec = _quick_spec(inp, legs={"faults": False, "availability": True, "imperfections": False, "foil": False},
                       availability={"blocks": "tables/my_blocks.csv", "n_trials": 20})
    save_spec(spec, tmp_path / "spec.json")
    spec = load_spec(tmp_path / "spec.json")
    assert spec.availability["blocks"] == str(table.resolve())
    c = ReliabilityCampaign.create(tmp_path / "run", spec)
    pinned = tmp_path / "run" / "legs" / "availability" / "blocks.csv"
    assert pinned.exists() and "copied from" in pinned.read_text(encoding="utf-8")
    table.unlink()                                              # the original is gone: the copy serves
    c.run(legs=["availability"], serial=True)
    av = json.loads((tmp_path / "run" / "legs" / "availability" / "availability.json").read_text())
    assert av.get("error") is None and av["n_blocks"] == 2 and av["source"].startswith("campaign copy")
    job = export_job(c, tmp_path / "job")
    assert (job / "legs" / "availability" / "blocks.csv").exists()
    remote = ReliabilityCampaign.load(job)                      # the remote loads it (blocks path kept: identity)
    assert remote.spec.availability["blocks"] == str(table.resolve())
    shutil.rmtree(job / "legs" / "availability" / "availability.json", ignore_errors=True)
    (job / "legs" / "availability" / "availability.json").unlink(missing_ok=True)
    remote.run(legs=["availability"], serial=True)             # reads the pinned copy, not the vanished path
    av2 = json.loads((job / "legs" / "availability" / "availability.json").read_text())
    assert av2.get("error") is None and av2["n_blocks"] == 2
    # a missing table is refused at creation, before any folder exists
    bad = _quick_spec(inp, availability={"blocks": str(tmp_path / "nope.csv")})
    with pytest.raises(FileNotFoundError, match="availability.blocks"):
        ReliabilityCampaign.create(tmp_path / "run2", bad)
    assert not (tmp_path / "run2").exists()


def test_apply_preset_refuses_non_object_fields(tmp_path):
    from linac_gen.reliability.spec import default_spec
    inp = _demo(tmp_path)
    with pytest.raises(ValueError, match="error_budget"):
        default_spec(str(inp), error_budget="x.json")
    with pytest.raises(ValueError, match="error_budget"):
        default_spec(str(inp), error_budget=[])
    with pytest.raises(ValueError, match="availability"):
        default_spec(str(inp), availability=5)
