"""Job export / import: run the heavy waves of a campaign elsewhere.

``export_job`` writes ``job.json`` (the spec, both hashes, the expected
items), the manifests, pins, circuits, draws and a copy of the deck at
the same relative depth (``--bundle-fields`` also copies the field-map
files).  ``reliability run <job dir>`` on the remote executes the pending
items with resume.  ``import_results`` checks both hashes and every
item's status / results file before copying into the campaign, never
overwrites an ``ok`` item, and writes ``import.json``.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from linac_gen.reliability.spec import spec_sha256

__all__ = ["export_job", "import_results"]

_JOB_KIND = "linac_gen_reliability_job"


def _relative_lattice_layout(campaign, job_dir: Path, bundle_fields: bool) -> str:
    """Copy the deck (and its .lgproj) into the job at ``lattice/<name>``,
    keeping a project's relative deck path so its bytes — and the digest —
    are unchanged; return the job-relative input path."""
    src = campaign.input_path
    dst_dir = job_dir / "lattice"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)
    if src.suffix == ".lgproj":
        try:
            doc = json.loads(src.read_text(encoding="utf-8"))
            rel = Path(doc.get("lattice_path", ""))
            deck = src.parent / rel
            if deck.is_file():
                target = dst_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(deck, target)
        except (OSError, json.JSONDecodeError):
            pass
    if bundle_fields:
        lat, _c, _v = campaign._load()
        for el in lat.elements:
            fp = getattr(el, "field_path", None) or getattr(getattr(el, "field_data", None), "path", None)
            if fp and Path(fp).is_file():
                target = dst_dir / Path(fp).name
                if not target.exists():
                    shutil.copy2(fp, target)
    return f"lattice/{src.name}"


def export_job(campaign, job_dir, *, legs=None, bundle_fields: bool = False) -> Path:
    job_dir = Path(job_dir)
    if (job_dir / "job.json").exists():
        raise FileExistsError(f"{job_dir}/job.json already exists")
    job_dir.mkdir(parents=True, exist_ok=True)
    plan = campaign.plan(legs)
    legs = list(plan)
    expected = [{"id": it["id"], "leg": it["leg"], "kind": it.get("kind", "point")}
                for items in plan.values() for it in items]
    rel = _relative_lattice_layout(campaign, job_dir, bundle_fields)
    # completed items of the exported legs travel with the job, so the
    # remote resumes instead of re-running them
    for items in plan.values():
        for it in items:
            if campaign._is_complete(it):
                src = campaign._status_path(it)
                dst = job_dir / src.relative_to(campaign.dir)
                if it.get("kind") == "point":
                    shutil.copytree(src.parent, dst.parent, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
    spec = campaign.spec
    (job_dir / "plan").mkdir(exist_ok=True)
    for p in campaign.plan_dir.glob("manifest*.json"):
        if p.name != "manifest.json" and not json.loads(p.read_text(encoding="utf-8")):
            continue                              # an empty dynamic wave is not a plan
        shutil.copy2(p, job_dir / "plan" / p.name)
    for name in ("pins.json", "circuits.json"):
        if (campaign.dir / name).exists():
            shutil.copy2(campaign.dir / name, job_dir / name)
    draws = campaign.legs_dir / "imperfections" / "draws"
    if draws.exists():
        shutil.copytree(draws, job_dir / "legs" / "imperfections" / "draws", dirs_exist_ok=True)
    blocks = campaign.legs_dir / "availability" / "blocks.csv"
    if blocks.exists():                           # the campaign's pinned block table
        (job_dir / "legs" / "availability").mkdir(parents=True, exist_ok=True)
        shutil.copy2(blocks, job_dir / "legs" / "availability" / "blocks.csv")
    # the job's spec is the campaign's DOCUMENT with only the relocated paths
    # patched (paths are outside the identity), never a re-serialisation of
    # the preset-filled spec: a default key added since the campaign was
    # created would change the stored hash and the remote would refuse it
    doc = json.loads((campaign.dir / "campaign.json").read_text(encoding="utf-8"))
    doc["input"] = rel
    doc["circuits"] = None
    # availability.blocks stays as written (it is part of the identity); the
    # remote reads the pinned copy above first and never needs that path
    doc["lattice_sha256"] = spec.lattice_sha256
    doc["spec_sha256"] = spec.spec_sha256
    (job_dir / "campaign.json").write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    job = {"__kind__": _JOB_KIND, "__version__": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "campaign_dir": str(campaign.dir), "name": spec.name, "legs": legs,
           "lattice_sha256": spec.lattice_sha256, "spec_sha256": spec.spec_sha256,
           "expected": expected}
    (job_dir / "job.json").write_text(json.dumps(job, indent=1) + "\n", encoding="utf-8")
    legs_opt = ",".join(legs)
    (job_dir / "README.txt").write_text(
        "HELIX reliability job\n\n"
        f"Legs: {legs_opt}\n\n"
        f"Run on the compute machine (resume-by-default; the legs above are the default):\n"
        f"    PYTHONPATH=.:gui python -m linac_gen reliability run {job_dir.name} --legs {legs_opt} --parallel N\n\n"
        f"Then copy the folder back and import into the campaign:\n"
        f"    python -m linac_gen reliability import {job_dir.name} --into {campaign.dir}\n",
        encoding="utf-8")
    return job_dir


def import_results(job_dir, campaign, *, allow_partial: bool = False) -> dict:
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if job.get("__kind__") != _JOB_KIND:
        raise ValueError(f"{job_dir}/job.json is not a {_JOB_KIND}")
    spec = campaign.spec
    if job.get("lattice_sha256") != spec.lattice_sha256:
        raise RuntimeError("job lattice sha differs from the campaign's -- refusing")
    if job.get("spec_sha256") != spec.spec_sha256 or spec_sha256(spec) != spec.spec_sha256:
        raise RuntimeError("job spec sha differs from the campaign's -- refusing")
    # the job's NON-EMPTY dynamic manifests join the campaign's plan (an
    # empty one means the wave had nothing to plan from over there)
    for p in (job_dir / "plan").glob("manifest_*.json"):
        dst = campaign.plan_dir / p.name
        if not dst.exists() and json.loads(p.read_text(encoding="utf-8")):
            shutil.copy2(p, dst)
    plan = campaign.plan()
    by_id = {it["id"]: it for items in plan.values() for it in items}
    from linac_gen.reliability.campaign import ReliabilityCampaign, deck_sha256
    deck_sha = deck_sha256(campaign.input_path)
    remote = ReliabilityCampaign.__new__(ReliabilityCampaign)
    remote.dir = job_dir
    remote.spec = spec
    remote.legs_dir = job_dir / "legs"
    remote.plan_dir = job_dir / "plan"
    # everything the job's own plan lists for its legs — the waves it
    # planned while running are part of it, not only the items frozen at
    # export time
    job_legs = job.get("legs") or sorted({e["leg"] for e in job.get("expected", [])})
    remote_items = {}
    for exp in job.get("expected", []):
        remote_items[exp["id"]] = exp
    for p in (job_dir / "plan").glob("manifest_*.json"):
        for it in json.loads(p.read_text(encoding="utf-8")) or []:
            if it.get("leg") in job_legs:
                remote_items.setdefault(it["id"], {"id": it["id"], "leg": it["leg"],
                                                   "kind": it.get("kind", "point")})
    imported, skipped, missing, bad, to_copy = [], [], [], [], []
    for exp in remote_items.values():
        it = by_id.get(exp["id"])
        if it is None:
            bad.append(f"{exp['id']}: not in the campaign plan")
            continue
        src_status = remote._status_path(it)
        if not src_status.exists():
            missing.append(exp["id"])
            continue
        if it.get("kind") == "point":
            if not (remote.item_dir(it) / "results.h5").exists() or (remote.item_dir(it) / "results.h5.part").exists():
                bad.append(f"{exp['id']}: no complete results.h5")
                continue
            try:
                from linac_gen.study.observables import read_provenance
                prov = read_provenance(str(remote.item_dir(it) / "results.h5"))
            except Exception:                     # noqa: BLE001
                prov = {}
            sha = prov.get("lattice_sha256")
            if sha and sha != deck_sha:
                bad.append(f"{exp['id']}: results.h5 provenance sha differs")
                continue
        cur = campaign.item_status(it)
        if cur is not None and not cur.get("error") and cur.get("status", "ok") == "ok":
            skipped.append(exp["id"])
            continue
        to_copy.append((it, src_status))
    # validate everything first: a refused import copies nothing
    if bad:
        raise RuntimeError("job results refused:\n  - " + "\n  - ".join(bad))
    if missing and not allow_partial:
        raise RuntimeError(f"{len(missing)} expected item(s) missing from the job "
                           f"(use --allow-partial to import the rest): {', '.join(missing[:8])}")
    for it, src_status in to_copy:
        dst_status = campaign._status_path(it)
        dst_status.parent.mkdir(parents=True, exist_ok=True)
        if it.get("kind") == "point":
            shutil.copytree(remote.item_dir(it), campaign.item_dir(it), dirs_exist_ok=True)
        else:
            shutil.copy2(src_status, dst_status)
        imported.append(it["id"])
    rec = {"job": str(job_dir), "imported": imported, "skipped_ok": skipped, "missing": missing,
           "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (campaign.dir / "import.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec
