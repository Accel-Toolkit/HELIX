"""``ReliabilityCampaign`` — one resumable, folder-backed campaign over
the four legs (structure after ``linac_gen.study.engine.StudyManager``).

Folder::

    <dir>/campaign.json  circuits.json  pins.json  lattice/<deck>
      plan/manifest.json (+ manifest_<wave>.json for the dynamic waves)
      legs/faults/runs/<id>/{results.h5,status.json}
      legs/faults/compensation/<case>_<strategy>.json
      legs/imperfections/draws/seed_NNNN.json  legs/imperfections/runs/…
      legs/foil/runs/…   legs/availability/{blocks.csv,availability.json}
      legs/*/*.csv  figures/*.png  summary.json  report.html

Every scan-pool run is one ``ScanPoint`` (``points.make_point``) whose
``status.json`` carries the worker row and the ``observables.evaluate_run``
extras; deltas, criticality and rankings are derived views built by
:meth:`summarize` from those facts.  ``create`` pins the deck's SHA-256,
harvests the frozen-phase pins for both engines and refuses a malformed
error budget; ``load`` refuses a drifted deck, spec or plan.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import logging
import re
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from linac_gen.reliability.spec import (ReliabilitySpec, load_spec, save_spec,
                                        spec_sha256)

__all__ = ["ReliabilityCampaign", "CampaignProgress", "LEGS", "compensation_job",
           "run_availability_leg"]

_log = logging.getLogger(__name__)
LEGS = ("faults", "imperfections", "foil", "availability")
_STRATEGIES = ("k_out_of_n", "spares", "section_rematch", "l_neighboring_lattices")


@dataclass
class CampaignProgress:
    leg: str
    phase: str
    done: int
    failed: int
    total: int
    mean_elapsed: float | None = None
    eta_s: float | None = None


def _sha256(path) -> str:
    """Digest of the input file — and, for a .lgproj, of the deck it points
    to as well, so an edited deck behind an unchanged project is caught."""
    h = hashlib.sha256()
    files = [Path(path)]
    if Path(path).suffix == ".lgproj":
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            deck = Path(path).parent / str(doc.get("lattice_path", ""))
            if deck.is_file():
                files.append(deck)
        except (OSError, json.JSONDecodeError):
            pass
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _pointed_deck(input_path) -> Path | None:
    p = Path(input_path)
    if p.suffix != ".lgproj":
        return p
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        deck = p.parent / str(doc.get("lattice_path", ""))
        return deck if deck.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def deck_sha256(input_path) -> str:
    """Digest of the deck TEXT alone (a .lgproj resolves to its deck) — what
    the HDF5 provenance group records per run."""
    p = Path(input_path)
    if p.suffix == ".lgproj":
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            deck = p.parent / str(doc.get("lattice_path", ""))
            if deck.is_file():
                p = deck
        except (OSError, json.JSONDecodeError):
            pass
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mode_tag(mode: str) -> str:
    return "mp" if mode == "mp" else "env"


# ---------------------------------------------------------------------------
# pool workers (module level: picklable)
# ---------------------------------------------------------------------------
def compensation_job(job: dict) -> dict:
    """One (case × strategy) compensation in a worker process."""
    t0 = time.time()
    try:
        from linac_gen.cli import common
        from linac_gen.core.config import BeamConfig
        from linac_gen.failures import CompensationConfig, compensate
        from linac_gen.failures.failure_mode import FailureKind, FailureMode
        from linac_gen.failures.scenario import FailureScenario
        with contextlib.redirect_stdout(io.StringIO()):
            lattice, _cfg, _conv = common.load_input(job["lattice_path"],
                                                     tracewin_ini=job.get("tracewin_ini"))
        for sel, val in job.get("pins") or ():
            common.apply_element_override(lattice, sel, val)
        cfg = BeamConfig(**job["beam_config"])
        fails = tuple((n, FailureMode(FailureKind(k), amp_scale=a, phase_deg=p))
                      for n, k, a, p in job["failures"])
        scenario = FailureScenario(failures=fails, label=job.get("label", ""))
        comp = dict(job["comp"])
        strategy = job["strategy"]
        kw = dict(algorithm=comp.get("algorithm", "least_squares"),
                  cost_solver=comp.get("cost_solver", "envelope"),
                  max_iter=int(comp.get("max_iter", 120)),
                  phase_bounds_deg=float(comp.get("phase_bounds_deg", 30.0)),
                  family_of=job.get("family_of"),
                  amp_bounds_by_family=job.get("amp_bounds_by_family"),
                  objectives=job.get("objectives"))
        if strategy == "k_out_of_n":
            ccfg = CompensationConfig(strategy="k_out_of_n", k=int(comp.get("k", 1)), **kw)
        elif strategy == "l_neighboring_lattices":
            ccfg = CompensationConfig(strategy="l_neighboring_lattices", l=int(comp.get("l", 1)), **kw)
        elif strategy == "spares":
            ccfg = CompensationConfig(strategy="manual", manual_names=list(job.get("manual_names") or []),
                                      extra_types=("spare",), **kw)
        elif strategy == "section_rematch":
            ccfg = CompensationConfig(strategy="manual", manual_names=list(job.get("manual_names") or []),
                                      **kw)
        else:
            raise ValueError(f"unknown compensation strategy {strategy!r}")
        res = compensate(lattice, cfg, scenario, job["name_to_class"], job["baseline_metrics"], ccfg)
        return {"case_id": job["case_id"], "strategy": strategy, "error": None,
                "recovered": bool(res.recovered), "match_success": bool(res.match_success),
                "compensator_names": list(res.compensator_names),
                "settings": dict(res.settings), "residual_cost": float(res.residual_cost),
                "metrics_after": dict(res.metrics_after), "message": str(res.message),
                "elapsed": time.time() - t0}
    except Exception as exc:            # noqa: BLE001
        import traceback
        return {"case_id": job.get("case_id"), "strategy": job.get("strategy"),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                "elapsed": time.time() - t0}


def run_availability_leg(leg_dir: Path, spec: ReliabilitySpec, class_split: dict | None) -> dict:
    """Leg B in process: blocks (CSV or the surrogate template), the fault
    classes, one Monte Carlo per SRF-trip variant.  ``class_split`` is
    keyed by the fault leg's scenario class; it reaches a block through
    the block's ``scenario_class`` column."""
    from linac_gen.reliability.availability import (
        DEFAULT_FAULT_CLASSES, Block, FaultClass, read_blocks_csv, simulate,
        write_blocks_csv, write_blocks_template)
    av = spec.availability or {}
    leg_dir.mkdir(parents=True, exist_ok=True)
    blocks_csv = leg_dir / "blocks.csv"
    src = av.get("blocks")
    if blocks_csv.exists():
        # the campaign's own copy: pinned from the user's table at creation
        # (so an exported job or a moved campaign never needs the original
        # path), or the table `--force` kept
        blocks = read_blocks_csv(blocks_csv)
        source = f"campaign copy of {src}" if src else "campaign copy"
    elif src:
        blocks = read_blocks_csv(src)
        write_blocks_csv(blocks, blocks_csv, header_note=f"copied from {src}")
        source = str(src)
    else:
        write_blocks_template(blocks_csv)
        blocks = read_blocks_csv(blocks_csv)
        source = "SURROGATE template (replace with the project's MTBF/MTTR data)"
    classes = dict(DEFAULT_FAULT_CLASSES)
    for name, d in (av.get("fault_classes") or {}).items():
        classes[name] = FaultClass(name, recovery=d.get("recovery"),
                                   downtime=bool(d.get("downtime", True)),
                                   trip=bool(d.get("trip", True)))
    _write_json(leg_dir / "fault_classes.json",
                {k: {"recovery": v.recovery, "downtime": v.downtime, "trip": v.trip}
                 for k, v in classes.items()})
    variants = av.get("variants") or {"base": None}
    bins = tuple(av.get("budget_bins_s") or ())
    applied = {b.name: dict(class_split[b.scenario_class]) for b in blocks
               if class_split and b.scenario_class and b.scenario_class in class_split}
    out = {"source": source, "n_blocks": len(blocks), "class_split_by_class": class_split,
           "class_split_applied": applied,
           "blocks_without_split": [b.name for b in blocks if b.name not in applied],
           "variants": {}}
    for vname, srf_mtbf_h in variants.items():
        rows = list(blocks)
        if srf_mtbf_h:
            rows.append(Block("SRF_TRIP_" + vname, float(srf_mtbf_h), 10.0 / 3600.0, "RF",
                              "fixed", fault_class="auto_rephase"))
        r = simulate(rows, classes, hours_per_year=float(av.get("hours_per_year", 5000.0)),
                     n_trials=int(av.get("n_trials", 200)), seed=int(av.get("seed", 0)),
                     class_split=applied or None,
                     bins_s=bins or (10.0, 60.0, 300.0, 1200.0, 3600.0, 14400.0))
        out["variants"][vname] = r.summary()
    _write_json(leg_dir / "availability.json", out)
    return out


# ---------------------------------------------------------------------------
class ReliabilityCampaign:
    def __init__(self, campaign_dir, spec: ReliabilitySpec):
        self.dir = Path(campaign_dir)
        self.spec = spec
        self.legs_dir = self.dir / "legs"
        self.plan_dir = self.dir / "plan"
        self.figures_dir = self.dir / "figures"
        self._input_path = self._resolve_input()
        self._lattice = None
        self._beam_cfg = None
        self._conv = None
        self._circuits = None
        self._pins = None
        self._elem_index = None

    # ------------------------------------------------------------------ setup
    def _resolve_input(self) -> Path:
        p = Path(self.spec.input)
        if not p.is_absolute():
            cand = (self.dir / p)
            p = cand.resolve() if cand.exists() else p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"campaign input not found: {p}")
        return p

    @property
    def input_path(self) -> Path:
        return self._input_path

    def _load(self):
        if self._lattice is None:
            from linac_gen.cli import common
            with contextlib.redirect_stdout(io.StringIO()):
                lat, cfg, conv = common.load_input(str(self._input_path),
                                                   tracewin_ini=self.spec.tracewin_ini)
            for k, v in (self.spec.beam or {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            self._lattice, self._beam_cfg, self._conv = lat, cfg, conv
            self._elem_index = {}
            for i, e in enumerate(lat.elements):
                nm = getattr(e, "name", None)
                if nm and nm not in self._elem_index:
                    self._elem_index[nm] = i
        return self._lattice, self._beam_cfg, self._conv

    @property
    def circuits(self):
        if self._circuits is None:
            from linac_gen.reliability.circuits import CircuitMap, build_circuit_map
            p = self.dir / "circuits.json"
            if p.exists():
                self._circuits = CircuitMap.load(p)
            elif self.spec.circuits:
                src = Path(self.spec.circuits)
                if not src.is_absolute():
                    src = (self._input_path.parent / src)
                self._circuits = CircuitMap.load(src)
            else:
                lat, _c, _v = self._load()
                self._circuits = build_circuit_map(lat)
        return self._circuits

    @property
    def landmarks(self) -> dict:
        lm = dict(self.spec.landmarks or {})
        cm = self.circuits
        return {"treaty_element": lm.get("treaty_element") or cm.treaty_element,
                "foil_element": lm.get("foil_element") or cm.foil_element}

    def pins(self, mode: str) -> tuple:
        if self._pins is None:
            self._pins = _read_json(self.dir / "pins.json") or {}
        d = self._pins.get(_mode_tag(mode)) or {}
        return tuple((sel, val) for sel, val in d.get("overrides", []))

    @classmethod
    def create(cls, campaign_dir, spec: ReliabilitySpec) -> "ReliabilityCampaign":
        campaign_dir = Path(campaign_dir)
        if (campaign_dir / "campaign.json").exists():
            raise FileExistsError(f"{campaign_dir}/campaign.json already exists -- "
                                  "use load() / `reliability resume`")
        spec.validate_shape()
        if any(ch in str(spec.name) for ch in "/\\") or ".." in str(spec.name) or str(spec.name).startswith("."):
            raise ValueError(f"campaign name {spec.name!r} must be a plain folder name")
        c = cls(campaign_dir, spec)
        lat, cfg, conv = c._load()
        missing = c.circuits.check(lat)
        if missing:
            raise ValueError("circuits.json names elements the lattice does not have: "
                             + ", ".join(missing[:8]))
        lm = c.landmarks
        for key in ("treaty_element", "foil_element"):
            v = lm.get(key)
            if v and v not in c._elem_index:
                raise ValueError(f"landmark {key}={v!r} is not in the lattice")
        if (spec.legs or {}).get("foil", True) and not lm.get("foil_element"):
            _log.warning("no foil element: the foil leg will be skipped")
        # pins for both engines
        from linac_gen.cli.common import make_sc_config, make_step_config
        from linac_gen.reliability.pins import harvest_pins
        env_solver = (spec.forward or {}).get("env_solver", "matrix")
        pins = {}
        p_env = harvest_pins(lat, cfg, mode="envelope", env_solver=env_solver)
        pins["env"] = {"overrides": [list(o) for o in p_env.overrides], "pins": p_env.pins,
                       "kinds": p_env.kinds, "skipped": list(p_env.skipped)}
        if (spec.forward or {}).get("mode") == "mp" or (spec.forward or {}).get("mp_verify", {}).get("critical") \
                or (spec.compensation or {}).get("verify_mp"):
            sc = make_sc_config(cfg, conv, dict(spec.numerics or {}))
            st = make_step_config(conv, dict(spec.numerics or {}))
            p_mp = harvest_pins(lat, cfg, mode="mp", sc_config=sc, step_config=st)
            pins["mp"] = {"overrides": [list(o) for o in p_mp.overrides], "pins": p_mp.pins,
                          "kinds": p_mp.kinds, "skipped": list(p_mp.skipped)}
        # error budget audit
        if (spec.legs or {}).get("imperfections", True) and (spec.error_budget or {}).get("element"):
            from linac_gen.reliability.errors import (ErrorBudget, audit_or_raise,
                                                      build_error_study)
            study = build_error_study(copy.deepcopy(lat), cfg,
                                      ErrorBudget.from_dict(spec.error_budget))
            audit_or_raise(study)
        blocks_src = (spec.availability or {}).get("blocks")
        if blocks_src and not Path(blocks_src).is_file():
            raise FileNotFoundError(f"availability.blocks not found: {blocks_src}")
        campaign_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("lattice", "plan", "legs", "figures"):
            (campaign_dir / sub).mkdir(exist_ok=True)
        try:
            shutil.copy2(c._input_path, campaign_dir / "lattice" / c._input_path.name)
            deck = _pointed_deck(c._input_path)
            if deck is not None and deck != c._input_path:
                shutil.copy2(deck, campaign_dir / "lattice" / deck.name)
        except OSError as exc:
            _log.warning("campaign snapshot copy failed: %s", exc)
        c.circuits.save(campaign_dir / "circuits.json")
        spec.circuits_sha256 = _sha256(campaign_dir / "circuits.json")
        if blocks_src:
            # pin the user's block table into the campaign now: the leg, an
            # exported job and a campaign copied to another machine all read
            # this copy, never the original path
            from linac_gen.reliability.availability import read_blocks_csv, write_blocks_csv
            rows = read_blocks_csv(blocks_src)
            (campaign_dir / "legs" / "availability").mkdir(parents=True, exist_ok=True)
            write_blocks_csv(rows, campaign_dir / "legs" / "availability" / "blocks.csv",
                             header_note=f"copied from {blocks_src}")
        _write_json(campaign_dir / "pins.json", pins)
        c._pins = pins
        spec.lattice_sha256 = _sha256(c._input_path)
        spec.stored_sha256 = None
        spec.spec_sha256 = spec_sha256(spec)
        save_spec(spec, campaign_dir / "campaign.json")
        manifest = c._static_plan()
        _write_json(campaign_dir / "plan" / "manifest.json", manifest)
        return c

    @classmethod
    def load(cls, campaign_dir) -> "ReliabilityCampaign":
        campaign_dir = Path(campaign_dir)
        spec = load_spec(campaign_dir / "campaign.json")
        c = cls(campaign_dir, spec)
        if spec.lattice_sha256:
            now = _sha256(c._input_path)
            if now != spec.lattice_sha256:
                raise RuntimeError(f"lattice {c._input_path} changed since the campaign was "
                                   f"created (sha {now[:12]} != pinned {spec.lattice_sha256[:12]}) "
                                   "-- refusing to mix two machines")
        if spec.spec_sha256 and spec_sha256(spec) != spec.spec_sha256:
            raise RuntimeError("campaign.json was edited after creation (spec hash drift) "
                               "-- start a new campaign or restore the spec")
        cj = campaign_dir / "circuits.json"
        if spec.circuits_sha256 and cj.exists() and _sha256(cj) != spec.circuits_sha256:
            raise RuntimeError("circuits.json changed since the campaign was created -- the "
                               "groups, sections and landmarks are part of the plan; refusing")
        stored = _read_json(campaign_dir / "plan" / "manifest.json")
        if stored is not None and stored != c._static_plan():
            raise RuntimeError("the recorded plan no longer matches the spec -- refusing to "
                               "mix plans")
        return c

    # ------------------------------------------------------------------ plan
    def _cases(self):
        from linac_gen.reliability.scenarios import enumerate_cases
        lat, _c, _v = self._load()
        return enumerate_cases(lat, self.spec, self.circuits)

    def _case_item(self, case, mode: str, extra_ov=(), suffix: str = "") -> dict:
        _cases, n2c = self._cases_cached()
        ov = list(self.pins(mode)) if case.bracket == "frozen" else []
        ov += list(case.scenario.element_overrides(n2c))
        ov += list(extra_ov)
        return {"id": f"A_{case.case_id}{suffix}_{_mode_tag(mode)}", "leg": "faults", "kind": "point",
                "case_id": case.case_id, "cls": case.cls, "bracket": case.bracket,
                "label": case.label, "elements": list(case.element_names), "mode": mode,
                "seed": 42, "overrides": [list(o) for o in ov]}

    def _cases_cached(self):
        if not hasattr(self, "_case_cache"):
            self._case_cache = self._cases()
        return self._case_cache

    def _baseline_items(self, leg: str, mode: str, prefix: str) -> list:
        items = []
        brackets = (self.spec.phase_pin or {}).get("brackets", ["rephased", "frozen"])
        for br in brackets:
            ov = list(self.pins(mode)) if br == "frozen" else []
            items.append({"id": f"{prefix}_base_{br[:2]}_{_mode_tag(mode)}", "leg": leg,
                          "kind": "point", "case_id": "baseline", "cls": "baseline",
                          "bracket": br, "label": f"baseline ({br})", "elements": [],
                          "mode": mode, "seed": 42, "overrides": [list(o) for o in ov]})
        return items

    def _plan_faults(self) -> list:
        if not (self.spec.legs or {}).get("faults", True):
            return []
        mode = (self.spec.forward or {}).get("mode", "envelope")
        cases, _n2c = self._cases_cached()
        return self._baseline_items("faults", mode, "A") + [self._case_item(c, mode) for c in cases]

    def _plan_foil(self) -> list:
        if not (self.spec.legs or {}).get("foil", True):
            return []
        foil = self.landmarks.get("foil_element")
        if not foil:
            return []
        fs = self.spec.foil or {}
        mode = (self.spec.forward or {}).get("mode", "envelope")
        lat, _c, _v = self._load()
        el = lat.elements[self._elem_index[foil]]
        nominal = float(getattr(el, "thickness_ug_cm2", 0.0))
        snaps = (foil,) if mode == "mp" else ()
        common_ov = []
        if mode == "mp":
            common_ov = [[f"{foil}.strip_model", fs.get("strip_model", "two_step")],
                         [f"{foil}.seed", 12345]]
            if fs.get("extent_mm"):
                ext = fs["extent_mm"]
                ex, ey = (ext, ext) if isinstance(ext, (int, float)) else ext
                common_ov.append([f"{foil}.extent_mm", f"{ex},{ey}"])

        def item(tag, ov, label):
            return {"id": f"D_{tag}_{_mode_tag(mode)}", "leg": "foil", "kind": "point",
                    "case_id": tag, "cls": "foil", "bracket": "rephased", "label": label,
                    "elements": [foil], "mode": mode, "seed": 42,
                    "overrides": common_ov + [list(o) for o in ov],
                    "snapshot_elements": list(snaps)}
        items = [item("nominal", [], f"nominal foil ({nominal:g} ug/cm2)")]
        for t in fs.get("thickness_ug_cm2") or []:
            if abs(float(t) - nominal) < 1e-9:
                continue
            items.append(item(f"thick_{float(t):g}", [(f"{foil}.thickness_ug_cm2", float(t))],
                              f"thickness {float(t):g} ug/cm2" if t else "foil missing"))
        for dx, dy in fs.get("offsets_mm") or []:
            items.append(item(f"off_{dx:g}_{dy:g}", [(f"{foil}.dx", float(dx)), (f"{foil}.dy", float(dy))],
                              f"foil offset dx={dx:g} dy={dy:g} mm"))
        fr = fs.get("thinned_fraction")
        if fr:
            items.append(item("thinned", [(f"{foil}.thickness_ug_cm2", nominal * float(fr))],
                              f"thinned foil x{float(fr):g}"))
        return items

    def _plan_imperfections(self) -> list:
        if not (self.spec.legs or {}).get("imperfections", True):
            return []
        if not (self.spec.error_budget or {}).get("element") and not (self.spec.error_budget or {}).get("beam"):
            return []
        sd = self.spec.seeds or {}
        n = int(sd.get("n", 0))
        base = int(sd.get("base_seed", 0))
        mode = (self.spec.forward or {}).get("mode", "envelope")
        items = [{"id": f"C_control_{_mode_tag(mode)}", "leg": "imperfections", "kind": "point",
                  "case_id": "control", "cls": "control", "bracket": "rephased",
                  "label": "control (no errors)", "elements": [], "mode": mode, "seed": 42,
                  "overrides": []}]
        for i in range(n):
            seed = base + i
            items.append({"id": f"C_draw_{seed:04d}", "leg": "imperfections", "kind": "draw",
                          "case_id": f"seed_{seed:04d}", "cls": "draw", "bracket": "rephased",
                          "label": f"draw seed {seed}", "elements": [], "mode": mode, "seed": seed,
                          "overrides": []})
        for i in range(n):
            seed = base + i
            items.append({"id": f"C_seed_{seed:04d}_{_mode_tag(mode)}", "leg": "imperfections",
                          "kind": "point", "case_id": f"seed_{seed:04d}", "cls": "seed",
                          "bracket": "rephased", "label": f"error seed {seed}", "elements": [],
                          "mode": mode, "seed": seed + 1000, "draw": f"C_draw_{seed:04d}",
                          "overrides": []})
        return items

    def _static_plan(self) -> dict:
        return {"faults": self._plan_faults(), "imperfections": self._plan_imperfections(),
                "foil": self._plan_foil(),
                "availability": ([{"id": "B_availability", "leg": "availability",
                                   "kind": "availability", "case_id": "availability",
                                   "cls": "availability", "bracket": "", "label": "block-diagram Monte Carlo",
                                   "elements": [], "mode": "", "seed": 0, "overrides": []}]
                                 if (self.spec.legs or {}).get("availability", True) else [])}

    def plan(self, legs=None) -> dict:
        m = _read_json(self.plan_dir / "manifest.json") or self._static_plan()
        for wave in ("comp", "mp", "fos"):
            extra = _read_json(self.plan_dir / f"manifest_{wave}.json")
            if extra:
                for it in extra:
                    m.setdefault(it["leg"], []).append(it)
        if legs:
            m = {k: v for k, v in m.items() if k in legs}
        return m

    # ------------------------------------------------------------------ status
    def item_dir(self, item: dict) -> Path:
        kind = item.get("kind")
        if kind == "draw":
            return self.legs_dir / "imperfections" / "draws"
        if kind == "comp":
            return self.legs_dir / "faults" / "compensation"
        if kind == "availability":
            return self.legs_dir / "availability"
        return self.legs_dir / item["leg"] / "runs" / item["id"]

    def _status_path(self, item: dict) -> Path:
        kind = item.get("kind")
        if kind == "draw":
            return self.item_dir(item) / f"{item['case_id']}.json"
        if kind == "comp":
            return self.item_dir(item) / f"{item['case_id']}_{item['strategy']}.json"
        if kind == "availability":
            return self.item_dir(item) / "availability.json"
        return self.item_dir(item) / "status.json"

    def item_status(self, item: dict) -> dict | None:
        return _read_json(self._status_path(item))

    def _is_complete(self, item: dict, *, retry_failed: bool = False) -> bool:
        st = self.item_status(item)
        if st is None:
            return False
        if item.get("kind") in ("draw", "comp", "availability"):
            if st.get("error"):
                return not retry_failed
            return True
        if st.get("status") == "ok":
            return (self.item_dir(item) / "results.h5").exists()
        if st.get("status") == "failed":
            return not retry_failed
        return False

    def pending(self, leg: str | None = None, *, retry_failed: bool = False) -> list:
        out = []
        for lg, items in self.plan().items():
            if leg and lg != leg:
                continue
            out += [it for it in items if not self._is_complete(it, retry_failed=retry_failed)]
        return out

    def status(self) -> dict:
        """Per leg: total, done (complete by the same rule as ``pending``),
        failed."""
        out = {}
        for lg, items in self.plan().items():
            done = failed = 0
            for it in items:
                st = self.item_status(it)
                if st is None:
                    continue
                if st.get("error") or st.get("status") == "failed":
                    failed += 1
                elif self._is_complete(it):
                    done += 1
            out[lg] = {"total": len(items), "done": done, "failed": failed}
        return out

    # ------------------------------------------------------------------ run
    def _point_for(self, item: dict):
        from linac_gen.reliability.points import make_point
        run_dir = self.item_dir(item)
        run_dir.mkdir(parents=True, exist_ok=True)
        ov = [tuple(o) for o in item.get("overrides", [])]
        beam_ov = dict(item.get("beam_overrides") or {})
        if item.get("draw"):
            d = _read_json(self.legs_dir / "imperfections" / "draws" / f"{item['case_id']}.json") or {}
            ov = [tuple(o) for o in d.get("overrides", [])] + [tuple(o) for o in d.get("kick_overrides", [])] + ov
            beam_ov = dict(d.get("beam_config") or {}) | beam_ov
        return make_point(self.spec, str(self._input_path), overrides=ov, mode=item["mode"],
                          seed=int(item.get("seed", 42)), out_path=run_dir / "results.h5",
                          beam_overrides=beam_ov or None,
                          snapshot_elements=item.get("snapshot_elements") or (),
                          loss_metrics=True, tracewin_ini=self.spec.tracewin_ini)

    def _sections(self):
        lat, _c, _v = self._load()
        return self.circuits.section_ranges(lat)

    def _extras(self, item: dict) -> dict:
        from linac_gen.reliability.observables import evaluate_run
        _lat, cfg, _v = self._load()
        h5 = self.item_dir(item) / "results.h5"
        if not h5.exists():
            return {}
        try:
            return evaluate_run(str(h5), element_index=self._elem_index, landmarks=self.landmarks,
                                sections=self._sections(), current_mA=float(cfg.current),
                                duty_pct=float(cfg.duty_cycle),
                                hits_per_particle=float((self.spec.foil or {}).get("hits_per_particle", 1.0)))
        except Exception as exc:          # noqa: BLE001
            _log.warning("extras failed for %s: %s", item["id"], exc)
            return {"_error": f"{type(exc).__name__}: {exc}"}

    def _run_points(self, items: list, *, max_workers, serial, progress_cb, should_stop,
                    leg: str, phase: str, counters: dict) -> bool:
        """False when stopped."""
        from linac_gen.parallel.scan_pool import run_scan_points, run_scan_points_serial
        if not items:
            return True
        for it in items:
            if (self.item_dir(it) / "results.h5.part").exists():
                (self.item_dir(it) / "results.h5.part").unlink()
        points = [self._point_for(it) for it in items]
        elapsed_q: deque = deque(maxlen=10)
        stopped = {"v": False}

        def _on_done(i, row):
            it = items[i]
            ok = row.get("error") is None
            extras = self._extras(it) if ok else {}
            status = {"status": "ok" if ok else "failed",
                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "elapsed": row.get("elapsed"), "error": row.get("error"),
                      "item": {k: it.get(k) for k in ("id", "leg", "kind", "case_id", "cls", "bracket",
                                                      "label", "elements", "mode", "seed", "draw",
                                                      "strategy", "variant")},
                      "overrides": it.get("overrides", []),
                      "metrics": {k: v for k, v in row.items() if k not in ("error", "traceback")},
                      "extras": extras}
            _write_json(self._status_path(it), status)
            counters["done"] += 1
            if not ok:
                counters["failed"] += 1
                _log.warning("%s FAILED: %s", it["id"], row.get("error"))
            if row.get("elapsed"):
                elapsed_q.append(float(row["elapsed"]))
            if progress_cb is not None:
                mean = sum(elapsed_q) / len(elapsed_q) if elapsed_q else None
                remaining = counters["total"] - counters["done"]
                width = max(1, min(max_workers or 1, remaining or 1))
                progress_cb(CampaignProgress(leg=leg, phase=phase, done=counters["done"],
                                             failed=counters["failed"], total=counters["total"],
                                             mean_elapsed=mean,
                                             eta_s=(remaining * mean / width) if mean else None))

        def _stop():
            s = bool(should_stop()) if should_stop is not None else False
            stopped["v"] = stopped["v"] or s
            return s

        if serial or (max_workers or 1) <= 1:
            run_scan_points_serial(points, on_done=_on_done, should_stop=_stop)
        else:
            run_scan_points(points, on_done=_on_done, max_workers=max_workers, should_stop=_stop)
        return not stopped["v"]

    def _run_jobs(self, fn, items: list, jobs: list, *, max_workers, progress_cb, should_stop,
                  leg: str, phase: str, counters: dict) -> bool:
        from linac_gen.reliability.comp_pool import run_jobs
        if not items:
            return True
        stopped = {"v": False}

        def _on_done(i, row):
            it = items[i]
            row = dict(row)
            row["item"] = {k: it.get(k) for k in ("id", "leg", "kind", "case_id", "cls", "label", "seed",
                                                  "strategy")}
            row["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write_json(self._status_path(it), row)
            counters["done"] += 1
            if row.get("error"):
                counters["failed"] += 1
                _log.warning("%s FAILED: %s", it["id"], row.get("error"))
            if progress_cb is not None:
                progress_cb(CampaignProgress(leg=leg, phase=phase, done=counters["done"],
                                             failed=counters["failed"], total=counters["total"]))

        def _stop():
            s = bool(should_stop()) if should_stop is not None else False
            stopped["v"] = stopped["v"] or s
            return s

        run_jobs(fn, jobs, max_workers=max_workers, on_done=_on_done, should_stop=_stop)
        return not stopped["v"]

    # ---- fault leg helpers -------------------------------------------------
    def fault_rows(self) -> list:
        """Status rows of the fault leg with deltas / criticality vs the
        matching baseline (bracket + mode)."""
        from linac_gen.failures.criticality import criticality_score
        from linac_gen.reliability.observables import criticality_rule
        items = self.plan().get("faults", [])
        rows = []
        base = {}
        for it in items:
            st = self.item_status(it)
            if st is None or it.get("kind") != "point":
                continue
            m = dict(st.get("metrics") or {})
            m.update({k: v for k, v in (st.get("extras") or {}).items() if not k.startswith("_")})
            m["ok"] = st.get("status") == "ok"
            rec = {"item": it, "status": st, "row": m}
            rows.append(rec)
            if it.get("case_id") == "baseline":
                base[(it["bracket"], it["mode"])] = m
        crit = self.spec.criticality or {}
        secs = self._sections()
        for rec in rows:
            it, m = rec["item"], rec["row"]
            b = base.get((it["bracket"], it["mode"]))
            rec["baseline"] = b
            if b is None or not m["ok"] or it.get("case_id") == "baseline":
                rec["criticality"] = None
                continue
            score, terms, lost = criticality_score(b, m, weights=crit.get("weights"),
                                                   lost_threshold_pct=float(crit.get("lost_threshold_pct", 1.0)))
            critical, rterms = criticality_rule(b, m, crit.get("rule") or {}, secs)
            d_e = (abs(m["ref_w_kin"] - b["ref_w_kin"]) if m.get("ref_w_kin") is not None
                   and b.get("ref_w_kin") is not None else None)
            d_phi = None
            if m.get("ref_phi_s_end") is not None and b.get("ref_phi_s_end") is not None:
                d_phi = (m["ref_phi_s_end"] - b["ref_phi_s_end"] + 180.0) % 360.0 - 180.0
            rec["criticality"] = {"score": score, "terms": terms, "beam_lost": lost,
                                  "critical": critical or lost, "rule_terms": rterms,
                                  "d_energy_mev": d_e, "d_phi_end_deg": d_phi}
        return rows

    def ranking(self, bracket: str = "rephased") -> list:
        rows = [r for r in self.fault_rows()
                if r.get("criticality") and r["item"]["bracket"] == bracket
                and r["item"]["mode"] == (self.spec.forward or {}).get("mode", "envelope")]
        rows.sort(key=lambda r: -r["criticality"]["score"])
        return rows

    def _family_of(self, names) -> dict:
        pat = (self.spec.compensation or {}).get("family_pattern") or r"^([A-Za-z]+)"
        rx = re.compile(pat)
        out = {}
        for n in names:
            m = rx.match(n)
            out[n] = m.group(1) if m else n
        return out

    def _plan_compensation(self) -> list:
        comp = self.spec.compensation or {}
        top_n = int(comp.get("top_n", 0))
        if top_n <= 0:
            return []
        ranked = self.ranking("rephased")
        chosen = [r for r in ranked if r["criticality"]["critical"]][:top_n]
        if len(chosen) < top_n:
            chosen += [r for r in ranked if r not in chosen][:top_n - len(chosen)]
        _cases, n2c = self._cases_cached()
        strategies = [s for s in comp.get("strategies") or ["k_out_of_n"] if s in _STRATEGIES]
        spares = list(self.circuits.spares) + list(comp.get("spare_names") or [])
        items = []
        for r in chosen:
            it = r["item"]
            if it["cls"] == "baseline":
                continue
            fails = [[n, m.kind.value, m.amp_scale, m.phase_deg]
                     for n, m in self._case_by_id(it["case_id"]).scenario.failures]
            for strat in strategies:
                if strat == "spares" and not spares:
                    continue
                items.append({"id": f"A_comp_{it['case_id']}_{strat}", "leg": "faults", "kind": "comp",
                              "case_id": it["case_id"], "cls": it["cls"], "bracket": it["bracket"],
                              "label": it["label"], "elements": it["elements"], "mode": it["mode"],
                              "seed": 42, "strategy": strat, "failures": fails, "overrides": []})
        return items

    def _case_by_id(self, case_id: str):
        cases, _n2c = self._cases_cached()
        for c in cases:
            if c.case_id == case_id:
                return c
        raise KeyError(case_id)

    def _section_members(self, names: tuple, category_names: list) -> list:
        """Same-category elements downstream of the failed ones to the end
        of their section (the section_rematch zone)."""
        lat, _c, _v = self._load()
        idx = self._elem_index
        pos = [idx[n] for n in names if n in idx]
        if not pos:
            return []
        start = max(pos)
        s = 0.0
        exit_s = {}
        for i, el in enumerate(lat.elements):
            s += float(getattr(el, "length", 0.0) or 0.0)
            exit_s[i] = s
        end_i = len(lat.elements) - 1
        for name, s0, s1 in self._sections():
            if s0 * 1e3 <= exit_s[start] <= s1 * 1e3 + 1e-9:
                for i in range(start, len(lat.elements)):
                    if exit_s[i] > s1 * 1e3 + 1e-9:
                        end_i = i - 1
                        break
                break
        return [n for n in category_names if n in idx and start < idx[n] <= end_i]

    def _comp_jobs(self, items: list) -> list:
        from linac_gen.failures.element_filter import ALL_TYPES, EXTRA_TYPES, failable_elements
        lat, cfg, _v = self._load()
        _cases, n2c = self._cases_cached()
        comp = self.spec.compensation or {}
        base_rows = {(r["item"]["bracket"], r["item"]["mode"]): r["row"] for r in self.fault_rows()
                     if r["item"]["case_id"] == "baseline"}
        labels = {n: lbl for (n, lbl, _c) in failable_elements(lat, tuple(ALL_TYPES) + tuple(EXTRA_TYPES))}
        fam = self._family_of(list(n2c))
        headroom = comp.get("headroom") or {"default": 1.5}
        bounds = {f: (0.0, float(headroom.get(f, headroom.get("default", 1.5)))) for f in set(fam.values())}
        spares = list(self.circuits.spares) + list(comp.get("spare_names") or [])
        jobs = []
        for it in items:
            case = self._case_by_id(it["case_id"])
            b = base_rows.get((it["bracket"], it["mode"])) or {}
            obj_spec = comp.get("objectives") or {}
            objectives = {"ke_out_min": bool(obj_spec.get("ke_out_min", True)),
                          "transmission": bool(obj_spec.get("transmission", True)),
                          "emit_growth": bool(obj_spec.get("emit_growth", True))}
            if obj_spec.get("phase_out") and b.get("ref_phi_s_end") is not None:
                objectives["phase_out"] = {"phase_deg": float(b["ref_phi_s_end"]), "weight": 5.0,
                                           "tol_deg": 0.0}
            if obj_spec.get("size_max_mm"):
                sm = float(obj_spec["size_max_mm"])
                objectives["size_max"] = {"x_mm": sm, "y_mm": sm, "weight": 1.0}
            manual = []
            if it["strategy"] == "spares":
                manual = list(spares)
            elif it["strategy"] == "section_rematch":
                cat = {labels.get(n) for n in case.element_names}
                same = [n for n, lbl in labels.items()
                        if (lbl in cat) or (lbl == "spare" and "cavity" in cat)]
                manual = self._section_members(case.element_names, same)
            jobs.append({"case_id": it["case_id"], "strategy": it["strategy"], "label": it["label"],
                         "lattice_path": str(self._input_path), "tracewin_ini": self.spec.tracewin_ini,
                         "beam_config": asdict(cfg), "failures": it["failures"],
                         "pins": [list(o) for o in (self.pins(it["mode"]) if it["bracket"] == "frozen" else ())],
                         "comp": {k: comp.get(k) for k in ("algorithm", "cost_solver", "max_iter",
                                                            "phase_bounds_deg", "k", "l")},
                         "name_to_class": n2c,
                         "baseline_metrics": {k: b.get(k) for k in ("ref_w_kin", "transmission", "emit_nx",
                                                                     "emit_ny", "emit_nz")},
                         "family_of": fam, "amp_bounds_by_family": bounds,
                         "objectives": objectives, "manual_names": manual})
        return jobs

    def compensation_rows(self) -> list:
        """Compensation results with the RULE verdict: ``recovered_by_rule``
        = the matcher's energy/transmission verdict AND the compensated
        beam not critical by the study rule against the case's baseline
        (a magnet fault leaves the energy untouched, so the energy verdict
        alone would call any magnet compensation recovered)."""
        from linac_gen.reliability.observables import criticality_rule
        base_rows = {(r["item"]["bracket"], r["item"]["mode"]): r["row"] for r in self.fault_rows()
                     if r["item"]["case_id"] == "baseline"}
        rule = (self.spec.criticality or {}).get("rule") or {}
        out = []
        for it in self.plan().get("faults", []):
            if it.get("kind") != "comp":
                continue
            st = self.item_status(it)
            if st is None:
                continue
            after_critical = None
            b = base_rows.get((it["bracket"], it["mode"]))
            if not st.get("error") and b and st.get("metrics_after"):
                after_critical, _terms = criticality_rule(b, st["metrics_after"], rule, ())
            st = dict(st)
            st["after_critical"] = after_critical
            st["recovered_by_rule"] = (bool(st.get("recovered")) and after_critical is False)
            out.append({"item": it, "result": st})
        return out

    def _plan_mp_verification(self) -> list:
        fw = (self.spec.forward or {}).get("mp_verify") or {}
        if (self.spec.forward or {}).get("mode") == "mp":
            return []
        if not fw.get("critical") and not int(fw.get("top_noncritical", 0)) \
                and not (self.spec.compensation or {}).get("verify_mp"):
            return []
        ranked = self.ranking("rephased") + self.ranking("frozen")
        chosen = []
        if fw.get("critical"):
            chosen += [r for r in ranked if r["criticality"]["critical"]]
        top = int(fw.get("top_noncritical", 0))
        if top:
            chosen += [r for r in ranked if r not in chosen][:top]
        items = self._baseline_items("faults", "mp", "A")
        for r in chosen:
            items.append(self._case_item(self._case_by_id(r["item"]["case_id"]), "mp"))
        if (self.spec.compensation or {}).get("verify_mp"):
            for c in self.compensation_rows():
                res = c["result"]
                if res.get("error") or not res.get("recovered_by_rule"):
                    continue
                case = self._case_by_id(c["item"]["case_id"])
                ov = [(k, v) for k, v in (res.get("settings") or {}).items()]
                it = self._case_item(case, "mp", extra_ov=ov, suffix=f"_comp_{c['item']['strategy']}")
                it["strategy"] = c["item"]["strategy"]
                items.append(it)
        seen, out = set(), []
        for it in items:
            if it["id"] not in seen:
                seen.add(it["id"]); out.append(it)
        return out

    # ---- imperfections helpers --------------------------------------------
    def _draw_jobs(self, items: list) -> list:
        _lat, cfg, _v = self._load()
        corr = dict(self.spec.correction or {})
        return [{"lattice_path": str(self._input_path), "tracewin_ini": self.spec.tracewin_ini,
                 "beam_config": asdict(cfg), "budget": dict(self.spec.error_budget or {}),
                 "seed": int(it["seed"]), "base_seed": 0,
                 "correction": corr if corr.get("enabled") else None} for it in items]

    def _plan_faults_on_seeds(self) -> list:
        fos = (self.spec.seeds or {}).get("faults_on_seeds") or {}
        top_n, n_seeds = int(fos.get("top_n", 0)), int(fos.get("n_seeds", 0))
        if top_n <= 0 or n_seeds <= 0:
            return []
        seeds = [it for it in self.plan().get("imperfections", []) if it.get("cls") == "seed"][:n_seeds]
        if not seeds:
            return []
        ranked = self.ranking("rephased")[:top_n]
        comps = {c["item"]["case_id"]: c["result"] for c in self.compensation_rows()
                 if not c["result"].get("error") and c["result"].get("recovered_by_rule")}
        _cases, n2c = self._cases_cached()
        items = []
        for r in ranked:
            case = self._case_by_id(r["item"]["case_id"])
            fault_ov = [list(o) for o in case.scenario.element_overrides(n2c)]
            settings = comps.get(case.case_id, {}).get("settings") if fos.get("after_compensation", True) else None
            for sd in seeds:
                for variant, extra in (("before", []), ("after", [[k, v] for k, v in (settings or {}).items()])):
                    if variant == "after" and not settings:
                        continue
                    items.append({"id": f"C_fos_{case.case_id}_{sd['case_id']}_{variant}_{_mode_tag(sd['mode'])}",
                                  "leg": "imperfections", "kind": "point", "case_id": case.case_id,
                                  "cls": "faults_on_seeds", "bracket": case.bracket, "label": case.label,
                                  "elements": list(case.element_names), "mode": sd["mode"],
                                  "seed": sd["seed"], "draw": sd["draw"], "variant": variant,
                                  "overrides": fault_ov + extra})
        return items

    # ---- the run ---------------------------------------------------------
    def run(self, legs=None, *, max_workers: int | None = None, serial: bool = False,
            force: bool = False, retry_failed: bool = False, progress_cb=None,
            should_stop=None):
        """Execute every pending item of the selected legs (all by
        default), waves in dependency order, then :meth:`summarize`.
        Returns the summary path, or ``None`` when stopped."""
        legs = tuple(legs) if legs else LEGS
        ex = self.spec.execution or {}
        if max_workers is None:
            max_workers = int(ex.get("max_workers", 1))
        serial = serial or bool(ex.get("serial", False))
        if force and self.legs_dir.exists():
            # scoped to the selected legs; the user's block table survives
            for lg in legs:
                d = self.legs_dir / lg
                if not d.exists():
                    continue
                keep = d / "blocks.csv"
                kept = keep.read_bytes() if keep.exists() else None
                shutil.rmtree(d)
                if kept is not None:
                    d.mkdir(parents=True, exist_ok=True)
                    keep.write_bytes(kept)
            for wave, owner in (("comp", "faults"), ("mp", "faults"), ("fos", "imperfections")):
                p = self.plan_dir / f"manifest_{wave}.json"
                if p.exists() and (owner in legs or "faults" in legs):
                    p.unlink()
        self.legs_dir.mkdir(exist_ok=True)
        if retry_failed:
            # a re-queued fault scenario changes the ranking the dynamic
            # waves were planned from: plan them again (completed items
            # keep their ids and are skipped)
            faults_pending = [it for it in self.pending("faults", retry_failed=True)
                              if it.get("kind") == "point" and not it.get("strategy")]
            if faults_pending:
                for wave in ("comp", "mp", "fos"):
                    p = self.plan_dir / f"manifest_{wave}.json"
                    if p.exists():
                        p.unlink()

        def counters_for(items):
            done = sum(1 for it in items if self._is_complete(it))
            return {"done": done, "failed": 0, "total": len(items)}

        def points_wave(leg, phase, items):
            todo = [it for it in items if not self._is_complete(it, retry_failed=retry_failed)]
            c = counters_for(items)
            return self._run_points(todo, max_workers=max_workers, serial=serial, progress_cb=progress_cb,
                                    should_stop=should_stop, leg=leg, phase=phase, counters=c)

        def jobs_wave(fn, leg, phase, items, jobs_fn):
            todo = [it for it in items if not self._is_complete(it, retry_failed=retry_failed)]
            c = counters_for(items)
            return self._run_jobs(fn, todo, jobs_fn(todo), max_workers=max_workers, progress_cb=progress_cb,
                                  should_stop=should_stop, leg=leg, phase=phase, counters=c)

        def dyn_manifest(wave, planner):
            # an EMPTY stored manifest is never frozen: it means the wave
            # had nothing to plan from when it was written (a job run
            # without the fault leg, a stopped run) — plan again
            p = self.plan_dir / f"manifest_{wave}.json"
            items = _read_json(p)
            if not items:
                items = planner()
                _write_json(p, items)
            return items

        static = self.plan()
        if "faults" in legs and static.get("faults"):
            base = [it for it in static["faults"] if it.get("kind") == "point" and not it.get("strategy")
                    and not it["id"].endswith("_mp") or (it.get("kind") == "point" and it["mode"] != "mp")]
            base = [it for it in static["faults"] if it.get("kind") == "point"
                    and it["mode"] == (self.spec.forward or {}).get("mode", "envelope")
                    and "_comp_" not in it["id"]]
            if not points_wave("faults", "scenarios", base):
                return None
            comp_items = dyn_manifest("comp", self._plan_compensation)
            if not jobs_wave(compensation_job, "faults", "compensation", comp_items, self._comp_jobs):
                return None
            mp_items = dyn_manifest("mp", self._plan_mp_verification)
            if not points_wave("faults", "mp verification", mp_items):
                return None
        if "imperfections" in legs and static.get("imperfections"):
            from linac_gen.reliability.errors import draw_seed
            draws = [it for it in static["imperfections"] if it.get("kind") == "draw"]
            if not jobs_wave(draw_seed, "imperfections", "draws", draws, self._draw_jobs):
                return None
            seeds = [it for it in static["imperfections"] if it.get("kind") == "point"
                     and it.get("cls") in ("seed", "control")]
            if not points_wave("imperfections", "seeds", seeds):
                return None
            if "faults" in legs or (self.plan_dir / "manifest_comp.json").exists():
                fos = dyn_manifest("fos", self._plan_faults_on_seeds)
                if not points_wave("imperfections", "faults on seeds", fos):
                    return None
        if "foil" in legs and static.get("foil"):
            if not points_wave("foil", "foil scenarios", static["foil"]):
                return None
        if "availability" in legs and static.get("availability"):
            it = static["availability"][0]
            if not self._is_complete(it, retry_failed=retry_failed):
                if progress_cb is not None:
                    progress_cb(CampaignProgress("availability", "Monte Carlo", 0, 0, 1))
                try:
                    run_availability_leg(self.legs_dir / "availability", self.spec, self._class_split())
                except Exception as exc:          # noqa: BLE001
                    _write_json(self._status_path(it), {"error": f"{type(exc).__name__}: {exc}"})
                    _log.warning("availability leg FAILED: %s", exc)
                if progress_cb is not None:
                    progress_cb(CampaignProgress("availability", "Monte Carlo", 1, 0, 1))
        return self.summarize()

    def _class_split(self) -> dict | None:
        """Recovered-by fractions of the fault leg → a class split for the
        cavity / magnet blocks of the availability leg (``{'CAVITY': {...}}``
        keyed by the element label, matched to blocks by name prefix)."""
        rows = self.fault_rows()
        comps = {}
        for c in self.compensation_rows():
            res = c["result"]
            if res.get("error"):
                continue
            comps.setdefault(c["item"]["case_id"], []).append(bool(res.get("recovered_by_rule")))
        buckets: dict = {}
        for r in rows:
            crit = r.get("criticality")
            it = r["item"]
            if crit is None or it["cls"] in ("baseline",) or it["bracket"] != "rephased" \
                    or it.get("strategy") or it["mode"] != (self.spec.forward or {}).get("mode", "envelope"):
                continue
            key = it["cls"]
            b = buckets.setdefault(key, {"auto_rephase": 0, "operator_retune": 0, "downtime": 0})
            if not crit["critical"]:
                b["auto_rephase"] += 1
            elif it["case_id"] not in comps:
                continue                      # critical but never attempted: unknown, not downtime
            elif any(comps[it["case_id"]]):
                b["operator_retune"] += 1
            else:
                b["downtime"] += 1
        out = {}
        for key, b in buckets.items():
            n = sum(b.values())
            if n:
                out[key] = {k: v / n for k, v in b.items()}
        return out or None

    # ------------------------------------------------------------------ summary
    def summarize(self):
        from linac_gen.reliability import summary as _summary
        return _summary.write_summary(self)
