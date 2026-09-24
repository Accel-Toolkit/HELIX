"""The campaign spec (``campaign.json``): what a Reliability Study runs.

One JSON document, ``__kind__ = "linac_gen_reliability"``, with a
``preset`` (``quick`` | ``full`` | ``custom``) whose defaults
:func:`apply_preset` fills in and every knob of the four legs.  Unknown
keys are tolerated on load (forward compatibility, the study-spec rule);
``spec_sha256`` is the hash of the canonical document without the two
hash fields, pinned at creation so a campaign refuses a spec that drifted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

__all__ = ["ReliabilitySpec", "PRESETS", "CLASS_IDS", "apply_preset",
           "save_spec", "load_spec", "spec_sha256", "default_spec"]

_KIND = "linac_gen_reliability"
_VERSION = 1

#: Scenario classes of the fault leg.
CLASS_IDS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")
CLASS_LABELS = {
    "S1": "single cavity off", "S2": "single magnet off",
    "S3": "cavity detuned (amplitude / phase)", "S4": "magnet partial",
    "S5": "steerer off", "S6": "adjacent cavity pair off",
    "S7": "cryomodule off", "S8": "RF station off",
    "S9": "magnet circuit off (doublet / arc bus)",
    "S10": "single cavity off, RF phases frozen",
}


@dataclass
class ReliabilitySpec:
    name: str = "reliability"
    input: str = ""                     # deck or .lgproj
    tracewin_ini: str | None = None
    preset: str = "quick"
    beam: dict = field(default_factory=dict)          # BeamConfig overrides
    sc: dict = field(default_factory=dict)            # SpaceChargeConfig overrides
    numerics: dict = field(default_factory=dict)      # nx / grid_extent / step1 / step2
    legs: dict = field(default_factory=dict)          # {faults, availability, imperfections, foil}
    forward: dict = field(default_factory=dict)       # {mode, env_solver, mp_verify{critical, top_noncritical}}
    phase_pin: dict = field(default_factory=dict)     # {brackets: [rephased, frozen]}
    landmarks: dict = field(default_factory=dict)     # {treaty_element, foil_element}
    circuits: str | None = None                       # circuits.json path (relative to the spec)
    classes: list = field(default_factory=list)       # subset of CLASS_IDS
    class_options: dict = field(default_factory=dict) # {S3: {amp_scale, phase_deg}, S4: {amp_scale}}
    criticality: dict = field(default_factory=dict)
    compensation: dict = field(default_factory=dict)
    error_budget: dict = field(default_factory=dict)  # {element: [...], beam: [...]}
    correction: dict = field(default_factory=dict)
    seeds: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)
    foil: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    lattice_sha256: str | None = None
    spec_sha256: str | None = None
    circuits_sha256: str | None = None

    def validate_shape(self) -> None:
        if self.preset not in PRESETS and self.preset != "custom":
            raise ValueError(f"preset must be one of {sorted(PRESETS)} or 'custom', got {self.preset!r}")
        if not self.input:
            raise ValueError("spec.input (deck or .lgproj) is required")
        bad = [c for c in self.classes if c not in CLASS_IDS]
        if bad:
            raise ValueError(f"unknown scenario class(es) {bad}; known: {list(CLASS_IDS)}")
        mode = (self.forward or {}).get("mode", "envelope")
        if mode not in ("envelope", "mp"):
            raise ValueError(f"forward.mode must be envelope|mp, got {mode!r}")
        for b in (self.phase_pin or {}).get("brackets", []):
            if b not in ("rephased", "frozen"):
                raise ValueError(f"phase_pin.brackets entries must be rephased|frozen, got {b!r}")
        pairing = (self.correction or {}).get("pairing", "cards")
        if pairing not in ("cards", "auto"):
            raise ValueError(f"correction.pairing must be cards|auto, got {pairing!r}")
        if not isinstance(self.error_budget, dict):
            raise ValueError("error_budget must be an object with element / beam rows "
                             "(a file reference is resolved by load_spec)")
        if int((self.seeds or {}).get("n", 0)) < 0:
            raise ValueError("seeds.n must be >= 0")


_COMMON = {
    "legs": {"faults": True, "availability": True, "imperfections": True, "foil": True},
    "forward": {"mode": "envelope", "env_solver": "matrix",
                "mp_verify": {"critical": False, "top_noncritical": 0}},
    "phase_pin": {"brackets": ["rephased", "frozen"]},
    "landmarks": {"treaty_element": None, "foil_element": None},
    "class_options": {"S3": {"amp_scale": 0.9, "phase_deg": 5.0}, "S4": {"amp_scale": 0.9}},
    "criticality": {
        "weights": None, "lost_threshold_pct": 1.0,
        "rule": {"emit_growth_pct": 5.0, "loss_frac": 1e-4, "energy_pct": 0.5,
                 "w_per_m": {"default": 1.0, "LINAC": 0.1, "BTL": 1.0}},
    },
    "compensation": {
        "strategies": ["k_out_of_n"], "k": 1, "l": 1, "spare_names": [],
        "headroom": {"default": 1.5}, "phase_bounds_deg": 30.0,
        "algorithm": "least_squares", "cost_solver": "envelope", "max_iter": 120,
        "objectives": {"ke_out_min": True, "transmission": True, "phase_out": True,
                       "emit_growth": True, "size_max_mm": None},
        "top_n": 5, "verify_mp": False, "family_pattern": r"^([A-Za-z]+)",
    },
    "error_budget": {"element": [], "beam": []},
    "correction": {"enabled": False, "method": None, "n_iter": 5, "tol_mm": 0.05,
                   "bpm_noise": 0.0, "reading_backend": "envelope", "pairing": "cards"},
    "seeds": {"n": 5, "base_seed": 0,
              "faults_on_seeds": {"top_n": 3, "n_seeds": 2, "after_compensation": True}},
    "availability": {"blocks": None, "fault_classes": {}, "hours_per_year": 5000.0,
                     "n_trials": 200, "seed": 0,
                     "variants": {"srf_fdr": 62.5, "srf_sns": 6.0},   # MTBF (h) of an SRF trip
                     "budget_bins_s": [10.0, 60.0, 300.0, 1200.0, 3600.0, 14400.0]},
    "foil": {"element": None, "thickness_ug_cm2": [0.0, 435.0, 600.0, 800.0],
             "offsets_mm": [[1.0, 0.0], [0.0, 1.0]], "thinned_fraction": 0.5,
             "strip_model": "two_step", "extent_mm": None, "hits_per_particle": 1.0},
    "execution": {"max_workers": 1, "serial": False, "seeds_per_task": 4},
}

PRESETS = {
    "quick": {
        **copy.deepcopy(_COMMON),
        "classes": ["S1", "S2", "S5", "S10"],
    },
    "full": {
        **copy.deepcopy(_COMMON),
        "classes": list(CLASS_IDS),
    },
}
PRESETS["full"]["forward"] = {"mode": "envelope", "env_solver": "matrix",
                              "mp_verify": {"critical": True, "top_noncritical": 30}}
PRESETS["full"]["compensation"]["top_n"] = 20
PRESETS["full"]["compensation"]["verify_mp"] = True
PRESETS["full"]["compensation"]["strategies"] = ["k_out_of_n", "spares", "section_rematch"]
PRESETS["full"]["seeds"] = {"n": 200, "base_seed": 0,
                            "faults_on_seeds": {"top_n": 20, "n_seeds": 50, "after_compensation": True}}
PRESETS["full"]["availability"]["n_trials"] = 2000
PRESETS["full"]["execution"] = {"max_workers": 4, "serial": False, "seeds_per_task": 4}


#: Mappings a user REPLACES rather than extends: a table of variants or
#: fault classes given in a spec is the whole table (a preset entry must
#: not creep back in), unlike the knob dictionaries, whose unset knobs
#: are filled from the preset.
_ATOMIC_TABLES = frozenset({("availability", "variants"), ("availability", "fault_classes")})


def _deep_fill(dst: dict, src: dict, _path: tuple = ()) -> dict:
    """Fill missing keys of ``dst`` from ``src`` recursively (dst wins);
    a mapping listed in :data:`_ATOMIC_TABLES` is taken from ``dst`` as
    given when present."""
    out = copy.deepcopy(src)
    for k, v in dst.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and (_path + (k,)) not in _ATOMIC_TABLES:
            out[k] = _deep_fill(v, out[k], _path + (k,))
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_preset(spec: ReliabilitySpec) -> ReliabilitySpec:
    """Fill every unset knob from the spec's preset (``custom`` fills from
    ``quick``'s shape but keeps the given classes).  Returns ``spec``."""
    base = copy.deepcopy(PRESETS.get(spec.preset, PRESETS["quick"]))
    for f in fields(spec):
        if f.name not in base:
            continue
        cur = getattr(spec, f.name)
        if isinstance(base[f.name], dict):
            if cur is not None and not isinstance(cur, dict):
                raise ValueError(f"spec.{f.name} must be a JSON object, got {type(cur).__name__}"
                                 + (" (a file reference is resolved by load_spec only)"
                                    if f.name == "error_budget" and isinstance(cur, str) else ""))
            setattr(spec, f.name, _deep_fill(cur or {}, base[f.name], (f.name,)))
        elif f.name == "classes":
            if not cur and spec.preset != "custom":
                spec.classes = list(base["classes"])
            elif not cur:
                spec.classes = list(PRESETS["quick"]["classes"])
    return spec


def default_spec(input_path: str, *, preset: str = "quick", name: str | None = None,
                 **overrides) -> ReliabilitySpec:
    spec = ReliabilitySpec(name=name or f"reliability_{preset}", input=str(input_path),
                           preset=preset, **overrides)
    return apply_preset(spec)


def _filtered(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def _canonical(spec_or_doc) -> str:
    d = dict(spec_or_doc) if isinstance(spec_or_doc, dict) else asdict(spec_or_doc)
    # paths are not part of the identity (an exported job relocates the
    # deck; the lattice digest pins its content) — nor are the hashes or
    # the file header
    for k in ("lattice_sha256", "spec_sha256", "circuits_sha256", "input", "tracewin_ini", "circuits",
              "__kind__", "__version__"):
        d.pop(k, None)
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def spec_sha256(spec_or_doc) -> str:
    """Identity of a spec — of the DOCUMENT as saved when a file was loaded
    (``load_spec`` keeps it as ``spec.stored_sha256``), so a default key a
    later version adds to the preset does not invalidate a campaign."""
    if not isinstance(spec_or_doc, dict):
        stored = getattr(spec_or_doc, "stored_sha256", None)
        if stored:
            return stored
    return hashlib.sha256(_canonical(spec_or_doc).encode("utf-8")).hexdigest()


def save_spec(spec: ReliabilitySpec, path) -> None:
    doc = {"__kind__": _KIND, "__version__": _VERSION, **asdict(spec)}
    Path(path).write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    spec.stored_sha256 = spec_sha256(doc)


def load_spec(path) -> ReliabilitySpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("__kind__") != _KIND:
        raise ValueError(f"{path}: not a {_KIND} file (missing/wrong __kind__)")
    data = {k: v for k, v in data.items() if not k.startswith("__")}
    spec = _filtered(ReliabilitySpec, data)
    spec.stored_sha256 = spec_sha256(data)      # the document's identity, before preset filling
    if isinstance(spec.error_budget, str):
        # a budget kept in its own file, relative to the spec; inlined here so
        # the campaign.json a campaign saves is self-contained
        bp = Path(spec.error_budget)
        if not bp.is_absolute():
            bp = Path(path).parent / bp
        if not bp.is_file():
            raise FileNotFoundError(f"{path}: error_budget file not found: {bp}")
        budget = json.loads(bp.read_text(encoding="utf-8"))
        if not isinstance(budget, dict):
            raise ValueError(f"{bp}: an error budget is a JSON object with element / beam rows")
        spec.error_budget = {"element": list(budget.get("element") or []),
                             "beam": list(budget.get("beam") or [])}
    blocks = (spec.availability or {}).get("blocks") if isinstance(spec.availability, dict) else None
    if isinstance(blocks, str) and blocks and not Path(blocks).is_absolute():
        spec.availability["blocks"] = str((Path(path).parent / blocks).resolve())   # relative to the spec
    apply_preset(spec)
    spec.validate_shape()
    return spec
