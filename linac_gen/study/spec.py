"""Parameter-study specification: dataclasses ↔ ``study.json``.

A study is fully described by a :class:`StudySpec`; the JSON file uses
the same ``__kind__`` sentinel convention as ``io/project.py`` and
tolerates unknown keys (schema drift never crashes a load).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Union

_KIND = "linac_gen_study"
_VERSION = 1

#: structural knobs a ParamSpec may target besides beam/element params
STRUCTURAL_VARS = ("nx", "grid_extent", "step1", "step2")

STRATEGIES = ("oat", "zip", "grid", "random", "lhs")


@dataclass
class ParamSpec:
    """One varied quantity.

    ``selector`` addressing (same grammar as ``--vary`` in the scan
    CLI / ``apply_element_override``):

    * ``"NAME.attr"`` / ``"@N.attr"`` (1-based) — element parameter
    * a bare :class:`BeamConfig` field name — beam parameter
    * one of ``nx / grid_extent / step1 / step2`` — numerics

    Values come either from an explicit ``values`` list or from
    ``start``/``stop``/``n`` with ``spacing`` ``"lin"`` or ``"log"``.
    ``baseline`` (the nominal value) is REQUIRED for the ``oat``
    strategy; the GUI fills it from the live lattice, the engine
    backfills it from the snapshot when omitted.
    """

    selector: str
    kind: str = "auto"          # auto | element | beam | structural
    values: list | None = None
    start: float | None = None
    stop: float | None = None
    n: int | None = None
    spacing: str = "lin"        # lin | log
    # verification / display metadata (optional in hand-written specs)
    display_name: str | None = None
    elem_class: str | None = None
    baseline: float | None = None

    def resolved_kind(self) -> str:
        if self.kind != "auto":
            return self.kind
        if "." in self.selector or self.selector.startswith("@"):
            return "element"
        if self.selector in STRUCTURAL_VARS:
            return "structural"
        return "beam"


@dataclass
class ObservableSpec:
    """A scalar extracted from each run's results file.

    ``quantity`` is any ``envelope/`` array name (``sigma_x``,
    ``emit_ny``, ``transmission``, …).  ``at`` is ``"end"`` (default),
    ``{"s_m": 12.5}``, or ``{"element": "NAME"}`` — the element form is
    resolved to ``s_m`` against the study's lattice snapshot at
    validation time, so evaluation itself never needs a lattice.
    """

    name: str
    quantity: str
    at: Union[str, dict] = "end"
    s_m: float | None = None      # filled by validation for non-"end"


@dataclass
class StudySpec:
    name: str
    input: str                       # deck or .lgproj (study-dir relative)
    mode: str = "envelope"           # envelope | mp
    env_solver: str = "matrix"
    strategy: str = "grid"
    parameters: list = field(default_factory=list)    # [ParamSpec]
    observables: list = field(default_factory=list)   # [ObservableSpec]
    seed: int = 42
    repeats: int = 1                 # seeds seed .. seed+repeats-1
    n_samples: int | None = None     # random / lhs
    sampler_seed: int = 1234
    beam: dict = field(default_factory=dict)          # fixed overrides
    sc: dict = field(default_factory=dict)
    numerics: dict = field(default_factory=dict)      # nx/extent/steps
    execution: dict = field(default_factory=dict)     # max_workers, ...
    lattice_sha256: str | None = None

    def validate_shape(self) -> None:
        """Cheap structural validation (no lattice access)."""
        if self.mode not in ("envelope", "mp"):
            raise ValueError(f"mode must be envelope|mp, got {self.mode!r}")
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"strategy must be one of {STRATEGIES}, got "
                f"{self.strategy!r}")
        if not self.parameters:
            raise ValueError("a study needs at least one parameter")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.strategy in ("random", "lhs") and not self.n_samples:
            raise ValueError(
                f"strategy {self.strategy!r} needs n_samples")
        for p in self.parameters:
            if p.values is None and None in (p.start, p.stop, p.n):
                raise ValueError(
                    f"parameter {p.selector!r}: give either values=[...] "
                    "or start/stop/n")
            if p.spacing not in ("lin", "log"):
                raise ValueError(
                    f"parameter {p.selector!r}: spacing must be lin|log")


def _filtered(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def save_spec(spec: StudySpec, path) -> None:
    doc = {"__kind__": _KIND, "__version__": _VERSION, **asdict(spec)}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def load_spec(path) -> StudySpec:
    data = json.loads(Path(path).read_text())
    if data.get("__kind__") != _KIND:
        raise ValueError(
            f"{path}: not a {_KIND} file (missing/wrong __kind__)")
    data = {k: v for k, v in data.items()
            if not k.startswith("__")}
    data["parameters"] = [
        p if isinstance(p, ParamSpec) else _filtered(ParamSpec, p)
        for p in data.get("parameters", [])]
    data["observables"] = [
        o if isinstance(o, ObservableSpec) else _filtered(ObservableSpec, o)
        for o in data.get("observables", [])]
    spec = _filtered(StudySpec, data)
    spec.validate_shape()
    return spec
