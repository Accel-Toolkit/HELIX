"""Named space-charge methods.

A *method* is a named combination of the settings in
:class:`~linac_gen.core.config.SpaceChargeConfig` that decide how the
bunched-beam space-charge field is computed: the charge-assignment kernel,
whether the grid is fixed or follows the beam, the solver engine, and so on.
Each method has

* **pinned** settings, which define it (change one and it is a different
  method), and
* **params** with the method's own defaults, which may be changed without
  leaving the method (grid size, grid extent, Green's function and how it
  is built, whether ``HELIX_SC_GRID`` cards apply).

The name is a label: solvers run from the resolved field values, never from
the name, so ``cic_fixed`` runs exactly the code a fixed CIC grid always ran.

Resolution rules for ``SpaceChargeConfig(...)`` (applied in its
``__post_init__`` through :func:`resolve_in_place`):

* no method and no method-controlled field set: the default method
  (:data:`DEFAULT_METHOD`; ``None`` means the classic values below);
* method-controlled fields set but no method: those fields on the classic
  base, labelled with the method they match (or ``"custom"``);
* ``method="X"``: X's pinned settings and param defaults, with any field
  given explicitly taking precedence; a given field that contradicts one of
  X's pinned settings makes it the method the settings match, with that
  method's defaults for the fields not given (``method="tsc_adaptive",
  grid_mode="fixed"`` is ``method="tsc_fixed"``);
* ``method="custom"``: the given fields on the classic base;
* ``method="default"``: :data:`DEFAULT_METHOD`.

After construction ``method`` names the method whose pinned settings the
config matches -- the one named, unless a given field contradicts it --
or ``"custom"`` when it matches none, so a label always describes the
settings, also after ``dataclasses.replace``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

REGISTRY_VERSION = 1

#: Values of the method-controlled fields that ``SpaceChargeConfig()``
#: produced before named methods existed.
CLASSIC: Mapping[str, object] = MappingProxyType({
    "nx": 96, "ny": 96, "nz": 96,
    "grid_extent": 5.0,
    "grid_mode": "fixed",
    "kernel": "cic",
    "green_kind": "igf",
    "sc_backend": "numpy",
    "grid_card": "apply",
    "green_build": "exact",
    "reuse_every": 1,
})

#: Fields a method controls (pinned or as a param).
METHOD_FIELDS: tuple[str, ...] = tuple(CLASSIC)

#: Method used when nothing is chosen.  ``None`` keeps the classic values.
DEFAULT_METHOD: str | None = None

CUSTOM = "custom"
DEFAULT_ALIAS = "default"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    summary: str
    pinned: Mapping[str, object]
    params: Mapping[str, object]
    default_candidate: bool = True
    experimental: bool = False


_GRID48_FIXED = {"nx": 48, "ny": 48, "nz": 48, "grid_extent": 7.0,
                 "green_kind": "igf"}
# An adaptive grid follows the beam, so a HELIX_SC_GRID card's wider extent
# would only coarsen its cells: the named adaptive methods ignore the card
# while their grid is adaptive ("auto").
# An adaptive grid rebuilds its Green's function every kick, so the named
# adaptive methods of the numpy PIC use the symmetric (mirrored) build.
_GRID48_ADAPTIVE = {"nx": 48, "ny": 48, "nz": 48, "grid_extent": 6.0,
                    "green_kind": "igf", "grid_card": "auto"}
_NUMPY_ADAPTIVE = {**_GRID48_ADAPTIVE, "green_build": "symmetric"}


def _numpy(kernel: str, grid_mode: str, reuse_every: int = 1) -> dict:
    """Pinned settings of a numpy-PIC method."""
    return {"kernel": kernel, "grid_mode": grid_mode, "sc_backend": "numpy",
            "reuse_every": reuse_every}


def _reuse_summary(kernel_text: str, k: int) -> str:
    nth = {2: "2nd", 4: "4th"}[k]
    return (f"{kernel_text} on a grid resized to the beam; the field is solved "
            f"on every {nth} kick and, on the kicks between, the last solved "
            "field is rescaled to the beam's current centre, rms sizes and "
            "charge (one gather instead of a deposit and a Poisson solve).")

_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "cic_fixed", "CIC, fixed grid",
        "Cloud-in-cell charge assignment (each particle shared among 8 grid "
        "nodes) on a grid sized from the beam at the first space-charge "
        "kick and kept for the rest of the run.",
        pinned=_numpy("cic", "fixed"),
        params=_GRID48_FIXED),
    MethodSpec(
        "tsc_fixed", "TSC, fixed grid",
        "Triangular-shaped-cloud charge assignment (each particle shared "
        "among 27 grid nodes; smoother, less grid noise) on a grid sized "
        "from the beam at the first space-charge kick and kept.",
        pinned=_numpy("tsc", "fixed"),
        params=_GRID48_FIXED),
    MethodSpec(
        "cic_adaptive", "CIC, adaptive grid",
        "Cloud-in-cell charge assignment on a grid resized to the beam at "
        "every kick, so the cells keep the same size relative to the beam.",
        pinned=_numpy("cic", "adaptive"),
        params=_NUMPY_ADAPTIVE),
    MethodSpec(
        "tsc_adaptive", "TSC, adaptive grid",
        "Triangular-shaped-cloud charge assignment on a grid resized to the "
        "beam at every kick.",
        pinned=_numpy("tsc", "adaptive"),
        params=_NUMPY_ADAPTIVE),
    *(MethodSpec(
        f"{kern}_adaptive_reuse{k}",
        f"{kern.upper()}, adaptive grid, field solved every "
        f"{ {2: '2nd', 4: '4th'}[k] } kick",
        _reuse_summary(text, k),
        pinned=_numpy(kern, "adaptive", k),
        params=_NUMPY_ADAPTIVE)
      for kern, text in (("cic", "Cloud-in-cell charge assignment"),
                         ("tsc", "Triangular-shaped-cloud charge assignment"))
      for k in (2, 4)),
    MethodSpec(
        "differentiable_pic", "Differentiable PIC (PyTorch)",
        "The same 3-D PIC written in PyTorch so gradients can flow through "
        "the space-charge kick; used for gradient-based matching.  Always "
        "resizes its grid to the beam.",
        pinned={"sc_backend": "torch", "grid_mode": "adaptive"},
        # the torch PIC has no grid-extent setting a card could change
        params={**_GRID48_ADAPTIVE, "kernel": "cic", "grid_card": "apply"},
        default_candidate=False),
    MethodSpec(
        "halo_pic", "HALO-PIC (experimental)",
        "Coarse PIC with a learned correction for the halo field; needs a "
        "trained corrector.",
        pinned={"sc_backend": "halo"},
        params={},
        default_candidate=False, experimental=True),
)

_BY_NAME = {s.name: s for s in _SPECS}


def names() -> tuple[str, ...]:
    """Registry names in display order (``custom`` last)."""
    return tuple(s.name for s in _SPECS) + (CUSTOM,)


def get(name: str) -> MethodSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown space-charge method {name!r}; valid names: "
            f"{', '.join(names())} (or 'default')") from None


def table() -> str:
    """Plain-text list of the methods (``--sc-method list``)."""
    lines = []
    for spec in _SPECS:
        tag = "  [default]" if spec.name == DEFAULT_METHOD else ""
        lines.append(f"{spec.name:20s} {spec.label}{tag}")
        lines.append(f"{'':20s} {spec.summary}")
    lines.append(f"{CUSTOM:20s} Every setting chosen individually.")
    if DEFAULT_METHOD is None:
        lines.append("")
        lines.append("With no method chosen, a project keeps its own settings "
                     "and a bare lattice uses the classic ones (96^3 grid, "
                     "+-5 sigma, fixed grid, CIC).")
    return "\n".join(lines)


def base_fields(name: str | None) -> dict:
    """Field values a method starts from (classic values for None/custom)."""
    if name in (None, CUSTOM):
        return dict(CLASSIC)
    spec = get(name)
    return {**CLASSIC, **spec.params, **spec.pinned}


def infer(fields: Mapping[str, object]) -> str:
    """The method whose pinned settings all match ``fields`` (most pinned
    settings wins; registry order breaks ties), else ``custom``."""
    best, best_n = CUSTOM, -1
    for spec in _SPECS:
        if all(fields.get(k) == v for k, v in spec.pinned.items()):
            if len(spec.pinned) > best_n:
                best, best_n = spec.name, len(spec.pinned)
    return best


def _label(name, fields: Mapping[str, object]) -> str:
    """``name`` if its pinned settings all hold in ``fields``, else the
    inferred method (``custom`` when none matches)."""
    if name in _BY_NAME and all(fields.get(k) == v
                                for k, v in _BY_NAME[name].pinned.items()):
        return name
    return infer(fields)


def card_policy(cfg) -> str:
    """What a HELIX_SC_GRID card does under ``cfg``: "apply" or "ignore"
    ("auto" ignores the card on an adaptive grid, applies it on a fixed
    one)."""
    policy = getattr(cfg, "grid_card", None) or "apply"
    if policy == "auto":
        return ("ignore" if getattr(cfg, "grid_mode", "fixed") == "adaptive"
                else "apply")
    return policy


def effective_method(cfg) -> str:
    """Method label consistent with ``cfg``'s current field values (fields
    may have been set after construction)."""
    fields = {k: getattr(cfg, k, None) for k in METHOD_FIELDS}
    return _label(getattr(cfg, "method", None), fields)


def _resolve_fields(name, given: Mapping) -> tuple[dict, str]:
    """Method-controlled field values and label for method ``name`` (None:
    the classic values) with the fields ``given`` set explicitly: the
    method's pinned settings and defaults, the given fields on top; a given
    field that contradicts a pinned setting makes it the method the settings
    match, with that method's defaults for the fields not given."""
    resolved = {**base_fields(name), **given}
    label = _label(name, resolved)
    if name not in (None, CUSTOM) and label not in (name, CUSTOM):
        resolved = {**base_fields(label), **given}
        label = _label(label, resolved)
    return resolved, label


def resolve_in_place(cfg) -> None:
    """Fill ``cfg``'s unset method-controlled fields and set its label."""
    name = getattr(cfg, "method", None)
    if name == DEFAULT_ALIAS:
        name = DEFAULT_METHOD
    given = {k: getattr(cfg, k) for k in METHOD_FIELDS
             if getattr(cfg, k) is not None}
    if name is None and not given and DEFAULT_METHOD is not None:
        name = DEFAULT_METHOD
    resolved, label = _resolve_fields(name, given)   # raises: unknown name
    for k, v in resolved.items():
        setattr(cfg, k, v)
    cfg.method = label


# ---------------------------------------------------------------------------
# Project files and command lines
# ---------------------------------------------------------------------------
#: Project-file ("convergence" block) keys that hold settings of the
#: method-controlled fields.
METHOD_PROJECT_KEYS = ("grid_nx", "grid_extent_sigma", "kernel", "green_kind",
                       "grid_mode")
#: Project-file keys that hold space-charge settings.  A project with any of
#: them but no ``sc_method`` is resolved exactly as before named methods
#: existed (the "legacy rule").
LEGACY_PROJECT_KEYS = METHOD_PROJECT_KEYS + ("dc_kernel", "backend",
                                             "csr_enabled")

#: Fields for which the text "auto" / "none" / "null" means ``None`` (a value
#: with its own meaning: train_images=None engages the images automatically).
_NONE_IS_A_VALUE = ("train_images",)


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def coerce_field(name: str, raw):
    """Convert a command-line, job-file or project-file value to the type of
    SpaceChargeConfig field ``name``: bool, int, float, dict (a JSON object)
    or str; "auto"/"none"/"null" give None for ``train_images`` only.  A
    value of the field's type, a dict or None passes unchanged; any other
    non-text value is converted through its text, as before named methods
    (a JSON 100.0 for an int field gives 100)."""
    import dataclasses
    import json

    from linac_gen.core.config import SpaceChargeConfig
    ftypes = {f.name: f.type for f in dataclasses.fields(SpaceChargeConfig)}
    if name not in ftypes:
        raise ValueError(f"unknown space-charge parameter '{name}'")
    tp = ftypes[name]
    t = tp.__name__ if isinstance(tp, type) else str(tp)
    if raw is None:
        return None
    if isinstance(raw, dict):
        if not t.startswith("dict"):
            raise ValueError(f"{name}: expected a single value, got {raw!r}")
        return raw
    if not isinstance(raw, str):
        if ((t.startswith("bool") and isinstance(raw, bool))
                or (t.startswith("int") and isinstance(raw, int)
                    and not isinstance(raw, bool))
                or (t.startswith("float") and isinstance(raw, float))):
            return raw
        raw = str(raw)
    text = raw.strip()
    low = text.lower()
    if name in _NONE_IS_A_VALUE and low in ("auto", "none", "null"):
        return None
    if t.startswith("bool"):
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name}: expected true/false, got {raw!r}")
    if t.startswith("int"):
        return int(float(text))
    if t.startswith("float"):
        return float(text)
    if t.startswith("dict"):
        try:
            val = json.loads(text)
        except ValueError:
            val = None
        if not isinstance(val, dict):
            raise ValueError(f"{name}: expected a JSON object, got {raw!r}")
        return val
    return text


def legacy_kwargs(conv: Mapping, cli: Mapping) -> dict:
    """SpaceChargeConfig keyword arguments exactly as ``make_sc_config``
    built them before named methods (CLI flag > project key > classic)."""
    c = CLASSIC
    nx = int(_first(cli.get("nx"), conv.get("grid_nx"), c["nx"]))
    return dict(
        nx=nx, ny=nx, nz=nx,
        grid_extent=float(_first(cli.get("grid_extent"),
                                 conv.get("grid_extent_sigma"),
                                 c["grid_extent"])),
        kernel=str(_first(cli.get("kernel"), conv.get("kernel"), c["kernel"])),
        green_kind=str(conv.get("green_kind", c["green_kind"])),
        grid_mode=str(conv.get("grid_mode", c["grid_mode"])),
        dc_kernel=str(conv.get("dc_kernel", "uniform")),
        use_gpu=str(_first(cli.get("backend"), conv.get("backend"), "auto")),
        csr_enabled=bool(conv.get("csr_enabled", False)),
    )


def _field_names() -> list[str]:
    import dataclasses

    from linac_gen.core.config import SpaceChargeConfig
    return [f.name for f in dataclasses.fields(SpaceChargeConfig)]


def project_kwargs(conv: Mapping) -> dict:
    """Every space-charge setting a project with ``sc_method`` stores: any
    key named like a config field, then the keys HELIX itself writes
    (``grid_nx`` for all three grid sizes, ``grid_extent_sigma``,
    ``backend``), which win over a field-named duplicate.  Text values are
    converted to the field's type; keys holding ``null`` are skipped."""
    kw: dict = {}
    for name in _field_names():
        if name != "method" and conv.get(name) is not None:
            kw[name] = coerce_field(name, conv[name])
    if conv.get("grid_nx") is not None:
        n = int(float(conv["grid_nx"]))
        kw.update(nx=n, ny=n, nz=n)
    if conv.get("grid_extent_sigma") is not None:
        kw["grid_extent"] = float(conv["grid_extent_sigma"])
    if conv.get("backend") is not None:
        kw["use_gpu"] = str(conv["backend"])
    return kw


def _general_kwargs(conv: Mapping) -> dict:
    """The project's settings outside the method-controlled fields (engine
    device, DC kernel, CSR, bunch-train images, ...); they carry over to any
    method.  A project without ``sc_method`` gives them as it always ran."""
    if conv.get("sc_method") is not None:
        return {k: v for k, v in project_kwargs(conv).items()
                if k not in METHOD_FIELDS}
    legacy = legacy_kwargs(conv, {})
    return {k: legacy[k] for k in ("use_gpu", "dc_kernel", "csr_enabled")}


def _own_settings(conv: Mapping):
    """``(settings, method, clashes)`` the project itself stores, or
    ``(None, None, [])`` when it stores no method setting.

    With ``sc_method`` the named method's pinned settings win over stored
    keys that contradict them; ``clashes`` lists those as (key, stored,
    method's) for the notes.  Without it, a project with legacy method keys
    gives its legacy settings and the method they match.
    """
    proj_name = conv.get("sc_method")
    if proj_name is not None:
        own = _canonical(proj_name)
        kw = project_kwargs(conv)
        clashes = []
        if own not in (None, CUSTOM):
            for key, val in get(own).pinned.items():
                if key in kw and kw[key] != val:
                    clashes.append((key, kw.pop(key), val))
        return kw, own, clashes
    if any(k in conv for k in METHOD_PROJECT_KEYS):
        # as such a project always ran: the classic value of every method
        # setting it does not store (cards applied, the exact build, ...)
        kw = {**CLASSIC, **legacy_kwargs(conv, {})}
        return kw, infer(kw), []
    return None, None, []


def own_method(conv: Mapping | None):
    """The method a project's settings belong to: its ``sc_method``
    (``default`` read as :data:`DEFAULT_METHOD`), else the method its
    space-charge keys match, else ``None`` -- a project that stores no
    method setting follows the default method.  A command line naming this
    method keeps the project's settings; any other name starts from that
    method's own defaults."""
    return _own_settings(dict(conv or {}))[1]


def _apply_cli_flags(kw: dict, cli: Mapping) -> dict:
    if cli.get("nx") is not None:
        n = int(cli["nx"])
        kw.update(nx=n, ny=n, nz=n)
    if cli.get("grid_extent") is not None:
        kw["grid_extent"] = float(cli["grid_extent"])
    if cli.get("kernel") is not None:
        kw["kernel"] = str(cli["kernel"])
    if cli.get("backend") is not None:
        kw["use_gpu"] = str(cli["backend"])
    return kw


def _canonical(name):
    """``"default"`` -> :data:`DEFAULT_METHOD` (``None``: the classic values)."""
    return DEFAULT_METHOD if name == DEFAULT_ALIAS else name


def check_name(name) -> None:
    """Raise ``ValueError`` (listing the valid names) unless ``name`` is
    ``None``, a registry name, ``custom`` or ``default``."""
    if name is not None and name not in (CUSTOM, DEFAULT_ALIAS):
        get(name)


def cli_method(cli: Mapping):
    """The method a command line names (``--sc-method`` or ``--sc
    method=``), checked; naming two different ones is an error."""
    flag = cli.get("sc_method")
    over = (cli.get("sc") or {}).get("method")
    check_name(flag)
    check_name(over)
    if flag is not None and over is not None and flag != over:
        raise ValueError(f"--sc-method {flag!r} conflicts with "
                         f"--sc method={over!r}")
    return _first(flag, over)


def resolve_sc(conv: Mapping | None = None, cli: Mapping | None = None, *,
               notes: list | None = None):
    """The SpaceChargeConfig a project ``convergence`` block plus command-line
    options describe (:func:`resolve_sc_kwargs`, then constructed, which
    fills the method's defaults and checks every value)."""
    from linac_gen.core.config import SpaceChargeConfig
    return SpaceChargeConfig(**resolve_sc_kwargs(conv, cli, notes=notes))


def resolve_sc_kwargs(conv: Mapping | None = None,
                      cli: Mapping | None = None, *,
                      notes: list | None = None) -> dict:
    """Keyword arguments of the SpaceChargeConfig a project ``convergence``
    block plus command-line options describe (not yet checked).

    Method choice: ``cli["sc_method"]`` or ``cli["sc"]["method"]``, then the
    project's ``sc_method``; an unknown name is an error.  With neither, a
    project that has any space-charge key resolves exactly as before named
    methods existed (classic values + its keys + CLI flags), and a project
    with none gets :data:`DEFAULT_METHOD`.  A project's ``sc_method`` decides
    the pinned settings; its other stored keys adjust the params.  When the
    command line names the project's own method, or ``custom``, the
    project's settings are kept; a different method starts from its own
    defaults, keeping the project's settings outside the method fields (one
    note is appended to ``notes``).  Then CLI flags
    (``nx``/``grid_extent``/``kernel``/``backend``) and last the generic
    ``cli["sc"]`` NAME=VALUE overrides (a null value overrides nothing,
    except for ``train_images``, where None means automatic).
    """
    conv = dict(conv or {})
    cli = dict(cli or {})
    cli_name = cli_method(cli)
    sc_over = dict(cli.get("sc") or {})
    sc_over.pop("method", None)
    proj_name = conv.get("sc_method")
    check_name(proj_name)
    has_legacy = any(k in conv for k in LEGACY_PROJECT_KEYS)
    if cli_name is None and proj_name is None:
        if has_legacy or DEFAULT_METHOD is None:
            kw = legacy_kwargs(conv, cli)
        else:
            kw = _apply_cli_flags({"method": DEFAULT_METHOD,
                                   **_general_kwargs(conv)}, cli)
    else:
        own_kw, own, clashes = _own_settings(conv)
        shown = cli_name if cli_name is not None else proj_name
        target = _canonical(shown)
        if own_kw is not None and target in (own, CUSTOM):
            if target == own:
                kw = dict(own_kw)
                if own is not None:
                    kw["method"] = own
            else:
                # every setting individually: start from the project's own
                kw = {**base_fields(own), **own_kw, "method": CUSTOM}
            # a stored key the method overrules, unless the command line
            # sets that field anyway
            later = set(sc_over)
            if cli.get("kernel") is not None:
                later.add("kernel")
            if notes is not None:
                notes.extend(
                    f"the project's {key}={stored!r} contradicts its "
                    f"space-charge method {proj_name!r} ({key}={val!r}); "
                    "the method's setting is used"
                    for key, stored, val in clashes if key not in later)
        else:
            kw = _general_kwargs(conv)
            if target is not None:
                kw["method"] = target
            if own_kw is not None and notes is not None:
                what = ("the classic settings" if target is None
                        else "the method's own defaults")
                notes.append(
                    f"space-charge method {shown!r} replaces the project's "
                    f"settings ({own or 'classic'}); {what} are used")
        kw = _apply_cli_flags(kw, cli)
    # a setting outside the legacy keys that a project without sc_method
    # stores by its field name
    if "gpu_particles" not in kw and conv.get("gpu_particles") is not None:
        kw["gpu_particles"] = coerce_field("gpu_particles",
                                           conv["gpu_particles"])
    fields = set(_field_names())
    for key, raw in sc_over.items():
        if key not in fields:
            raise ValueError(f"unknown space-charge parameter '{key}'")
        val = coerce_field(key, raw)
        if val is None and key not in _NONE_IS_A_VALUE:
            continue        # null: not set -- the project's and method's stand
        kw[key] = val
    return kw


def filled_kwargs(kw: Mapping) -> dict:
    """``kw`` with every method-controlled field filled the way
    SpaceChargeConfig fills it, without checking any value (a scan point
    carries these; its worker builds and checks the config only when the
    point runs with space charge, as before named methods)."""
    kw = dict(kw)
    name = _canonical(kw.get("method"))
    given = {k: kw[k] for k in METHOD_FIELDS if kw.get(k) is not None}
    if name is None and not given and DEFAULT_METHOD is not None:
        name = DEFAULT_METHOD
    resolved, label = _resolve_fields(name, given)
    return {**kw, **resolved, "method": label}
