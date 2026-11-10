"""Base classes for HELIX surrogate elements.

M1 scope (per ``docs/plans/surrogates.md``):
* ``SurrogateFieldMap(FieldMapElement, nn.Module)`` — a drop-in for
  any ``FieldMapElement`` instance in **envelope-mode** tracking.
* ``fitted_matrix(ref)`` is implemented by an MLP that predicts the
  flattened 6x6 transfer matrix from (ref kinematics, element params).
* Out-of-scope inputs raise :class:`OutOfScopeError`; callers fall
  back to the wrapped element.

Where a registered surrogate actually engages:

* **Envelope, current = 0, no per-sub-step recording, no interior
  markers** — the pure-linear path requests the FULL-element matrix
  once per traversal through ``envelope._full_matrix_at``; that is the
  quantity the NN was trained on, so the surrogate serves it (speedup).
* **Envelope, current > 0 (SC bundles) or per-sub-step walks** — the
  solver requests PARTIAL slices; ``fitted_matrix_slice`` delegates
  every partial slice to the wrapped RK4 (see its docstring), so those
  runs make zero NN queries and see no speedup — by design, stated in
  the manual and the compare report.
* **Multi-particle** — hybrid/fast paths only under the explicit
  double opt-in (``registry.set_mp_enabled`` and the fast-path flag);
  ``track_rk4`` otherwise falls back to the wrapped element.
"""
from __future__ import annotations

import contextlib
import copy as _copy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from linac_gen.elements.base import FieldMapElement
from linac_gen.parallel.threads import sync_torch as _sync_torch

# a run that loads torch here (an envelope or tracking run meeting a field
# map) keeps torch's CPU work on one thread
_sync_torch()


F64 = torch.float64


# ---------------------------------------------------------------------------
class OutOfScopeError(RuntimeError):
    """Raised when a surrogate is queried outside its trained scope."""


# ---------------------------------------------------------------------------
class MlpHead(nn.Module):
    """Configurable MLP: input -> hidden(s) -> output.

    FP64 throughout, smooth activation (default SiLU) so the Jacobian
    stays clean for downstream gradient-matcher use (M7).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: Sequence[int] = (128, 128, 128),
                 activation: str = "silu") -> None:
        super().__init__()
        act_cls = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h), dtype=F64))
            layers.append(act_cls())
            prev = int(h)
        layers.append(nn.Linear(prev, int(output_dim), dtype=F64))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
@dataclass
class Scope:
    """Training-set scope bounds for OOD admissibility.

    The check is elementwise on the input vector: each entry must
    lie in ``[lo, hi]`` (with a small ``eps`` tolerance) to be in
    scope.
    """
    input_names: list[str]
    input_lo: np.ndarray   # shape (input_dim,)
    input_hi: np.ndarray   # shape (input_dim,)

    def is_in_scope(self, x: np.ndarray, eps: float = 1e-9) -> bool:
        x = np.asarray(x, dtype=np.float64)
        return bool(np.all(x >= self.input_lo - eps)
                    and np.all(x <= self.input_hi + eps))

    def to_dict(self) -> dict:
        return {
            "input_names": list(self.input_names),
            "input_lo": [float(v) for v in self.input_lo],
            "input_hi": [float(v) for v in self.input_hi],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scope":
        return cls(
            input_names=list(d["input_names"]),
            input_lo=np.asarray(d["input_lo"], dtype=np.float64),
            input_hi=np.asarray(d["input_hi"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
@dataclass
class SurrogateMetadata:
    """Bookkeeping persisted alongside a trained weights file.

    Required for reproducibility (seed, commit SHA, lattice hash),
    runtime admissibility (scope), and inference (normalisation).
    """
    element_key: str
    element_class: str
    architecture: dict
    scope: Scope
    input_norm: dict   # {"mean": [...], "std": [...]}
    output_norm: dict  # {"mean": [...], "std": [...]}
    training_seed: int
    n_samples: int
    epochs: int
    val_mape: float
    helix_commit_sha: str
    lattice_hash: str
    created_iso: str
    # Training kinematics (added 2026-07-18 with the H- mass fix): the
    # species the transfer maps were sampled with.  Defaults keep legacy
    # metadata loadable; load_surrogate warns when they are absent or when
    # the recorded mass no longer matches the current species table.
    species_name: str = ""
    species_mass_mev: float = 0.0
    # SHA-256 of weights.pt, computed by load_surrogate() at READ time
    # (never trusted from metadata.json): carried on the metadata so
    # every load route — CLI, GUI, and the plain Python API — feeds
    # the HDF5 provenance manifest uniformly.  Empty for a surrogate
    # never loaded from disk (in-memory training).
    weights_sha256: str = ""
    # What the targets were taken at: the clock, after ``_at_element_frequency``,
    # and the reference phase at the entrance.  0 = not recorded.
    train_frequency_mhz: float = 0.0
    train_phi_s_deg: float = 0.0
    # Set by load_surrogate, never saved, as weights_sha256 is: a file that records
    # no clock, and the longitudinal determinant if it says the targets hold a jump.
    legacy_weights: bool = False
    stale_factor: float | None = None
    # Set by load_surrogate, never saved: the species a file that records none was
    # trained for.  The Surrogates tab and the CLI trained H⁻ only until files
    # recorded their species, so such a file counts as H⁻.
    species_assumed: str = ""
    # Set by load_surrogate, never saved: whether the file records the phase it was
    # trained at (one from before counts as trained at 0), and the outcome of the one
    # check of that against the native element, for a map that follows the running
    # phase: None not run, "" it fits, else why it does not.
    phase_recorded: bool = True
    phase_check: str | None = None
    # Set by load_surrogate, never saved: the weights of a file are checked against their
    # native element once per load, at their first use (``SurrogateFieldMap.
    # _accuracy_refusal``); weights built in memory are not.  ``accuracy_check``: None not
    # run, "" within the limits, else why not; ``accuracy_block`` and ``accuracy_det``: the
    # largest block and determinant errors it found; ``accuracy_seconds``: what it took.
    accuracy_due: bool = False
    accuracy_check: str | None = None
    accuracy_block: float | None = None
    accuracy_det: float | None = None
    accuracy_seconds: float = 0.0
    # The element's settings its weights hold fixed (``settings_of``), recorded in training
    # (saved); for a file from before, or metadata built in memory without them, taken from
    # the element the first surrogate built on it wraps (``settings_snapshot``, never saved).
    train_settings: dict | None = None
    settings_snapshot: bool = False
    # How the network was fitted (``training.train_surrogate``'s report: optimiser, learning
    # rate and its schedule, batch, split, output floor, kept epoch and its validation error,
    # torch and numpy versions) and what the element was (``structure_record``: its class and
    # ``STRUCTURE``, the phase reference ``p_flag`` among it), recorded in training (saved);
    # None for a file from before.  Records: nothing checks them.
    train_recipe: dict | None = None
    train_structure: dict | None = None

    @property
    def trained_species(self) -> str:
        """The species the maps were sampled with: the recorded one, else the one
        assumed at load for a file that records none, else "" (metadata built in memory
        from a reference with no species)."""
        return self.species_name or self.species_assumed

    def to_json(self) -> dict:
        return {
            "element_key": self.element_key,
            "element_class": self.element_class,
            "architecture": self.architecture,
            "scope": self.scope.to_dict(),
            "input_norm": self.input_norm,
            "output_norm": self.output_norm,
            "training_seed": self.training_seed,
            "n_samples": self.n_samples,
            "epochs": self.epochs,
            "val_mape": self.val_mape,
            "helix_commit_sha": self.helix_commit_sha,
            "lattice_hash": self.lattice_hash,
            "created_iso": self.created_iso,
            "species_name": self.species_name,
            "species_mass_mev": self.species_mass_mev,
            "train_frequency_mhz": self.train_frequency_mhz,
            "train_phi_s_deg": self.train_phi_s_deg,
            "train_settings": self.train_settings,
            "train_recipe": self.train_recipe,
            "train_structure": self.train_structure,
            # informational only — load_surrogate() recomputes from
            # the actual file bytes and overwrites this.
            "weights_sha256": self.weights_sha256,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SurrogateMetadata":
        return cls(
            element_key=str(d["element_key"]),
            element_class=str(d["element_class"]),
            architecture=dict(d["architecture"]),
            scope=Scope.from_dict(d["scope"]),
            input_norm=dict(d["input_norm"]),
            output_norm=dict(d["output_norm"]),
            training_seed=int(d["training_seed"]),
            n_samples=int(d["n_samples"]),
            epochs=int(d["epochs"]),
            val_mape=float(d["val_mape"]),
            helix_commit_sha=str(d["helix_commit_sha"]),
            lattice_hash=str(d["lattice_hash"]),
            species_name=str(d.get("species_name", "")),
            species_mass_mev=float(d.get("species_mass_mev", 0.0)),
            train_frequency_mhz=float(d.get("train_frequency_mhz", 0.0)),
            train_phi_s_deg=float(d.get("train_phi_s_deg", 0.0)),
            train_settings=(dict(d["train_settings"]) if d.get("train_settings") is not None
                            else None),
            train_recipe=(dict(d["train_recipe"]) if d.get("train_recipe") is not None
                          else None),
            train_structure=(dict(d["train_structure"]) if d.get("train_structure") is not None
                             else None),
            created_iso=str(d["created_iso"]),
        )


#: a shift of the running phase that weights with ``phase`` not swept cannot follow
PHASE_TOLERANCE_DEG = 0.01


def _wrap180(deg: float) -> float:
    return ((float(deg) + 180.0) % 360.0) - 180.0


def species_name_of(mass_mev: float, charge: int) -> str | None:
    """The name of the species with this mass and charge, for a caller with only those."""
    from linac_gen.core import particle as pt
    for p in (pt.PROTON, pt.H_MINUS, pt.DEUTERON):
        if p.charge == charge and abs(p.mass - float(mass_mev)) < 1e-6:
            return p.name
    return None


#: what the registry recognises an element by (``registry.get_for_element``): what it is,
#: not how it is set.  The lookup decides once per run, so the weights also refuse a query
#: once the element's structure is no longer what it was when they were built on it
#: (``SurrogateFieldMap._structure_reason``).
STRUCTURE = ("length", "field_file", "geom", "aperture", "n_steps", "p_flag", "ki", "ka")
#: misalignments the tracking seams apply around the element (``dz``, pitch and yaw are inert
#: in every mode), exact around a surrogate as around the native element
SEAM = ("dx", "dy", "dz", "tilt_deg", "pitch_deg", "yaw_deg", "hom_kick_x", "hom_kick_y")
#: the error knobs ``SurrogateFieldMap._folds`` folds into an input of the weights, or refuses
FOLDED = ("voltage_rel", "phase_offset", "frequency_offset")
#: what the parser records of how a card was written, not what the element does: the
#: ``SUPERPOSE_MAP 0`` idiom before a map (the same physics)
PROVENANCE = ("superpose_z0",)
_NOT_SETTINGS = (frozenset(STRUCTURE) | frozenset(SEAM) | frozenset(PROVENANCE)
                 | {"name", "label"})
#: settings kept on the class, which the element's own attributes do not show: the field-map
#: integrator and interpolation (Simulation Settings), the NCELLS synchronism factor
CLASS_SETTINGS = ("integrator_kind", "interp_kind", "synchronism")
#: inputs an element class exposes under another name than the attribute that holds them
_ALIASES = {"NCells": {"frequency": "frequency_mhz"}}
#: two values of a setting are the same within this part of the larger: a deck written and
#: read back keeps 10 significant digits (``tracewin_writer``), and the map moves far less
SETTING_TOLERANCE = 1e-9
_NOT_SCALAR = object()


def structure_of(element) -> tuple:
    """What the registry recognises ``element`` by: its class and its ``STRUCTURE``, floats
    to 9 digits."""
    out = [type(element).__name__]
    for a in STRUCTURE:
        v = getattr(element, a, None)
        if isinstance(v, float):
            v = round(v, 9)
        out.append(v)
    return tuple(out)


def structure_record(element) -> dict:
    """``structure_of`` as a weights file records it (``SurrogateMetadata.train_structure``):
    ``{"class": ..., name: value}`` for each ``STRUCTURE`` attribute the element has, the field
    file by its name alone (the parser holds an absolute path, which belongs to the machine)."""
    out: dict = {"class": type(element).__name__}
    for a in STRUCTURE:
        if not hasattr(element, a):
            continue
        v = getattr(element, a)
        if a == "field_file" and v is not None:
            v = Path(str(v)).name
        elif isinstance(v, float):
            v = float(round(v, 9))
        else:
            v = _scalar(v)
            if v is _NOT_SCALAR:
                v = repr(getattr(element, a))
        out[a] = v
    return out


def _scalar(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _NOT_SCALAR


def _folds_knobs(element) -> bool:
    """Whether ``_folds`` takes this element's error knobs (a field map, an ``NCELLS``)."""
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.elements.ncells import NCells
    return isinstance(element, (FieldMap, FieldMap3D, NCells))


def settings_of(element, inputs) -> dict:
    """``{name: value}`` of the settings of ``element`` that weights with the inputs ``inputs``
    hold fixed: its public scalar attributes and the numerics kept on its class
    (``CLASS_SETTINGS``), but those inputs (read live at each query, under the name of the
    attribute that holds them), its structure and name, the misalignments the seams apply,
    what the parser records of how its card was written (``PROVENANCE``) and the error knobs
    the folds take (on a field map or an ``NCELLS``).  A change of one of these the weights
    cannot follow, a pinned calibration (``sync_phase_pin``) among them."""
    folded = frozenset(FOLDED) if _folds_knobs(element) else frozenset()
    alias = _ALIASES.get(type(element).__name__, {})
    inputs = set(inputs) | {alias[n] for n in inputs if n in alias}
    out = {}
    for name, value in vars(element).items():
        if name.startswith("_") or name in _NOT_SETTINGS or name in inputs or name in folded:
            continue
        value = _scalar(value)
        if value is not _NOT_SCALAR:
            out[name] = value
    for name in CLASS_SETTINGS:
        if name in out or name in inputs or not hasattr(type(element), name):
            continue
        value = _scalar(getattr(element, name))
        if value is not _NOT_SCALAR:
            out[name] = value
    return out


def _same_setting(a, b) -> bool:
    if (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return a == b or abs(a - b) <= SETTING_TOLERANCE * max(abs(a), abs(b)) + 1e-12
    return a == b


def design_copy(element):
    """A copy of ``element`` as its weights hold it: its error knobs at zero where the folds
    take them (``FOLDED``, folded into the inputs at each query, ``SurrogateFieldMap._folds``);
    elsewhere they are settings the weights hold, and the copy keeps them.  What the training
    takes its targets on, and the accuracy check its native maps."""
    work = _copy.deepcopy(element)
    if _folds_knobs(work):
        for knob in FOLDED:
            if hasattr(work, knob):
                setattr(work, knob, 0.0)
    return work


def design_reference(ref):
    """A copy of ``ref`` without the RF shift and correction of SET_BEAM_PHASE_ERROR, which
    a query folds into the inputs, or refuses."""
    out = ref.copy()
    for shift in ("rf_shift", "rf_rephase"):
        if hasattr(out, shift):
            setattr(out, shift, 0.0)
    return out


def _inert_settings(element) -> frozenset:
    """Settings that cannot move the map of ``element``, read off the channels of its field
    file: a field map's ``phase`` and ``frequency`` when no channel is RF (its clock is
    converted, ``SurrogateFieldMap._magnet_clock_ratio``), its ``ke`` when none is electric,
    its ``kb`` when none is magnetic."""
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D
    if not isinstance(element, (FieldMap, FieldMap3D)):
        return frozenset()
    chans = list(getattr(getattr(element, "field_data", None), "channels", None) or ())
    if not chans:
        return frozenset()
    out = set()
    if not any(getattr(c, "is_rf", False) for c in chans):
        out |= {"phase", "frequency"}
    if not any(getattr(c, "is_electric", False) for c in chans):
        out.add("ke")
    if not any(getattr(c, "is_magnetic", False) for c in chans):
        out.add("kb")
    return frozenset(out)


def _acting_ks(element) -> list:
    """The amplitudes of a field map that act: ``ke`` with an electric channel, ``kb`` with a
    magnetic one, each not zero (with ``scale``)."""
    fd = getattr(element, "field_data", None)
    chans = list(fd.channels) if fd is not None else []
    scale = float(getattr(element, "scale", 1.0))
    return [k for k, used in (
        ("ke", any(c.is_electric for c in chans) and scale * float(getattr(element, "ke", 0.0))),
        ("kb", any(c.is_magnetic for c in chans) and scale * float(getattr(element, "kb", 0.0))))
        if used]


_TALLY = threading.local()


class Tally:
    """What every registered weights did in one thread inside ``tally()``: ``calls``, the
    answers of ``SurrogateFieldMap.fitted_matrix``, and ``refused``, the traversals they did
    not answer, by reason."""
    __slots__ = ("calls", "refused")

    def __init__(self) -> None:
        self.calls = 0
        self.refused: dict[str, int] = {}


@contextlib.contextmanager
def tally():
    """Inside it, count what the weights do in this thread (``Tally``): a study or a scan
    counts its own runs so, whatever another thread runs on the same weights meanwhile.  An
    outer tally takes an inner one's counts too."""
    outer = getattr(_TALLY, "current", None)
    t = Tally()
    _TALLY.current = t
    try:
        yield t
    finally:
        _TALLY.current = outer
        if outer is not None:
            outer.calls += t.calls
            for k, n in t.refused.items():
                outer.refused[k] = outer.refused.get(k, 0) + n


#: a question that counts nothing (``SurrogateFieldMap.drives_mp``), per thread
_UNCOUNTED = threading.local()


def _counted() -> bool:
    return not getattr(_UNCOUNTED, "on", False)


def _tally_call() -> None:
    t = getattr(_TALLY, "current", None)
    if t is not None:
        t.calls += 1


def _tally_refusal(reason: str) -> None:
    t = getattr(_TALLY, "current", None)
    if t is not None:
        t.refused[reason] = t.refused.get(reason, 0) + 1


_COMPARE = threading.local()
#: one accuracy check at a time: a study's thread and a toolbar run may meet the same
#: weights first together
_CHECK_LOCK = threading.Lock()


@contextlib.contextmanager
def accuracy_refusals_suspended():
    """Inside it, in this thread, weights that failed their accuracy check still answer:
    for the compare view, which shows how far such weights are from their native element.
    The check itself still runs and keeps its verdict."""
    before = getattr(_COMPARE, "on", False)
    _COMPARE.on = True
    try:
        yield
    finally:
        _COMPARE.on = before


def _refusals_suspended() -> bool:
    return bool(getattr(_COMPARE, "on", False))


def _say(message: str) -> None:
    """``warnings.warn(message)`` shown at every call, as ``training._file_notice`` is: the
    accuracy check runs once per load, and a file loaded again is checked, and says so,
    again (Python would show a warning once per text and line)."""
    import sys
    import warnings
    caller = sys._getframe(1)
    registry = caller.f_globals.setdefault("__warningregistry__", {})
    registry.pop((message, UserWarning, caller.f_lineno), None)
    warnings.warn_explicit(message, UserWarning, caller.f_code.co_filename, caller.f_lineno,
                           module=caller.f_globals.get("__name__", "<unknown>"),
                           registry=registry)


def drift_scale_z(beta: float, gamma: float, mass_mev: float, wavelength_mm: float) -> float:
    """The Δφ–ΔW term of a 1 m drift (deg/MeV), in absolute value, at the RF wavelength
    (mm, as ``ReferenceParticle.wavelength``): what ``block_error`` divides the longitudinal
    plane by, so that a drift has the same off-diagonal term in every plane (1 mm/mrad, the
    transverse one of a 1 m drift)."""
    return 360.0 * 1000.0 / (beta ** 3 * gamma ** 3 * mass_mev * wavelength_mm)


def block_error(native: np.ndarray, other: np.ndarray, scale_z: float) -> float:
    """The largest 2x2-block difference between two 6x6 maps, each block's over the largest
    norm of the native blocks in its row and its column: a diagonal block's own norm in an
    uncoupled map, and in a solenoid turned near 90°, whose diagonal blocks vanish, its
    coupling blocks'.  Taken in units where a drift of 1 m has the same off-diagonal term in
    every plane: x, x', y, y' in mm and mrad, and Δφ/√s, ΔW·√s with s = ``scale_z`` (a 1 m
    drift's Δφ–ΔW term, ``drift_scale_z``), which makes that term 1 there too.  In (deg,
    MeV) a cavity's Δφ–ΔW term outweighs its longitudinal focusing by 10³ to 10⁵, and the
    norm of the block does not see the focusing.  NaN when either map holds a non-finite
    entry."""
    a0, b0 = np.asarray(native, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if not (np.all(np.isfinite(a0)) and np.all(np.isfinite(b0))):
        return float("nan")
    t = np.array([1.0, 1.0, 1.0, 1.0, 1.0 / np.sqrt(scale_z), np.sqrt(scale_z)])
    a = a0 * t[:, None] / t[None, :]
    b = b0 * t[:, None] / t[None, :]
    blocks = (0, 2, 4)
    norms = np.array([[np.linalg.norm(a[i:i + 2, j:j + 2]) for j in blocks] for i in blocks])
    worst = 0.0
    for p, i in enumerate(blocks):
        for q, j in enumerate(blocks):
            ref = max(norms[p, :].max(), norms[:, q].max(), 1e-300)
            worst = max(worst, float(np.linalg.norm(b[i:i + 2, j:j + 2] - a[i:i + 2, j:j + 2]) / ref))
    return worst


def determinant_error(native: np.ndarray, other: np.ndarray) -> float:
    """The larger relative difference of the transverse 4x4 and the longitudinal 2x2
    determinants, ``other`` against ``native``.  The emittances follow them (ε_out = |det| ε_in
    for the 4-D transverse and the longitudinal emittance); the 2x2 transverse blocks alone
    are no invariant where the planes couple (a solenoid's are cos²θ of it).  Against the
    native map, not against p_in/p_out: a native solenoid map misses that by itself (its
    magnetic kick is not area-preserving, about 1.5e-3 at the 0.2 mm step), and the check is
    of the weights, not of the integrator.  NaN when either map holds a non-finite entry."""
    a = np.asarray(native, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")             # LAPACK gives 0 for some NaN placements
    worst = 0.0
    for sl in (slice(0, 4), slice(4, 6)):
        da, db = np.linalg.det(a[sl, sl]), np.linalg.det(b[sl, sl])
        worst = max(worst, float(abs(db - da) / max(abs(da), 1e-300)))
    return worst


# ---------------------------------------------------------------------------
def fast_slice(M_full, log_M_full, length, ds) -> np.ndarray:
    """The multiparticle fast path's map over ``ds`` mm of an element ``length`` mm long whose
    whole map is ``M_full`` (``log_M_full`` its logarithm): ``M_full`` itself over the whole
    element, else ``real(expm(log_M_full · ds / length))``.  The backtracker inverts the same
    maps."""
    from scipy.linalg import expm
    if abs(float(ds) - float(length)) < 1e-9:
        return M_full
    ratio = float(ds) / max(float(length), 1e-12)
    return np.real(expm(log_M_full * ratio))


def _fast_log(M_full):
    """The logarithm the fast path cuts its slices from — its real part when ``logm`` returns
    one with a non-trivial imaginary part (accepting a small per-slice unitarity error) — or
    None where it cannot be taken."""
    from scipy.linalg import logm
    try:
        log_m = logm(M_full)
        if np.max(np.abs(np.imag(log_m))) > 1e-6:
            log_m = np.real(log_m)
        return log_m
    except Exception:                                       # noqa: BLE001 - the element runs native
        return None


def push_settings_of(element) -> dict:
    """What a field map's push depends on of ``element`` besides the reference at its entrance
    and its step schedule, as the record of a traversal holds it (``SurrogateFieldMap.
    _push_record``, ``native_push_record``) for the backtracker to check against the element
    as it stands: its settings (``settings_of``, the weights' inputs among them), its
    structure (``STRUCTURE``: what the registry recognises it by, so whether weights push it
    at all, as they refuse an element whose structure is no longer the one they were built
    on; its phase reference and its step count, which also sets SET_SYNC_PHASE's
    calibration, move the native element's push too) and the error knobs the folds take
    (``FOLDED``).  Not its misalignments, which the tracking seams apply around the element
    and the backtracker undoes as they stand."""
    out = settings_of(element, ())
    for name in STRUCTURE + FOLDED:
        if hasattr(element, name):
            value = getattr(element, name)
            scalar = _scalar(value)
            # a value that is not a scalar (a field file given as a pathlib.Path) by its repr,
            # which for such a value is its content and is the same in a copy of the record
            out[name] = repr(value) if scalar is _NOT_SCALAR else scalar
    return out


def _entrance_of(ref) -> dict:
    """The reference at an element's entrance as the record of a traversal holds it, under the
    names the backtracker's replay table (``BoundaryState``) gives them: its energy, phase,
    frequency and clock, and SET_BEAM_PHASE_ERROR's RF shift and its correction, all that the
    push reads of the reference."""
    return {"w_kin": float(ref.w_kin), "phi_s": float(ref.phi_s),
            "frequency": float(ref.frequency),
            "clock": float(getattr(ref, "_clock", ref.frequency)),
            "rf_shift": float(getattr(ref, "rf_shift", 0.0)),
            "rf_rephase": float(getattr(ref, "rf_rephase", 0.0))}


def native_push_record(element, ref, n_calls: int, ds: float) -> dict:
    """The record of a traversal of ``element`` that no weights stand for, its native element
    making all ``n_calls`` calls of ``ds`` from the entrance reference ``ref``
    (``Tracker._track_field_map``): no call of the weights, and what the backtracker checks
    every record by, as ``SurrogateFieldMap._record_call`` takes it of a traversal through
    them (its step schedule, the reference at its entrance and ``push_settings_of``)."""
    return {"calls": [], "M_full": None, "log_M_full": None, "length": float(element.length),
            "n_calls": int(n_calls), "ds": float(ds), "entry": _entrance_of(ref),
            "settings": push_settings_of(element)}


# ---------------------------------------------------------------------------
class SurrogateFieldMap(FieldMapElement, nn.Module):
    """ML surrogate for any FieldMapElement, envelope-mode (M1)."""

    #: what the tracking modules check, without importing this module (and torch), to
    #: walk the reference through the native element behind a surrogate placed in the
    #: lattice (``tracker.field_map_behind``)
    stands_for_wrapped = True

    #: one traversal's push through the weights, for the backtracker: the index of the next
    #: ``track_rk4`` call, and the record of the traversal (its step schedule, the reference
    #: at its entrance and the element's settings, ``_record_call``) and of the calls the
    #: weights made (``_record_push``; both reset with the run state)
    _call_index = 0
    _push_record = None

    def __init__(self,
                 wrapped: FieldMapElement,
                 mlp: MlpHead,
                 metadata: SurrogateMetadata) -> None:
        FieldMapElement.__init__(
            self,
            name=wrapped.name,
            length=wrapped.length,
            aperture=wrapped.aperture,
            n_steps=wrapped.n_steps,
        )
        nn.Module.__init__(self)
        self._wrapped = wrapped
        self.mlp = mlp
        self.metadata = metadata
        # Normalisation tensors (FP64) — kept on CPU.
        self._in_mean = torch.as_tensor(metadata.input_norm["mean"], dtype=F64)
        self._in_std = torch.as_tensor(metadata.input_norm["std"], dtype=F64)
        self._out_mean = torch.as_tensor(metadata.output_norm["mean"], dtype=F64)
        self._out_std = torch.as_tensor(metadata.output_norm["std"], dtype=F64)
        self._param_names: list[str] = list(
            metadata.architecture.get("param_names", []))
        # Ensure inference mode by default.
        self.mlp.eval()
        # M3 envelope hook reads this — set False to bypass surrogate.
        self._enabled: bool = True
        # Cumulative count of SUCCESSFUL NN predictions actually
        # returned by :meth:`fitted_matrix` over the lifetime of this
        # object — across modes (the MP fast path's ``_init_fast_path``
        # also queries ``fitted_matrix``) and NOT reset by
        # ``reset_run_state``.  Per-run counts are deltas (see
        # ``compare_envelope``).  ``fitted_matrix_torch`` (matcher arm)
        # is deliberately not counted.  Kept in a list that the views of
        # this surrogate (``bound_to``) share, so the count is the load's.
        self._calls: list[int] = [0]
        # The traversals the weights did not answer, by reason (``refusals``): shared by the
        # views of the load, as the call count is.
        self._refused: dict[str, int] = {}
        # What the element is (``structure_of``) as the weights were built on it: a view
        # keeps its root's, and a query refuses once the element is no longer that.
        self._structure = structure_of(wrapped)
        if metadata.train_settings is None:
            # a file that records no settings: they are those of the element it is built on
            metadata.train_settings = settings_of(wrapped, set(self._param_names))
            metadata.settings_snapshot = True
        # M7 hybrid MP knob: how many RK4 substeps the residual pass
        # uses inside :meth:`track_rk4`.  Lower = faster + less
        # accurate; higher = slower + closer to baseline RK4.
        # Default 15 is the "accuracy first" choice from the plan.
        self.residual_n_steps: int = 15
        # M7 fast-path cache: ``scipy.linalg.logm + expm`` is ~10 ms on
        # a 6x6 matrix (eigendecomposition).  Across ~160 substeps per
        # cavity × 20 cavities that dominates the run wall-clock.
        # Slice matrices are constant per ``(ds, ref.w_kin)`` within
        # one element's inner loop, so cache the most recent.
        self._slice_cache_key: tuple | None = None
        self._slice_cache_M: np.ndarray | None = None
        # M7-followup fast-path per-element state.  All cleared by
        # `reset_run_state` so each element traversal re-inits cleanly.
        self._fast_init_done: bool = False
        self._fast_init_failed: bool = False
        self._M_full: np.ndarray | None = None
        self._log_M_full: np.ndarray | None = None
        self._electric_channels: list | None = None
        # The reference phase at this traversal's entrance: the scope check below
        # runs at every substep, on a reference that keeps moving.
        self._phi_s_entry: float | None = None

    # The clock is the element's: a surrogate placed in the lattice (the compare flows)
    # makes the jump to its cavity's clock where the native element would, in every
    # mode that reads these off the element in hand (the envelope's seam above all).
    @property
    def frequency(self) -> float:
        return float(getattr(self._wrapped, "frequency", 0.0) or 0.0)

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._wrapped.frequency = value

    @property
    def effective_frequency(self) -> float:
        w = self._wrapped
        return float(getattr(w, "effective_frequency", 0.0) or getattr(w, "frequency", 0.0) or 0.0)

    def _has_electric_channel(self) -> bool:
        return bool(getattr(self._wrapped, "_has_electric_channel", lambda: False)())

    @property
    def nn_calls(self) -> int:
        return self._calls[0]

    @property
    def refusals(self) -> dict:
        """``{reason: count}`` of the traversals these weights did not answer (the load's)."""
        return dict(self._refused)

    def follows(self, attr: str) -> bool:
        """Whether these weights answer for the element after a change of ``attr`` (inside
        their training range): an input of theirs, a misalignment the seams apply, an error
        knob the folds take on this element, a setting its map cannot feel
        (``_inert_settings``).  A change of any other setting, or of its structure, has them
        refuse."""
        names = self._param_names
        alias = _ALIASES.get(type(self._wrapped).__name__, {})
        if attr in names or attr in SEAM or any(alias.get(n) == attr for n in names):
            return True
        if attr in _inert_settings(self._wrapped):
            return True
        return attr in FOLDED and _folds_knobs(self._wrapped) and self._folds_knob(attr)

    def _folds_knob(self, attr: str) -> bool:
        """Whether ``_folds`` takes a change of the error knob ``attr`` on this element."""
        from linac_gen.elements.ncells import NCells
        w, names = self._wrapped, self._param_names
        if isinstance(w, NCells):
            # its voltage_rel scales the gap voltages, which no input stands for, and
            # SET_SYNC_PHASE reads θs without the offset; it has no frequency_offset
            return attr == "phase_offset" and not w.sync_phase and "theta_s_deg" in names
        driven = self._rf_driven()
        if attr == "phase_offset":
            return not driven or "phase" in names
        if attr == "frequency_offset":
            return not driven or "frequency" in names
        if attr == "voltage_rel":
            ks = _acting_ks(w)
            return not ks or "scale" in names or all(k in names for k in ks)
        return False

    @nn_calls.setter
    def nn_calls(self, value: int) -> None:
        self._calls[0] = int(value)

    def __deepcopy__(self, memo) -> SurrogateFieldMap:
        """A copy of a lattice holding these weights (an error or failure study's) holds a
        view of them for its copy of the element: the network, the metadata (so the load's
        one-time checks and their verdict) and the call count stay those of the load."""
        return self.bound_to(_copy.deepcopy(self._wrapped, memo))

    def bound_to(self, element) -> SurrogateFieldMap:
        """A view of these weights for ``element``, the same element in another object
        (a reloaded deck, a study's copy of the lattice): it reads the parameters of
        ``element`` and has its own traversal state, and shares the MLP, the metadata
        (so the load's one-time checks) and the NN-call count with this surrogate."""
        view = SurrogateFieldMap(element, self.mlp, self.metadata)
        view._calls = self._calls
        view._refused = self._refused
        view._structure = self._structure
        view._enabled = self._enabled
        view.residual_n_steps = self.residual_n_steps
        if hasattr(self, "weights_dir"):
            view.weights_dir = self.weights_dir
        return view

    # ------------------------------------------------------------------
    # Envelope-mode contract
    # ------------------------------------------------------------------
    def fitted_matrix(self, ref) -> np.ndarray:
        """Predict the linearised 6x6 transfer matrix.

        Raises :class:`OutOfScopeError` if the input falls outside the
        training-set scope (caller falls back to the wrapped element).
        """
        if not self._enabled:
            return self._wrapped.fitted_matrix(ref)
        own = self._on_own_clock(ref)
        reason = self._refusal_reason(getattr(getattr(ref, "species", None), "name", None),
                                      getattr(own, "phi_s", 0.0), own)
        if reason is not None:
            self._refuse(*reason)
        why = self._clock_refusal(float(getattr(ref, "frequency", 0.0) or 0.0))
        if why is not None:
            self._refuse("clock", why)
        x = self._make_input(own)
        if not self.metadata.scope.is_in_scope(x):
            self._refuse("outside its range",
                         f"surrogate '{self.metadata.element_key}': input "
                         f"{x.tolist()} outside training scope")
        why = self._unrecorded_phase_refusal(own, x)
        if why:
            self._refuse("phase check", why)
        why = self._accuracy_refusal(own, x)
        if why:
            self._refuse("accuracy check", why)
        x_t = torch.as_tensor(x, dtype=F64)
        x_norm = (x_t - self._in_mean) / self._in_std
        with torch.no_grad():
            y_norm = self.mlp(x_norm)
        y = y_norm * self._out_std + self._out_mean
        M = y.detach().numpy().reshape(6, 6)
        # Output sanity gate: a corrupted/NaN weight file must not
        # inject garbage into the beamline — raise OutOfScopeError so
        # every caller takes its existing wrapped-element fallback.
        # Two unambiguous rejections only (a condition-number threshold
        # would be arbitrary and could refuse legitimate strongly-
        # focusing optics): non-finite entries, and an outright
        # singular matrix — no physical transfer map is singular
        # (Liouville: |det| ~ 1 up to adiabatic damping).
        if not np.all(np.isfinite(M)):
            self._refuse("non-finite map",
                         f"surrogate '{self.metadata.element_key}': NN produced "
                         "a non-finite transfer matrix — falling back to the "
                         "wrapped element (weights may be corrupt)")
        if abs(np.linalg.det(M)) < 1e-9:
            self._refuse("singular map",
                         f"surrogate '{self.metadata.element_key}': NN produced "
                         "a SINGULAR transfer matrix (|det|="
                         f"{abs(np.linalg.det(M)):.3g}) — falling back to the "
                         "wrapped element (weights may be corrupt)")
        if self._legacy_magnet():
            why = self._m45_refusal(float(M[4, 5]), ref.beta, ref.gamma,
                                    ref.species.mass, ref.wavelength)
            if why is not None:
                self._refuse("clock", why)
        if _counted():
            self.nn_calls += 1      # successful NN prediction actually returned
            _tally_call()
        # The weights hold the map alone.  Asked at another clock, give what
        # the native element gives there: the jump it makes on entry, then the
        # map (M D, so the Δφ column takes the factor).
        ratio = self._jump_ratio(ref)
        if ratio != 1.0:
            M[:, 4] *= ratio
        # A magnet knows no clock, so only Δφ's degrees change: D M D⁻¹.
        ratio = self._magnet_clock_ratio(float(getattr(ref, "frequency", 0.0) or 0.0))
        if ratio != 1.0:
            M[4, :] *= ratio
            M[:, 4] /= ratio
        return M

    def _refuse(self, reason: str, why: str):
        """Count a traversal these weights do not answer, by ``reason``, and refuse it."""
        if _counted():
            self._refused[reason] = self._refused.get(reason, 0) + 1
            _tally_refusal(reason)
        raise OutOfScopeError(why)

    def drives_mp(self, ref, *, fast_path: bool | None = None) -> bool:
        """Whether a multiparticle run pushes through these weights at this element, ``ref``
        at its entrance (the fast path of ``track_rk4``), rather than through the native
        element: for the backtracker, whose exact inverse is the native element's.  As the
        fast path decides it at the entrance (``_init_fast_path``): a map it can push (one
        with 3-D channel interpolators, which it samples; a 1-D map runs native), no
        frequency jump, and an answer in range.  Nothing is counted.  ``fast_path`` True
        answers for a run with the fast path on, whatever the registry says now (the
        backtracker, of a beam that carries no record of its run)."""
        from linac_gen.surrogates import registry as _reg
        fast = _reg.is_fast_path_enabled() if fast_path is None else bool(fast_path)
        if not (self._enabled and _reg.is_mp_enabled() and fast):
            return False
        if getattr(self._wrapped, "_interpolators", None) is None or self._jump_ratio(ref) != 1.0:
            return False
        if not self.metadata.scope.is_in_scope(
                self._make_input(ref, phi_s=float(getattr(ref, "phi_s", 0.0)))):
            return False
        before = getattr(_UNCOUNTED, "on", False)
        _UNCOUNTED.on = True
        try:
            self.fitted_matrix(ref.copy())
            return True
        except OutOfScopeError:
            return False
        finally:
            _UNCOUNTED.on = before

    def fast_path_map(self, ref):
        """For the backtracker, where the beam's record of its forward run does not cover the
        element or no longer matches it: the whole map and its logarithm the fast path would
        push this element with from the entrance ``ref``, as ``_init_fast_path`` takes them
        (asked without counting), or None where it would leave the element to the native one
        (``drives_mp``)."""
        if not self.drives_mp(ref):
            return None
        before = getattr(_UNCOUNTED, "on", False)
        _UNCOUNTED.on = True
        try:
            M_full = np.asarray(self.fitted_matrix(ref.copy()), dtype=np.float64)
        except OutOfScopeError:
            return None
        finally:
            _UNCOUNTED.on = before
        log_m = _fast_log(M_full)
        return None if log_m is None else (M_full, log_m)

    def in_fast_scope(self, ref, phi_s_entry: float) -> bool:
        """The fast path's check at each call: the reference as it stands and the phase at the
        element's entrance inside the weights' range."""
        return bool(self.metadata.scope.is_in_scope(self._make_input(ref, phi_s=phi_s_entry)))

    def _record_call(self, call: int, beam, ds: float) -> None:
        """This traversal's record (``_push_record``, which the tracker keeps on the beam) of
        its step schedule: the calls made so far and their step ``ds``; at its first call
        also the reference at the element's entrance (``beam.ref``) and the settings its push
        depends on (``push_settings_of``).  The backtracker takes the traversal from the
        record, the weights' calls and the native element's alike, only while the walk it
        makes has all three the same."""
        rec = self._push_record
        if rec is None:
            return
        rec["n_calls"] = call + 1
        rec["ds"] = float(ds)
        if call == 0:
            rec["entry"] = _entrance_of(beam.ref)
            rec["settings"] = push_settings_of(self._wrapped)

    def _record_push(self, call: int) -> None:
        """This traversal's record (``_push_record``, which the tracker keeps on the beam):
        the weights made call ``call``, cutting its slice from the map they took at the
        entrance.  The backtracker undoes each such call with the inverse of the same slice,
        whatever the registry says when it runs, while the record covers its walk
        (``_record_call``)."""
        rec = self._push_record
        if rec is None:
            return
        if rec["M_full"] is None:
            rec["M_full"] = np.array(self._M_full, dtype=np.float64)
            rec["log_M_full"] = np.array(self._log_M_full)
        rec["calls"].append(int(call))

    def _refusal(self, species_name, phi_s, ref=None) -> str | None:
        """Why these weights must not answer for this beam, or None.  ``phi_s`` is the
        reference phase at the entrance, None for a caller that does not know it;
        ``ref`` carries the RF shifts of SET_BEAM_PHASE_ERROR, if any (a
        ``ReferenceParticle`` or the torch arm's kinematics).
        ``fitted_matrix`` turns a refusal into the native element by the existing
        route; the torch arm, which bypasses it, into its own hard error."""
        reason = self._refusal_reason(species_name, phi_s, ref)
        return None if reason is None else reason[1]

    def _refusal_reason(self, species_name, phi_s, ref=None) -> tuple[str, str] | None:
        """``_refusal`` with its reason for the count: ``(reason, why)`` or None."""
        meta = self.metadata
        key = meta.element_key
        reason = self._structure_reason()
        if reason is not None:
            return reason
        if meta.stale_factor is not None:
            return "stale", (f"surrogate '{key}': the longitudinal determinant of its map is "
                             f"{meta.stale_factor:.3g}, not 1, as when a cavity is trained at "
                             "another clock than its own and its targets hold the ratio.  "
                             "Retrain it.")
        trained = meta.trained_species
        if trained and species_name and species_name != trained:
            if not meta.species_name:
                return "species", (
                    f"surrogate '{key}': its file records no species, so it counts as trained "
                    f"for {trained} (the only species files were trained for before they "
                    f"recorded one), asked for {species_name}")
            return "species", f"surrogate '{key}': trained for {trained}, asked for {species_name}"
        why = self._folds(ref)[1]
        if why is not None:
            return "error knob", why
        running = self._running_phase_input()
        if running is not None:
            if phi_s is None:
                return "running phase", (f"surrogate '{key}': its cavity's field follows the "
                                         "running phase, which this path does not know")
            if (running[0] not in self._param_names
                    and abs(_wrap180(phi_s - meta.train_phi_s_deg)) > PHASE_TOLERANCE_DEG):
                return "running phase", (
                    f"surrogate '{key}': its cavity's field follows the running phase, trained "
                    f"at {meta.train_phi_s_deg:.3f}° with its own phase not swept")
        return self._settings_reason()

    def _structure_reason(self) -> tuple[str, str] | None:
        """The element's structure (``structure_of``) is no longer what it was when the
        weights were built on it, as after an edit or a scan of its length: ``(reason, why)``,
        or None.  The lookup checks the structure once per run, this at every query."""
        now = structure_of(self._wrapped)
        if now == self._structure:
            return None
        names = ("class",) + STRUCTURE
        i = next(i for i, (a, b) in enumerate(zip(now, self._structure, strict=True)) if a != b)
        return f"its {names[i]} changed", (
            f"surrogate '{self.metadata.element_key}': its {names[i]} is {now[i]!r}, "
            f"{self._structure[i]!r} when its weights were built on it, and its weights hold "
            "the map of that element")

    def _settings_reason(self) -> tuple[str, str] | None:
        """A setting of the element that is not an input of the weights and is not what they
        hold fixed (``settings_of``, against ``train_settings``, within
        ``SETTING_TOLERANCE``), a setting its map cannot feel aside (``_inert_settings``):
        ``(reason, why)``, or None."""
        meta = self.metadata
        held = meta.train_settings or {}
        now = settings_of(self._wrapped, set(self._param_names))
        inert = _inert_settings(self._wrapped)
        for name in sorted((set(held) | set(now)) - inert):
            if not _same_setting(held.get(name), now.get(name)):
                when = ("when its weights were loaded" if meta.settings_snapshot
                        else "in training")
                return f"its {name} changed", (
                    f"surrogate '{meta.element_key}': its {name} is {now.get(name)!r}, "
                    f"{held.get(name)!r} {when}, and {name} is not an input of its weights")
        return None

    def _rf_driven(self) -> bool:
        """Whether the RF drives a field in the map: an RF channel, electric or magnetic,
        whose amplitude (``scale`` times ``ke`` or ``kb``, with its error) is not zero.
        Only such a field turns with the running phase: a static channel does not, and a
        switched-off channel adds nothing, so the native map is the same at every phase."""
        w = self._wrapped
        fd = getattr(w, "field_data", None)
        if fd is None:
            return False
        scale = float(getattr(w, "scale", 1.0))
        for ch in fd.channels:
            if getattr(ch, "is_rf", False):
                k = (getattr(w, "effective_ke", getattr(w, "ke", 0.0)) if ch.is_electric
                     else getattr(w, "effective_kb", getattr(w, "kb", 0.0)))
                if scale * float(k) != 0.0:
                    return True
        return False

    def _running_phase_input(self) -> tuple[str, float] | None:
        """The input the running phase moves, and with which sign, or None for a map
        that does not depend on it.  φ_s is not an input, and metadata built in memory
        says nothing about φ_s at training.

        * A field map whose ``p_flag`` is not 1 has a field that follows the running
          phase where the RF drives it, so its map depends on phase + φ_s only:
          ("phase", +1).
        * An ``NCELLS`` with P=1 fires each gap at φ_s − θs, so its map depends on
          φ_s − θs only: ("theta_s_deg", −1).  Its P=0 and P=2 count from its first
          gap, and SET_SYNC_PHASE calibrates at entry: those maps do not depend on φ_s.
        """
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements.ncells import NCells
        meta = self.metadata
        if not (meta.train_frequency_mhz > 0 or meta.legacy_weights):
            return None
        w = self._wrapped
        if (isinstance(w, (FieldMap, FieldMap3D)) and getattr(w, "p_flag", 1) != 1
                and self._rf_driven()):
            return ("phase", 1.0)
        if (isinstance(w, NCells) and w.p_flag == 1 and not w.sync_phase
                and w.eot_v_per_m * (1.0 + w.voltage_rel) != 0.0):
            return ("theta_s_deg", -1.0)
        return None

    def _folds(self, ref=None) -> tuple[list, str | None]:
        """How the element's error knobs and the reference's RF shift move its inputs, as
        ``(input, "add" | "mul", amount)``, and None; or ``[]`` and why one of them cannot
        be folded, its input not swept.  The weights hold the map of the design element:

        * a field map the RF drives fires at ``phase + phase_offset``, plus the RF shift
          where its field follows the running phase; at ``p_flag`` 1 its calibration reads
          the phase with the offset but without the shift, which no input stands for (a
          shift of 5° is 2e-5 off a phase change), so a shift refuses;
          ``frequency_offset`` adds to ``frequency``; a map the RF does not drive takes
          neither.  ``voltage_rel`` scales ``ke`` and ``kb`` alike, so it goes into
          ``scale``, or else into each of ``ke`` and ``kb`` that acts;
        * an ``NCELLS`` fires at ``theta_s_deg + phase_offset`` plus the RF shift, or
          minus it with P=1 (which counts θs the other way); under ``SET_SYNC_PHASE`` the
          calibration reads θs without them, and ``voltage_rel`` scales its gap
          voltages, which no input stands for: those refuse.
        """
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements.ncells import NCells
        w, names, key = self._wrapped, self._param_names, self.metadata.element_key
        shift = float(getattr(ref, "rf_shift", 0.0) or 0.0) if ref is not None else 0.0

        def unswept(what, name):
            return [], (f"surrogate '{key}': {what}, and its {name!r} was not swept in "
                        "training")

        folds = []
        if isinstance(w, (FieldMap, FieldMap3D)):
            driven = self._rf_driven()
            calibrated = getattr(w, "p_flag", 1) == 1
            if driven and calibrated and shift != 0.0:
                return [], (f"surrogate '{key}': the RF shift of SET_BEAM_PHASE_ERROR comes after "
                            "its p_flag 1 calibration, which no input stands for")
            if driven and calibrated and getattr(w, "sync_phase_pin", None) is not None:
                # train mode and frozen reliability pins keep a calibration taken once, while
                # every training sample was calibrated afresh: the weights hold that map only
                # where the pin was taken, which an error or another entrance energy leaves
                return [], (f"surrogate '{key}': its p_flag 1 calibration is pinned (train mode, "
                            "a frozen reliability pin), and its weights hold the map calibrated "
                            "afresh at each input")
            dphi = (float(getattr(w, "phase_offset", 0.0) or 0.0) + shift) if driven else 0.0
            if dphi != 0.0:
                if "phase" not in names:
                    return unswept(f"its phase is moved {dphi:+.6g}° (phase_offset, RF shift)",
                                   "phase")
                folds.append(("phase", "add", dphi))
            v = float(getattr(w, "voltage_rel", 0.0) or 0.0)
            if v != 0.0:
                ks = _acting_ks(w)
                if not ks:
                    pass                    # no channel acts: the error changes nothing
                elif "scale" in names:
                    folds.append(("scale", "mul", 1.0 + v))
                elif all(k in names for k in ks):
                    folds += [(k, "mul", 1.0 + v) for k in ks]
                else:
                    return unswept(f"its voltage_rel is {v:+.6g}", "scale' or '" + "' and '".join(ks))
            df = float(getattr(w, "frequency_offset", 0.0) or 0.0) if driven else 0.0
            if df != 0.0:
                if "frequency" not in names:
                    return unswept(f"its frequency_offset is {df:+.6g} MHz", "frequency")
                folds.append(("frequency", "add", df))
        elif isinstance(w, NCells):
            if w.voltage_rel != 0.0:
                return [], (f"surrogate '{key}': its voltage_rel is {w.voltage_rel:+.6g}, which "
                            "scales the NCELLS gap voltages and no input stands for")
            if w.sync_phase:
                if w.phase_offset != 0.0 or shift != 0.0:
                    return [], (f"surrogate '{key}': its SET_SYNC_PHASE calibration reads θs "
                                "without its phase_offset or the RF shift, which no input "
                                "stands for")
            else:
                dth = w.phase_offset + (-shift if w.p_flag == 1 else shift)
                if dth != 0.0:
                    if "theta_s_deg" not in names:
                        return unswept(f"its θs is moved {dth:+.6g}° (phase_offset, RF shift)",
                                       "theta_s_deg")
                    folds.append(("theta_s_deg", "add", dth))
        elif shift != 0.0:
            return [], (f"surrogate '{key}': the RF shift of SET_BEAM_PHASE_ERROR moves its "
                        "phase, which no input stands for")
        return folds, None

    #: the phase steps of the one check of a file that records no training phase (deg)
    PHASE_CHECK_STEP = 0.25

    #: the S4 check refuses when the assumed input's residual is this many times the best
    #: fit's, and above it by this part of the scan's range.  A network's own error looks
    #: like a phase shift: correctly trained phase-0 files reached a ratio of 2.77 (360
    #: queries, files trained as the Surrogates tab and the CLI train), well-trained files
    #: 6° off exceeded it, files 20° and 40° off nearly always did.
    PHASE_CHECK_RATIO = 3.0
    PHASE_CHECK_GAP = 0.05

    def _unrecorded_phase_refusal(self, own, x) -> str | None:
        """For a file that records no training phase and so counts as trained at 0: whether
        its map fits the native element there.  Run once per load, at the first use of a
        map that follows the running phase with that input swept.  The map is scanned over
        the input's sweep against one native evaluation at ``own``: it refuses when the
        input the assumption gives fits clearly worse than the best (``PHASE_CHECK_RATIO``
        times its residual, and by ``PHASE_CHECK_GAP`` of the scan's range), and says
        near which phase it fits, an estimate (biased for a roughly trained file), or that
        the best fit is at the edge of the sweep.  A file whose phase was not swept cannot
        be checked.  ``x`` is the input vector at ``own``."""
        meta = self.metadata
        if meta.phase_recorded:
            return None
        if meta.phase_check is not None:
            return meta.phase_check
        running = self._running_phase_input()
        if running is None or running[0] not in self._param_names:
            return None
        name, sign = running
        j = 3 + self._param_names.index(name)
        k = list(meta.scope.input_names).index(name)
        grid = np.arange(float(meta.scope.input_lo[k]), float(meta.scope.input_hi[k]) + 1e-9,
                         self.PHASE_CHECK_STEP)
        X = np.repeat(np.asarray(x, dtype=np.float64)[None, :], len(grid) + 1, axis=0)
        X[1:, j] = grid
        with torch.no_grad():
            Y = (self.mlp((torch.as_tensor(X, dtype=F64) - self._in_mean) / self._in_std)
                 * self._out_std + self._out_mean).numpy()
        native = np.asarray(_copy.deepcopy(self._wrapped).fitted_matrix(own.copy()),
                            dtype=np.float64).reshape(36)
        spread = np.maximum(np.abs(self._out_std.numpy()), 1e-12)
        r = np.linalg.norm((Y - native) / spread, axis=1)
        r_assumed, r_grid = r[0], r[1:]
        lo, hi = float(r_grid.min()), float(r_grid.max())
        if not (r_assumed > self.PHASE_CHECK_RATIO * lo
                and r_assumed - lo > self.PHASE_CHECK_GAP * (hi - lo)):
            meta.phase_check = ""
            return None
        best = int(np.argmin(r_grid))
        head = (f"surrogate '{meta.element_key}': its file records no training phase and so "
                f"counts as trained at a reference phase of {meta.train_phi_s_deg:g}°, where "
                "its map does not fit its native element")
        if 0 < best < len(grid) - 1:
            fit = _wrap180(meta.train_phi_s_deg - sign * _wrap180(grid[best] - x[j]))
            where = f"; it fits best near a reference phase of {fit:.0f}° (an estimate)"
        else:
            where = "; it fits best at the edge of its phase sweep"
        meta.phase_check = (head + where + ".  It was probably trained at another phase: if "
                            "you know it, write it into its metadata.json as "
                            "\"train_phi_s_deg\", which is then believed.")
        return meta.phase_check

    #: the accuracy check of weights loaded from a file (``_accuracy_refusal``): the largest
    #: ``block_error`` and ``determinant_error`` against the native map, and the number of
    #: Latin-hypercube points of the training range besides the input of the first use
    ACCURACY_TOLERANCE = 3e-3
    ACCURACY_DET_TOLERANCE = 1e-3
    ACCURACY_POINTS = 16

    def _accuracy_refusal(self, own, x) -> str | None:
        """For weights loaded from a file: why they must not answer, or None.  Their map is
        compared with the native element's once per load, at their first use by any view of
        them: at the input ``x`` of that use and at ``ACCURACY_POINTS`` points of their
        training range (``_accuracy_points``), the largest ``block_error`` must be within
        ``ACCURACY_TOLERANCE`` and the largest ``determinant_error`` within
        ``ACCURACY_DET_TOLERANCE``.  Weights that fail go native in every run, and say by how
        much once per load; the compare view still has them answer
        (``accuracy_refusals_suspended``).  ``own`` is the reference on the element's clock at
        that use.  Weights built in memory are not checked (the training's own validation
        describes them)."""
        meta = self.metadata
        if not meta.accuracy_due:
            return None
        if meta.accuracy_check is None:
            with _CHECK_LOCK:
                if meta.accuracy_check is None:
                    self._run_accuracy_check(own, x)
        if meta.accuracy_check and not _refusals_suspended():
            return meta.accuracy_check
        return None

    def accuracy_refusal_kinematics(self, kin, x) -> str | None:
        """``_accuracy_refusal`` for the torch arm, which carries the beam's kinematics
        (``RefKinematics``), not a reference: a first use builds the reference of the check,
        the beam's species at the energy of ``x`` on the arm's clock, moved to the element's
        clock as ``fitted_matrix`` moves it."""
        meta = self.metadata
        own = None
        if meta.accuracy_due and meta.accuracy_check is None:
            from linac_gen.core import particle as pt
            from linac_gen.core.constants import C_LIGHT
            from linac_gen.core.reference import ReferenceParticle
            species = next((p for p in (pt.PROTON, pt.H_MINUS, pt.DEUTERON)
                            if p.charge == kin.charge and abs(p.mass - float(kin.mass)) < 1e-6),
                           None)
            if species is None:
                return (f"surrogate '{meta.element_key}': its accuracy check needs the beam's "
                        f"species, and no species HELIX knows has mass {float(kin.mass):g} MeV")
            ref = ReferenceParticle(species, float(x[0]), C_LIGHT / (kin.wavelength * 1e3))
            own = self._on_own_clock(ref)
        return self._accuracy_refusal(own, x)

    def _accuracy_points(self, x) -> list[np.ndarray]:
        """``x`` and ``ACCURACY_POINTS`` points of the training range (a fixed seed): the
        energy and the element's inputs on a Latin hypercube; β and γ follow from the energy
        when the native map is taken."""
        scope = self.metadata.scope
        free = [0] + list(range(3, len(scope.input_names)))
        k = int(self.ACCURACY_POINTS)
        rng = np.random.default_rng(0)
        u = (np.argsort(rng.random((k, len(free))), axis=0) + rng.random((k, len(free)))) / k
        lo = np.asarray(scope.input_lo, dtype=np.float64)[free]
        hi = np.asarray(scope.input_hi, dtype=np.float64)[free]
        points = [np.array(x, dtype=np.float64)]
        for row in u:
            p = np.array(x, dtype=np.float64)
            p[free] = lo + row * (hi - lo)
            points.append(p)
        return points

    def _run_accuracy_check(self, own, x) -> None:
        """The check of ``_accuracy_refusal``; its verdict and largest errors go on the
        metadata, which every view shares.  The native maps are taken as the training took
        its targets: one copy of the element as the weights hold it (``design_copy``: its
        error knobs at zero where the inputs hold them, ``_folds``), each input set and the run
        state reset, the reference a copy of ``own`` without RF shifts (``design_reference``:
        folded into the inputs too), at the point's energy and the
        training phase, on the map's clock.  A pinned calibration gets here only as the
        weights hold it (``settings_of``), and never on a field map the RF drives, where
        ``_folds`` refuses it."""
        import time

        from linac_gen.surrogates.training import _at_element_frequency
        meta = self.metadata
        started = time.perf_counter()
        work = design_copy(self._wrapped)
        template = design_reference(own)
        worst_block = worst_det = 0.0
        points = self._accuracy_points(x)
        for p in points:
            for j, name in enumerate(self._param_names):
                setattr(work, name, float(p[3 + j]))
            if hasattr(work, "reset_run_state"):
                work.reset_run_state()
            ref = template.copy()
            ref.w_kin = float(p[0])
            ref.phi_s = float(meta.train_phi_s_deg)
            _at_element_frequency(work, ref)
            try:
                native = np.asarray(work.fitted_matrix(ref), dtype=np.float64)
            except Exception:                                   # noqa: BLE001
                # the native element cannot be taken there: the weights cannot be shown
                # accurate over their range (and the run must not stop over the check)
                native = np.full((6, 6), np.nan)
            p[1], p[2] = ref.beta, ref.gamma
            with torch.no_grad():
                nn_map = (self.mlp((torch.as_tensor(p, dtype=F64) - self._in_mean) / self._in_std)
                          * self._out_std + self._out_mean).numpy().reshape(6, 6)
            # what fitted_matrix gives: a magnet's Δφ row in degrees of this clock
            ratio = self._magnet_clock_ratio(float(getattr(ref, "frequency", 0.0) or 0.0))
            if ratio != 1.0:
                nn_map[4, :] *= ratio
                nn_map[:, 4] /= ratio
            scale = drift_scale_z(ref.beta, ref.gamma, ref.species.mass, ref.wavelength)
            b, d = block_error(native, nn_map, scale), determinant_error(native, nn_map)
            # a non-finite map, either one, is outside every limit
            worst_block = max(worst_block, b if np.isfinite(b) else np.inf)
            worst_det = max(worst_det, d if np.isfinite(d) else np.inf)
        meta.accuracy_block, meta.accuracy_det = worst_block, worst_det
        meta.accuracy_seconds = time.perf_counter() - started
        if worst_block <= self.ACCURACY_TOLERANCE and worst_det <= self.ACCURACY_DET_TOLERANCE:
            meta.accuracy_check = ""
            return
        meta.accuracy_check = (
            f"surrogate '{meta.element_key}': its map is not within the accuracy limits of its "
            f"native element (largest 2x2-block error {worst_block:.2g}, limit "
            f"{self.ACCURACY_TOLERANCE:g}; determinant error {worst_det:.2g}, limit "
            f"{self.ACCURACY_DET_TOLERANCE:g}; over {len(points)} points of its training "
            "range), so the native element tracks in every run.  Retrain it.")
        _say(meta.accuracy_check)

    def _phase_follows_clock(self) -> bool:
        """Whether the map depends on the running phase (see ``_running_phase_input``)."""
        return self._running_phase_input() is not None

    def _into_phase_scope(self, phase: float, name: str = "phase") -> float:
        """``phase``, the value of the input ``name``, or the same phase a turn away if
        that one is in the scope: the map is periodic in it."""
        scope = self.metadata.scope
        if name not in scope.input_names:
            return phase
        j = list(scope.input_names).index(name)
        lo, hi = float(scope.input_lo[j]), float(scope.input_hi[j])
        for turn in (0.0, -360.0, 360.0):
            if lo <= phase + turn <= hi:
                return phase + turn
        return phase

    def _is_magnet(self) -> bool:
        return not getattr(self._wrapped, "_has_electric_channel", lambda: True)()

    def _legacy_magnet(self) -> bool:
        return self.metadata.legacy_weights and self._is_magnet()

    def _m45_refusal(self, m45, beta, gamma, mass, wavelength) -> str | None:
        """Weights that record no clock: a magnet's Δφ–ΔW term is a drift's, to 2e-14
        natively, so it shows the clock they were trained on."""
        drift = -360.0 * self.length / (beta ** 3 * gamma ** 3 * mass * wavelength)
        if drift != 0.0 and not 0.8 <= m45 / drift <= 1.25:
            return (f"surrogate '{self.metadata.element_key}': its Δφ–ΔW term is {m45 / drift:.3g} "
                    "times a drift's at this clock, so it was trained at another one.  Retrain it.")
        return None

    def _magnet_clock_ratio(self, clock_mhz: float) -> float:
        """clock / training clock for a magnet whose weights record one, else 1."""
        train = self.metadata.train_frequency_mhz
        if train > 0 and clock_mhz > 0 and self._is_magnet() and abs(clock_mhz / train - 1.0) > 1e-12:
            return clock_mhz / train
        return 1.0

    def _clock_refusal(self, clock_mhz: float) -> str | None:
        """A cavity with no frequency of its own runs on the clock, so weights that record
        one answer only on it."""
        train = self.metadata.train_frequency_mhz
        own = float(getattr(self._wrapped, "effective_frequency", 0.0)
                    or getattr(self._wrapped, "frequency", 0.0) or 0.0)
        if (train > 0 and clock_mhz > 0 and own == 0.0 and not self._is_magnet()
                and abs(clock_mhz / train - 1.0) > 1e-12):
            return (f"surrogate '{self.metadata.element_key}': its cavity runs on the clock, "
                    f"{clock_mhz:.6g} MHz here, and it was trained at {train:.6g} MHz")
        return None

    def _on_own_clock(self, ref):
        """``ref`` as the cavity sees it on entry.  A cavity at another frequency
        than the clock switches it there, which moves the reference phase as a FREQ
        card does (``ReferenceParticle.switch_clock``): ``ref`` itself when nothing
        moves, else a copy."""
        if self._jump_ratio(ref) == 1.0:
            return ref
        own = ref.copy()
        self.switch_to_own_clock(own)
        return own

    def _jump_ratio(self, ref) -> float:
        """f_effective / f_clock for a cavity at another frequency than ``ref``,
        else 1.  A map with no electric channel carries its frequency as metadata
        only."""
        return self._jump_ratio_at(float(getattr(ref, "frequency", 0.0) or 0.0))

    def _jump_ratio_at(self, clock_mhz: float, tol: float = 0.0) -> float:
        """``_jump_ratio`` at a clock given as a number.  ``tol``: a ratio within it of 1
        counts as 1, for a caller whose clock went through λ = c/f and back."""
        eff = float(getattr(self._wrapped, "effective_frequency", 0.0)
                    or getattr(self._wrapped, "frequency", 0.0) or 0.0)
        clock = float(clock_mhz or 0.0)
        if (eff > 0 and clock > 0 and eff != clock and abs(eff / clock - 1.0) > tol
                and getattr(self._wrapped, "_has_electric_channel", lambda: True)()):
            return eff / clock
        return 1.0

    def rf_shifts_on_own_clock(self, kin, clock_mhz: float):
        """``kin`` (the torch arm's kinematics, which stay on the line's clock) with
        SET_BEAM_PHASE_ERROR's RF shifts counted on this cavity's clock, as
        ``switch_to_own_clock`` counts them for the numpy path."""
        if (self._jump_ratio_at(clock_mhz, tol=1e-12) == 1.0
                or not (getattr(kin, "rf_shift", 0.0) or getattr(kin, "rf_rephase", 0.0))):
            return kin
        import dataclasses
        eff = float(getattr(self._wrapped, "effective_frequency", 0.0)
                    or getattr(self._wrapped, "frequency", 0.0) or 0.0)
        s = (float(getattr(self._wrapped, "frequency", 0.0) or 0.0) or eff) / clock_mhz
        return dataclasses.replace(kin, rf_shift=kin.rf_shift * s, rf_rephase=kin.rf_rephase * s)

    def switch_to_own_clock(self, ref) -> bool:
        """Switch ``ref`` in place to this cavity's clock, as the native element does on
        entry when it is at another frequency than the clock (``_on_own_clock`` does it on
        a copy); False when nothing moves.  For the torch walks, which compose the map
        and must carry the clock on past it."""
        if self._jump_ratio(ref) == 1.0:
            return False
        eff = float(getattr(self._wrapped, "effective_frequency", 0.0)
                    or getattr(self._wrapped, "frequency", 0.0) or 0.0)
        ref.switch_clock(float(getattr(self._wrapped, "frequency", 0.0) or 0.0) or eff, eff)
        return True

    def fitted_matrix_torch(self, kin_tensor: torch.Tensor, kin=None) -> torch.Tensor:
        """Autograd-differentiable 6x6 transfer matrix (M8 matcher arm).

        Parameters
        ----------
        kin_tensor : torch.Tensor, shape (3,), dtype F64
            ``[w_kin, beta, gamma]`` of the reference particle.  May
            carry ``requires_grad=True`` so the matcher's chain rule
            connects (today the matcher tunes upstream linear knobs;
            surrogated cavities are passive blocks the gradient flows
            through).
        kin : RefKinematics, optional
            Carries SET_BEAM_PHASE_ERROR's RF shift, folded into the phase
            input with the element's error knobs (see ``_folds``).

        Returns
        -------
        torch.Tensor, shape (6, 6), dtype F64
            Same MLP output as :meth:`fitted_matrix` but with the
            autograd graph intact -- no ``torch.no_grad``.  Element
            params (``ke``, ``kb``, ``phase``, ...) are read once from
            the wrapped element as constants in this iteration; future
            work can lift them to differentiable overrides if matching
            ever needs to tune them directly.

        Notes
        -----
        Caller is responsible for the OOD scope check -- the matcher
        arm in :mod:`linac_gen.tracking.torch_tracking` does this
        before invoking, raising ``ValueError`` (hard error, never identity)
        when the surrogate would refuse.  This method itself never
        raises ``OutOfScopeError``; it computes whatever the MLP
        gives, which is what the matcher's gradient path needs.
        """
        params = torch.as_tensor(self._param_values(ref=kin), dtype=F64)
        x = torch.cat([kin_tensor, params])
        x_norm = (x - self._in_mean) / self._in_std
        # NO torch.no_grad -- autograd graph stays connected.
        y_norm = self.mlp(x_norm)
        y = y_norm * self._out_std + self._out_mean
        return y.reshape(6, 6)

    def fitted_matrix_slice(self, ref, ds_mm: float) -> np.ndarray:
        """Surrogate-NN slice when transporting the FULL element;
        delegate to wrapped RK4 for partial slices.

        Why the split:

        * **Full-element transport** (``ds_mm == self.length``): the
          surrogate's ``fitted_matrix(ref)`` IS the trained NN
          prediction for the whole element.  Returning it here gives
          the order-of-magnitude speedup the user expected when
          ticking Use, with no accuracy loss vs the wrapped RK4
          ``fitted_matrix_slice`` evaluated over the same full
          element (both predict the same M_full, surrogate via NN +
          training data, wrapped via direct RK4 integration).
        * **Partial slice** (``ds_mm < self.length``, e.g. an envelope
          SC bundle that splits the element to apply SC kicks every
          few substeps): the surrogate's ``M_full`` is the LINEAR
          end-to-end matrix.  Composing a sub-slice via
          ``expm(logm(M_full) × ratio)`` ONLY equals the true sub-slice
          when the per-substep dynamics is constant-coefficient
          (Drift / Quad / Solenoid).  For FieldMap RF cavities the
          field is z-varying ``sin(ωt)`` and ``expm(logm)`` gives a
          time-averaged approximation that diverges from RK4 -- this
          is what caused the env-with-surrogate vs env-without-
          surrogate result divergence in user testing.  RK4 is the
          honest answer for these partial slices; we delegate.

        Net effect for users:

        * **No-SC envelope, no per-sub-step recording** -- the solver's
          pure-linear path does not call this method at all: it requests
          the FULL-element matrix once via ``envelope._full_matrix_at``
          -> :meth:`fitted_matrix` (NN used, speedup).
        * **SC-active envelope on Drift/Quad/Solenoid-only lattice** --
          partial slices through CONSTANT-COEFF elements would
          actually be correct via logm/expm, but the wrapped
          fitted_matrix_slice on Drift/Quad is already analytical
          (not RK4) so the surrogate doesn't help anyway.
        * **SC-active envelope through FieldMap RF cavities, and I=0
          per-sub-step / interior-marker walks** -- partial slices
          delegate to RK4, zero NN queries, NO surrogate speedup.
          Future work: train a slice-aware surrogate that takes ds_mm
          as an input to gain speedup here too.

        OOD safety: surrogate's ``fitted_matrix`` raises
        :class:`OutOfScopeError` for inputs outside the trained scope;
        we catch and fall back to RK4 in that case too.
        """
        if abs(float(ds_mm) - float(self.length)) < 1e-9:
            try:
                return self.fitted_matrix(ref)
            except OutOfScopeError:
                pass  # OOD -- fall through to wrapped RK4.
        # Partial slice OR OOD -- delegate to RK4 for accuracy.
        return self._wrapped.fitted_matrix_slice(ref, ds_mm)

    def _slice_matrix(self, ref, ds_mm: float) -> np.ndarray:
        """Matrix log/exp slice of the surrogate's full-element matrix.

        Used by the M7 hybrid MP path (and any other slice-aware
        caller that wants a surrogate-predicted slice).  Mathematics:

            M_slice = expm(logm(M_full) * (ds / L))

        Slices for any partition of the element compose to M_full by
        construction.  Valid under the smooth-dynamics assumption;
        falls back to the wrapped element's RK4 slice if the matrix
        log isn't real (very rare for symplectic transport).

        ``scipy.linalg.logm + expm`` cost ~5-10 ms each on a 6x6 (an
        eigendecomposition + reconstruction), so a per-substep call
        across ~3000 substeps dominates the run.  The result is
        constant per ``(ds, ref.w_kin)`` within one element's inner
        loop, so cache the most recent and reuse if the inputs match.
        """
        from scipy.linalg import expm, logm
        # Cache key: (ds, w_kin rounded to 1e-9 MeV).  Within an SC
        # bundle the reference doesn't change between same-ds calls;
        # across elements both ds and w_kin shift so the cache misses
        # naturally.
        key = (round(float(ds_mm), 9), round(float(ref.w_kin), 9))
        if self._slice_cache_key == key and self._slice_cache_M is not None:
            return self._slice_cache_M

        try:
            if abs(float(ds_mm) - float(self.length)) < 1e-9:
                M = self.fitted_matrix(ref)
            else:
                M_full = self.fitted_matrix(ref)
                try:
                    log_M = logm(M_full)
                except Exception:
                    # Pathological matrix -- can't take a real log.
                    return self._wrapped.fitted_matrix_slice(ref, ds_mm)
                ratio = float(ds_mm) / max(float(self.length), 1e-12)
                M = np.real(expm(log_M * ratio))
        except OutOfScopeError:
            # OOD (or non-finite NN output): the wrapped element is
            # always the safe answer — never abort the caller's run
            # over a matrix that would not have been used.
            return self._wrapped.fitted_matrix_slice(ref, ds_mm)

        self._slice_cache_key = key
        self._slice_cache_M = M
        return M

    # ------------------------------------------------------------------
    # Multi-particle contract (M7 + M7-followup)
    #
    # Two opt-in gates from the registry control behaviour here:
    #   - is_mp_enabled()       (M7)       -- if False, delegate to
    #                                          wrapped (bit-identical
    #                                          to baseline).
    #   - is_fast_path_enabled() (M7-followup) -- if False (default),
    #                                          delegate.  If True, run
    #                                          the linear-matrix fast
    #                                          path (analytic ref
    #                                          advance + batched M @
    #                                          particles, no per-substep
    #                                          wrapped.track_rk4 call).
    # The fast path also falls back to the wrapped element on any
    # error (logm failure, OOD, missing helpers on the wrapped) so
    # the run is always physically defensible.
    # ------------------------------------------------------------------
    def _init_fast_path(self, ref) -> None:
        """Element-entry initialisation for the fast path.

        Runs once per element traversal (cleared by reset_run_state).
        Predicts M_full, caches logm(M_full), triggers the wrapped's
        SET_SYNC_PHASE calibration so per-substep `_phi_sync_rad`
        returns the right thing, and caches the electric-channel list.

        A cavity whose effective frequency differs from the clock has a jump
        to make on entry (Δφ and the clock), which only the native element
        does: this traversal goes to it.  So does one the native element has
        already started, its first steps out of scope: this set-up is for the
        entrance (it restarts the cursor and the entrance phase).  So does a map
        with no channel interpolators to sample (a 1-D map), before the weights
        are asked: an answer the push cannot use is not counted.
        """
        if (getattr(self._wrapped, "_step_idx", 0) > 0 or self._jump_ratio(ref) != 1.0
                or getattr(self._wrapped, "_interpolators", None) is None):
            self._fast_init_failed = True
            return
        try:
            self._M_full = self.fitted_matrix(ref)
        except OutOfScopeError:
            self._fast_init_failed = True
            return
        log_m = _fast_log(self._M_full)
        if log_m is None:
            self._fast_init_failed = True
            return
        self._log_M_full = log_m

        wrapped = self._wrapped
        # Trigger the cavity-entry sync-phase calibration so
        # `_phi_sync_rad` returns the right value under p_flag == 1.
        if hasattr(wrapped, "_calibrate_sync_phase"):
            try:
                wrapped._calibrate_sync_phase(ref)
            except Exception:
                pass
        if hasattr(wrapped, "_step_idx"):
            wrapped._step_idx = 0
        if hasattr(wrapped, "_phi_s_at_entrance"):
            wrapped._phi_s_at_entrance = ref.phi_s
        # NOTE: no lazy interpolator build is needed — FieldMap3D creates
        # ``_interpolators`` eagerly in __init__ via
        # ``_build_channel_interpolators(field_data)``.  (An old
        # ``hasattr(wrapped, "_build_interpolators")`` guard here matched a
        # method that never existed, so it was permanently-dead code.)
        # Cache electric-channel list for the per-substep dW_ref sum.
        interps = wrapped._interpolators
        self._electric_channels = [
            (ch_enum, comp_interps)
            for ch_enum, comp_interps in interps.items()
            if getattr(ch_enum, "is_electric", False)
        ]
        # Reset the slice cache so the entry's M_slice is computed
        # fresh -- prevents bleed-through from a previous element.
        self._slice_cache_key = None
        self._slice_cache_M = None

    def _get_cached_slice_from_log(self, ds: float, ref_w_kin: float) -> np.ndarray:
        """Like `_slice_matrix` but uses the cached log_M_full.

        Skips the per-substep `logm` (the expensive part); only the
        much cheaper `expm` runs on a cache miss.
        """
        key = (round(float(ds), 9), round(float(ref_w_kin), 9))
        if self._slice_cache_key == key and self._slice_cache_M is not None:
            return self._slice_cache_M
        M = fast_slice(self._M_full, self._log_M_full, self.length, ds)
        self._slice_cache_key = key
        self._slice_cache_M = M
        return M

    def track_rk4(self, beam, ds: float) -> None:
        """MP push -- safe delegate (M7) or fast path (M7-followup).

        Gating (cheap checks, top to bottom):

          * surrogate disabled                  -> delegate
          * MP-engagement (registry) off        -> delegate
          * fast-path (registry) off            -> delegate (safe M7)
          * ref OOD                             -> delegate
          * fast init failed earlier            -> delegate
          * wrapped lacks required helpers      -> delegate (defensive)

        If all checks pass, run the linear-matrix fast path:
        analytic ref advance using wrapped's helpers + batched
        M_slice @ alive particles.  Each call is recorded for the backtracker (its
        step, ``_record_call``), and so is each call the weights make (``_record_push``).
        """
        call = self._call_index
        self._call_index = call + 1
        self._record_call(call, beam, ds)
        if not self._enabled:
            self._wrapped.track_rk4(beam, ds)
            return
        from linac_gen.surrogates import registry as _reg
        if not _reg.is_mp_enabled() or not _reg.is_fast_path_enabled():
            self._wrapped.track_rk4(beam, ds)
            return
        # Cheap ref-only scope check (per-particle would be too slow).
        if self._phi_s_entry is None:
            self._phi_s_entry = float(getattr(beam.ref, "phi_s", 0.0))
        if not self.in_fast_scope(beam.ref, self._phi_s_entry):
            self._wrapped.track_rk4(beam, ds)
            return

        # First call within this element traversal: initialise.
        if not self._fast_init_done:
            self._init_fast_path(beam.ref)
            self._fast_init_done = True
        if self._fast_init_failed:
            self._wrapped.track_rk4(beam, ds)
            return

        wrapped = self._wrapped
        # Defensive: the fast path relies on these private helpers.
        # If the wrapped class doesn't expose them, fall back.
        for attr in ("_z_map_start", "_step_idx", "_advance_z_cursor",
                     "_phi_sync_rad", "_phasor", "_sample_onaxis",
                     "_scale_factor", "field_data"):
            if not hasattr(wrapped, attr):
                self._wrapped.track_rk4(beam, ds)
                return

        # ---- Per-substep ref advance (analytic) -------------------
        ref = beam.ref
        z_snap = wrapped._z_cursor_state()
        z_pos = wrapped._z_map_start + wrapped._advance_z_cursor(ds) + ds / 2.0

        if ref.wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref.phi_s + dphi_to_mid
        try:
            phi_sync_rad = wrapped._phi_sync_rad(ref, phi_s_mid)
        except Exception:
            # Calibration not ready, etc.  Roll back the position
            # cursor and fall back.
            wrapped._restore_z_cursor_state(z_snap)
            self._wrapped.track_rk4(beam, ds)
            return

        dW_ref = 0.0
        ds_m = ds * 1e-3
        for ch_enum, comp_interps in self._electric_channels:
            ch_data = wrapped.field_data.channels[ch_enum]
            Ez_ax = wrapped._sample_onaxis(ch_enum, comp_interps, z_pos)
            phasor = float(wrapped._phasor(
                ch_enum, np.array([phi_sync_rad]))[0])
            amp = wrapped._scale_factor(ch_enum, ch_data)
            dW_ref += ref.species.charge * Ez_ax * amp * phasor * ds_m

        ref.w_kin += dW_ref
        ref.s += ds
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

        # ---- Per-substep particle push (batched matmul) -----------
        alive = beam.alive_mask
        if alive.any():
            M_slice = self._get_cached_slice_from_log(ds, ref.w_kin)
            beam.particles[alive] = (M_slice @ beam.particles[alive].T).T
        self._record_push(call)

    def advance_ref(self, ref) -> None:
        """Delegate reference-particle advance to the wrapped element.

        Without this, a directly-substituted surrogate inherited the
        FieldMapElement no-op placeholder — the reference energy AND s
        froze across the element, silently corrupting every downstream
        map (latent hazard; the registry route never hit it because it
        advances via the native lattice element)."""
        self._wrapped.advance_ref(ref)

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Delegate sub-range reference advance (envelope SC sub-steps)
        to the wrapped element."""
        self._wrapped.advance_ref_over(ref, z_from_mm, z_to_mm)

    def reset_run_state(self) -> None:
        super().reset_run_state()
        if hasattr(self._wrapped, "reset_run_state"):
            self._wrapped.reset_run_state()
        # M7-followup fast-path state: each element traversal starts
        # fresh, so the next _init_fast_path() re-runs the sync-phase
        # calibration + M_full prediction against the current ref
        # state (not whatever the previous element left behind).
        self._fast_init_done = False
        self._fast_init_failed = False
        self._M_full = None
        self._log_M_full = None
        self._electric_channels = None
        self._slice_cache_key = None
        self._slice_cache_M = None
        self._phi_s_entry = None
        self._call_index = 0
        self._push_record = {"calls": [], "M_full": None, "log_M_full": None,
                             "length": float(self.length), "n_calls": 0, "ds": None,
                             "entry": None, "settings": None}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_input(self, ref, phi_s: float | None = None) -> np.ndarray:
        """Build the MLP input vector from (ref kinematics + element params).

        Convention: [w_kin (MeV), beta, gamma, *param_values].
        ``param_values`` are read by name from the wrapped element
        (see ``self._param_names``).  A cavity whose field follows the running
        phase (a field map whose ``p_flag`` is not 1) takes its ``phase`` input shifted
        by how far ``phi_s``, at its entrance, is from the training's: the map depends on
        their sum alone.
        """
        at = getattr(ref, "phi_s", 0.0) if phi_s is None else phi_s
        return np.array(
            [float(ref.w_kin), float(ref.beta), float(ref.gamma)]
            + self._param_values(at, ref),
            dtype=np.float64,
        )

    def _param_values(self, phi_s: float | None = None, ref=None) -> list[float]:
        """The element's inputs after the kinematics, in the weights' order: read from the
        wrapped element, with its error knobs and the RF shift of ``ref`` folded in
        (``_folds``).  Given the reference phase at the entrance, the input the running
        phase moves is shifted by how far it is from the training's (see
        ``_running_phase_input``; SET_BEAM_PHASE_ERROR's correction on the clock moves the
        running phase too).  A phase input that moved is turned back into the scope."""
        names = self._param_names
        params = [float(getattr(self._wrapped, p)) for p in names]
        moved = set()
        for name, kind, amount in self._folds(ref)[0]:
            k = names.index(name)
            params[k] = params[k] * amount if kind == "mul" else params[k] + amount
            moved.add(name)
        running = self._running_phase_input()
        if phi_s is not None and running is not None and running[0] in names:
            name, sign = running
            rephase = float(getattr(ref, "rf_rephase", 0.0) or 0.0) if ref is not None else 0.0
            k = names.index(name)
            params[k] += sign * _wrap180(phi_s + rephase - self.metadata.train_phi_s_deg)
            moved.add(name)
        for name in moved & {"phase", "theta_s_deg"}:
            k = names.index(name)
            params[k] = self._into_phase_scope(params[k], name)
        return params
