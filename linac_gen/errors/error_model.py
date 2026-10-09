# linac_gen/errors/error_model.py
"""Error model and Monte Carlo engine for error studies."""
import copy
import fnmatch
import warnings
from dataclasses import dataclass, field

import numpy as np

from linac_gen.distributions.factory import create_beam
from linac_gen.errors.beam_error import BeamErrorDef
from linac_gen.tracking.tracker import Tracker


@dataclass
class ErrorDef:
    """Definition of one error type on one set of elements."""
    pattern: str          # element name pattern with wildcards (e.g., "QUAD_*")
    parameter: str        # attribute name (e.g., "dx", "gradient_rel")
    distribution: str = "gaussian"
    sigma: float = 0.0    # RMS for gaussian
    half_width: float = 0.0  # half-width for uniform
    cutoff: float = 3.0   # sigma cutoff for gaussian
    # Optional element-type gate applied AFTER the name pattern:
    # "cavity" / "solenoid" (FieldMap family, by live electric channel)
    # or an exact class name ("Quadrupole", "Steerer", …).  ``None``
    # keeps the historical name-only matching.  This exists because name
    # patterns alone cannot separate solenoid field maps from cavity
    # field maps in decks where both are ``FMAP_*``.
    element_kind: str | None = None
    # Power-supply circuits.  ``circuits`` maps a circuit name to its member
    # elements (element ``label`` or ``name``, e.g. the BTL deck's MAD names
    # ``QD101``).  ``circuit_mode`` says how the matched elements' draws
    # relate (see :meth:`ErrorStudy.add_error`).  ``None`` / "independent"
    # keeps the historical one-draw-per-element stream bit for bit.
    circuits: dict | None = None
    circuit_mode: str = "independent"
    assignment: str = "random"
    # Land a ``*_rel`` draw in the element's own relative slot
    # (``gradient_rel`` / ``field_rel``) even when the design attribute
    # exists, leaving the design value untouched — so a later circuit
    # re-match (one shared absolute setpoint) keeps the spread.
    into_slot: bool = False
    # Sorting on MEASURED values: σ of the measurement error (same units as
    # the draw, e.g. 5e-4 = 0.05 % for gradient_rel); 0 = perfect.
    sort_noise: float = 0.0
    # Draw only for matched elements that sit on a listed circuit; the
    # rest of the pattern's matches get nothing (e.g. one supply family
    # out of a ``BEND_*`` pattern).
    listed_only: bool = False


_CIRCUIT_MODES = ("independent", "about_mean", "common")
_STREAM_TAG = 0x5EEDE770            # see ErrorStudy(rng_stream=...)


def stream_entropy(seed: int, *key: int) -> tuple:
    """Entropy for an auxiliary per-seed stream (at least one ``key``).

    Tagged and at least 4 words long, so for seeds and keys below 2³² it
    cannot coincide with an :class:`ErrorStudy` element / beam stream or
    ``create_beam``'s integer seeds (no SeedSequence zero-padding
    collisions; astronomically large integers can still alias words)."""
    if not key:
        raise ValueError("stream_entropy needs at least one key")
    return (int(seed), _STREAM_TAG + 1) + tuple(int(k) for k in key) + (0,)
_ASSIGNMENTS = ("random", "sorted")


def _elem_keys(elem) -> tuple:
    """Names a circuit list may use for ``elem``: deck label, then name."""
    lab = getattr(elem, "label", None)
    return tuple(k for k in (lab, getattr(elem, "name", None)) if k)


def _draw_value(rng, err) -> float | None:
    """One draw of ``err``'s distribution (None for an unknown one)."""
    if err.distribution == "gaussian":
        return _draw_truncated_gaussian(rng, err.sigma, err.cutoff)
    if err.distribution == "uniform":
        return float(rng.uniform(-err.half_width, err.half_width))
    return None


def circuit_groups(matched, circuits) -> tuple[list, list]:
    """Partition ``matched`` elements by circuit.

    Returns ``(groups, unlisted)``: ``groups`` is a list of
    ``(circuit_name, [indices into matched])`` in order of first
    appearance (only circuits with at least one matched member),
    ``unlisted`` the indices of matched elements on no listed circuit.
    """
    owner = {}
    for cname, members in (circuits or {}).items():
        for m in members:
            owner.setdefault(str(m), cname)
    order, groups, unlisted = [], {}, []
    for i, elem in enumerate(matched):
        cname = next((owner[k] for k in _elem_keys(elem) if k in owner), None)
        if cname is None:
            unlisted.append(i)
            continue
        if cname not in groups:
            groups[cname] = []
            order.append(cname)
        groups[cname].append(i)
    return [(c, groups[c]) for c in order], unlisted


def magnet_type(elem) -> tuple:
    """Production family of a magnet: class and length (mm, 0.1 µm).

    Sorting draws from one production batch per family — an arc-dipole
    outlier cannot be installed in a short chicane-magnet position."""
    return (type(elem).__name__, round(float(getattr(elem, "length", 0.0) or 0.0), 4))


def _centred_block_order(sizes):
    """Block sizes (descending) arranged largest-in-the-middle, the rest
    alternating outward — the densest part of a batch to the largest
    string."""
    order = []
    for k, n in enumerate(sorted(sizes, reverse=True)):
        if k % 2:
            order.append(n)
        else:
            order.insert(0, n)
    return order


def sorted_assignment(batch, groups, unlisted, rng, keys=None) -> np.ndarray:
    """Install a production batch of magnet values by sorting.

    The rule of the BTL sorting exercise: the magnets furthest from the
    batch median go to the individually powered positions (single-magnet
    circuits and magnets on no listed circuit), where their own supply
    absorbs their error; the rest, sorted by value, are cut into
    contiguous runs — one run per series string, the largest string on
    the densest (central) run and the others alternating outward — so
    each string carries near-equal magnets.  Inside a string the run is
    shuffled over the positions.  Returns the value per matched index.
    Call it per magnet family (:func:`magnet_type`).

    ``keys``: the values the sorting DECIDES on — e.g. the measured GdL,
    true value + measurement error — while the true ``batch`` values are
    what gets installed.  Default: the true values (perfect measurement).
    """
    batch = np.asarray(batch, dtype=float)
    keys = batch if keys is None else np.asarray(keys, dtype=float)
    out = np.empty_like(batch)
    singles = list(unlisted) + [ix[0] for _, ix in groups if len(ix) == 1]
    multis = sorted((ix for _, ix in groups if len(ix) > 1), key=len, reverse=True)
    order = np.argsort(-np.abs(keys - np.median(keys)), kind="stable")
    for pos, b in zip(singles, order[:len(singles)]):
        out[pos] = batch[b]
    tail = order[len(singles):]
    rest = batch[tail[np.argsort(keys[tail], kind="stable")]]
    # contiguous runs, largest string centred; strings of equal size take
    # the runs in their (deterministic) lattice order
    by_size = {}
    for ix in multis:
        by_size.setdefault(len(ix), []).append(ix)
    k = 0
    for n in _centred_block_order([len(ix) for ix in multis]):
        ix = by_size[n].pop(0)
        run = rest[k:k + n]
        k += n
        out[np.asarray(ix)[rng.permutation(n)]] = run
    return out


@dataclass
class SeedDraw:
    """One seed's draws in transportable form (:meth:`ErrorStudy.draw_overrides`).

    ``overrides`` — ``((f"@{index}.{attr}", value), …)`` reproducing every
    changed element attribute on a fresh copy of the design lattice
    through :func:`linac_gen.cli.common.apply_element_override`;
    ``beam_config`` — the per-seed ``BeamConfig`` copy; ``lattice`` — the
    errored copy itself (for in-process consumers; drop it before
    pickling); ``untransportable`` — ``((element, parameter), …)`` draws
    that landed on an attribute the design element does not carry
    (``_apply_errors``'s alignment ``setattr`` fallback on e.g. a
    marker), which the override transport cannot express; ``applied`` /
    ``beam_applied`` — the seed's provenance log entries.
    """
    seed: int
    overrides: tuple
    beam_config: object
    lattice: object = field(default=None, repr=False)
    untransportable: tuple = ()
    applied: list = field(default_factory=list)
    beam_applied: list = field(default_factory=list)


_MISSING = object()


def lattice_overrides(design, errored) -> tuple[tuple, tuple]:
    """Diff two same-shape lattices into ``@index.attr`` overrides.

    Every public numeric attribute (int/float, not bool) of an element
    of ``errored`` that differs from the same attribute of the
    corresponding ``design`` element becomes ``(f"@{i+1}.{attr}",
    float(value))``.  Attributes the design element lacks are reported
    in the second tuple as ``(element_name, attr)`` instead of emitted.
    """
    overrides: list = []
    untransportable: list = []
    if len(design.elements) != len(errored.elements):
        raise ValueError("lattice_overrides: element counts differ "
                         f"({len(design.elements)} vs {len(errored.elements)})")
    for i, (a, b) in enumerate(zip(design.elements, errored.elements)):
        va = vars(a)
        for k, v in vars(b).items():
            if k.startswith("_") or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            old = va.get(k, _MISSING)
            if old is _MISSING:
                untransportable.append((getattr(b, "name", f"@{i + 1}"), k))
            elif old != v:
                overrides.append((f"@{i + 1}.{k}", float(v)))
    return tuple(overrides), tuple(untransportable)


def element_kind(elem) -> str:
    """Classify an element for :attr:`ErrorDef.element_kind` matching.

    ``FieldMap`` / ``FieldMap3D`` become ``"cavity"`` (live electric
    channel, ``ke != 0``) or ``"solenoid"`` (pure magnetic, ``ke == 0``)
    — the same rule as ``linac_gen.matching.variables.
    categorize_fieldmap`` (kept in lockstep by a regression test).
    Everything else classifies as its class name.
    """
    cls = type(elem).__name__
    if cls in ("FieldMap", "FieldMap3D"):
        ke = float(getattr(elem, "ke", 0.0) or 0.0)
        return "solenoid" if abs(ke) < 1e-12 else "cavity"
    return cls


def _draw_truncated_gaussian(rng, sigma: float, cutoff: float) -> float:
    """TraceWin-style truncated gaussian: REDRAW until |v| ≤ cutoff·σ.

    The old clip-based truncation piled ~0.3 % of draws (at cutoff 3)
    onto exactly ±cutoff·σ, distorting the tails vs TraceWin's redraw
    convention.  Falls back to a clip after 100 rejections.  ``cutoff
    ≤ 0`` keeps the legacy behaviour (clip to ±0 → 0.0).
    """
    if sigma == 0.0 or cutoff <= 0.0:
        return 0.0
    lim = cutoff * sigma
    v = float(rng.normal(0.0, sigma))
    for _ in range(100):
        if abs(v) <= lim:
            return v
        v = float(rng.normal(0.0, sigma))
    return float(np.clip(v, -lim, lim))


class ErrorStudyResults:
    """Aggregated results from a Monte Carlo error study."""

    def __init__(self, recorders, corrections=None):
        self.n_seeds = len(recorders)
        self._recorders = recorders
        # ``corrections[i]`` is the dict returned by
        # ``run_correction_from_lattice`` for seed ``i`` (or ``None`` if
        # correction was disabled for that seed).
        self._corrections = list(corrections or [])

    def corrected_kicks(self, seed_idx: int) -> dict | None:
        """Return the steerer-kick dict applied for *seed_idx* (or
        ``None`` if no correction was run)."""
        if 0 <= seed_idx < len(self._corrections):
            entry = self._corrections[seed_idx]
            return entry.get("kicks") if isinstance(entry, dict) else None
        return None

    def correction_history(self, seed_idx: int) -> list | None:
        """Per-iteration RMS / saturation log for *seed_idx*."""
        if 0 <= seed_idx < len(self._corrections):
            entry = self._corrections[seed_idx]
            return entry.get("history") if isinstance(entry, dict) else None
        return None

    def correction_status(self, seed_idx: int) -> str | None:
        """Correction status for *seed_idx* — ``"converged"``,
        ``"saturated"``, ``"max_iter"``, ``"beam_lost"`` or ``"none"``;
        ``None`` when no correction ran for that seed (or the stored
        dict predates the status key)."""
        if 0 <= seed_idx < len(self._corrections):
            entry = self._corrections[seed_idx]
            return entry.get("status") if isinstance(entry, dict) else None
        return None

    def mean(self, quantity):
        """Mean of quantity across all seeds.  Returns 1-D array vs. element index."""
        arrays = [np.asarray(getattr(r, quantity)) for r in self._recorders]
        return np.mean(arrays, axis=0)

    def std(self, quantity):
        """Std of quantity across all seeds."""
        arrays = [np.asarray(getattr(r, quantity)) for r in self._recorders]
        return np.std(arrays, axis=0)

    def percentile(self, quantity, p):
        """p-th percentile of quantity across all seeds."""
        arrays = [np.asarray(getattr(r, quantity)) for r in self._recorders]
        return np.percentile(arrays, p, axis=0)

    def transmission_stats(self):
        """Return mean, min, max, std of final transmission across seeds."""
        trans = [r.transmission[-1] for r in self._recorders]
        return {
            "mean": float(np.mean(trans)),
            "min":  float(np.min(trans)),
            "max":  float(np.max(trans)),
            "std":  float(np.std(trans)),
        }


class ErrorStudy:
    """Monte Carlo error study engine.

    Sources of errors:

    1. **Programmatic** — call :meth:`add_error` and :meth:`add_beam_error`.
    2. **Lattice-attached** — when constructed with a lattice carrying
       ``lattice.errors`` / ``lattice.beam_errors`` (typically populated
       by the TraceWin parser from ``ERROR_*`` directives), those are
       absorbed automatically.

    Programmatic errors win on conflicts (same pattern + parameter): the
    user's explicit ``add_error`` call appends *after* the lattice-derived
    list, so it overrides during the per-seed apply pass.
    """

    def __init__(self, lattice, beam_config, n_seeds=100, sc_config=None,
                 base_seed: int = 0, rng_stream=0):
        self.lattice = lattice
        self.beam_config = beam_config
        self.n_seeds = n_seeds
        self.sc_config = sc_config
        # Offset added to every per-seed RNG draw so two studies with the
        # same configuration but different ``base_seed`` produce
        # statistically independent ensembles (useful for chained Monte
        # Carlo runs where the user wants to extend an existing study).
        self.base_seed = int(base_seed)
        # Entropy key: 0 is the historical stream (element draws
        # ``(seed, 0)``, beam draws ``(seed, 2)``).  Any other int or tuple
        # of ints gives streams ``(seed, TAG, *key, 0|1)`` — e.g. per-pulse
        # jitter on top of a frozen static seed, which must never replay
        # that seed's static draws.  The constant TAG word matters:
        # SeedSequence zero-pads short entropy, so a bare ``(seed, *key)``
        # would make key 0/(0,0) replay the default stream and key 2 the
        # default BEAM stream (adversarial-review finding).
        key = rng_stream if isinstance(rng_stream, (tuple, list)) else (rng_stream,)
        self._rng_key = tuple(int(k) for k in key)
        if not self._rng_key:
            raise ValueError("rng_stream: an empty key")
        # Absorb errors attached to the lattice (TraceWin ERROR_* family).
        # Copy so user mutations on this list don't leak into the lattice.
        self._errors: list[ErrorDef] = list(getattr(lattice, "errors", []) or [])
        self._beam_errors: list[BeamErrorDef] = list(
            getattr(lattice, "beam_errors", []) or []
        )
        # Orbit-correction settings (disabled by default).
        self._correction_enabled = False
        self._correction_kwargs: dict = {}
        # Per-seed provenance: every draw actually written to a lattice
        # copy / beam config, recorded by ``_apply_errors`` /
        # ``_apply_beam_errors`` for archival (element name, parameter,
        # drawn value, applied flag).  Keyed by seed.  Retained for the
        # study's lifetime (~0.4 KB per draw — subdominant to the
        # recorders); ``.clear()`` them between chained mega-ensembles
        # if memory matters.
        self.applied_log: dict[int, list[dict]] = {}
        self.beam_applied_log: dict[int, list[dict]] = {}

    def add_error(self, pattern, parameter, distribution="gaussian",
                  sigma=0.0, half_width=0.0, cutoff=3.0,
                  element_kind=None, *, circuits=None,
                  circuit_mode="independent", assignment="random",
                  into_slot=False, listed_only=False, sort_noise=0.0):
        """Register an element-level error definition.

        ``element_kind`` (optional) gates the name pattern by element
        type: ``"cavity"`` / ``"solenoid"`` for field maps, or an exact
        class name such as ``"Quadrupole"`` — see
        :func:`element_kind`.  Use it whenever a name glob could reach
        more than one element species (e.g. ``FMAP_*`` decks where
        solenoids and cavities share the prefix).

        ``distribution`` must be ``"gaussian"`` or ``"uniform"`` —
        anything else would be silently skipped by the draw loop, so it
        is rejected HERE (adversarial-review finding: a typo like
        ``"Gaussian"`` used to pass registration and the audit while
        producing zero draws).

        Power-supply circuits (``circuits = {name: [member labels or
        names]}``) relate the matched elements' draws:

        * ``circuit_mode="independent"`` — one draw per element (the
          historical behaviour; circuits are then only bookkeeping);
        * ``"about_mean"`` — one draw per magnet minus its circuit's mean.
          A series string is set to its mean transfer function, so only
          the magnet-to-magnet spread survives; a magnet alone on its
          supply, or on no listed circuit, keeps no error (its own
          current absorbs it).  This is the model of a "BdL within x %
          of each other" specification;
        * ``"common"`` — one draw per circuit shared by its magnets (a
          power-supply excursion); an unlisted magnet gets its own draw.

        ``assignment="sorted"`` (``about_mean`` only) treats the drawn
        values as a production batch installed by sorting
        (:func:`sorted_assignment`) instead of in drawn order.
        ``into_slot=True`` lands ``*_rel`` draws in the element's relative
        slot (``gradient_rel``) instead of scaling the design value.
        ``listed_only=True`` draws only for matched elements on a listed
        circuit (one supply family out of a broader pattern).
        ``sort_noise`` (sorted only): σ of the magnet-measurement error the
        sorting decides on — magnets are sorted on true + N(0, σ) and the
        true values installed.

        The circuit map is checked against the lattice at registration:
        a member that matches no element, a magnet listed on two circuits,
        and a circuit only partly inside the pattern (its mean would be
        taken over the matched members only) each raise a warning.
        """
        if circuit_mode not in _CIRCUIT_MODES:
            raise ValueError(f"add_error: circuit_mode {circuit_mode!r} — use "
                             f"one of {_CIRCUIT_MODES}")
        if assignment not in _ASSIGNMENTS:
            raise ValueError(f"add_error: assignment {assignment!r} — use one "
                             f"of {_ASSIGNMENTS}")
        if circuit_mode != "independent" and not circuits:
            raise ValueError(f"add_error: circuit_mode={circuit_mode!r} needs "
                             "a circuits map")
        if assignment == "sorted" and circuit_mode != "about_mean":
            raise ValueError("add_error: assignment='sorted' sorts magnets "
                             "onto series strings — it needs "
                             "circuit_mode='about_mean'")
        if into_slot and not str(parameter).endswith("_rel"):
            raise ValueError(f"add_error: into_slot applies to *_rel "
                             f"parameters, not {parameter!r}")
        if sort_noise and assignment != "sorted":
            raise ValueError("add_error: sort_noise applies to assignment='sorted'")
        if float(sort_noise) < 0.0:
            raise ValueError("add_error: sort_noise must be >= 0")
        if listed_only and not circuits:
            raise ValueError("add_error: listed_only needs a circuits map")
        if circuits:
            self._check_circuits(pattern, element_kind, circuits,
                                 sorted_mode=(assignment == "sorted"))
        if distribution not in ("gaussian", "uniform"):
            raise ValueError(
                f"add_error({pattern!r}, {parameter!r}): unknown "
                f"distribution {distribution!r} — use 'gaussian' or "
                "'uniform' (anything else draws NOTHING)")
        # dz / pitch / yaw draws are written onto matched elements but the
        # tracker applies only dx/dy/tilt_deg — such an error is inert.
        # The .dat-directive path already warns at parse time
        # (tracewin_error_parsing); this closes the programmatic path.
        if parameter in ("dz", "pitch_deg", "yaw_deg"):
            warnings.warn(
                f"add_error({pattern!r}, {parameter!r}): {parameter} is "
                "stored on matched elements but the tracker applies only "
                "dx/dy/tilt_deg — this error has NO effect on tracking "
                "results.",
                UserWarning,
                stacklevel=2,
            )
        self._errors.append(ErrorDef(
            pattern=pattern, parameter=parameter,
            distribution=distribution, sigma=sigma,
            half_width=half_width, cutoff=cutoff,
            element_kind=element_kind,
            circuits=({str(k): [str(m) for m in v] for k, v in circuits.items()}
                      if circuits else None),
            circuit_mode=circuit_mode, assignment=assignment,
            into_slot=bool(into_slot), listed_only=bool(listed_only),
            sort_noise=float(sort_noise),
        ))

    def _check_circuits(self, pattern, kind, circuits, sorted_mode=False) -> None:
        """Warn about circuit-map entries that cannot mean what they say."""
        keys = {}
        for e in self.lattice.elements:
            for k in _elem_keys(e):
                keys.setdefault(k, e)
        seen, problems = {}, []
        for cname, members in circuits.items():
            for m in members:
                m = str(m)
                if m not in keys:
                    problems.append(f"member {m!r} of circuit {cname!r} matches no "
                                    "element (label or name) — it would be "
                                    "silently unlisted")
                if m in seen and seen[m] != cname:
                    problems.append(f"{m!r} is listed on {seen[m]!r} and {cname!r} "
                                    "— the first wins")
                seen.setdefault(m, cname)
            inside = [m for m in members if str(m) in keys
                      and fnmatch.fnmatch(keys[str(m)].name, pattern)
                      and (kind is None or element_kind(keys[str(m)]) == kind)]
            if sorted_mode and len({magnet_type(keys[str(m)]) for m in inside}) > 1:
                problems.append(f"circuit {cname!r} mixes magnet families — sorting "
                                "draws one batch per family, so the string is "
                                "sorted piecewise")
            if inside and len(inside) < len(members):
                problems.append(f"circuit {cname!r}: only {len(inside)} of "
                                f"{len(members)} members match {pattern!r} — the "
                                "string mean / single-supply status would use "
                                "the matched members only")
        for msg in problems[:12]:
            warnings.warn(f"add_error({pattern!r}): {msg}", UserWarning, stacklevel=3)
        if len(problems) > 12:
            warnings.warn(f"add_error({pattern!r}): {len(problems) - 12} more "
                          "circuit-map problems", UserWarning, stacklevel=3)

    def add_beam_error(self, parameter: str, distribution: str = "gaussian",
                       sigma: float = 0.0, half_width: float = 0.0,
                       cutoff: float = 3.0) -> None:
        """Register a beam-input error (centroid jitter, ε scale, etc.).

        See :class:`linac_gen.errors.beam_error.BeamErrorDef` for the
        list of supported parameters.
        """
        if distribution not in ("gaussian", "uniform"):
            raise ValueError(
                f"add_beam_error({parameter!r}): unknown distribution "
                f"{distribution!r} — use 'gaussian' or 'uniform' "
                "(anything else draws NOTHING)")
        self._beam_errors.append(BeamErrorDef(
            parameter=parameter, distribution=distribution,
            sigma=sigma, half_width=half_width, cutoff=cutoff,
        ))

    _INERT_ALIGN = ("dz", "pitch_deg", "yaw_deg")   # stored, tracker ignores

    @staticmethod
    def _applicable(elem, parameter: str) -> bool:
        """Mirror of the ``_apply_errors`` branch logic WITHOUT drawing:
        True iff a draw for ``parameter`` would actually land on
        ``elem``.  Kept in lockstep with ``_apply_errors`` by a
        regression test (mirror drift = silent audit lies)."""
        if parameter.endswith("_rel"):
            return (hasattr(elem, parameter[:-4])
                    or hasattr(elem, parameter))
        if parameter in ("dx", "dy", "dz",
                         "tilt_deg", "pitch_deg", "yaw_deg"):
            return True                      # setattr fallback always lands
        if parameter == "phase":
            return hasattr(elem, "phase")
        return hasattr(elem, parameter)

    def audit_errors(self) -> dict:
        """Pre-run audit of every registered :class:`ErrorDef`.

        Classifies each definition against the DESIGN lattice without
        consuming any RNG entropy: which elements match (name pattern +
        ``element_kind`` gate), which of those would actually receive
        the draw, which would be warn-skipped, and which elements are
        selected for the *same parameter* by more than one definition
        (usually a malformed selector).  Returns::

            {"defs": [{"index", "pattern", "element_kind", "parameter",
                       "matched", "applicable", "skipped", "inert",
                       "bad_distribution"}, ...],
             "beam_defs": [{"index", "parameter", "applicable",
                            "bad_distribution"}, ...],
             "multi": {(element_name, parameter): [def indices]}}

        A def with an unknown ``distribution`` draws NOTHING — its row
        carries ``bad_distribution=True`` with empty applicable/skipped
        (the applicable↔apply mirror holds only for valid
        distributions).  ``multi`` lists (element, parameter) pairs hit
        by more than one DISTINCT def — duplicate-named elements under
        a single def are legitimate per-element draws, not overlaps.

        Policy (raise / warn / print) is the caller's job — e.g. a study
        driver prints the table and refuses to run on anomalies.
        """
        rows: list[dict] = []
        hits: dict[tuple, list[int]] = {}
        for i, err in enumerate(self._errors):
            bad_dist = err.distribution not in ("gaussian", "uniform")
            matched, applicable, skipped = [], [], []
            listed = None
            if err.circuits and err.listed_only:
                listed = {str(m) for v in err.circuits.values() for m in v}
            for elem in self.lattice.elements:
                if not fnmatch.fnmatch(elem.name, err.pattern):
                    continue
                if (err.element_kind is not None
                        and element_kind(elem) != err.element_kind):
                    continue
                if listed is not None and not any(k in listed for k in _elem_keys(elem)):
                    continue          # listed_only: no draw lands here
                matched.append(elem.name)
                if bad_dist:
                    continue          # draws nothing — can't overlap/apply
                lst = hits.setdefault((elem.name, err.parameter), [])
                if i not in lst:      # identity, not name-count: two
                    lst.append(i)     # same-named elems under ONE def
                                      # are separate draws, not overlap
                if self._applicable(elem, err.parameter):
                    applicable.append(elem.name)
                else:
                    skipped.append((elem.name, type(elem).__name__))
            row = {
                "index": i, "pattern": err.pattern,
                "element_kind": err.element_kind,
                "parameter": err.parameter,
                "matched": matched, "applicable": applicable,
                "skipped": skipped,
                "inert": err.parameter in self._INERT_ALIGN,
                "bad_distribution": bad_dist,
            }
            if err.circuits:
                elems = [e for e in self.lattice.elements
                         if fnmatch.fnmatch(e.name, err.pattern)
                         and (err.element_kind is None
                              or element_kind(e) == err.element_kind)]
                groups, unlisted = circuit_groups(elems, err.circuits)
                row["circuits"] = {
                    "mode": err.circuit_mode, "assignment": err.assignment,
                    "groups": {c: [elems[k].name for k in ix] for c, ix in groups},
                    "unlisted": [elems[k].name for k in unlisted],
                }
            rows.append(row)
        beam_rows: list[dict] = []
        cfg = self.beam_config
        for i, be in enumerate(self._beam_errors):
            p = be.parameter
            if p.endswith("_rel"):
                ok = hasattr(cfg, p[:-4])
            else:
                ok = hasattr(cfg, p)
            beam_rows.append({
                "index": i, "parameter": p, "applicable": bool(ok),
                "bad_distribution":
                    be.distribution not in ("gaussian", "uniform"),
            })
        multi = {k: v for k, v in hits.items() if len(v) > 1}
        return {"defs": rows, "beam_defs": beam_rows, "multi": multi}

    def enable_correction(self, method: str | None = None,
                          n_iter: int = 5, tol_mm: float = 0.05,
                          bpm_noise: float = 0.0,
                          rcond: float | None = None,
                          targets=None,
                          reading_backend: str = "mp") -> None:
        """Enable orbit correction between error injection and tracking.

        The correction pass scans the per-seed lattice copy for
        ``ADJUST_STEERER`` cards (parsed from the .dat as
        :class:`AdjustSteerer` instances) and runs
        :func:`linac_gen.errors.correction.run_correction_from_lattice`.
        Corrections — and their per-iteration histories — are stored on
        the :class:`ErrorStudyResults` instance returned by :meth:`run`.

        Parameters
        ----------
        method : {"one_to_one", "svd", None}
            Force a specific algorithm.  ``None`` (default) lets the
            driver pick: one-to-one when ``#cards == #BPMs`` with clean
            1:1 pairing, SVD otherwise.
        n_iter, tol_mm, bpm_noise, rcond : same as
            :func:`apply_correction`.
        targets : None | "deck" | dict
            Per-BPM set-points (see :func:`apply_correction`): ``None``
            keeps the historical steer-to-zero; ``"deck"`` corrects each
            sample onto the lattice's ``DIAG_POSITION`` targets.
        reading_backend : {"mp", "envelope"}
            How each per-seed correction reads BPMs: ``"mp"`` tracks a
            fresh particle beam per reading (legacy, noisy, slow);
            ``"envelope"`` uses the envelope solver's centroid —
            deterministic and ~an order of magnitude faster per seed.
        """
        self._correction_enabled = True
        self._correction_kwargs = dict(
            override_method=method, n_iter=int(n_iter),
            tol_mm=float(tol_mm), bpm_noise=float(bpm_noise),
            rcond=rcond, targets=targets,
            reading_backend=str(reading_backend),
        )

    def run(self, n_workers=1, should_stop=None, progress_cb=None):
        """Run Monte Carlo error study.

        Returns ErrorStudyResults with aggregated statistics.

        ``n_workers`` is accepted for forward compatibility but the seed
        loop currently runs SERIALLY — a value > 1 emits a warning
        instead of silently pretending to parallelize.  For parallel
        ensembles, run several studies with different ``base_seed`` via
        ``linac_gen.parallel.scan_pool``.

        ``should_stop`` (optional zero-arg callable) is polled at the top
        of the seed loop AND re-checked after tracking: a stop request
        mid-seed discards that seed entirely — a truncated recorder must
        never enter the ensemble (its arrays are shorter than the
        others', corrupting every mean/std/percentile).  The returned
        results carry only whole seeds; ``results.n_requested`` records
        how many were asked for so callers can report "k of n".

        ``progress_cb`` (optional) is called as ``progress_cb(i_done,
        n_requested)`` after each completed seed.
        """
        all_results = []
        all_corrections: list = []
        if int(n_workers) > 1:
            warnings.warn(
                "ErrorStudy.run(n_workers>1): the seed loop is serial — "
                "n_workers is currently ignored.  For parallel ensembles "
                "run several studies with different base_seed via "
                "linac_gen.parallel.scan_pool.",
                stacklevel=2,
            )
        n_requested = int(self.n_seeds)
        for i in range(n_requested):
            if should_stop is not None and should_stop():
                break
            seed = self.base_seed + i
            # 1. Create a fresh lattice copy with errors applied
            lattice_copy = self._apply_errors(seed)
            # 2. Create a fresh beam config + beam (with input-beam errors)
            beam_cfg = self._apply_beam_errors(seed)
            # 3. Optional orbit correction (steerer kicks land on
            #    ``lattice_copy`` BEFORE tracking, so the recorded orbit
            #    reflects the post-correction state).
            correction_info = None
            if self._correction_enabled:
                from linac_gen.core.cancelled import OperationCancelled
                from linac_gen.errors.correction import run_correction_from_lattice
                # Capture loop variable for the closure.
                _bcfg = beam_cfg
                _seed = seed
                def _factory(_bcfg=_bcfg, _seed=_seed):
                    return create_beam(_bcfg, seed=_seed + 1000)
                try:
                    correction_info = run_correction_from_lattice(
                        lattice_copy, _factory,
                        should_stop=should_stop,
                        # Per-seed BPM-noise stream — a fixed default_rng(0)
                        # made noise 100 % correlated across the ensemble.
                        noise_seed=_seed,
                        # The envelope reading backend needs the config,
                        # not particles — the per-seed beam_cfg already
                        # carries this sample's beam errors.
                        beam_config=_bcfg,
                        **self._correction_kwargs,
                    )
                except OperationCancelled:
                    break   # discard this seed, return the whole ones

            beam = create_beam(beam_cfg, seed=seed + 1000)
            # 4. Optional PIC solver
            pic = None
            if self.sc_config:
                from linac_gen.pic.pic_solver import PicSolver
                pic = PicSolver(self.sc_config)
            # 5. Track — forward the stop hook so cancellation lands at
            #    the next element instead of the next seed.
            tracker = Tracker(lattice_copy, beam, pic_solver=pic,
                              should_abort=should_stop)
            recorder = tracker.run()
            if getattr(tracker, "aborted", False):
                break   # truncated recorder — never aggregate it
            # Whole seed completed: commit recorder + correction together
            # so the two lists stay index-aligned.
            all_results.append(recorder)
            all_corrections.append(correction_info)
            if progress_cb is not None:
                progress_cb(i + 1, n_requested)

        results = ErrorStudyResults(all_results, corrections=all_corrections)
        results.n_requested = n_requested
        # Seeds whose orbit correction KILLED the beam track a dead beam
        # — count them and warn once, so the ensemble statistics are
        # never silently biased by rows full of zeros.
        n_lost = sum(1 for c in all_corrections
                     if isinstance(c, dict) and c.get("status") == "beam_lost")
        results.n_correction_beam_lost = n_lost
        if n_lost:
            first_idx = next(
                i for i, c in enumerate(all_corrections)
                if isinstance(c, dict) and c.get("status") == "beam_lost")
            warnings.warn(
                f"ErrorStudy: orbit correction lost the beam in "
                f"{n_lost}/{len(all_corrections)} seed(s) (first: seed "
                f"{self.base_seed + first_idx}, at "
                f"{all_corrections[first_idx].get('beam_lost_at')}) — "
                "those seeds track a dead beam",
                stacklevel=2,
            )
        return results

    def draw_overrides(self, seed: int) -> SeedDraw:
        """One seed's draws as transportable overrides (reliability
        campaigns run seeds through the scan pool).

        Calls the unchanged :meth:`_apply_errors` and
        :meth:`_apply_beam_errors` — the RNG streams, the kind gate and
        the provenance logs are exactly those of :meth:`run` — and diffs
        the errored copy against the design lattice with
        :func:`lattice_overrides`.  Draws the transport cannot express
        are listed in ``untransportable`` and warned about once per
        study.
        """
        lattice_copy = self._apply_errors(seed)
        beam_cfg = self._apply_beam_errors(seed)
        overrides, untransportable = lattice_overrides(self.lattice, lattice_copy)
        if untransportable and not getattr(self, "_warned_untransportable", False):
            self._warned_untransportable = True
            warnings.warn(
                "ErrorStudy.draw_overrides: "
                f"{len(untransportable)} draw(s) landed on attributes the "
                "design elements do not carry (e.g. dx on a marker) and "
                "cannot be transported as overrides: "
                + ", ".join(f"{n}.{p}" for n, p in untransportable[:6])
                + (" …" if len(untransportable) > 6 else ""),
                stacklevel=2)
        return SeedDraw(
            seed=int(seed), overrides=overrides, beam_config=beam_cfg,
            lattice=lattice_copy, untransportable=untransportable,
            applied=list(self.applied_log.get(int(seed), [])),
            beam_applied=list(self.beam_applied_log.get(int(seed), [])))

    def _apply_beam_errors(self, seed: int):
        """Return a deep-copy of the BeamConfig with beam errors applied.

        Returns the original config unchanged when no beam errors are
        registered (cheap fast-path).
        """
        if not self._beam_errors:
            return self.beam_config
        # Distinct entropy STREAM (not an integer offset) so beam-side
        # draws can never collide with another seed's element-error
        # stream: with the old ``seed + 50_000`` scheme, seed i's beam
        # stream was bit-identical to seed i+50 000's element stream.
        rng = np.random.default_rng(
            (int(seed), 2) if self._rng_key == (0,)
            else (int(seed), _STREAM_TAG) + self._rng_key + (1,))
        cfg = copy.deepcopy(self.beam_config)
        blog: list[dict] = []
        for be in self._beam_errors:
            if be.distribution == "gaussian":
                if be.sigma == 0.0:
                    continue
                val = _draw_truncated_gaussian(rng, be.sigma, be.cutoff)
            elif be.distribution == "uniform":
                if be.half_width == 0.0:
                    continue
                val = float(rng.uniform(-be.half_width, +be.half_width))
            else:
                warnings.warn(
                    f"beam error {be.parameter!r}: unknown distribution "
                    f"{be.distribution!r} — NO draw made.", stacklevel=2)
                blog.append({"parameter": be.parameter, "value": 0.0,
                             "applied": False})
                continue
            ok = self._apply_one_beam_error(cfg, be.parameter, val)
            blog.append({"parameter": be.parameter, "value": float(val),
                         "applied": bool(ok)})
        self.beam_applied_log[int(seed)] = blog
        return cfg

    @staticmethod
    def _apply_one_beam_error(cfg, parameter: str, val: float) -> bool:
        """Apply a single beam-error draw to a BeamConfig copy.

        Returns True when the draw landed on a real config field."""
        # Relative perturbations: param_rel name maps to design field × (1+δ)
        if parameter.endswith("_rel"):
            base = parameter[:-4]
            if hasattr(cfg, base):
                setattr(cfg, base, getattr(cfg, base) * (1.0 + val))
                return True
            warnings.warn(
                f"beam error parameter {parameter!r}: BeamConfig has "
                f"no field {base!r} — error NOT applied.",
                stacklevel=2,
            )
            return False
        # Mismatch-style fields: additive (existing semantics — already %).
        if parameter.startswith("mismatch_"):
            if hasattr(cfg, parameter):
                setattr(cfg, parameter, getattr(cfg, parameter) + val)
                return True
            warnings.warn(
                f"beam error parameter {parameter!r} does not exist "
                "on BeamConfig — error NOT applied.", stacklevel=2)
            return False
        # Centroid offsets: additive in the same units as the BeamConfig field.
        if hasattr(cfg, parameter):
            setattr(cfg, parameter, getattr(cfg, parameter) + val)
            return True
        warnings.warn(
            f"beam error parameter {parameter!r} does not exist on "
            "BeamConfig — error NOT applied.", stacklevel=2)
        return False

    def _grouped_values(self, err, matched, rng) -> list:
        """Draws for ``matched`` (lattice order) related by circuit.

        See :meth:`add_error` for the modes.  RNG use is deterministic per
        seed: one draw per drawn element in lattice order ("independent" /
        "about_mean"), or one per circuit in order of first appearance
        plus one per unlisted element ("common"); with ``listed_only``
        unlisted elements draw nothing.  "sorted" then consumes one
        permutation per series string, family by family
        (:func:`magnet_type`).  ``None`` marks an element with no draw.
        """
        n = len(matched)
        if n == 0:
            return []
        mode = err.circuit_mode if err.circuits else "independent"
        groups, unlisted = circuit_groups(matched, err.circuits)
        unl = set(unlisted)
        skip = unl if err.listed_only else set()
        out = [None] * n
        if mode == "common":
            done = set()
            for i in range(n):
                if i in skip:
                    continue
                g = next((ix for _, ix in groups if i in ix), None)
                if g is None:
                    out[i] = float(_draw_value(rng, err))
                elif id(g) not in done:
                    done.add(id(g))
                    v = float(_draw_value(rng, err))
                    for k in g:
                        out[k] = v
            return out
        drawn = [i for i in range(n) if i not in skip]
        vals = np.zeros(n)
        for i in drawn:
            vals[i] = _draw_value(rng, err)
        if mode == "independent":
            for i in drawn:
                out[i] = float(vals[i])
            return out
        if err.assignment == "sorted":
            fams = {}
            for i in drawn:
                fams.setdefault(magnet_type(matched[i]), []).append(i)
            for members in fams.values():
                mset = set(members)
                pos = {i: k for k, i in enumerate(members)}
                g_f = [(c, [pos[i] for i in ix if i in mset]) for c, ix in groups]
                g_f = [(c, ix) for c, ix in g_f if ix]
                u_f = [pos[i] for i in members if i in unl]
                keys = None
                if err.sort_noise > 0.0:     # measured = true + error
                    keys = vals[members] + rng.normal(0.0, err.sort_noise, len(members))
                vals[members] = sorted_assignment(vals[members], g_f, u_f, rng, keys=keys)
        for i in drawn:
            out[i] = 0.0                      # unlisted / single: absorbed
        for _, ix in groups:
            if len(ix) > 1:
                ix = np.asarray(ix)
                for k, v in zip(ix, vals[ix] - vals[ix].mean()):
                    out[k] = float(v)
        return out

    def _apply_draw(self, elem, err, val, _log) -> None:
        """Write one drawn value onto ``elem`` (shared by both draw paths)."""
        if err.parameter.endswith("_rel") and err.into_slot \
                and hasattr(elem, err.parameter):
            # Requested: keep the design value, add to the relative slot
            # (``effective_gradient = gradient·(1+gradient_rel)``).
            setattr(elem, err.parameter, getattr(elem, err.parameter) + val)
            _log(elem, err, val, True, "slot-add")
        elif err.parameter.endswith("_rel"):
            # Relative error: multiply the design field × (1+δ).
            # Strip the trailing "_rel" and look up the design slot
            # (e.g. "gradient_rel" → "gradient").  This is the
            # legacy behaviour kept stable for the existing test
            # suite — equivalent in *physics* to setting the
            # element's ``gradient_rel`` perturbation slot.
            base_param = err.parameter[:-4]
            if hasattr(elem, base_param):
                setattr(elem, base_param,
                        getattr(elem, base_param) * (1.0 + val))
                _log(elem, err, val, True, "scale")
            elif hasattr(elem, err.parameter):
                # No design slot, but the element carries the
                # relative-perturbation slot itself: FieldMap /
                # FieldMap3D consume ``voltage_rel`` via
                # effective_ke/kb = k·(1+voltage_rel), Dipole
                # consumes ``field_rel`` via effective_angle.
                # ADD the draw so multiple ErrorDefs (and any
                # hand-set design offset) compose.  Before this
                # branch, ERROR_CAV amplitude errors on field-map
                # cavities and ERROR_BEND field errors were
                # silently dropped — tolerance studies ran with
                # zero RF-amplitude jitter on SRF sections.
                setattr(elem, err.parameter,
                        getattr(elem, err.parameter) + val)
                _log(elem, err, val, True, "slot-add")
            else:
                # Neither slot exists on this element type: the
                # error cannot be applied.  Warn ONCE per
                # (pattern, parameter, type) so a mis-targeted
                # ErrorDef is never again a silent no-op.
                warned = getattr(self, "_warned_skipped", None)
                if warned is None:
                    warned = self._warned_skipped = set()
                key = (err.pattern, err.parameter,
                       type(elem).__name__)
                if key not in warned:
                    warned.add(key)
                    warnings.warn(
                        f"ErrorDef({err.pattern!r}, "
                        f"{err.parameter!r}) matched element "
                        f"'{elem.name}' "
                        f"({type(elem).__name__}) but the element "
                        f"has neither '{base_param}' nor "
                        f"'{err.parameter}' — error NOT applied.",
                        stacklevel=2,
                    )
                _log(elem, err, val, False, "no-slot")
        elif err.parameter in ("dx", "dy", "dz",
                               "tilt_deg", "pitch_deg", "yaw_deg"):
            # Alignment slots: ADD to any design / accumulated
            # value so multiple ErrorDefs on the same parameter
            # compose correctly, and so an ERROR_*_STAT directive
            # layered on top of a hand-edited dx still works.
            if hasattr(elem, err.parameter):
                setattr(elem, err.parameter,
                        getattr(elem, err.parameter) + val)
            else:
                setattr(elem, err.parameter, val)
            _log(elem, err, val, True, "align-add")
        elif err.parameter == "phase":
            if hasattr(elem, "phase"):
                elem.phase += val
                _log(elem, err, val, True, "add")
            else:
                # Same warn-once contract as the other skip
                # branches — this was the last silent no-op path.
                warned = getattr(self, "_warned_skipped", None)
                if warned is None:
                    warned = self._warned_skipped = set()
                key = (err.pattern, err.parameter,
                       type(elem).__name__)
                if key not in warned:
                    warned.add(key)
                    warnings.warn(
                        f"ErrorDef({err.pattern!r}, 'phase') "
                        f"matched element '{elem.name}' "
                        f"({type(elem).__name__}) but the element "
                        "has no 'phase' slot — error NOT applied.",
                        stacklevel=2,
                    )
                _log(elem, err, val, False, "no-slot")
        else:
            # Additive default for everything else (phase_offset,
            # frequency_offset, raw gradient, raw field, etc.)
            if hasattr(elem, err.parameter):
                setattr(elem, err.parameter,
                        getattr(elem, err.parameter) + val)
                _log(elem, err, val, True, "add")
            else:
                # Same warn-once contract as the voltage_rel branch
                # above: a matched-but-unsupported parameter must
                # never be a silent no-op (ERROR_CAV phase errors on
                # an element without a phase_offset slot used to be
                # dropped without a trace — tolerance studies ran
                # with zero RF-phase jitter).
                warned = getattr(self, "_warned_skipped", None)
                if warned is None:
                    warned = self._warned_skipped = set()
                key = (err.pattern, err.parameter,
                       type(elem).__name__)
                if key not in warned:
                    warned.add(key)
                    warnings.warn(
                        f"ErrorDef({err.pattern!r}, "
                        f"{err.parameter!r}) matched element "
                        f"'{elem.name}' "
                        f"({type(elem).__name__}) but the element "
                        f"has no '{err.parameter}' slot — error "
                        f"NOT applied.",
                        stacklevel=2,
                    )
                _log(elem, err, val, False, "no-slot")

    def _apply_errors(self, seed):
        """Create a copy of the lattice with random errors applied."""
        # Tuple entropy → a stream disjoint from the beam-error stream
        # ((seed, 2)) and from create_beam's integer seeds.
        rng = np.random.default_rng(
            (int(seed), 0) if self._rng_key == (0,)
            else (int(seed), _STREAM_TAG) + self._rng_key + (0,))
        lattice_copy = copy.deepcopy(self.lattice)
        applied_log: list[dict] = []

        def _log(elem, err, val, applied, how):
            applied_log.append({
                "element": elem.name, "class": type(elem).__name__,
                "parameter": err.parameter, "value": float(val),
                "applied": bool(applied), "mode": how,
            })

        for err in self._errors:
            if (err.circuits is not None or err.into_slot) \
                    and err.distribution in ("gaussian", "uniform"):
                # Circuit-aware / slot-writing definitions: draw for all
                # matched elements first, relate the draws by circuit,
                # then apply.  Definitions without these options never
                # reach this branch — their RNG stream and application
                # below are unchanged bit for bit.
                matched = [e for e in lattice_copy.elements
                           if fnmatch.fnmatch(e.name, err.pattern)
                           and (err.element_kind is None
                                or element_kind(e) == err.element_kind)]
                for elem, val in zip(matched, self._grouped_values(err, matched, rng)):
                    if val is not None:
                        self._apply_draw(elem, err, val, _log)
                continue
            for elem in lattice_copy.elements:
                if not fnmatch.fnmatch(elem.name, err.pattern):
                    continue
                # Optional element-type gate — BEFORE the draw, so
                # filtered elements consume no RNG entropy (and a study
                # without kinds keeps its historical draw stream).
                if (err.element_kind is not None
                        and element_kind(elem) != err.element_kind):
                    continue
                # Generate random error value
                if err.distribution == "gaussian":
                    val = _draw_truncated_gaussian(rng, err.sigma,
                                                   err.cutoff)
                elif err.distribution == "uniform":
                    val = float(rng.uniform(-err.half_width, err.half_width))
                else:
                    # Unknown distribution: draws nothing.  add_error
                    # rejects these at registration; this guards defs
                    # constructed directly / absorbed from elsewhere.
                    warned = getattr(self, "_warned_skipped", None)
                    if warned is None:
                        warned = self._warned_skipped = set()
                    key = (err.pattern, "distribution", err.distribution)
                    if key not in warned:
                        warned.add(key)
                        warnings.warn(
                            f"ErrorDef({err.pattern!r}, "
                            f"{err.parameter!r}): unknown distribution "
                            f"{err.distribution!r} — NO draws made.",
                            stacklevel=2,
                        )
                    _log(elem, err, 0.0, False, "bad-distribution")
                    continue

                self._apply_draw(elem, err, val, _log)

        self.applied_log[int(seed)] = applied_log
        return lattice_copy
