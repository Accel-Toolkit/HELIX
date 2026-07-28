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
                 base_seed: int = 0):
        self.lattice = lattice
        self.beam_config = beam_config
        self.n_seeds = n_seeds
        self.sc_config = sc_config
        # Offset added to every per-seed RNG draw so two studies with the
        # same configuration but different ``base_seed`` produce
        # statistically independent ensembles (useful for chained Monte
        # Carlo runs where the user wants to extend an existing study).
        self.base_seed = int(base_seed)
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
                  element_kind=None):
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
        """
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
        ))

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
            for elem in self.lattice.elements:
                if not fnmatch.fnmatch(elem.name, err.pattern):
                    continue
                if (err.element_kind is not None
                        and element_kind(elem) != err.element_kind):
                    continue
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
            rows.append({
                "index": i, "pattern": err.pattern,
                "element_kind": err.element_kind,
                "parameter": err.parameter,
                "matched": matched, "applicable": applicable,
                "skipped": skipped,
                "inert": err.parameter in self._INERT_ALIGN,
                "bad_distribution": bad_dist,
            })
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
        return results

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
        rng = np.random.default_rng((int(seed), 2))
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

    def _apply_errors(self, seed):
        """Create a copy of the lattice with random errors applied."""
        # Tuple entropy → a stream disjoint from the beam-error stream
        # ((seed, 2)) and from create_beam's integer seeds.
        rng = np.random.default_rng((int(seed), 0))
        lattice_copy = copy.deepcopy(self.lattice)
        applied_log: list[dict] = []

        def _log(elem, err, val, applied, how):
            applied_log.append({
                "element": elem.name, "class": type(elem).__name__,
                "parameter": err.parameter, "value": float(val),
                "applied": bool(applied), "mode": how,
            })

        for err in self._errors:
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

                # Apply error
                if err.parameter.endswith("_rel"):
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

        self.applied_log[int(seed)] = applied_log
        return lattice_copy
