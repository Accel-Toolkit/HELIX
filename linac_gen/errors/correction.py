# linac_gen/errors/correction.py
"""Orbit correction routines: one-to-one and SVD-based global correction.

Two public entry points:

* :func:`apply_correction` — algorithm-level driver.  Pick a method
  (``"one_to_one"`` / ``"svd"``), pass element-list or glob-pattern
  selectors for steerers and BPMs, get the kicks that flatten the orbit.
  Honours per-steerer ``vmax`` clips, iterates until the residual RMS
  BPM reading falls below ``tol_mm`` (or every steerer saturates).

* :func:`run_correction_from_lattice` — TraceWin-card-driven driver.
  Walks the lattice for ``ADJUST_STEERER`` / ``ADJUST_STEERER_BX`` /
  ``ADJUST_STEERER_BY`` cards, resolves each card's ``diag_n`` to the
  N-th ``is_bpm`` marker and pairs it with the next ``Steerer`` after
  the card, picks one-to-one or SVD based on cardinality, and calls
  :func:`apply_correction`.

The orbit-correction algorithm matches TraceWin's behaviour: when
``#steerers == #BPMs`` the system is solved by direct (one-to-one)
matching; otherwise an SVD-truncated pseudo-inverse is built from a
finite-difference response matrix.  ``vmax`` from the
``ADJUST_STEERER`` card is interpreted as an integrated kick (T·m) on
the partner ``Steerer`` since HELIX steerers are zero-length thin
kicks.
"""
import fnmatch
import logging
from typing import Iterable, Optional

import numpy as np

from linac_gen.tracking.tracker import Tracker
from linac_gen.units import SVD_RCOND_DEFAULT

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_correction(lattice, beam_factory, bpm_pattern="BPM_*",
                     steerer_pattern="STEER_*", method="one_to_one",
                     bpm_noise=0.0, rcond: Optional[float] = None,
                     n_iter: int = 1, tol_mm: float = 0.05,
                     vmax: Optional[object] = None,
                     steerers: Optional[Iterable] = None,
                     bpms: Optional[Iterable] = None,
                     planes: Optional[dict] = None,
                     paired_bpms: bool = False,
                     noise_seed: int = 0,
                     history: bool = False,
                     should_stop=None,
                     targets=None,
                     reading_backend: str = "mp",
                     beam_config=None):
    """Apply orbit correction to *lattice* in-place.

    ``targets`` selects the per-BPM set-points (see
    :func:`_resolve_targets`): ``None`` steers every BPM to zero —
    bit-identical to the historical behaviour; ``"deck"`` uses the
    ``DIAG_POSITION`` operands (file overrides win); a dict gives
    explicit ``{name: (tx, ty)}`` values in mm.  Readings are ALWAYS
    computed by tracking — targets only shift the set-point.

    ``reading_backend`` selects HOW readings are computed:
    ``"mp"`` (default, bit-compatible) tracks a fresh multi-particle
    beam from ``beam_factory`` per reading; ``"envelope"`` runs the
    envelope solver — whose results now carry the beam centroid — and
    needs ``beam_config`` (a BeamConfig) instead of particles.
    Envelope readings are deterministic (no sampling noise), so
    ``n_iter=1`` usually suffices; ``bpm_noise`` still applies
    (additive, models the diagnostic not the beam).

    Parameters
    ----------
    lattice : Lattice
        The lattice to correct.  Steerer ``bx_l``/``by_l`` are modified in-place.
    beam_factory : callable
        Zero-argument callable returning a fresh :class:`~linac_gen.core.beam.Beam`.
        Called once per tracking pass.
    bpm_pattern : str
        Glob pattern selecting BPM elements by name.  Ignored when
        ``bpms`` is given.  When the pattern matches nothing, falls back
        to every ``Marker`` with ``is_bpm=True``.
    steerer_pattern : str
        Glob pattern selecting :class:`~linac_gen.elements.steerer.Steerer`
        elements.  Ignored when ``steerers`` is given.
    method : {"one_to_one", "svd"}
        Correction algorithm.
    bpm_noise : float
        Optional Gaussian noise (mm) added to BPM readings.
    rcond : float, optional
        (SVD method only) Singular values below ``rcond * s_max`` are
        treated as zero.  Defaults to
        :data:`linac_gen.units.SVD_RCOND_DEFAULT`.
    n_iter : int
        Number of correction passes.  ``n_iter=1`` is single-shot
        (legacy behaviour).  Each pass adds an incremental kick so
        non-linear residual shrinks geometrically.
    tol_mm : float
        Convergence tolerance — stop when RMS over all BPM readings
        (combined x and y) falls below this value.
    vmax : float | dict | None
        Per-steerer cap on ``|bx_l|`` and ``|by_l|`` in T·m.  Scalar
        applies to every steerer; ``dict[steerer_name, float]`` is
        per-element; ``None`` disables clipping.
    steerers, bpms : iterable of elements, optional
        Explicit element-list overrides.  When both are given, the glob
        patterns are bypassed entirely — used by
        :func:`run_correction_from_lattice` to honour TraceWin
        ``ADJUST_STEERER`` resolution.
    planes : dict[str, tuple[bool, bool]], optional
        Per-steerer plane authorization ``{name: (allow_x, allow_y)}``.
        ``allow_x`` gates the X-plane branch (the ``by_l`` knob),
        ``allow_y`` the Y-plane branch (``bx_l``).  Steerers absent from
        the dict — or ``planes=None`` — are corrected in both planes
        (legacy behaviour).  Set by ``run_correction_from_lattice`` from
        ``ADJUST_STEERER_BX`` / ``_BY`` cards.
    history : bool
        Return per-iteration diagnostics alongside the kicks dict.
    should_stop : callable, optional
        Polled at the top of each correction pass; when it returns True
        an :class:`~linac_gen.core.cancelled.OperationCancelled` is
        raised.  NOTE: passes already applied have mutated the steerers
        in place — callers wanting all-or-nothing semantics must operate
        on a copy of the lattice.

    Returns
    -------
    dict
        ``{steerer_name: {"bx_l": float, "by_l": float}}`` of applied
        settings.  When ``history=True``, returns
        ``(kicks, [{"iter": k, "rms_orbit_mm": x, "n_saturated": m}, ...])``.
    """
    # ----- Resolve steerers / BPMs --------------------------------------
    if steerers is None:
        steerers = [e for e in lattice.elements
                    if fnmatch.fnmatch(e.name, steerer_pattern)]
    else:
        steerers = list(steerers)
    if bpms is None:
        bpms = [e for e in lattice.elements
                if fnmatch.fnmatch(e.name, bpm_pattern)]
        if not bpms:
            # Fallback: any Marker with the is_bpm flag (parser sets this
            # for DIAG_POSITION / BPM cards).
            bpms = [e for e in lattice.elements
                    if getattr(e, "is_bpm", False)]
    else:
        bpms = list(bpms)

    if not steerers or not bpms:
        return ({}, []) if history else {}

    resolved_targets = _resolve_targets(bpms, targets)

    # ----- Reading backend ------------------------------------------------
    # One zero-arg ``reader()`` producing a results object with
    # ``centroid`` + ``element_exit_idx``; rigidity hoisted so the
    # solvers don't burn a tracking pass each just for brho.  NOTE the
    # two backends deliberately differ: mp keeps the legacy EXIT
    # rigidity (ref tracked through the lattice — do not "fix" to
    # entrance, it would break bit-compat), envelope uses the entrance
    # rigidity.  brho cancels exactly in the kick algebra (probe and
    # apply share it), so the correction is rigidity-choice-invariant.
    if reading_backend == "envelope":
        if beam_config is None:
            raise ValueError(
                "reading_backend='envelope' requires beam_config= (a "
                "BeamConfig): envelope readings are built from Twiss, "
                "not particles")
        from linac_gen.cli.common import build_ref, run_envelope_sim

        def reader():
            return run_envelope_sim(lattice, beam_config)

        brho0 = build_ref(beam_config).brho
    elif reading_backend == "mp":
        def reader():
            return Tracker(lattice, beam_factory()).run()

        # Legacy brho semantics: measured on a tracked beam's ref (the
        # run advances it; rigidity of a static line is s-independent
        # only for unaccelerated beams — keep the historical value).
        _beam_ref = beam_factory()
        Tracker(lattice, _beam_ref).run()
        brho0 = _beam_ref.ref.brho
    else:
        raise ValueError(
            f"Unknown reading_backend: '{reading_backend}' "
            "(choose 'mp' or 'envelope')")

    # ----- Iteration loop -----------------------------------------------
    # noise_seed: BPM-noise stream.  ErrorStudy passes the per-seed value —
    # a fixed default_rng(0) made noise 100 % correlated across every
    # Monte Carlo seed, biasing residual-orbit statistics.
    rng = np.random.default_rng(noise_seed)
    iter_history: list[dict] = []
    prev_rms: Optional[float] = None
    last_corrections: dict = {}

    for k in range(max(1, int(n_iter))):
        if should_stop is not None and should_stop():
            from linac_gen.core.cancelled import OperationCancelled
            raise OperationCancelled("orbit correction cancelled")
        if method == "one_to_one":
            last_corrections = _one_to_one(
                lattice, reader, steerers, bpms, brho0,
                bpm_noise=bpm_noise, rng=rng,
                planes=planes, paired=paired_bpms,
                targets=resolved_targets,
            )
        elif method == "svd":
            last_corrections = _svd_correction(
                lattice, reader, steerers, bpms, brho0,
                bpm_noise=bpm_noise,
                rng=rng, rcond=rcond, planes=planes,
                targets=resolved_targets,
            )
        else:
            raise ValueError(
                f"Unknown method: '{method}'. Choose 'one_to_one' or 'svd'."
            )

        # vmax clipping → record saturation
        n_saturated = _clip_all_to_vmax(steerers, vmax)
        # Refresh the post-clip values for the steerers that the inner
        # method actually corrected (preserving its keying — e.g.
        # `_one_to_one` excludes steerers with no downstream BPM).
        for s in steerers:
            if s.name in last_corrections:
                last_corrections[s.name] = {
                    "bx_l": float(s.bx_l),
                    "by_l": float(s.by_l),
                }

        rms = _orbit_rms_mm(lattice, reader, bpms, bpm_noise, rng,
                            targets=resolved_targets)
        iter_history.append({
            "iter": k + 1,
            "rms_orbit_mm": rms,
            "n_saturated": n_saturated,
        })

        # Convergence: residual below tolerance.
        if rms < tol_mm:
            break
        # Saturation stall: every steerer at vmax AND orbit no longer
        # improving by ≥10 % between iterations.
        if (n_saturated == len(steerers) and prev_rms is not None
                and rms > 0.9 * prev_rms):
            _log.info(
                "Correction stalled at iter %d (all %d steerers saturated)",
                k + 1, n_saturated,
            )
            break
        prev_rms = rms

    return (last_corrections, iter_history) if history else last_corrections


# ---------------------------------------------------------------------------
# Card-driven driver
# ---------------------------------------------------------------------------

def run_correction_from_lattice(lattice, beam_factory, *,
                                override_method: Optional[str] = None,
                                n_iter: int = 5, tol_mm: float = 0.05,
                                bpm_noise: float = 0.0,
                                rcond: Optional[float] = None,
                                noise_seed: int = 0,
                                history: bool = False,
                                should_stop=None,
                                targets=None,
                                reading_backend: str = "mp",
                                beam_config=None) -> dict:
    """Run orbit correction driven by TraceWin ``ADJUST_STEERER`` cards.

    Walks ``lattice.elements`` for ``AdjustSteerer`` /
    ``AdjustSteererBx`` / ``AdjustSteererBy`` cards, resolves each
    card's ``diag_n`` to the N-th ``is_bpm`` marker (1-indexed) and
    its partner steerer as the next ``Steerer`` strictly after the
    card (TraceWin convention; can be overridden via
    ``card.target_name``).  Picks one-to-one when ``#cards == #BPMs``
    and the pairing is geometrically clean, otherwise SVD.  Honours
    each card's ``vmax`` as a per-steerer T·m clip.

    Returns
    -------
    dict
        ``{"kicks": dict, "history": list, "method": str, "n_pairs": int}``.
    """
    from linac_gen.elements.lattice_commands import (
        AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
    )
    from linac_gen.elements.steerer import Steerer

    elements = list(lattice.elements)

    # Build the BPM table: 1-indexed list of is_bpm markers in order.
    bpm_list = [e for e in elements if getattr(e, "is_bpm", False)]

    # Collect ADJUST_STEERER cards and resolve to (steerer, plane_x, plane_y, vmax).
    pairs: list[dict] = []
    for idx, e in enumerate(elements):
        if not isinstance(e, (AdjustSteerer, AdjustSteererBx, AdjustSteererBy)):
            continue

        # Resolve partner steerer.
        target_name = getattr(e, "target_name", None)
        steerer = None
        if target_name:
            for cand in elements:
                if isinstance(cand, Steerer) and cand.name == target_name:
                    steerer = cand
                    break
        if steerer is None:
            for cand in elements[idx + 1:]:
                if isinstance(cand, Steerer):
                    steerer = cand
                    break
        if steerer is None:
            _log.warning(
                "ADJUST_STEERER %s: no partner Steerer found, skipping",
                getattr(e, "name", "?"),
            )
            continue
        if getattr(steerer, "elec", False):
            # The whole driver hard-codes the MAGNETIC model: by_l→x′ /
            # bx_l→y′ plane mapping, brho-based kick↔knob conversion
            # and a T·m vmax clip.  An electric steerer (same-plane
            # volts knobs, 1/(βc·Bρ) response) would get the wrong
            # plane AND a ~βc-times-wrong magnitude — refuse loudly.
            raise ValueError(
                f"ADJUST_STEERER {getattr(e, 'name', '?')}: partner "
                f"steerer '{steerer.name}' is ELECTRIC (elec=1) — "
                "orbit correction only supports magnetic steerers "
                "(magnetic plane mapping and T·m unit conversions); "
                "use a magnetic corrector or correct manually."
            )

        # Resolve diag_n → BPM (1-indexed).
        diag_n = int(getattr(e, "diag_n", 0))
        if diag_n < 1 or diag_n > len(bpm_list):
            _log.warning(
                "ADJUST_STEERER %s: diag_n=%d out of range (1..%d), skipping",
                getattr(e, "name", "?"), diag_n, len(bpm_list),
            )
            continue
        bpm = bpm_list[diag_n - 1]

        # Plane semantics follow the matcher (variables.py): an
        # ADJUST_STEERER_BX card authorizes the **bx_l** knob — a Bx field
        # kicks y′ — so it permits Y-plane correction only; _BY authorizes
        # by_l → x′ → X-plane only; plain ADJUST_STEERER permits both.
        # (The previous assignment was inverted, but harmlessly so — the
        # masks were computed and never used.  They are enforced now.)
        plane_x = not isinstance(e, AdjustSteererBx)
        plane_y = not isinstance(e, AdjustSteererBy)
        vmax = float(getattr(e, "vmax", 0.0))

        pairs.append({
            "card": e, "steerer": steerer, "bpm": bpm,
            "plane_x": plane_x, "plane_y": plane_y, "vmax": vmax,
        })

    if not pairs:
        return {"kicks": {}, "history": [], "method": "none", "n_pairs": 0}

    # Method: one_to_one when #cards == #BPMs and pairing is clean (each
    # card pairs with a distinct BPM), else SVD.
    distinct_bpms = {id(p["bpm"]) for p in pairs}
    method = override_method
    if method is None:
        if len(distinct_bpms) == len(pairs) == len(bpm_list):
            method = "one_to_one"
        else:
            method = "svd"

    # Build vmax dict and element lists.
    vmax_dict: dict[str, float] = {
        p["steerer"].name: p["vmax"] for p in pairs if p["vmax"] > 0.0
    }
    steerers = [p["steerer"] for p in pairs]
    bpms = [p["bpm"] for p in pairs] if method == "one_to_one" else bpm_list

    # Per-steerer plane authorization from the cards.  Multiple cards on
    # the same steerer compose with OR (a BX card plus a BY card on one
    # steerer authorizes both planes).
    planes: dict[str, tuple[bool, bool]] = {}
    for p in pairs:
        name = p["steerer"].name
        px, py = planes.get(name, (False, False))
        planes[name] = (px or p["plane_x"], py or p["plane_y"])

    out = apply_correction(
        lattice, beam_factory, method=method,
        bpm_noise=bpm_noise, rcond=rcond,
        n_iter=n_iter, tol_mm=tol_mm,
        vmax=vmax_dict if vmax_dict else None,
        steerers=steerers, bpms=bpms,
        planes=planes,
        # For one_to_one the bpms list above is card-resolved 1:1 with
        # steerers — honour that pairing instead of re-pairing
        # nearest-downstream inside the algorithm.
        paired_bpms=(method == "one_to_one"),
        noise_seed=noise_seed,
        history=True,
        should_stop=should_stop,
        targets=targets,
        reading_backend=reading_backend,
        beam_config=beam_config,
    )
    kicks, hist = out  # history=True ensures tuple
    return {
        "kicks": kicks,
        "history": hist,
        "method": method,
        "n_pairs": len(pairs),
    }


def apply_diagnostic_matching(lattice, beam_factory, *,
                              families=None,
                              targets="deck",
                              override_method: Optional[str] = None,
                              n_iter: int = 5, tol_mm: float = 0.05,
                              bpm_noise: float = 0.0,
                              rcond: Optional[float] = None,
                              noise_seed: int = 0,
                              should_stop=None,
                              reading_backend: str = "mp",
                              beam_config=None) -> dict:
    """Steer the beam onto ``DIAG_POSITION`` targets (TraceWin
    "matched with diagnostics", steerer special case).

    Selects the ``is_bpm`` markers and the ``Steerer`` elements whose
    immediately preceding plain ``ADJUST N v`` card carries a matching
    family number N (v=1 → the ``bx_l`` knob / y plane, v=2 → ``by_l``
    / x plane), then solves the linear system by SVD so every BPM
    reading — always computed by MP tracking — lands on its target.

    This is the direct-inversion special case TraceWin itself uses when
    the correctors are pure steerers; decks whose ADJUST cards bind
    quad gradients (fnalscl) need the nonlinear matching route
    (``linac_gen.matching.match`` with ``cost_solver="mp"``) instead —
    quad variables are deliberately NOT consumed here.

    Parameters mirror :func:`run_correction_from_lattice`;
    ``families=None`` accepts every family that has steerer ADJUSTs,
    else pass an iterable of family numbers.  ``targets`` as in
    :func:`apply_correction` (default ``"deck"``).
    """
    from linac_gen.elements.lattice_commands import Adjust, LatticeCommand
    from linac_gen.elements.steerer import Steerer

    fam_filter = None if families is None else {int(f) for f in families}

    def _fam_of(card):
        try:
            return int(card.target.strip())
        except (ValueError, AttributeError):
            return None

    # Steerers tagged by a plain ADJUST family card directly before them
    # (possibly several cards deep: two consecutive ADJUSTs both tag the
    # next element).  planes: v=1 → bx_l knob → y plane; v=2 → by_l → x.
    steerers, planes = [], {}
    elements = lattice.elements
    for i, e in enumerate(elements):
        if not isinstance(e, Adjust):
            continue
        fam = _fam_of(e)
        if fam is None or (fam_filter is not None and fam not in fam_filter):
            continue
        nxt = next((el for el in elements[i + 1:]
                    if not isinstance(el, LatticeCommand)), None)
        if not isinstance(nxt, Steerer) or e.param_idx not in (1, 2):
            continue
        if nxt not in steerers:
            steerers.append(nxt)
            planes[nxt.name] = (False, False)
        px, py = planes[nxt.name]
        if e.param_idx == 2:
            px = True          # by_l → horizontal
        else:
            py = True          # bx_l → vertical
        planes[nxt.name] = (px, py)

    # families=None means "every family that has steerer ADJUSTs" — NOT
    # every BPM in the lattice.  Deriving the filter from the steerers
    # actually collected keeps this route consistent with the matching
    # route's passive-monitor semantics (fnalscl family 12 is excluded
    # here exactly as collect_constraints excludes it).
    if fam_filter is None:
        fam_filter = set()
        for i, e in enumerate(elements):
            if isinstance(e, Adjust) and _fam_of(e) is not None:
                nxt = next((el for el in elements[i + 1:]
                            if not isinstance(el, LatticeCommand)), None)
                if isinstance(nxt, Steerer) and e.param_idx in (1, 2):
                    fam_filter.add(_fam_of(e))

    bpms = [m for m in elements
            if getattr(m, "is_bpm", False)
            and getattr(m, "diag_family", None) in fam_filter]

    if not steerers or not bpms:
        return {"kicks": {}, "history": [], "method": None,
                "n_steerers": len(steerers), "n_bpms": len(bpms)}

    out = apply_correction(
        lattice, beam_factory,
        method=override_method or "svd",
        bpm_noise=bpm_noise, rcond=rcond,
        n_iter=n_iter, tol_mm=tol_mm,
        steerers=steerers, bpms=bpms, planes=planes,
        noise_seed=noise_seed, history=True,
        should_stop=should_stop, targets=targets,
        reading_backend=reading_backend, beam_config=beam_config,
    )
    kicks, hist = out
    # ``history`` already rides inside the dict — one return shape only.
    return {"kicks": kicks, "history": hist,
            "method": override_method or "svd",
            "n_steerers": len(steerers), "n_bpms": len(bpms)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elem_index(lattice, elem):
    for i, e in enumerate(lattice.elements):
        if e is elem:
            return i
    raise KeyError(elem.name)


def _resolve_vmax_for(steerer, vmax) -> Optional[float]:
    if vmax is None:
        return None
    if isinstance(vmax, dict):
        return vmax.get(steerer.name)
    try:
        v = float(vmax)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _clip_all_to_vmax(steerers, vmax) -> int:
    """Clip every steerer's bx_l/by_l to ±vmax in-place.

    Returns the number of steerers where at least one plane saturated.
    """
    if vmax is None:
        return 0
    n_saturated = 0
    for s in steerers:
        cap = _resolve_vmax_for(s, vmax)
        if cap is None or cap <= 0:
            continue
        sat = False
        if abs(s.bx_l) > cap:
            s.bx_l = float(np.sign(s.bx_l) * cap)
            sat = True
        if abs(s.by_l) > cap:
            s.by_l = float(np.sign(s.by_l) * cap)
            sat = True
        if sat:
            n_saturated += 1
    return n_saturated


def _bpm_row(lattice, rec, bpm) -> int:
    """Recorder row for a BPM's exit state.

    Prefer the recorder's own ``element_exit_idx`` map (correct with
    substep recording on); fall back to the historical ``index + 1``
    convention (row 0 = INPUT) for minimal recorders.
    """
    ei = _elem_index(lattice, bpm)
    exit_idx = getattr(rec, "element_exit_idx", None)
    if exit_idx and ei < len(exit_idx):
        return int(exit_idx[ei])
    return ei + 1


def _resolve_targets(bpms, targets):
    """Normalize the ``targets`` option to ``{bpm_name: (tx, ty)}`` (mm).

    ``None`` in a plane = that plane is excluded from correction
    (TraceWin's 1e50 sentinel).

    * ``targets=None``  → every BPM steers to ``(0.0, 0.0)`` — bit-
      compatible with the historical flatten-to-zero behaviour.
    * ``targets="deck"`` → per-marker: runtime file overrides
      (``x_target_override_mm`` …) take precedence over the deck's
      ``DIAG_POSITION`` operands; a marker with no target at all steers
      to zero; a sentinel-disabled plane stays excluded.
    * ``targets=dict``  → explicit ``{name: (tx, ty)}``; missing names
      steer to zero.
    """
    out = {}
    for b in bpms:
        if targets is None:
            out[b.name] = (0.0, 0.0)
        elif targets == "deck":
            ov = getattr(b, "diag_target_override", None)
            if ov is not None:
                # Override PRESENCE wins — (None, None) genuinely frees
                # both planes instead of resurrecting the deck targets.
                tx, ty = ov[0], ov[1]
            else:
                tx = getattr(b, "x_target_mm", None)
                ty = getattr(b, "y_target_mm", None)
                if tx is None and ty is None:
                    tx, ty = 0.0, 0.0      # bare BPM — flatten as before
            out[b.name] = (tx, ty)
        else:
            out[b.name] = tuple(targets.get(b.name, (0.0, 0.0)))
    return out


def _orbit_rms_mm(lattice, reader, bpms, bpm_noise, rng,
                  targets=None) -> float:
    """RMS over both planes of all BPM (reading − target) errors (mm).

    ``reader`` is the zero-arg backend callable (MP tracking or
    envelope) returning a results object with ``centroid``.
    ``targets`` is the resolved ``{name: (tx, ty)}`` dict (``None`` ≡
    all-zero targets); an excluded (``None``) plane contributes nothing.
    """
    if not bpms:
        return 0.0
    tgt = targets or {}
    rec = reader()
    vals: list[float] = []
    for b in bpms:
        bi = _bpm_row(lattice, rec, b)
        if bi < len(rec.centroid):
            c = np.array(rec.centroid[bi])
            cx, cy = float(c[0]), float(c[2])
            if bpm_noise > 0:
                cx += float(rng.normal(0, bpm_noise))
                cy += float(rng.normal(0, bpm_noise))
            tx, ty = tgt.get(b.name, (0.0, 0.0))
            if tx is not None:
                vals.append(cx - tx)
            if ty is not None:
                vals.append(cy - ty)
    if not vals:
        return 0.0
    return float(np.sqrt(np.mean(np.square(vals))))


def _unit_kick_response(lattice, reader, steer, bpms, plane,
                        brho, delta_kick=1e-2):
    """Measure BPM response (mm) to a unit kick (mrad) from *steer* in *plane*.

    Returns dict: bpm_name -> response_mm_per_mrad
    """
    # Save original setting
    if plane == "x":
        original = steer.by_l
        steer.by_l += delta_kick * brho / 1e3   # convert mrad → T.m
    else:
        original = steer.bx_l
        steer.bx_l += delta_kick * brho / 1e3

    rec = reader()
    responses = {}
    for b in bpms:
        rec_idx = _bpm_row(lattice, rec, b)
        if rec_idx < len(rec.centroid):
            c = np.array(rec.centroid[rec_idx])
            responses[b.name] = float(c[0] if plane == "x" else c[2])

    # Restore
    if plane == "x":
        steer.by_l = original
    else:
        steer.bx_l = original

    return responses


# ---------------------------------------------------------------------------
# One-to-one correction
# ---------------------------------------------------------------------------

def _one_to_one(lattice, reader, steerers, bpms, brho,
                bpm_noise=0.0, rng=None, planes=None, paired=False,
                targets=None):
    """One-to-one orbit correction.

    Iterate over steerers; for each, find the nearest downstream BPM and
    apply a kick to drive its reading to the BPM's target (zero when no
    target).  ``planes`` (optional) gates the per-steerer X (``by_l``) /
    Y (``bx_l``) branches; an excluded (``None``) target plane is
    skipped like a masked plane.  ``paired=True`` (set by the card
    driver) takes ``zip(steerers, bpms)`` as the authoritative pairing
    instead of re-pairing nearest-downstream — an ``ADJUST_STEERER``
    card may legitimately pair a steerer with a farther BPM when the
    adjacent one is served by another corrector.
    """
    rng = rng or np.random.default_rng(0)
    tgt = targets or {}
    corrections = {}

    elem_idx = {e.name: _elem_index(lattice, e) for e in lattice.elements}

    if paired and len(bpms) == len(steerers):
        # Card-resolved pairing — honour it verbatim.
        pairs = list(zip(steerers, bpms))
    else:
        # Pair steerer → nearest downstream BPM
        pairs = []
        for steer in steerers:
            si = elem_idx[steer.name]
            downstream_bpms = [b for b in bpms if elem_idx[b.name] > si]
            if downstream_bpms:
                nearest = min(downstream_bpms,
                              key=lambda b: elem_idx[b.name])
                pairs.append((steer, nearest))

    if not pairs:
        return {}

    delta_kick = 1e-2  # mrad (used for response estimation)

    for steer, bpm in pairs:
        allow_x, allow_y = (planes or {}).get(steer.name, (True, True))
        tx, ty = tgt.get(bpm.name, (0.0, 0.0))
        # An excluded target plane behaves like a masked plane.
        allow_x = allow_x and tx is not None
        allow_y = allow_y and ty is not None
        # Fresh reading of the current orbit
        rec0 = reader()
        bpm_idx = _bpm_row(lattice, rec0, bpm)
        if bpm_idx >= len(rec0.centroid):
            continue
        c0 = np.array(rec0.centroid[bpm_idx])
        reading_x = float(c0[0])
        reading_y = float(c0[2])
        if bpm_noise > 0:
            reading_x += float(rng.normal(0, bpm_noise))
            reading_y += float(rng.normal(0, bpm_noise))

        # Estimate response of this BPM to a kick from this steerer.
        # Masked planes skip both the response measurement (two tracking
        # passes each) and the kick below.
        if allow_x:
            resp_x = _unit_kick_response(lattice, reader, steer,
                                         [bpm], "x", brho, delta_kick)
            r0_x = _unit_kick_response(lattice, reader, steer,
                                       [bpm], "x", brho, 0.0)
            R_x = (resp_x.get(bpm.name, 0.0)
                   - r0_x.get(bpm.name, 0.0)) / delta_kick
        else:
            R_x = 0.0
        if allow_y:
            resp_y = _unit_kick_response(lattice, reader, steer,
                                         [bpm], "y", brho, delta_kick)
            r0_y = _unit_kick_response(lattice, reader, steer,
                                       [bpm], "y", brho, 0.0)
            R_y = (resp_y.get(bpm.name, 0.0)
                   - r0_y.get(bpm.name, 0.0)) / delta_kick
        else:
            R_y = 0.0

        # Apply correction: kick to drive the reading onto the target
        # (target 0.0 = the historical flatten behaviour, bit-identical:
        # subtracting a literal 0.0 changes nothing).
        if allow_x and abs(R_x) > 1e-12:
            correction_xp = -(reading_x - tx) / R_x  # mrad
            steer.by_l += correction_xp * brho / 1e3
        if allow_y and abs(R_y) > 1e-12:
            correction_yp = -(reading_y - ty) / R_y  # mrad
            steer.bx_l += correction_yp * brho / 1e3

        corrections[steer.name] = {"bx_l": float(steer.bx_l), "by_l": float(steer.by_l)}

    return corrections


# ---------------------------------------------------------------------------
# SVD-based global correction
# ---------------------------------------------------------------------------

def _svd_correction(lattice, reader, steerers, bpms, brho,
                    bpm_noise=0.0, rng=None, rcond=None, planes=None,
                    targets=None):
    """Global SVD-based orbit correction.

    Builds a response matrix R[bpm, steerer] then inverts via truncated SVD.
    For simplicity the x and y planes are treated independently.
    ``planes`` (optional) gates the per-steerer X (``by_l``) / Y (``bx_l``)
    columns: a masked plane's response column stays zero and its kick is
    forced to zero after the solve.

    ``targets`` (resolved ``{name: (tx, ty)}``): the response matrices are
    built from RAW orbit differences — targets enter ONLY the solve's
    right-hand side (``err = orbit0 − t``).  An excluded (``None``)
    target plane masks BOTH its error entry and its R row: a zero error
    with a live row would wrongly constrain the kicks to "don't move
    this BPM", but an unconstrained plane must stay free.
    """
    rng = rng or np.random.default_rng(0)
    tgt = targets or {}
    if rcond is None:
        rcond = SVD_RCOND_DEFAULT
    n_bpm = len(bpms)
    n_steer = len(steerers)

    # Measure initial orbit
    rec0 = reader()
    orbit0_x = np.zeros(n_bpm)
    orbit0_y = np.zeros(n_bpm)
    for k, b in enumerate(bpms):
        bi = _bpm_row(lattice, rec0, b)
        if bi < len(rec0.centroid):
            c = np.array(rec0.centroid[bi])
            orbit0_x[k] = float(c[0])
            orbit0_y[k] = float(c[2])

    if bpm_noise > 0:
        orbit0_x += rng.normal(0, bpm_noise, n_bpm)
        orbit0_y += rng.normal(0, bpm_noise, n_bpm)

    delta_kick = 1e-2  # mrad

    # Build response matrices R_x and R_y: shape (n_bpm, n_steer)
    R_x = np.zeros((n_bpm, n_steer))
    R_y = np.zeros((n_bpm, n_steer))

    for j, steer in enumerate(steerers):
        allow_x, allow_y = (planes or {}).get(steer.name, (True, True))
        # x plane: by_l kick (skipped when the plane is masked — the
        # response column stays zero and the kick is zeroed after solve)
        if allow_x:
            steer.by_l += delta_kick * brho / 1e3
            rec_j = reader()
            for k, b in enumerate(bpms):
                bi = _bpm_row(lattice, rec_j, b)
                if bi < len(rec_j.centroid):
                    c = np.array(rec_j.centroid[bi])
                    R_x[k, j] = (float(c[0]) - orbit0_x[k]) / delta_kick
            steer.by_l -= delta_kick * brho / 1e3

        # y plane: bx_l kick
        if allow_y:
            steer.bx_l += delta_kick * brho / 1e3
            rec_j2 = reader()
            for k, b in enumerate(bpms):
                bi = _bpm_row(lattice, rec_j2, b)
                if bi < len(rec_j2.centroid):
                    c = np.array(rec_j2.centroid[bi])
                    R_y[k, j] = (float(c[2]) - orbit0_y[k]) / delta_kick
            steer.bx_l -= delta_kick * brho / 1e3

    # SVD inversion with truncation at rcond * s_max.
    def _svd_solve(R, orbit, plane_label):
        if np.allclose(R, 0) or np.allclose(orbit, 0):
            return np.zeros(R.shape[1])
        U, s_vals, Vt = np.linalg.svd(R, full_matrices=False)
        s_max = float(s_vals[0])
        threshold = rcond * s_max if s_max > 0 else 0.0
        n_kept = int(np.sum(s_vals > threshold))
        s_min = float(s_vals[-1])
        cond = (s_max / s_min) if s_min > 0 else float("inf")
        _log.debug(
            "SVD correction (%s): n_kept=%d/%d, cond=%.2e, rcond=%.1e",
            plane_label, n_kept, len(s_vals), cond, rcond,
        )
        # Masked target rows / plane columns produce exact-zero singular
        # values; guard the reciprocal so no RuntimeWarning fires.
        s_inv = np.zeros_like(s_vals)
        np.divide(1.0, s_vals, out=s_inv, where=s_vals > threshold)
        R_pinv = (Vt.T * s_inv) @ U.T
        return -R_pinv @ orbit

    # Targets enter ONLY here (RHS of the solve); excluded planes mask
    # both the error entry and the response row.
    err_x = orbit0_x.copy()
    err_y = orbit0_y.copy()
    for k, b in enumerate(bpms):
        tx, ty = tgt.get(b.name, (0.0, 0.0))
        if tx is None:
            err_x[k] = 0.0
            R_x[k, :] = 0.0
        else:
            err_x[k] -= tx
        if ty is None:
            err_y[k] = 0.0
            R_y[k, :] = 0.0
        else:
            err_y[k] -= ty

    kicks_x = _svd_solve(R_x, err_x, "x")  # mrad
    kicks_y = _svd_solve(R_y, err_y, "y")  # mrad

    corrections = {}
    for j, steer in enumerate(steerers):
        allow_x, allow_y = (planes or {}).get(steer.name, (True, True))
        if allow_x:
            steer.by_l += kicks_x[j] * brho / 1e3
        if allow_y:
            steer.bx_l += kicks_y[j] * brho / 1e3
        corrections[steer.name] = {"bx_l": float(steer.bx_l), "by_l": float(steer.by_l)}

    return corrections
