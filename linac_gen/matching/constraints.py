"""Optimiser constraints collected from ``SET_*`` lattice commands.

Each :class:`Constraint` produces a residual array (length ≥ 1) given an
:class:`linac_gen.tracking.envelope.EnvelopeResults` from the forward
simulation.  The matcher concatenates the per-constraint residuals into
one vector and feeds it to ``scipy.optimize.least_squares``.

Supported constraints
---------------------

* ``SET_TWISS family α_x β_x α_y β_y α_z β_z k…`` — selected by k-flag,
  evaluated at the **end of the lattice** (most common usage).  For
  finer placement, use a Marker-named family.
* ``SET_SIZE k σ_x σ_y σ_φ/z k2`` — equality constraint on σ, evaluated
  at the **end of the lattice** (same limitation as SET_TWISS).  A
  positive 4th operand is σ_φ in degrees, a NEGATIVE one is σ_z =
  |value| in mm (TW manual rule), converted via the local βλ.  ``k2``
  (include the beam centroid in the transverse sizes) is not modelled
  and warns at collect — sizes are about-centroid (the k2=0 form).
* ``SET_SIZE_MAX`` — one-sided bound; residual = ``max(0, σ - target)``
  per axis with k=1, evaluated over ``n_elems`` following elements.
* ``SET_SIZE_MIN`` — symmetric one-sided lower bound.
* ``SET_BEAM_PHASE_ADV`` — μ_x / μ_y / μ_z residual over the ``n_elems``
  non-command elements following the card, via the shared ∫ds/β
  integrator (``phase_advance._integrate_inverse_beta_split``) and the
  effective (z, z′) β for μ_z.
* ``SET_POSITION`` — live centroid constraint: both the MP recorder
  and envelope results carry the beam first moment, so the residual is
  (centroid − card values) at the card's own record row under either
  cost solver.

Unsupported constraints surface as a warning and a zero residual rather
than crashing — they preserve the matcher's continuity even if the user
attaches a not-yet-wired card.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from linac_gen.elements.lattice_commands import (
    Adjust, LatticeCommand,
    MinEmit4DGrowth, MinEmitGrowth, MinTransmission, SetAchromat,
    SetAdv, SetBeamPhaseAdv,
    SetKeOutMin, SetPosition, SetSeparation, SetSize, SetSizeMax,
    SetSizeMin, SetTwiss,
)


@dataclass
class Constraint:
    """One residual contribution.

    Attributes
    ----------
    label :
        Identifier for the report (e.g. ``"SET_TWISS:QUAD1"``).
    evaluator :
        Callable ``(EnvelopeResults, Lattice) -> np.ndarray``.  Returns
        a residual vector (``shape (n,)``); concatenated with peers.
    weight :
        Multiplied into the residual; comes from the TraceWin ``k`` arg.
    source :
        Originating ``LatticeCommand``.
    notes :
        Free-form descriptor for the report (e.g. "stub: not implemented").
    """

    label: str
    evaluator: Callable[..., np.ndarray]
    weight: float = 1.0
    source: Optional[LatticeCommand] = None
    notes: str = ""
    # Historical name; semantic today: "requires centroid-bearing
    # results".  Both the MP recorder AND EnvelopeResults carry the
    # beam centroid now, so position constraints fit under either cost
    # solver and the audit no longer gates on this flag — it is kept
    # for API stability and for third-party results objects without a
    # centroid, where the evaluator degrades to fixed-length zeros.
    requires_mp: bool = False

    def evaluate(self, results, lattice) -> np.ndarray:
        r = np.atleast_1d(np.asarray(self.evaluator(results, lattice),
                                     dtype=float))
        return r * float(self.weight)


# ---------------------------------------------------------------------------
# Evaluator helpers
# ---------------------------------------------------------------------------
def _final_index(results) -> int:
    """Index of the last record in EnvelopeResults."""
    return len(results.s) - 1


def _twiss_at_end(results) -> dict:
    """Twiss at the lattice exit.  Transverse in mm/mrad; longitudinal
    in the HELIX-internal (Δφ deg, ΔW MeV) convention — α_z = −TW's,
    β_z in deg/MeV at the exit machine clock (recorded since 2026-07 by
    both the envelope solver and the MP recorder; legacy results
    without the lists fall back to 0.0 / 1.0)."""
    i = _final_index(results)
    az = getattr(results, "alpha_z", None)
    bz = getattr(results, "beta_z", None)
    # len()-based emptiness: safe for both lists (live results) and
    # numpy arrays (loaded results) — truthiness on an array raises.
    return dict(
        alpha_x=results.alpha_x[i], beta_x=results.beta_x[i],
        alpha_y=results.alpha_y[i], beta_y=results.beta_y[i],
        alpha_z=az[i] if az is not None and len(az) else 0.0,
        beta_z=bz[i] if bz is not None and len(bz) else 1.0,
    )


def _z_twiss_degenerate(results) -> bool:
    """True when the end-of-line longitudinal Twiss is physically
    meaningless: the beam is still DC/continuous AT THE EXIT, or
    ε_z ~ 0 (cold beam) — the α_z/β_z numbers exist but carry no
    usable gradient.

    Deliberately does NOT read the scalar ``results.continuous``: the
    envelope solver sets it once at run START and it goes stale across
    a DC→bunched transition (RFQ/RFGap bunching), which would falsely
    kill the z axes on any DC-front-end deck with a bunched exit while
    the MP cost solver (per-row ``continuous_at``) evaluated them.
    The envelope's DC z-block is zeroed, so a genuinely-DC exit reads
    ε_z ~ 0 and the emit_z check catches it anyway."""
    ca = getattr(results, "continuous_at", None)   # MP: per-row truth
    if ca is not None and len(ca) and bool(ca[-1]):
        return True
    ez = getattr(results, "emit_z", None)
    return (ez is not None and len(ez)
            and float(ez[-1]) <= 1e-12)


#: residual value filling a SET_TWISS z-slot when a TRIAL point drives
#: the longitudinal plane degenerate after a healthy baseline: the
#: computed values blow up (β_z = Σ_φφ/ε_z → ∞ as ε_z → 0), so a large
#: finite constant is the honest "this region is bad" signal to the
#: optimizer — and the residual VECTOR LENGTH never changes (scipy
#: least_squares requires a fixed shape).
_Z_DEGENERATE_PENALTY = 1.0e3


def _has_z_record(results) -> bool:
    vals = getattr(results, "alpha_z", None)
    return vals is not None and len(vals) > 0


def audit_z_twiss_baseline(constraints, results,
                           allow_inert_constraints: bool) -> None:
    """Baseline gate for SET_TWISS kaz/kbz (same pattern as the k2 /
    stub audit): if the BASELINE evaluation carries no usable
    longitudinal record — legacy results without the arrays, or a
    degenerate exit (DC/continuous beam, ε_z ≈ 0) — the z axes have no
    gradient and the match would silently solve a different problem
    than the deck requested.  Refuse up front; the override keeps a
    warn-and-continue (z residual slots read 0)."""
    if results is None:
        return
    bad = []
    for c in constraints:
        src = getattr(c, "source", None)
        if not isinstance(src, SetTwiss):
            continue
        if not (getattr(src, "kaz", 0) == 1 or getattr(src, "kbz", 0) == 1):
            continue
        absent = not _has_z_record(results)
        if absent or _z_twiss_degenerate(results):
            reason = ("no longitudinal Twiss record (legacy/foreign "
                      "results)" if absent else
                      "degenerate longitudinal plane at the exit "
                      "(DC/continuous beam or ε_z ≈ 0)")
            bad.append(f"{c.label} [{reason}]")
    if bad:
        msg = ("SET_TWISS kaz/kbz cannot be enforced — baseline check: "
               + "; ".join(bad)
               + ".  Fix the deck/beam, set the z flags to 0, or pass "
               "allow_inert_constraints=True to warn-and-continue "
               "(the z residual slots then read a fixed penalty — a "
               "constant, gradient-free cost offset).")
        if allow_inert_constraints:
            import warnings
            warnings.warn(msg, stacklevel=3)
        else:
            raise ValueError(msg)


def _make_set_twiss_evaluator(cmd: SetTwiss):
    fam = str(getattr(cmd, "family", "") or "").strip()
    if fam and fam.upper() != "END":
        # TraceWin evaluates at "the output of the following element of
        # the named family"; HELIX evaluates at lattice END regardless.
        # Operand-level ignore — declare it once at collect time.
        # (family "END" designates the lattice end: intent and behavior
        # coincide, so no warning.)
        import warnings
        warnings.warn(
            f"SET_TWISS family='{fam}': family placement is not "
            "honoured — the constraint is evaluated at the END of the "
            "lattice.",
            stacklevel=3,
        )
    target = {
        "alpha_x": cmd.alpha_x, "beta_x": cmd.beta_x,
        "alpha_y": cmd.alpha_y, "beta_y": cmd.beta_y,
        "alpha_z": cmd.alpha_z, "beta_z": cmd.beta_z,
    }
    flags = {
        "alpha_x": cmd.kax, "beta_x": cmd.kbx,
        "alpha_y": cmd.kay, "beta_y": cmd.kby,
        "alpha_z": cmd.kaz, "beta_z": cmd.kbz,
    }

    # The residual length is FIXED at build time (one slot per k=1
    # flag) — it must never change between evaluations (scipy
    # least_squares requires a constant shape).  Unusable z slots are
    # FILLED with a STATELESS constant penalty, never dropped:
    # * strict mode: match()'s baseline audit guarantees the baseline
    #   is healthy, so any degenerate evaluation is a TRIAL point that
    #   left the physical region — the penalty is the correct signal;
    # * allow_inert override: every evaluation is degenerate, so the
    #   penalty is a constant offset — gradient-free, optimization-
    #   neutral, and the reported cost honestly shows the z targets
    #   as unmet.
    # Statelessness matters: cmaes_parallel workers rebuild these
    # closures and their first evaluation is a TRIAL, not x0 — any
    # first-call-defines-baseline state would let a worker score a
    # degenerate trial as perfect (adversarial-review finding).
    # Known edge (documented): approaching ε_z → 0 continuously, the
    # true residual diverges past the penalty before the 1e-12 gate
    # trips, so the cost has a downward cliff INTO degeneracy; the
    # real degeneracy modes (DC exit, legacy results) are discrete,
    # not approached continuously, and knob bounds keep ε_z ≥ 1e-6.
    zwarn = {"done": False}

    def evaluator(results, lattice):
        twiss = _twiss_at_end(results)
        need_z = flags["alpha_z"] == 1 or flags["beta_z"] == 1
        z_bad = False
        if need_z:
            z_bad = (not _has_z_record(results)
                     or _z_twiss_degenerate(results))
            if z_bad and not zwarn["done"]:
                zwarn["done"] = True
                import warnings
                warnings.warn(
                    "SET_TWISS: kaz/kbz flag set but the longitudinal "
                    "Twiss is unusable (legacy results, DC/continuous "
                    "beam, or ε_z ≈ 0) — the z residual slots read a "
                    f"fixed penalty of {_Z_DEGENERATE_PENALTY:g}.",
                    stacklevel=2,
                )
        out = []
        for name, k in flags.items():
            if k == 1:
                if name in ("alpha_z", "beta_z") and z_bad:
                    out.append(_Z_DEGENERATE_PENALTY)
                    continue
                out.append(twiss[name] - target[name])
        return np.asarray(out or [0.0], dtype=float)
    return evaluator


def _sigma_z_mm(results, i: int) -> float:
    """σ_z in mm at row *i* from σ_φ (deg) via the local βλ:
    z = φ/360° · βλ, λ_mm = 299792.458 / f_MHz."""
    beta = float(results.ref_beta[i])
    f_mhz = float(results.ref_frequency[i]) or 1.0
    lam_mm = 299792.458 / f_mhz
    return float(results.sigma_phi[i]) / 360.0 * beta * lam_mm


def _warn_k2(cmd) -> None:
    """TraceWin's k2 selects whether TRANSVERSE sizes include the beam
    centroid position (manual, "Set beam size") — NOT the deg/mm
    selector (that is the sign of the 4th operand).  HELIX evaluates
    sizes about the centroid (the k2=0 convention)."""
    if getattr(cmd, "k2", 0):
        import warnings
        warnings.warn(
            f"{type(cmd).KEYWORD} k2={cmd.k2} (include the beam "
            "centroid in the transverse sizes) is not modelled — "
            "sizes are evaluated about the beam centroid (the k2=0 "
            "convention).",
            stacklevel=4,
        )


def _make_set_size_evaluator(cmd: SetSize):
    """Equality residual on σ at lattice end (matching the most common
    usage; finer-placement caller can refactor).

    Longitudinal slot per the TraceWin manual: a POSITIVE 4th operand
    is σ_φ in degrees, a NEGATIVE one means the target is σ_z = |value|
    in mm (converted through the local βλ) — the z-form used to be
    silently dropped."""
    _warn_k2(cmd)

    def evaluator(results, lattice):
        i = _final_index(results)
        out = []
        if cmd.x_mm > 0:
            out.append(results.sigma_x[i] - cmd.x_mm)
        if cmd.y_mm > 0:
            out.append(results.sigma_y[i] - cmd.y_mm)
        if cmd.phi_or_z > 0:
            out.append(results.sigma_phi[i] - cmd.phi_or_z)
        elif cmd.phi_or_z < 0:
            out.append(_sigma_z_mm(results, i) - abs(cmd.phi_or_z))
        return np.asarray(out or [0.0], dtype=float)
    return evaluator


def _make_set_size_bound_evaluator(cmd, *, sign: int, cmd_index=None):
    """One-sided residual: ``sign=+1`` for SET_SIZE_MAX (penalise σ > target);
    ``sign=-1`` for SET_SIZE_MIN.

    Longitudinal slot follows the same TW rule as SET_SIZE: positive =
    σ_φ (deg), negative = σ_z bound of |value| mm via the local βλ.

    Window per the TW manual: "the N elements FOLLOWING this command" —
    an ELEMENT span anchored at the card, translated to recorder rows
    via ``element_record_span``.  The old evaluator sliced the last N
    recorder ROWS from the lattice end, so turning on record_substeps
    (or adding interior markers) silently changed which s-range the
    bound applied to — the objective depended on the recording cadence
    (2026-07 external review).  Legacy fallback (no ``cmd_index``):
    the last N *elements* of the line."""
    _warn_k2(cmd)
    _warned: set = set()

    def _warn_once(key: str, msg: str) -> None:
        if key not in _warned:
            _warned.add(key)
            import warnings
            warnings.warn(msg, stacklevel=3)

    def evaluator(results, lattice):
        from linac_gen.analysis.phase_advance import element_record_span
        # "Worst case" depends on the sign: a MAX ceiling is violated
        # by the LARGEST σ in the window, a MIN floor by the SMALLEST.
        # (Using max() for both hid any dip below the floor whenever
        # the window also held one large sample — SET_SIZE_MIN never
        # saw envelope pinches.)
        ne = int(getattr(cmd, "n_elems", 0) or 0)
        if ne <= 0:
            _warn_once("ne0", f"{type(cmd).KEYWORD} with N <= 0 — card "
                              "disabled (TW zero-means-off convention)")
            return np.asarray([0.0])
        elems = list(getattr(lattice, "elements", []) or [])
        if cmd_index is not None and elems:
            j = cmd_index
            counted = 0
            while j < len(elems) and counted < ne:
                if not isinstance(elems[j], LatticeCommand):
                    counted += 1
                j += 1
            if counted == 0:
                _warn_once("trailing", f"{type(cmd).KEYWORD} has no "
                                       "following elements — card "
                                       "disabled")
                return np.asarray([0.0])
            el_start, el_end = cmd_index, j
        else:
            el_end = len(elems)
            el_start = max(0, el_end - ne)
        n_rows = len(results.sigma_x)
        if el_end > 0:
            r0, r1 = element_record_span(results, el_start, el_end)
            i_start = max(0, min(int(r0), n_rows - 1))
            i_end = max(i_start, min(int(r1), n_rows - 1))
        else:
            # No lattice information at all — whole record.
            i_start, i_end = 0, _final_index(results)
        _worst = max if sign > 0 else min
        sx = _worst(results.sigma_x[i_start:i_end + 1])
        sy = _worst(results.sigma_y[i_start:i_end + 1])
        sphi = _worst(results.sigma_phi[i_start:i_end + 1])
        out = []
        if cmd.x_mm > 0:
            out.append(max(0.0, sign * (sx - cmd.x_mm)))
        if cmd.y_mm > 0:
            out.append(max(0.0, sign * (sy - cmd.y_mm)))
        if cmd.phi_or_z > 0:
            out.append(max(0.0, sign * (sphi - cmd.phi_or_z)))
        elif getattr(cmd, "phi_or_z", 0.0) < 0:
            # σ_z bound in mm: worst-case σ_z over the same window.
            _z = _worst(_sigma_z_mm(results, j)
                        for j in range(i_start, i_end + 1))
            out.append(max(0.0, sign * (_z - abs(cmd.phi_or_z))))
        return np.asarray(out or [0.0], dtype=float)
    return evaluator


def _make_set_beam_phase_adv_evaluator(cmd: SetBeamPhaseAdv,
                                       cmd_index: int | None = None):
    """Residual against target μ_x / μ_y / μ_z per TraceWin semantics.

    The card "applies to the entrance of the element where it is
    placed" (TW manual): the span is the ``n_elems`` non-command
    elements FOLLOWING the card's position — not the whole record
    window the old evaluator integrated.  A ZERO target disables that
    plane (TW rule; previously a zero target drove the measured phase
    toward zero).  Invalid β samples (all particles lost) are never
    bridged with a fabricated β=1: if a plane's span has fewer than two
    valid samples the residual takes a fixed penalty, keeping the
    residual-vector dimensionality stable for the optimiser.

    μ integration is shared with the analysis module
    (``_integrate_inverse_beta_split`` — contiguous-run trapz, mm/mrad
    units) and μ_z uses the effective (z, z′) β (the native deg/MeV β
    is not compatible with the transverse integrand).
    """
    _PENALTY = 1.0e3   # deg-scale residual for an unmeasurable plane
    _MRAD_TO_DEG = (180.0 / math.pi) * 1e-3

    _warned: set = set()          # one warning per condition per card

    def _warn_once(key: str, msg: str) -> None:
        if key not in _warned:
            _warned.add(key)
            import warnings
            warnings.warn(msg, stacklevel=3)

    def evaluator(results, lattice):
        from linac_gen.analysis.phase_advance import (
            _beta_z_eff_from_sigma, _integrate_inverse_beta_split,
            element_record_span,
        )
        s = np.asarray(getattr(results, "s", []), dtype=float)
        if s.size < 2:
            return np.zeros(3)

        # Ne ≤ 0: TW's pervasive "0 ⇒ not considered" convention (the
        # same rule this evaluator applies to zero TARGETS) — inventing
        # a 1-element span for an explicit Ne=0 would create a
        # constraint the file never asked for.
        if cmd.n_elems <= 0:
            _warn_once("ne0", "SET_BEAM_PHASE_ADV with Ne <= 0 — card "
                              "disabled (TW zero-means-off convention)")
            return np.zeros(3)

        # Element span: the n_elems non-command elements following the
        # card.  Falls back to the whole line when the card's position
        # is unknown (legacy call sites).
        elems = list(getattr(lattice, "elements", []) or [])
        if cmd_index is not None and elems:
            j = cmd_index
            counted = 0
            while j < len(elems) and counted < cmd.n_elems:
                if not isinstance(elems[j], LatticeCommand):
                    counted += 1
                j += 1
            if counted == 0:
                # Trailing card with nothing after it: an unmeasurable
                # span must not poison the fit with a permanent,
                # gradient-dead penalty — disable and say so.
                _warn_once("trailing", "SET_BEAM_PHASE_ADV has no "
                                       "following elements — card disabled")
                return np.zeros(3)
            el_start, el_end = cmd_index, j
        else:
            el_start, el_end = 0, max(len(elems), 1)

        r0, r1 = element_record_span(results, el_start, el_end)
        r0 = max(0, min(r0, s.size - 1))
        r1 = max(r0 + 1, min(r1, s.size - 1))
        s_slice = s[r0:r1 + 1]

        def _mu(beta_full):
            """Phase advance over the span, or the reason it is absent:
            ``None`` — the results carry NO usable data for the plane
            (nothing recorded / foreign length) — skip with a warning;
            ``nan`` — data recorded but invalid over the span (beam
            lost) — a real failure, penalized."""
            if beta_full is None:
                return None
            b = np.asarray(beta_full, dtype=float)
            if b.size != s.size:
                return None
            b = b[r0:r1 + 1]
            m = b > 0
            if m.sum() < 2:
                return float("nan")
            return _integrate_inverse_beta_split(s_slice, b, m) * _MRAD_TO_DEG

        def _residual(plane: str, target: float, mu) -> float:
            if mu is None:
                _warn_once(f"nodata_{plane}",
                           f"SET_BEAM_PHASE_ADV mu_{plane} target set but "
                           f"the results record no {plane}-plane data — "
                           "plane skipped (not penalized)")
                return 0.0
            if math.isnan(mu):
                return _PENALTY               # beam lost over the span
            return mu - target

        residuals = []
        for plane, target, beta_full in (
                ("x", cmd.mu_x_deg, getattr(results, "beta_x", None)),
                ("y", cmd.mu_y_deg, getattr(results, "beta_y", None))):
            if target > 0:
                residuals.append(_residual(plane, target, _mu(beta_full)))
            else:
                residuals.append(0.0)          # zero target ⇒ plane disabled
        if cmd.mu_z_deg > 0:
            residuals.append(
                _residual("z", cmd.mu_z_deg,
                          _mu(_beta_z_eff_from_sigma(results))))
        else:
            residuals.append(0.0)
        return np.asarray(residuals, dtype=float)
    return evaluator


# Residual (mm) reported per constrained plane when the beam is fully
# lost at the evaluation point.  A dead beam records centroid zeros —
# without this penalty, "kill the beam" would read as a PERFECT zero
# orbit and the optimizer could accept a beam-annihilating step as an
# improvement (observed: an unbounded steerer jumping to 1 T·m and the
# fit locking onto the dead-beam plateau).  Large-but-finite keeps
# least-squares arithmetic healthy while guaranteeing the step is
# rejected against any live-beam cost.
_DEAD_BEAM_RESIDUAL_MM = 1.0e3


def _centroid_row(results, elem_idx: int):
    """Map lattice element index → ``(centroid_list, row, dead)``.

    The MP tracker records one row per element of ``lattice.elements``
    (commands and markers included; row 0 = INPUT) and exposes the
    mapping as ``element_exit_idx``.  Fall back to ``elem_idx + 1``
    (clamped) for hand-built fixtures or abort-truncated runs.

    ``dead`` is True when the beam is fully lost at that row
    (transmission 0 %) — its centroid row is zeros and must NOT be
    read as a genuine on-axis orbit.
    """
    cent = getattr(results, "centroid", None)
    if cent is None or len(cent) == 0:
        return None, None, False              # envelope / no MP data
    exit_idx = getattr(results, "element_exit_idx", None) or []
    if elem_idx < len(exit_idx):
        row = int(exit_idx[elem_idx])
    else:
        row = elem_idx + 1
    row = min(row, len(cent) - 1)
    trans = getattr(results, "transmission", None)
    dead = bool(trans is not None and row < len(trans)
                and float(trans[row]) <= 0.0)
    return cent, row, dead


def _effective_targets(marker):
    """Resolve a BPM marker's (x_target, y_target, weight) with runtime
    file overrides taking precedence over the deck's card operands.
    ``None`` in a plane = unconstrained.

    The override is a single ``diag_target_override`` tuple whose
    PRESENCE wins — so a file row of ``nan nan`` genuinely frees both
    planes instead of silently falling back to the deck targets.
    """
    ov = getattr(marker, "diag_target_override", None)
    if ov is not None:
        return ov[0], ov[1], ov[2]
    return (getattr(marker, "x_target_mm", None),
            getattr(marker, "y_target_mm", None),
            None)


def _make_diag_position_evaluator(marker, elem_idx: int):
    """Residual (centroid − target) [mm] per constrained plane at the
    BPM marker's record row.  Works under BOTH cost solvers (MP
    recorder and EnvelopeResults carry the centroid); results without
    one degrade to fixed-length zeros (residual dimension must never
    change between evaluations)."""
    def evaluator(results, lattice):
        tx, ty, _ = _effective_targets(marker)
        n = int(tx is not None) + int(ty is not None)
        cent, row, dead = _centroid_row(results, elem_idx)
        if cent is None:
            return np.zeros(n)
        if dead:
            return np.full(n, _DEAD_BEAM_RESIDUAL_MM)
        c = np.asarray(cent[row], dtype=float)
        vals = []
        if tx is not None:
            vals.append(float(c[0]) - tx)
        if ty is not None:
            vals.append(float(c[2]) - ty)
        return np.asarray(vals, dtype=float)
    return evaluator


def _make_set_position_evaluator(cmd: SetPosition, cmd_idx: int):
    """Centroid residual [x−x_mm, x′−xp_mrad, y−y_mm, y′−yp_mrad] at the
    card's own recorder row (LatticeCommands get rows too).  Works
    under BOTH cost solvers; centroid-less results degrade to four
    zeros."""
    def evaluator(results, lattice):
        cent, row, dead = _centroid_row(results, cmd_idx)
        if cent is None:
            return np.zeros(4)
        if dead:
            return np.full(4, _DEAD_BEAM_RESIDUAL_MM)
        c = np.asarray(cent[row], dtype=float)
        return np.asarray([c[0] - cmd.x_mm,  c[1] - cmd.xp_mrad,
                           c[2] - cmd.y_mm,  c[3] - cmd.yp_mrad],
                          dtype=float)
    return evaluator


def z_clock_ratio(results) -> float:
    """``f_out/f_in`` for re-expressing an entrance deg·MeV (clock-
    referenced) longitudinal emittance at the exit RF clock.

    Returns 1.0 whenever the results carry no usable ``ref_frequency``
    (older recorders, HDF5-loaded results, zero entries) or the span is
    fixed-frequency — the anchor then degrades to a no-op, never a
    wrong division.  Shared by MIN_EMIT_GROWTH Z, the
    MIN_EMIT_4D_GROWTH z-arm, and the sequential-scan reversal
    threshold in the engine.
    """
    freq = getattr(results, "ref_frequency", None) or []
    if freq:
        f_in, f_out = float(freq[0]), float(freq[-1])
        if f_in > 0 and f_out > 0 and f_in != f_out:
            return f_out / f_in
    return 1.0


def _make_min_emit_growth_evaluator(cmd: MinEmitGrowth):
    """One-sided emittance-growth residual: ``max(0, ε_out − ε_in)``.

    For the transverse planes (``X``, ``Y``) the comparison is on the
    *normalised* emittance ``ε_geom · β γ`` so adiabatic damping under
    acceleration does not trivially satisfy the constraint.  For ``Z``
    the native ``emit_z`` (deg·MeV) is used — the conjugate-action pair,
    invariant under longitudinal acceleration — but degrees are
    frequency-referenced, so across a ``FREQ`` jump the raw value
    inflates by ``f_new/f_old`` with no physical growth.  The entrance
    emittance is therefore re-expressed at the *exit* clock
    (``ε_in · f_out/f_in``) before comparing; on a fixed-frequency span
    the ratio is 1 and the residual is unchanged.

    ``ε_in`` is read from the envelope-results record at index 0 (the
    seed beam projected onto the first element).  Drops below ``ε_in``
    contribute zero residual; recovery up to ``ε_in`` is also free.
    """
    attr = {"X": "emit_x", "Y": "emit_y", "Z": "emit_z"}[cmd.plane]
    normalise = cmd.plane in ("X", "Y")

    def evaluator(results, lattice):
        arr = getattr(results, attr)
        if not arr:
            return np.array([0.0])
        if normalise:
            beta = getattr(results, "ref_beta", None) or []
            gamma = getattr(results, "ref_gamma", None) or []
            bg_in = (beta[0] * gamma[0]) if beta and gamma else 1.0
            bg_out = (beta[-1] * gamma[-1]) if beta and gamma else 1.0
        else:
            bg_in = bg_out = 1.0
        eps_in = float(arr[0]) * float(bg_in)
        eps_out = float(arr[-1]) * float(bg_out)
        if not normalise:
            # deg·MeV is clock-referenced: express eps_in at the exit
            # frequency so a FREQ jump inside the span reads no growth.
            eps_in *= z_clock_ratio(results)
        return np.array([max(0.0, eps_out - eps_in)])
    return evaluator


def _make_set_ke_out_min_evaluator(cmd: SetKeOutMin):
    """One-sided floor on the reference particle's exit kinetic energy.

    Residual is ``max(0, E_floor − W_kin,out)``.  Zero when the matcher
    leaves end-of-line energy at or above the floor.
    """
    def evaluator(results, lattice):
        wkin = getattr(results, "ref_w_kin", None) or []
        if not wkin:
            return np.array([0.0])
        return np.array([max(0.0, cmd.energy_mev - float(wkin[-1]))])
    return evaluator


_MIN_TRANSMISSION_ENV_WARNED = False


def _make_min_transmission_evaluator(cmd):
    """End-of-line transmission floor (closes the loss-gaming gap).

    Without this, MIN_EMIT_4D_GROWTH (and the other ε metrics) are
    mathematically gameable in MP mode: ε is computed from *alive*
    particles only, so trials that scrape off the halo show smaller
    ε in survivors → lower cost → preferred.  This constraint
    explicitly penalises trials below a transmission floor.

    Residual is one-sided: ``r = max(0, threshold - T_final) / 100``
    -- normalised by 100 so a 1% loss below threshold produces a
    residual of ~0.01, comparable to the other constraint residuals so
    the user's weight knob lives in a familiar range.  The weight is
    NOT applied here: ``Constraint.evaluate()`` multiplies every
    evaluator's raw residual by ``weight`` exactly once (applying it
    here too made the effective weight k², unlike every other card --
    fixed 2026-07-10).

    Envelope mode: env tracks the RMS Σ matrix, not particles -- apertures
    are no-ops and ``results.transmission`` either doesn't exist or is
    a constant 100% array.  Return a zero residual and warn once so the
    user knows to switch to MP cost-solver if loss matters.
    """
    def evaluator(results, lattice):
        global _MIN_TRANSMISSION_ENV_WARNED
        trans = getattr(results, "transmission", None)
        if trans is None or len(trans) == 0:
            if not _MIN_TRANSMISSION_ENV_WARNED:
                import sys
                print("[match] MIN_TRANSMISSION found but results have no "
                      "transmission field (envelope mode?).  Constraint is "
                      "INERT -- switch to cost_solver='mp' to enforce the "
                      "transmission floor.", file=sys.stderr)
                _MIN_TRANSMISSION_ENV_WARNED = True
            return np.array([0.0])
        t_final = float(trans[-1])
        # 100% trace (envelope-synthesised) is also effectively inert --
        # the env solver fills transmission with constant 100.0 so MP
        # vs env distinction stays clean for the dropdown, but a real
        # MP run with losses produces a value < 100.
        deficit = max(0.0, cmd.threshold_pct - t_final)
        # Raw residual only -- Constraint.evaluate() applies the weight
        # once (was double-applied here as weight^2; fixed 2026-07-10).
        return np.array([deficit / 100.0])
    return evaluator


def _make_min_emit_4d_growth_evaluator(cmd: MinEmit4DGrowth):
    """Emittance-exchange-aware one-sided growth residual.

    Returns a 2-element residual vector:

    * ``r_4d = max(0, ε_4D_norm_out  - tol_4d · ε_4D_norm_in)``  where
      ``ε_4D_norm = sqrt(det(Σ_4x4_transverse)) · (β γ)²`` -- the 4-D
      coupled transverse normalised emittance.  Invariant under x-y
      coupling (solenoid rotation, coupling resonance), so the matcher
      is free to trade transverse area between planes.
    * ``r_z  = max(0, ε_z_norm_out  - tol_z  · ε_z_norm_in)``  where
      ``ε_z_norm = emit_z · (β γ)`` -- the normalised longitudinal
      emittance.

    Both residuals are clamped at zero (a drop below the seed is free).
    The tolerance factors are multiplicative bounds (``1.0`` = strict,
    ``1.1`` = allow 10 % growth).

    Falls back to a 2-element zero vector if the envelope didn't record
    the expected arrays (``emit_4d``, ``emit_z``, ``ref_beta``,
    ``ref_gamma``), so an inert constraint never crashes the matcher.
    """
    def evaluator(results, lattice):
        emit_4d = getattr(results, "emit_4d", None) or []
        emit_z = getattr(results, "emit_z", None) or []
        beta = getattr(results, "ref_beta", None) or []
        gamma = getattr(results, "ref_gamma", None) or []
        if (not emit_4d or not emit_z
                or len(beta) < 1 or len(gamma) < 1):
            return np.array([0.0, 0.0])
        bg_in = float(beta[0]) * float(gamma[0])
        bg_out = float(beta[-1]) * float(gamma[-1])
        # 4-D transverse: det(Σ_4x4) -> sqrt gives 4-D emit, then (βγ)²
        # promotes to normalised since each row of Σ has one factor of
        # βγ stripped by the unnormalising in tracking.
        eps_4d_in = float(emit_4d[0]) * (bg_in ** 2)
        eps_4d_out = float(emit_4d[-1]) * (bg_out ** 2)
        # Z: native deg·MeV scaled by βγ to put it on the same
        # normalised footing.  deg·MeV is clock-referenced, so express
        # the entrance value at the exit frequency first — a FREQ jump
        # inside the span must not read as growth (mirrors the plain
        # MIN_EMIT_GROWTH Z fix).
        eps_z_in = float(emit_z[0]) * bg_in * z_clock_ratio(results)
        eps_z_out = float(emit_z[-1]) * bg_out
        r_4d = max(0.0, eps_4d_out - cmd.tol_4d * eps_4d_in)
        r_z = max(0.0, eps_z_out - cmd.tol_z * eps_z_in)
        return np.array([r_4d, r_z])
    return evaluator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def collect_constraints(lattice) -> List[Constraint]:
    """Walk the lattice and return one :class:`Constraint` per SET_* card.

    Unsupported / not-yet-implemented commands surface as zero-residual
    constraints with a ``notes`` flag so they appear in the report.
    """
    out: List[Constraint] = []
    # Families that have at least one adjusting card — DIAG_POSITION
    # targets in a family nobody adjusts are passive monitors (TraceWin
    # semantics: fnalscl family 12) and mint no constraint.  Both card
    # languages count: plain ``ADJUST N v`` AND ``ADJUST_STEERER N``
    # (diag_n) — the latter also mints matcher variables.
    adjust_families = set()
    for e in lattice.elements:
        if isinstance(e, Adjust):
            try:
                adjust_families.add(int(e.target.strip()))
            except ValueError:
                pass
        else:
            diag_n = getattr(e, "diag_n", None)
            if diag_n is not None and isinstance(e, LatticeCommand):
                try:
                    adjust_families.add(int(diag_n))
                except (TypeError, ValueError):
                    pass
    for elem_idx, elem in enumerate(lattice.elements):
        if getattr(elem, "is_bpm", False):
            tx, ty, w_over = _effective_targets(elem)
            if tx is None and ty is None:
                continue                       # target-less BPM
            fam = getattr(elem, "diag_family", None)
            if fam is not None and fam not in adjust_families:
                continue                       # passive monitor family
            dm = float(getattr(elem, "accuracy_mm", 1.0) or 1.0)
            weight = (w_over if w_over is not None else 1.0) / dm
            if weight == 0.0:
                continue                       # zero-weighted = disabled
            out.append(Constraint(
                label=f"DIAG_POSITION:{elem.name}",
                evaluator=_make_diag_position_evaluator(elem, elem_idx),
                # Gauss–Markov: residual/dm ⇒ cost ((x−t)/dm)²; a file-
                # supplied weight multiplies on top.
                weight=weight,
                source=None, requires_mp=True,
                notes=f"family {fam}" if fam is not None else "",
            ))
            continue
        if not isinstance(elem, LatticeCommand):
            continue

        if isinstance(elem, SetTwiss):
            if not any([elem.kax, elem.kbx, elem.kay, elem.kby,
                        elem.kaz, elem.kbz]):
                continue   # all k-flags zero ⇒ inert
            out.append(Constraint(
                label=f"SET_TWISS:{elem.family or 'END'}",
                evaluator=_make_set_twiss_evaluator(elem),
                # NOTE: MatchResult.z_penalty_infeasible detects the
                # degeneracy penalty in the POST-weight residuals and
                # scales its comparison by this weight — if SET_TWISS
                # ever gains a card-level weight, that property (and
                # its tests) must move in lockstep.
                weight=1.0, source=elem,
            ))

        elif isinstance(elem, SetSize):
            if elem.k == 0:
                continue
            out.append(Constraint(
                label="SET_SIZE",
                evaluator=_make_set_size_evaluator(elem),
                weight=float(elem.k), source=elem,
            ))

        elif isinstance(elem, SetSizeMax):
            if elem.k == 0:
                continue
            out.append(Constraint(
                label="SET_SIZE_MAX",
                evaluator=_make_set_size_bound_evaluator(
                    elem, sign=+1, cmd_index=elem_idx),
                weight=float(elem.k), source=elem,
            ))

        elif isinstance(elem, SetSizeMin):
            if elem.k == 0:
                continue
            out.append(Constraint(
                label="SET_SIZE_MIN",
                evaluator=_make_set_size_bound_evaluator(
                    elem, sign=-1, cmd_index=elem_idx),
                weight=float(elem.k), source=elem,
            ))

        elif isinstance(elem, SetBeamPhaseAdv):
            if elem.k == 0:
                continue
            out.append(Constraint(
                label="SET_BEAM_PHASE_ADV",
                evaluator=_make_set_beam_phase_adv_evaluator(
                    elem, cmd_index=elem_idx),
                weight=float(elem.k), source=elem,
            ))

        elif isinstance(elem, SetPosition):
            if elem.k == 0:
                continue
            out.append(Constraint(
                label="SET_POSITION",
                evaluator=_make_set_position_evaluator(elem, elem_idx),
                weight=float(elem.k), source=elem,
                requires_mp=True,
            ))

        elif isinstance(elem, MinEmitGrowth):
            if elem.weight == 0.0:
                continue
            out.append(Constraint(
                label=f"MIN_EMIT_GROWTH:{elem.plane}",
                evaluator=_make_min_emit_growth_evaluator(elem),
                weight=elem.weight, source=elem,
            ))

        elif isinstance(elem, MinEmit4DGrowth):
            if elem.weight == 0.0:
                continue
            out.append(Constraint(
                label="MIN_EMIT_4D_GROWTH",
                evaluator=_make_min_emit_4d_growth_evaluator(elem),
                weight=elem.weight, source=elem,
            ))

        elif isinstance(elem, MinTransmission):
            if elem.weight == 0.0:
                continue
            out.append(Constraint(
                label=f"MIN_TRANSMISSION:{elem.threshold_pct:g}%",
                evaluator=_make_min_transmission_evaluator(elem),
                weight=elem.weight, source=elem,
            ))

        elif isinstance(elem, SetKeOutMin):
            if elem.weight == 0.0:
                continue
            out.append(Constraint(
                label=f"SET_KE_OUT_MIN:{elem.energy_mev:g}MeV",
                evaluator=_make_set_ke_out_min_evaluator(elem),
                weight=elem.weight, source=elem,
            ))

        elif isinstance(elem, (SetAchromat, SetSeparation, SetAdv)):
            # Parsed and reported; not yet a residual.  Zero placeholder
            # so the constraint shows up in the result table.
            out.append(Constraint(
                label=f"{elem.KEYWORD} (stub)",
                evaluator=lambda r, l: np.array([0.0]),
                weight=1.0, source=elem,
                notes="evaluator not implemented",
            ))

    return out
