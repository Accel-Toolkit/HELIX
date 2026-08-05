"""Structure and beam phase advance — per-period and along-s.

``structure_phase_advance`` builds the one-period 6×6 transfer matrix
from bare-element optics (no space-charge) and extracts μ_x and μ_y
via Courant-Snyder.  Accelerating sections are *not* refused — the
result is labelled at the entry energy and ΔW is reported separately.

``beam_phase_advance`` reads the matched-beam β(s) from a previously
computed :class:`linac_gen.tracking.envelope.EnvelopeResults` and
integrates ``∫ ds/β`` over the period span.  This number is meaningful
only for a beam that is matched to the period (β_start ≈ β_end); a
``matched`` flag in the output lets the caller decide whether to trust
the value.

Both functions expect a :class:`PeriodicStructure` from
``period_detect.detect_periods``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)


def _check_stop(should_stop) -> None:
    """Raise OperationCancelled when the caller's stop hook fires.

    Cancellation RAISES (never returns a partial dict): the results-tab
    popups cache whatever these functions return, and a partially
    walked lattice cached as valid σ₀ data would silently poison every
    later read.
    """
    if should_stop is not None and should_stop():
        from linac_gen.core.cancelled import OperationCancelled
        raise OperationCancelled("phase-advance computation cancelled")


# ---------------------------------------------------------------------------
# Structure phase advance (zero-current)
# ---------------------------------------------------------------------------
def structure_phase_advance(lattice, ref, period: PeriodicStructure,
                            *, cache: dict | None = None,
                            should_stop=None) -> dict:
    """Compute σ₀_x and σ₀_y for one period from the bare transfer matrix.

    Parameters
    ----------
    lattice : Lattice
        The lattice carrying the period.
    ref : ReferenceParticle
        Reference particle at lattice entry.  A copy is taken
        internally — repeated calls with the same ``ref`` are safe.
    period : PeriodicStructure
        The period to analyse.  Indexes into ``lattice.elements``.
    cache : dict, optional
        Opt-in transfer-matrix memoisation dict; see
        :func:`linac_gen.tracking.matrix_tracking.get_element_matrix`.
        Default ``None`` reproduces the original recompute-every-call
        behaviour exactly.  Pass a dict when calling repeatedly with an
        unchanged ``(lattice, ref)`` to amortise the FieldMap3D
        ``fitted_matrix`` cost.

    Returns
    -------
    dict with keys
        ``mu_x_deg``, ``mu_y_deg`` — phase advance per cell, degrees.
        ``alpha_x``, ``beta_x`` (mm/rad), ``alpha_y``, ``beta_y``
            — periodic Twiss for one cell at the entry of the period.
        ``stable_x``, ``stable_y`` — bool; whether |½ tr(M_2x2)| < 1.
        ``coupled`` — bool; True when ``compute_twiss`` rejected the
            lattice (solenoids, skew multipoles).  In that case the
            μ / α / β fields are ``None``.
        ``w_in``, ``w_out`` — kinetic energy at period entry / exit (MeV).
        ``dw`` — ``w_out - w_in`` (MeV).  Non-zero ⇒ accelerating
            section, label μ as "single-cell @ entry energy".
        ``M_period`` — the 6×6 one-period transfer matrix.
        ``reason`` — human-readable string when stable_*/coupled fail.
    """
    end_excl = period.inner_slice_end
    if end_excl <= period.start:
        raise ValueError("period covers no elements")
    # ``compute_transfer_matrix`` takes inclusive start/end indices.
    end_incl = end_excl - 1

    # We want both the matrix and the exit energy, so feed it a fresh
    # ref copy and re-replay manually for the exit-energy report.
    _check_stop(should_stop)
    ref_for_matrix = ref.copy()
    M = compute_transfer_matrix(lattice, ref_for_matrix,
                                start=period.start, end=end_incl,
                                cache=cache)

    _check_stop(should_stop)
    w_in, w_out = _energy_change(lattice, ref, period.start, end_incl)
    dw = w_out - w_in

    out: dict = {
        "mu_x_deg": None, "mu_y_deg": None, "mu_z_deg": None,
        "alpha_x": None, "beta_x": None,
        "alpha_y": None, "beta_y": None,
        "alpha_z": None, "beta_z": None,
        "stable_x": False, "stable_y": False, "stable_z": False,
        "coupled_xy": False,
        "w_in": w_in, "w_out": w_out, "dw": dw,
        "M_period": M,
        "reason": None,
    }

    for plane in ("x", "y", "z"):
        try:
            tw = compute_twiss(M, plane=plane)
            # Map μ to the unsigned representation in [0, 180°] so
            # period μ matches |Σ Δμ_element| from along-s.  This is
            # the canonical "magnitude of phase advance per cell"
            # used in linac design (TraceWin's σ₀ output).  The
            # full-circle value (360 - μ) is the other valid branch
            # and is reported as ``mu_{p}_branch_deg`` for users who
            # need it.
            mu_full = tw["mu"]
            mu_unsigned = mu_full if mu_full <= 180.0 else 360.0 - mu_full
            out[f"mu_{plane}_deg"] = mu_unsigned
            out[f"mu_{plane}_branch_deg"] = mu_full
            out[f"alpha_{plane}"] = tw["alpha"]
            out[f"beta_{plane}"] = tw["beta"]
            out[f"stable_{plane}"] = True
        except ValueError as exc:
            msg = str(exc)
            if "coupled" in msg.lower() and plane in ("x", "y"):
                out["coupled_xy"] = True
                # Keep going so the longitudinal plane still computes.
            if not out["reason"]:
                out["reason"] = f"{plane}: {msg}"

    # Back-compat alias used by older callers.
    out["coupled"] = out["coupled_xy"]
    return out


# ---------------------------------------------------------------------------
# Coupled-lattice eigenmode phase advance (Edwards-Teng style)
# ---------------------------------------------------------------------------
def coupled_phase_advance(lattice, ref, period: PeriodicStructure,
                          *, cache: dict | None = None) -> dict:
    """Eigenmode phase advances μ_I, μ_II for a transversely-coupled period.

    For a stable 4×4 symplectic transverse map M (solenoids, skew quads,
    strong RF transverse coupling), the eigenvalues lie on the unit
    circle in two complex-conjugate pairs:

        λ_1 = e^{+iμ_I},   λ_2 = e^{-iμ_I},
        λ_3 = e^{+iμ_II},  λ_4 = e^{-iμ_II}.

    The two distinct phases ``μ_I, μ_II ∈ [0, π]`` are the eigenmode
    tunes per period — the coupled-lattice generalisation of σ₀_x, σ₀_y.
    They are well-defined even when the planes can't be decoupled (so
    ``compute_twiss`` would refuse).

    For unstable planes (eigenvalues off the unit circle), the magnitude
    of one eigenvalue exceeds 1 and ``stable_I``/``stable_II`` is False
    accordingly.  We still report the phase from ``np.angle`` so the
    caller has *something* to display.

    Parameters mirror :func:`structure_phase_advance` for symmetry,
    including the optional ``cache`` for amortised per-element matrices.
    """
    end_excl = period.inner_slice_end
    if end_excl <= period.start:
        raise ValueError("period covers no elements")
    end_incl = end_excl - 1

    ref_for_matrix = ref.copy()
    M = compute_transfer_matrix(lattice, ref_for_matrix,
                                start=period.start, end=end_incl,
                                cache=cache)
    w_in, w_out = _energy_change(lattice, ref, period.start, end_incl)
    M4 = M[:4, :4]

    # Acceleration normalization: through an accelerating cell the 4×4
    # transverse map is conformally symplectic with det(M₄) = D² where
    # D = (βγ)_in/(βγ)_out (each plane contributes one factor of D).
    # Its eigenvalues are √D·e^{±iμ}: testing raw |λ| against 1 would
    # mislabel ordinary adiabatic damping as "unstable" and bias the
    # extracted phases.  Scalar-normalize by det(M₄)^{1/4} first —
    # exact for conformally symplectic maps; skipped (bit-identical)
    # for magnetostatic cells.
    det4 = float(np.linalg.det(M4))
    if det4 <= 0.0:
        raise ValueError(
            f"4×4 transverse block determinant {det4:.3e} <= 0 — not a "
            f"transport matrix"
        )
    damping = 1.0
    M4_norm = M4
    if abs(det4 - 1.0) > 1e-9:
        damping = det4 ** 0.25          # per-plane amplitude ratio √D
        M4_norm = M4 / damping

    eigvals = np.linalg.eigvals(M4_norm)
    phases = np.angle(eigvals)              # in (-π, π]
    moduli = np.abs(eigvals)

    # Sort by |phase| so conjugate pairs sit adjacent.
    order = np.argsort(np.abs(phases))
    phases_sorted = phases[order]
    moduli_sorted = moduli[order]

    # Unique mode phases (collapse the two conjugates per mode).  We
    # walk sorted-|phase| and take every other entry so [I, I*, II, II*]
    # collapses to [I, II].
    pos_phases = np.abs(phases_sorted)
    mode_phases: list[float] = []
    for p in pos_phases:
        if not mode_phases or abs(p - mode_phases[-1]) > 1e-6:
            mode_phases.append(float(p))
    while len(mode_phases) < 2:
        mode_phases.append(mode_phases[-1] if mode_phases else 0.0)

    mu_I = mode_phases[0]
    mu_II = mode_phases[1]

    # Stability: all |λ| ≈ 1 AFTER det-normalization (adiabatic damping
    # is physics, not instability).  A 1 % tolerance is generous but
    # reflects practical envelope-tracking precision.
    stable_I = bool(np.all(np.abs(moduli_sorted[:2] - 1.0) < 1e-2))
    stable_II = bool(np.all(np.abs(moduli_sorted[2:] - 1.0) < 1e-2))

    return {
        "mu_I_deg":  math.degrees(mu_I),
        "mu_II_deg": math.degrees(mu_II),
        "stable_I":  stable_I,
        "stable_II": stable_II,
        "eigvals":   eigvals,
        "damping":   damping,
        "M_period":  M,
        "w_in":  w_in, "w_out": w_out, "dw": w_out - w_in,
    }


# ---------------------------------------------------------------------------
# Coupled-mode normal-mode emittances and beta-functions from σ
# ---------------------------------------------------------------------------
def _normal_mode_beta_from_sigma(sigma4: np.ndarray) -> "tuple | None":
    """Extract eigenmode (β_I, β_II, ε_I, ε_II, v_I, v_II) from a 4×4 σ.

    Uses the standard Wolski normalisation: σ·J has eigenvalues
    ``±i·ε_I, ±i·ε_II`` (J = symplectic form for 4 transverse coords).
    We pick the +i eigenvectors v_I, v_II, normalise via
    ``v^† J v = i``, and project β_I onto the x-coordinate via the
    canonical Wolski-Beta:   β_I,x = 2 · |v_I[0]|² .

    Returns ``None`` for non-physical / singular σ.
    """
    if sigma4.shape != (4, 4) or not np.all(np.isfinite(sigma4)):
        return None
    # Symplectic form for 4-D transverse phase space (x, x', y, y').
    J = np.array([
        [ 0,  1,  0,  0],
        [-1,  0,  0,  0],
        [ 0,  0,  0,  1],
        [ 0,  0, -1,  0],
    ], dtype=float)
    M = sigma4 @ J
    eigvals, eigvecs = np.linalg.eig(M)
    # For matched σ, eigenvalues = ±i·ε_I, ±i·ε_II.  Keep the +i half.
    pos_mask = np.imag(eigvals) > 0
    if pos_mask.sum() < 2:
        # σ not symplectic / unmatched — fall back to picking the two
        # largest |Im(λ)| eigenvalues.
        order = np.argsort(-np.abs(np.imag(eigvals)))[:2]
    else:
        order = np.where(pos_mask)[0][:2]
    # Sort by emittance (smaller first → mode I).
    eps = np.imag(eigvals[order])
    sort = np.argsort(np.abs(eps))
    order = order[sort]
    eps_I, eps_II = abs(eps[sort[0]]), abs(eps[sort[1]])
    if eps_I < 1e-30 or eps_II < 1e-30:
        return None
    v_I  = eigvecs[:, order[0]]
    v_II = eigvecs[:, order[1]]
    # Wolski normalisation: v^† · J · v = i.  Multiplying v by a
    # complex scalar α gives (αv)^†·J·(αv) = |α|² · i·θ where
    # θ = -i · v^† J v.  We want θ → 1 with sign positive.
    def _renormalise(v):
        u = (v.conj() @ J @ v) / 1j
        u_real = np.real(u)
        if abs(u_real) < 1e-30:
            return v
        return v / np.sqrt(abs(u_real))
    v_I = _renormalise(v_I)
    v_II = _renormalise(v_II)
    # Wolski β_I projected on x-coord:   β_{I,x} = 2 · |v_I[0]|² .
    # For decoupled limit this reduces to the standard β_x of mode I
    # (= β_x when mode I aligns with x).
    beta_I_x = 2.0 * float(np.abs(v_I[0]) ** 2)
    beta_II_y = 2.0 * float(np.abs(v_II[2]) ** 2)
    # Also return the cross-projections so callers can compute mode β
    # along whichever axis is relevant.
    return {
        "eps_I": eps_I, "eps_II": eps_II,
        "beta_I_x":  beta_I_x,
        "beta_II_y": beta_II_y,
        "beta_I_y":  2.0 * float(np.abs(v_I[2]) ** 2),
        "beta_II_x": 2.0 * float(np.abs(v_II[0]) ** 2),
        "v_I": v_I, "v_II": v_II,
    }


def coupled_beam_phase_advance_per_cell(results,
                                          period: PeriodicStructure,
                                          ) -> dict:
    """Per-cell depressed eigenmode tunes σ_I, σ_II from coupled envelope σ.

    .. warning::
       This is an **approximate** implementation that is only correct
       in the decoupled limit.  For strongly-coupled lattices (solenoid
       channels with axisymmetric mode mixing) the projected formula
       ``β_{I,x} = 2|v_I[0]|²`` does not separate the two normal modes:
       both modes get the same β projection on x, so ∫ds/β returns
       the same number for both (effectively the geometric mean of the
       two true mode tunes).  A proper coupled-mode tune extraction
       requires either (a) full Mais-Ripken Twiss propagation along s
       or (b) the depressed transfer matrix M_period_dep from the
       envelope solver.  Neither is yet wired up; the function is kept
       as a placeholder so the decoupled-limit test still passes.

    For each recorded σ-matrix sample in the envelope output, we
    eigendecompose the 4×4 transverse block to extract the normal-mode
    β functions β_I,x(s) and β_II,y(s) (Wolski / Mais-Ripken
    convention).  The depressed mode tunes per cell are
    ``σ_I = ∫_cell ds / β_I,x``  and  ``σ_II = ∫_cell ds / β_II,y``.

    Eigenvector continuity across samples is enforced by picking the
    new mode whose v_I has maximum overlap with the previous v_I — this
    prevents mode-swapping at near-degenerate σ.

    Parameters
    ----------
    results : EnvelopeResults
        Must have populated ``sigma_matrix`` and ``s`` (mm).
    period : PeriodicStructure
        The period whose cells we tune; samples are looked up by
        element-index span on the recorded ``s`` grid.

    Returns
    -------
    dict
        ``cells`` (1..n_repeats), ``mu_I_deg``, ``mu_II_deg`` (per cell),
        ``mode_swap_count`` — diagnostic for how many continuity flips
        happened (high values mean weak coupling-dominance, results
        less reliable).
    """
    sm = getattr(results, "sigma_matrix", None)
    s_arr = np.asarray(getattr(results, "s", []), dtype=float)
    if sm is None or len(sm) == 0 or s_arr.size == 0:
        raise ValueError(
            "envelope results carry no sigma_matrix; "
            "run the envelope tracker (it records σ on every element by default)"
        )
    if len(sm) != s_arr.size:
        raise ValueError(
            f"σ-matrix samples ({len(sm)}) ≠ s samples ({s_arr.size})"
        )

    # Walk every sample, extract mode β with continuity tracking.
    n = s_arr.size
    beta_I_x  = np.full(n, np.nan)
    beta_II_y = np.full(n, np.nan)
    eps_I = np.full(n, np.nan)
    eps_II = np.full(n, np.nan)
    prev_vI = None
    prev_vII = None
    swap_count = 0
    for i in range(n):
        sigma = np.asarray(sm[i], dtype=float)
        if sigma.shape != (6, 6):
            continue
        out = _normal_mode_beta_from_sigma(sigma[:4, :4])
        if out is None:
            continue
        v_I, v_II = out["v_I"], out["v_II"]
        # Mode continuity: if v_I overlaps more with prev_vII, swap.
        if prev_vI is not None:
            o_II_to_I = abs(prev_vI.conj() @ v_II)
            o_I_to_I  = abs(prev_vI.conj() @ v_I)
            if o_II_to_I > o_I_to_I:
                v_I, v_II = v_II, v_I
                eps_I_v, eps_II_v = out["eps_II"], out["eps_I"]
                bIx, bIIy = out["beta_II_x"], out["beta_I_y"]
                swap_count += 1
            else:
                eps_I_v, eps_II_v = out["eps_I"], out["eps_II"]
                bIx, bIIy = out["beta_I_x"], out["beta_II_y"]
        else:
            eps_I_v, eps_II_v = out["eps_I"], out["eps_II"]
            bIx, bIIy = out["beta_I_x"], out["beta_II_y"]
        prev_vI, prev_vII = v_I, v_II
        beta_I_x[i] = bIx
        beta_II_y[i] = bIIy
        eps_I[i] = eps_I_v
        eps_II[i] = eps_II_v

    # Integrate ∫ds/β over each cell.  Cell spans come from
    # ``period.spans()`` — explicit per-repeat element-index ranges built
    # by walking *significant* elements (detector-supplied), falling back
    # to the legacy constant-raw-stride tiling for manually-built
    # periods.  A constant raw stride is wrong whenever markers /
    # commands are unevenly interleaved between repeats.
    cell_spans = period.spans()
    cells = np.arange(1, period.n_repeats + 1)
    mu_I = np.full(period.n_repeats, np.nan)
    mu_II = np.full(period.n_repeats, np.nan)
    _MRAD_TO_DEG = (180.0 / math.pi) * 1e-3
    for k in range(min(period.n_repeats, len(cell_spans))):
        el0, el1 = cell_spans[k]
        # Element span → record-row span (substep-safe).
        i0, i1 = element_record_span(results, el0, el1)
        if i1 > n - 1:
            continue
        s_slice = s_arr[i0:i1 + 1]
        bI = beta_I_x[i0:i1 + 1]
        bII = beta_II_y[i0:i1 + 1]
        if not (np.all(np.isfinite(bI)) and np.all(bI > 0)):
            continue
        if not (np.all(np.isfinite(bII)) and np.all(bII > 0)):
            continue
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        mu_I[k]  = _trapz(1.0 / bI,  s_slice) * _MRAD_TO_DEG
        mu_II[k] = _trapz(1.0 / bII, s_slice) * _MRAD_TO_DEG

    return {
        "cells": cells,
        "mu_I_deg": mu_I,
        "mu_II_deg": mu_II,
        "mode_swap_count": swap_count,
    }


def coupled_beam_phase_advance_per_cell_via_M(
    lattice, ref, period: PeriodicStructure, results,
    *, cache: dict | None = None,
) -> dict:
    """Per-cell depressed eigenmode tunes σ_I, σ_II via the depressed
    one-period transfer matrix.

    PREFERRED PATH — probe-bearing results (``EnvelopeSolver(...,
    phase_probe=True)``): the per-cell monodromy is assembled from the
    solver's EXACT tangent factors (midpoint-σ substepped Strang maps,
    freq-jump D, edges, SC-compensation state), det-normalized, and
    eigen-extracted with symplectic mode pairing.  This is exact with
    respect to the envelope run.

    LEGACY FALLBACK (deprecated, emits a UserWarning): without probe
    maps, the historical approximation is used — one whole-element bare
    map followed by one full-element-length SC kick evaluated at the
    recorded ENTRANCE σ.  That is NOT the solver's tangent map: on a
    converged matched FODO it reads ~9 % low versus the fine ∫ds/β.
    Rerun the envelope with ``phase_probe=True`` to get the exact
    number.

    Parameters
    ----------
    lattice : Lattice
        The lattice carrying the period.
    ref : ReferenceParticle
        Initial reference state (entry of the lattice).
    period : PeriodicStructure
        The period over which to build M_dep per cell.
    results : EnvelopeResults
        Probe-bearing results preferred; must at least carry
        ``sigma_matrix`` (list of 6×6) and ``current_mA``.

    Returns
    -------
    dict
        ``cells`` (1..n_repeats), ``mu_I_deg``, ``mu_II_deg``,
        ``stable_I``, ``stable_II``.
    """
    from linac_gen.tracking.matrix_tracking import get_element_matrix
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    from linac_gen.tracking.envelope import (
        _sc_kick_matrix_3d, _sc_kick_matrix_2d_dc,
    )

    # --- Preferred: exact probe-based monodromies -----------------------
    if getattr(results, "element_maps_dep", None):
        ch = channel_phase_advance(results, period)
        n = period.n_repeats
        if ch["coupled_xy"]:
            mu_I = np.asarray(ch["mu_I_dep_deg"], dtype=float)
            mu_II = np.asarray(ch["mu_II_dep_deg"], dtype=float)
        else:
            # Decoupled lattice reached through the coupled API: modes
            # are the planes; keep {I, II} = sorted plane tunes per the
            # historical convention (eigen |angle| ordering).
            mx = np.asarray(ch["mu_x_dep_deg"], dtype=float)
            my = np.asarray(ch["mu_y_dep_deg"], dtype=float)
            both = np.sort(np.vstack([mx, my]), axis=0)
            mu_I, mu_II = both[0], both[1]
        # Pad/trim to n_repeats for key-shape compatibility.
        def _fit(a):
            out = np.full(n, np.nan)
            out[:min(n, a.size)] = a[:min(n, a.size)]
            return out
        mu_I, mu_II = _fit(mu_I), _fit(mu_II)
        return {
            "cells": np.arange(1, n + 1),
            "mu_I_deg": mu_I,
            "mu_II_deg": mu_II,
            "stable_I": np.isfinite(mu_I),
            "stable_II": np.isfinite(mu_II),
        }

    # --- Legacy fallback (deprecated approximation) ---------------------
    import warnings as _warnings
    _warnings.warn(
        "coupled_beam_phase_advance_per_cell_via_M: results carry no "
        "phase-probe maps — falling back to the LEGACY entrance-σ "
        "whole-element approximation (~9 % low on a matched FODO). "
        "Rerun the envelope with EnvelopeSolver(..., phase_probe=True) "
        "for the exact depressed tunes.",
        UserWarning, stacklevel=2,
    )
    sm = getattr(results, "sigma_matrix", None)
    s_arr = np.asarray(getattr(results, "s", []), dtype=float)
    if not sm or len(sm) != s_arr.size:
        raise ValueError(
            "envelope results carry no sigma_matrix per element; "
            "run the envelope tracker (records σ on every element by default)"
        )
    current = float(getattr(results, "current_mA", 0.0))
    continuous = bool(getattr(results, "continuous", False))

    def _entry_sigma_row(el_idx: int) -> int:
        """Record row holding element ``el_idx``'s ENTRANCE σ — goes
        through element_exit_idx so substep-recorded results map to the
        right rows (raw ``sm[el_idx]`` is wrong there)."""
        return element_record_span(results, el_idx, el_idx + 1)[0]

    # Replay the reference particle to the period entry.
    rc = ref.copy()
    for el in lattice.elements[:period.start]:
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc)
        else:
            rc.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc)

    # Cell spans from ``period.spans()`` — explicit per-repeat element
    # ranges built by walking *significant* elements (legacy constant
    # stride only as fallback for manually-built periods).
    cell_spans = period.spans()
    n = period.n_repeats
    mu_I = np.full(n, np.nan)
    mu_II = np.full(n, np.nan)
    stable_I = np.zeros(n, dtype=bool)
    stable_II = np.zeros(n, dtype=bool)

    for k in range(min(n, len(cell_spans))):
        M_dep = np.eye(6)
        span_a, span_b = cell_spans[k]
        for i in range(span_a, span_b):
            sig_row = _entry_sigma_row(i)
            if i >= len(lattice.elements) or sig_row >= len(sm):
                break
            el = lattice.elements[i]
            try:
                M_bare = get_element_matrix(el, rc, cache=cache)
            except Exception:                                 # noqa: BLE001
                M_bare = np.eye(6)
            ds = float(el.length) if el.length else 0.0
            sigma_i = np.asarray(sm[sig_row], dtype=float)
            # SC kick at the recorded σ using the same kernel the
            # envelope solver applied.  ds=0 means thin element →
            # identity (drift kick already absorbed by M_bare).
            if current > 0.0 and ds > 0.0 and sigma_i.shape == (6, 6):
                sx = math.sqrt(max(sigma_i[0, 0], 0.0))
                sy = math.sqrt(max(sigma_i[2, 2], 0.0))
                if continuous:
                    M_sc = _sc_kick_matrix_2d_dc(
                        current_mA=current,
                        charge_state=rc.species.charge,
                        mass_MeV=rc.species.mass,
                        beta=rc.beta, gamma=rc.gamma,
                        sigma_x_mm=sx, sigma_y_mm=sy, ds_mm=ds,
                    )
                else:
                    sphi = math.sqrt(max(sigma_i[4, 4], 0.0))
                    M_sc = _sc_kick_matrix_3d(
                        current_mA=current,
                        charge_state=rc.species.charge,
                        mass_MeV=rc.species.mass,
                        beta=rc.beta, gamma=rc.gamma,
                        frequency_MHz=rc.frequency,
                        sigma_x_mm=sx, sigma_y_mm=sy,
                        sigma_phi_deg=sphi, ds_mm=ds,
                    )
            else:
                M_sc = np.eye(6)
            # Envelope-solver order: M_bare first, then SC kick.
            M_step = M_sc @ M_bare
            M_dep = M_step @ M_dep
            # Advance ref through the element (matches matrix_tracking).
            if isinstance(el, FieldMapElement):
                el.advance_ref(rc)
            else:
                rc.s += el.length
                if isinstance(el, ThinKickElement):
                    el.advance_ref(rc)

        # Extract eigenmode tunes from the 4×4 transverse depressed map.
        M4 = M_dep[:4, :4]
        eigvals = np.linalg.eigvals(M4)
        moduli = np.abs(eigvals)
        phases_abs = np.sort(np.abs(np.angle(eigvals)))
        modes: list[float] = []
        for p in phases_abs:
            if not modes or abs(p - modes[-1]) > 1e-6:
                modes.append(float(p))
        while len(modes) < 2:
            modes.append(modes[-1] if modes else 0.0)
        mu_I[k] = math.degrees(modes[0])
        mu_II[k] = math.degrees(modes[1])
        order = np.argsort(np.abs(np.angle(eigvals)))
        moduli_sorted = moduli[order]
        stable_I[k] = bool(np.all(np.abs(moduli_sorted[:2] - 1.0) < 1e-2))
        stable_II[k] = bool(np.all(np.abs(moduli_sorted[2:] - 1.0) < 1e-2))

    return {
        "cells": np.arange(1, n + 1),
        "mu_I_deg": mu_I,
        "mu_II_deg": mu_II,
        "stable_I": stable_I,
        "stable_II": stable_II,
    }


def coupled_phase_advance_along_s(lattice, ref,
                                     period: PeriodicStructure,
                                     *, cache: dict | None = None,
                                     should_stop=None) -> dict:
    """Cumulative eigenmode μ_I(s), μ_II(s) along s for a coupled period.

    For each cell ``k`` in 1..n_repeats we extract eigenmode tunes
    ``μ_I_k, μ_II_k`` from the per-cell 4×4 transfer matrix.  The
    cumulative phase advance at the *exit* of cell ``k`` is then
    ``Σ_{j=1..k} μ_{plane,j}``.

    Returns
    -------
    dict with keys
        ``s_cells``    — 1-D float array, length ``n_repeats+1`` —
                         absolute s in mm at each cell boundary,
                         starting at the period entry.
        ``mu_I_deg``   — 1-D float array, length ``n_repeats+1`` —
                         cumulative μ_I (deg); element 0 is 0.
        ``mu_II_deg``  — 1-D float array, length ``n_repeats+1`` —
                         cumulative μ_II (deg).
        ``stable``     — bool, all cells were stable.
        ``s_period_start`` — float, absolute s in mm at period entry.
    """
    # Cell spans from ``period.spans()`` — explicit per-repeat element
    # ranges built by walking *significant* elements (legacy constant
    # stride only as fallback for manually-built periods).
    cell_spans = period.spans()
    n_reps = min(period.n_repeats, len(cell_spans))

    # Replay reference particle to the period entry for absolute s.
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    rc = ref.copy()
    for el in lattice.elements[:period.start]:
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc)
        else:
            rc.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc)
    s_at_period_start = float(rc.s)

    s_cells = np.zeros(n_reps + 1)
    s_cells[0] = s_at_period_start
    mu_I_cum = np.zeros(n_reps + 1)
    mu_II_cum = np.zeros(n_reps + 1)
    all_stable = True

    # Per-cell eigenvalue extraction; replay ref through cell to update s.
    for k in range(n_reps):
        _check_stop(should_stop)
        span_a, span_b = cell_spans[k]
        cell = PeriodicStructure(
            start=span_a,
            end=span_b,
            inner_period_length=period.inner_period_length,
            inner_slice_end=span_b,
            n_repeats=1,
            label=f"{period.label} cell {k+1}",
            source=period.source,
        )
        try:
            res = coupled_phase_advance(lattice, ref, cell, cache=cache)
            mu_I_cum[k + 1] = mu_I_cum[k] + res["mu_I_deg"]
            mu_II_cum[k + 1] = mu_II_cum[k] + res["mu_II_deg"]
            if not (res["stable_I"] and res["stable_II"]):
                all_stable = False
        except Exception:                                    # noqa: BLE001
            mu_I_cum[k + 1] = mu_I_cum[k]
            mu_II_cum[k + 1] = mu_II_cum[k]
            all_stable = False

        # Advance reference particle through the cell to update s.
        for i in range(span_a, span_b):
            el = lattice.elements[i]
            if isinstance(el, FieldMapElement):
                el.advance_ref(rc)
            else:
                rc.s += el.length
                if isinstance(el, ThinKickElement):
                    el.advance_ref(rc)
        s_cells[k + 1] = float(rc.s)

    return {
        "s_cells": s_cells,
        "mu_I_deg": mu_I_cum,
        "mu_II_deg": mu_II_cum,
        "stable": all_stable,
        "s_period_start": s_at_period_start,
    }


def coupled_phase_advance_per_cell(lattice, ref,
                                     period: PeriodicStructure,
                                     *, cache: dict | None = None) -> dict:
    """Per-cell eigenmode μ_I, μ_II for every repeat of a coupled period.

    Splits the period span into ``n_repeats`` consecutive cells of
    ``inner_period_length`` elements and returns the eigenmode tunes for
    each cell.  In a perfectly periodic lattice every cell gives the
    same (μ_I, μ_II); in accelerating sections the energy change
    detunes them per cell, which the returned arrays capture.

    Returns
    -------
    dict with keys
        ``cells``      — 1-D int array, 1..n_repeats
        ``mu_I_deg``   — 1-D float array, μ_I per cell (deg)
        ``mu_II_deg``  — 1-D float array, μ_II per cell (deg)
        ``stable_I``   — 1-D bool array
        ``stable_II``  — 1-D bool array
    """
    # Cell spans from ``period.spans()`` — explicit per-repeat element
    # ranges built by walking *significant* elements (legacy constant
    # stride only as fallback for manually-built periods).
    cell_spans = period.spans()
    n = period.n_repeats
    cells = np.arange(1, n + 1)
    mu_I = np.full(n, np.nan)
    mu_II = np.full(n, np.nan)
    s_I = np.zeros(n, dtype=bool)
    s_II = np.zeros(n, dtype=bool)
    for k in range(min(n, len(cell_spans))):
        span_a, span_b = cell_spans[k]
        cell = PeriodicStructure(
            start=span_a,
            end=span_b,
            inner_period_length=period.inner_period_length,
            inner_slice_end=span_b,
            n_repeats=1,
            label=f"{period.label} cell {k+1}",
            source=period.source,
        )
        try:
            res = coupled_phase_advance(lattice, ref, cell, cache=cache)
            mu_I[k] = res["mu_I_deg"]
            mu_II[k] = res["mu_II_deg"]
            s_I[k] = res["stable_I"]
            s_II[k] = res["stable_II"]
        except Exception:                                    # noqa: BLE001
            continue
    return {
        "cells": cells,
        "mu_I_deg": mu_I,
        "mu_II_deg": mu_II,
        "stable_I": s_I,
        "stable_II": s_II,
    }


# ---------------------------------------------------------------------------
# Channel tunes from the envelope phase probe (σ_model / σ₀_model)
# ---------------------------------------------------------------------------
_S4 = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 0],
], dtype=float)


def _pos_im_modes(M4: np.ndarray) -> "list[tuple[float, np.ndarray]] | None":
    """Positive-Im eigenmodes of a det-normalized 4×4 map.

    Returns two ``(mu_rad, eigvec)`` pairs (μ ∈ (0, π)) sorted by μ, or
    None when two distinct complex-conjugate pairs can't be identified
    (degenerate / real-only spectrum — resonance or instability)."""
    eigvals, eigvecs = np.linalg.eig(M4)
    idx = np.where(np.imag(eigvals) > 1e-12)[0]
    if idx.size != 2:
        return None
    modes = [(float(np.angle(eigvals[i])), eigvecs[:, i]) for i in idx]
    modes.sort(key=lambda t: t[0])
    return modes


def _pair_to_reference(modes, ref_vecs):
    """Reorder two ``(mu, v)`` modes to match ``ref_vecs`` by symplectic
    eigenvector overlap |v_ref† S₄ v| — the invariant pairing metric
    (a mode's symplectic product with ITSELF is what normalizes it, and
    cross-mode products vanish for a symplectic map)."""
    o = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            o[a, b] = abs(ref_vecs[a].conj() @ _S4 @ modes[b][1])
    # Two-mode assignment: keep order unless the swapped pairing
    # dominates both rows.
    if o[0, 1] + o[1, 0] > o[0, 0] + o[1, 1]:
        return [modes[1], modes[0]]
    return list(modes)


def _transverse_coupling_norm(M6: np.ndarray) -> float:
    """Relative magnitude of the x↔y off-blocks of the transverse map."""
    off = (np.abs(M6[0:2, 2:4]).sum() + np.abs(M6[2:4, 0:2]).sum())
    on = max(np.abs(M6[0:2, 0:2]).sum() + np.abs(M6[2:4, 2:4]).sum(), 1e-30)
    return off / on


def channel_phase_advance_matched(lattice, ref, period: PeriodicStructure,
                                  current: float, base_initial: dict, *,
                                  longitudinal: str = "iterate",
                                  coupling_tol: float = 1e-6,
                                  should_stop=None,
                                  **fixed_point_kw) -> dict:
    """Channel tunes of the MATCHED channel — the primary σ_model.

    Solves the full-Σ matched fixed point of the period's first cell
    (``matching.periodic.find_matched_period_sigma`` — normalized-
    coordinate iteration with eigenemittance renormalization, prefix
    state preserved), then runs the phase probe over the whole period
    span seeded with the matched Σ and extracts the channel tunes.
    Unlike ``channel_phase_advance`` on a plain tracked run, the result
    does not depend on how well the USER's beam was matched.

    Returns the :func:`channel_phase_advance` dict plus
    ``matched_state`` (converged/n_iter/residual/sigma_model_approx).
    """
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.lattice_commands import LatticeCommand
    from linac_gen.matching.periodic import find_matched_period_sigma
    from linac_gen.tracking.envelope import EnvelopeSolver

    ms = find_matched_period_sigma(lattice, ref, period, current,
                                   base_initial,
                                   longitudinal=longitudinal,
                                   **fixed_point_kw)
    spans = period.spans()
    a0 = spans[0][0]
    b_end = spans[-1][1]

    span_lat = Lattice()
    for e in lattice.elements[a0:b_end]:
        span_lat.add(e)
    step_cfg = getattr(lattice, "step_config", None)
    if step_cfg is not None:
        span_lat.step_config = step_cfg

    init = dict(base_initial)
    init["continuous"] = ms["continuous"]
    env = EnvelopeSolver(span_lat, ms["ref_entry"].copy(), init,
                         current=current,
                         initial_sigma=ms["sigma_entry"],
                         bunch_frequency=ms["bunch_frequency"],
                         sc_factor=ms["sc_factor"],
                         phase_probe=True)
    for cmd in (e for e in lattice.elements[:a0]
                if isinstance(e, LatticeCommand)):
        try:
            cmd.apply_command(env.track_state)
        except Exception:                                    # noqa: BLE001
            pass
    res = env.run()

    shifted = PeriodicStructure(
        start=0, end=b_end - a0,
        inner_period_length=period.inner_period_length,
        inner_slice_end=spans[0][1] - a0,
        n_repeats=period.n_repeats,
        label=period.label, source=period.source,
        repeat_spans=tuple((a - a0, b - a0) for (a, b) in spans),
    )
    out = channel_phase_advance(res, shifted, coupling_tol=coupling_tol,
                                should_stop=should_stop)
    out["matched_state"] = {
        "converged": ms["converged"],
        "n_iter": ms["n_iter"],
        "residual": ms["residual"],
        "sigma_model_approx": ms["sigma_model_approx"],
        "longitudinal": ms["longitudinal"],
    }
    return out


def run_phase_probe(lattice, ref, initial: dict, current: float = 0.0,
                    **solver_kwargs):
    """Convenience: run the envelope with the phase probe enabled and
    return the probe-bearing :class:`EnvelopeResults`.

    Use this to (re)generate the exact per-element tangent maps that
    :func:`channel_phase_advance` and the probe-based via-M path
    consume when an existing results object predates the probe.
    ``solver_kwargs`` pass through to ``EnvelopeSolver`` (e.g.
    ``record_substeps=True``, ``initial_sigma=…``,
    ``bunch_frequency=…``).
    """
    from linac_gen.tracking.envelope import EnvelopeSolver
    solver_kwargs.setdefault("phase_probe", True)
    return EnvelopeSolver(lattice, ref.copy(), initial, current=current,
                          **solver_kwargs).run()


def channel_phase_advance(results, period: PeriodicStructure, *,
                          coupling_tol: float = 1e-6,
                          should_stop=None) -> dict:
    """Channel (matched-model) tunes σ₀_model and σ_model per cell from
    the envelope phase probe — the depressed tune of the linear channel,
    independent of the tracked beam's match state.

    Requires probe-bearing results: run
    ``EnvelopeSolver(..., phase_probe=True)``.  The per-element maps are
    the solver's EXACT tangent factors (identical Strang slices,
    midpoint-σ SC kicks, freq-jump D, edges), so the per-cell monodromy
    M_dep = Π maps is the one-period map with the frozen linearized SC
    field of the run — the OPAL/TRACE-3D-style depressed map — and
    M_bare (SC factors → identity, same slices) gives the consistent
    zero-current counterpart.  Never build σ₀ from a separate I=0 run:
    full-element FieldMap maps differ from the SC walk's slicing.

    Extraction: det-normalized per-plane Twiss when the cell is
    transversely uncoupled; normal modes I/II (det(M₄)^{1/4}
    normalization) when coupled, with bare↔depressed and cell↔cell mode
    pairing by symplectic eigenvector overlap.  Machine totals are the
    sums of the paired per-cell principal increments — never the
    aliased eigen-angle of a whole multi-cell product.

    Returns
    -------
    dict
        ``cells`` — 1..n; ``coupled_xy`` — bool;
        uncoupled: ``mu_{x,y,z}_bare_deg``, ``mu_{x,y,z}_dep_deg``,
        ``eta_{x,y,z}`` (per-cell arrays, NaN where extraction failed);
        coupled: ``mu_{I,II}_bare_deg``, ``mu_{I,II}_dep_deg``,
        ``eta_{I,II}``;
        both: ``damping`` (per-cell det(M₄_dep)^{1/4}), machine totals
        ``total_mu_*`` for every reported μ series, and
        ``eta_machine_*`` = total_dep/total_bare.
    """
    maps_dep = getattr(results, "element_maps_dep", None) or []
    maps_bare = getattr(results, "element_maps_bare", None) or []
    if not maps_dep or len(maps_dep) != len(maps_bare):
        raise ValueError(
            "results carry no phase-probe maps — run the envelope with "
            "EnvelopeSolver(..., phase_probe=True) (or the "
            "run_phase_probe() helper).  MP tracking results cannot "
            "provide channel tunes: the PIC field has no recorded "
            "tangent map, and the linearized-ellipsoid kick is not the "
            "tangent field of a nonlinear PIC distribution."
        )

    cell_spans = [sp for sp in period.spans() if sp[1] <= len(maps_dep)]
    n = len(cell_spans)
    if n == 0:
        raise ValueError("period spans no probed elements")
    cells = np.arange(1, n + 1)

    # Per-cell monodromies from the probe factors.
    M_dep_cells: list = []
    M_bare_cells: list = []
    for (a, b) in cell_spans:
        _check_stop(should_stop)
        Md = np.eye(6)
        Mb = np.eye(6)
        for j in range(a, b):
            Md = maps_dep[j] @ Md
            Mb = maps_bare[j] @ Mb
        M_dep_cells.append(Md)
        M_bare_cells.append(Mb)

    coupled = any(
        _transverse_coupling_norm(M) > coupling_tol
        for M in (*M_dep_cells, *M_bare_cells)
    )

    damping = np.full(n, np.nan)
    out: dict = {"cells": cells, "coupled_xy": coupled}

    if not coupled:
        for plane in ("x", "y", "z"):
            for tag, mats in (("bare", M_bare_cells), ("dep", M_dep_cells)):
                arr = np.full(n, np.nan)
                for k, M in enumerate(mats):
                    try:
                        tw = compute_twiss(M, plane, coupling_tol=1e-3)
                        arr[k] = tw["mu_folded"]
                    except ValueError:
                        pass
                out[f"mu_{plane}_{tag}_deg"] = arr
            b = out[f"mu_{plane}_bare_deg"]
            d = out[f"mu_{plane}_dep_deg"]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"eta_{plane}"] = np.where(b > 0, d / b, np.nan)
        for k, M in enumerate(M_dep_cells):
            det4 = float(np.linalg.det(M[:4, :4]))
            damping[k] = det4 ** 0.25 if det4 > 0 else np.nan
        mu_keys = [f"mu_{p}_{t}_deg" for p in ("x", "y", "z")
                   for t in ("bare", "dep")]
        eta_pairs = [("x", "x"), ("y", "y"), ("z", "z")]
    else:
        mu_b = {m: np.full(n, np.nan) for m in ("I", "II")}
        mu_d = {m: np.full(n, np.nan) for m in ("I", "II")}
        ref_vecs = None
        for k in range(n):
            _check_stop(should_stop)
            Mb4 = M_bare_cells[k][:4, :4]
            Md4 = M_dep_cells[k][:4, :4]
            det_b = float(np.linalg.det(Mb4))
            det_d = float(np.linalg.det(Md4))
            if det_b <= 0 or det_d <= 0:
                continue
            damping[k] = det_d ** 0.25
            modes_b = _pos_im_modes(Mb4 / det_b ** 0.25)
            modes_d = _pos_im_modes(Md4 / det_d ** 0.25)
            if modes_b is None or modes_d is None:
                continue
            if ref_vecs is None:
                # Mode labels seeded from the first cell's bare
                # monodromy (μ ascending → I, II).
                ref_vecs = [modes_b[0][1], modes_b[1][1]]
            else:
                modes_b = _pair_to_reference(modes_b, ref_vecs)
            # Depressed modes pair against the SAME cell's bare modes.
            modes_d = _pair_to_reference(
                modes_d, [modes_b[0][1], modes_b[1][1]])
            mu_b["I"][k] = math.degrees(modes_b[0][0])
            mu_b["II"][k] = math.degrees(modes_b[1][0])
            mu_d["I"][k] = math.degrees(modes_d[0][0])
            mu_d["II"][k] = math.degrees(modes_d[1][0])
            # Cell-to-cell continuity: carry this cell's bare basis.
            ref_vecs = [modes_b[0][1], modes_b[1][1]]
        for m in ("I", "II"):
            out[f"mu_{m}_bare_deg"] = mu_b[m]
            out[f"mu_{m}_dep_deg"] = mu_d[m]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"eta_{m}"] = np.where(mu_b[m] > 0,
                                           mu_d[m] / mu_b[m], np.nan)
        # The longitudinal plane usually stays decoupled from the x-y
        # rotation — extract it per cell like the uncoupled path.
        for tag, mats in (("bare", M_bare_cells), ("dep", M_dep_cells)):
            arr = np.full(n, np.nan)
            for k, M in enumerate(mats):
                try:
                    tw = compute_twiss(M, "z", coupling_tol=1e-3)
                    arr[k] = tw["mu_folded"]
                except ValueError:
                    pass
            out[f"mu_z_{tag}_deg"] = arr
        with np.errstate(divide="ignore", invalid="ignore"):
            out["eta_z"] = np.where(out["mu_z_bare_deg"] > 0,
                                    out["mu_z_dep_deg"]
                                    / out["mu_z_bare_deg"], np.nan)
        mu_keys = [f"mu_{m}_{t}_deg" for m in ("I", "II", "z")
                   for t in ("bare", "dep")]
        eta_pairs = [("I", "I"), ("II", "II"), ("z", "z")]

    out["damping"] = damping
    # Machine totals: sums of paired per-cell principal increments.
    for key in mu_keys:
        vals = out[key]
        fin = vals[np.isfinite(vals)]
        out[f"total_{key[:-4]}"] = float(fin.sum()) if fin.size else float("nan")
    for pa, pb in eta_pairs:
        tb = out.get(f"total_mu_{pa}_bare")
        td = out.get(f"total_mu_{pb}_dep")
        out[f"eta_machine_{pa}"] = (td / tb if tb and np.isfinite(tb)
                                    and tb > 0 and np.isfinite(td)
                                    else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Beam phase advance (uses envelope results)
# ---------------------------------------------------------------------------
def beam_phase_advance(
    results, period: PeriodicStructure, *,
    sigma0: Optional[dict] = None,
    matched_tol: float = 0.05,
) -> dict:
    """Compute σ_x, σ_y for one period from a matched envelope run.

    Parameters
    ----------
    results : EnvelopeResults
        Output of the envelope solver — must contain populated
        ``s``, ``beta_x``, ``beta_y`` arrays.
    period : PeriodicStructure
        Period to integrate over (s-positions are looked up from
        ``results.s`` based on the period's element-index span).
    sigma0 : dict, optional
        ``structure_phase_advance`` output; if provided, the returned
        dict includes ``sigma_over_sigma0_x`` / ``_y`` (tune
        depression).
    matched_tol : float
        Maximum fractional difference ``|β_end − β_start| / β_start``
        before the beam is flagged as not matched to the period.

    Returns
    -------
    dict with keys
        ``mu_x_deg``, ``mu_y_deg`` — beam phase advance (deg).
        ``matched`` — bool; ``False`` ⇒ value is unreliable.
        ``mismatch_x``, ``mismatch_y`` — fractional β mismatch.
        ``sigma_over_sigma0_x``, ``sigma_over_sigma0_y`` — tune
            depression ratio (only set when ``sigma0`` is provided).
        ``s_start``, ``s_end`` — physical span used (mm).
    """
    s = np.asarray(results.s, dtype=float)
    if s.size < 2:
        raise ValueError(
            "envelope results have fewer than 2 sample points; cannot integrate"
        )
    n_elements_total = len(getattr(results, "element_names", []) or [])

    # Translate element-index span → record-row span.  With
    # record_substeps off this is the legacy identity mapping (row i =
    # exit of element i−1); with substeps on it MUST go through
    # element_exit_idx or the span silently lands on the wrong rows.
    r0, r1 = element_record_span(results, period.start,
                                 period.inner_slice_end)
    s_start_idx = max(0, min(r0, s.size - 1))
    s_end_idx = max(s_start_idx + 1, min(r1, s.size - 1))
    s_slice = s[s_start_idx:s_end_idx + 1]
    if s_slice.size < 2:
        raise ValueError("period covers fewer than 2 envelope samples")

    bx = np.asarray(results.beta_x[s_start_idx:s_end_idx + 1], dtype=float)
    by = np.asarray(results.beta_y[s_start_idx:s_end_idx + 1], dtype=float)

    # NumPy 2.0 renamed trapz → trapezoid; keep both code paths working.
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    # Units: this codebase stores lengths in mm and angles in mrad, and
    # β from σ²/ε has units mm/mrad ≡ m/rad numerically.  ∫ds[mm]/β[mm/mrad]
    # has units of mrad, so the conversion to degrees is mrad → deg
    # (× 180/π / 1000), not rad → deg.  Equivalently: convert s to metres
    # before integrating — same numerical answer.
    _MRAD_TO_DEG = (180.0 / math.pi) * 1e-3

    # MP runs write β=0 at any step where all particles have been lost
    # (recorder.py:221).  Treat those samples as gaps: integrate the
    # phase advance over each contiguous run of surviving samples and
    # sum — NEVER trapz across a gap (bridging fabricates phase advance
    # over a span where β is unknown).  For fully-valid spans this is
    # numerically identical to a single trapz call.
    mask_x = bx > 0
    mask_y = by > 0
    n_skip_x = int(mask_x.size - mask_x.sum())
    n_skip_y = int(mask_y.size - mask_y.sum())
    if mask_x.sum() < 2 or mask_y.sum() < 2:
        raise ValueError(
            "period has fewer than 2 surviving envelope samples "
            f"(β_x>0 at {mask_x.sum()}/{mask_x.size}, "
            f"β_y>0 at {mask_y.sum()}/{mask_y.size})"
        )
    mu_x = _integrate_inverse_beta_split(s_slice, bx, mask_x) * _MRAD_TO_DEG
    mu_y = _integrate_inverse_beta_split(s_slice, by, mask_y) * _MRAD_TO_DEG

    # Longitudinal: EFFECTIVE β̃_z in locally-normalized (z, z′)
    # coordinates — the native (Δφ, ΔW) β is in deg/MeV and must not be
    # fed to the transverse mrad→deg integrand (that error made μ_z
    # ~685× too small at PIP-II injection energies).
    bz_full = _beta_z_eff_from_sigma(results)
    mu_z = None
    mismatch_z = None
    if bz_full is not None and bz_full.size == np.asarray(results.s).size:
        bz = bz_full[s_start_idx:s_end_idx + 1]
        mask_z = bz > 0
        if mask_z.sum() >= 2:
            mu_z = _integrate_inverse_beta_split(
                s_slice, bz, mask_z) * _MRAD_TO_DEG
            bz_valid = bz[mask_z]
            mismatch_z = float(abs(bz_valid[-1] - bz_valid[0]) / bz_valid[0])

    # Use the first / last *surviving* samples for the mismatch check so
    # a trailing all-lost tail doesn't poison the comparison.
    bx_valid = bx[mask_x]; by_valid = by[mask_y]
    mismatch_x = float(abs(bx_valid[-1] - bx_valid[0]) / bx_valid[0])
    mismatch_y = float(abs(by_valid[-1] - by_valid[0]) / by_valid[0])
    matched_xy = bool((mismatch_x <= matched_tol) and (mismatch_y <= matched_tol))
    matched_z  = (mismatch_z is None) or bool(mismatch_z <= matched_tol)

    # Covariance-based mismatch (BMAG): β-endpoint agreement alone
    # misses α mismatch (an envelope oscillating with ~cell period).
    # BMAG = ½(β₁γ₂ − 2α₁α₂ + γ₁β₂) ≥ 1, computed on the DISPERSION-
    # SUBTRACTED betatron Twiss so dispersive beam size doesn't read as
    # mismatch; using the LOCAL emittance in β = σ²/ε makes the metric
    # invariant under acceleration/frequency scaling.
    bmag: dict = {"x": None, "y": None, "z": None}
    sm_all = getattr(results, "sigma_matrix", None)
    if sm_all is not None and len(sm_all) == s.size:
        S_in = np.asarray(sm_all[s_start_idx], dtype=float)
        S_out = np.asarray(sm_all[s_end_idx], dtype=float)
        for plane, (i, j) in (("x", (0, 1)), ("y", (2, 3)), ("z", (4, 5))):
            t1 = _betatron_twiss_from_sigma(S_in, i, j)
            t2 = _betatron_twiss_from_sigma(S_out, i, j)
            if t1 is not None and t2 is not None:
                bmag[plane] = mismatch_factor(t1[0], t1[1], t2[0], t2[1])
    bmag_ok = all(v is None or v <= 1.0 + matched_tol for v in bmag.values())

    # Coupled-interior guard: ∫ds/β of the PROJECTED β is not a
    # normal-mode tune inside x–y-coupled regions (solenoid interiors —
    # measured +16–20 % over the PIP-II HWR with substep records).
    # Element-boundary records are usually near-decoupled, so this
    # fires mostly on substep-recorded runs — exactly where it matters.
    projected_only = False
    if sm_all is not None and len(sm_all) == s.size:
        for irec in range(s_start_idx, s_end_idx + 1):
            S = np.asarray(sm_all[irec], dtype=float)
            denom = math.sqrt(max(S[0, 0] * S[2, 2], 1e-30))
            if np.abs(S[0:2, 2:4]).max() > 0.05 * denom:
                projected_only = True
                break
    if projected_only:
        import warnings as _warnings
        _warnings.warn(
            "beam_phase_advance: records inside this span are x-y "
            "coupled (solenoid interior) — the projected-β ∫ds/β is "
            "NOT a normal-mode tune there.  Use the channel tunes "
            "(channel_phase_advance) as the authoritative number.",
            UserWarning, stacklevel=2,
        )

    # Resolution: samples per full betatron period (2π).  Below ~20 the
    # trapezoid through thick elements is quantitatively unreliable
    # (measured −4.5 % over the PIP-II MEBT at element-boundary
    # sampling).
    n_valid = int(min(mask_x.sum(), mask_y.sum()))
    mu_max = max(mu_x, mu_y)
    samples_per_period = (n_valid / (mu_max / 360.0)) if mu_max > 0 else float("inf")
    resolution_ok = bool(samples_per_period >= 20.0)

    out = {
        "mu_x_deg": mu_x,
        "mu_y_deg": mu_y,
        "mu_z_deg": mu_z,
        "matched": matched_xy and matched_z and bmag_ok,
        "mismatch_x": mismatch_x,
        "mismatch_y": mismatch_y,
        "mismatch_z": mismatch_z,
        "mismatch_bmag_x": bmag["x"],
        "mismatch_bmag_y": bmag["y"],
        "mismatch_bmag_z": bmag["z"],
        "projected_only": projected_only,
        "n_samples": n_valid,
        "samples_per_period": samples_per_period,
        "resolution_ok": resolution_ok,
        "s_start": float(s_slice[0]),
        "s_end": float(s_slice[-1]),
        "n_skipped_x": n_skip_x,
        "n_skipped_y": n_skip_y,
        "n_total": int(mask_x.size),
    }
    if sigma0 is not None:
        for plane in ("x", "y", "z"):
            # Branch-consistent denominator: the beam integral is an
            # unwrapped magnitude; prefer the oriented branch value
            # (equal to the principal one below 180°).
            mu_struct = (sigma0.get(f"mu_{plane}_branch_deg")
                         or sigma0.get(f"mu_{plane}_deg"))
            mu_beam = out[f"mu_{plane}_deg"]
            if mu_struct and mu_beam is not None and mu_struct > 0:
                out[f"sigma_over_sigma0_{plane}"] = mu_beam / mu_struct
            else:
                out[f"sigma_over_sigma0_{plane}"] = None
    return out


def mismatch_factor(alpha1: float, beta1: float,
                    alpha2: float, beta2: float) -> float:
    """Courant-Snyder mismatch factor BMAG ≥ 1 between two Twiss sets.

    BMAG = ½(β₁γ₂ − 2α₁α₂ + γ₁β₂); 1 = perfectly matched.  The
    conventional mismatch amplitude is BMAG + √(BMAG² − 1).
    """
    g1 = (1.0 + alpha1 * alpha1) / beta1
    g2 = (1.0 + alpha2 * alpha2) / beta2
    return 0.5 * (beta1 * g2 - 2.0 * alpha1 * alpha2 + g1 * beta2)


def _betatron_twiss_from_sigma(S: np.ndarray, i: int, j: int):
    """Dispersion-subtracted (α, β) of one plane from a 6×6 Σ.

    Returns None for degenerate blocks (zero emittance / lost beam).
    Subtracting the dispersive correlation with ΔW (Σ[:,5]) isolates
    the betatron ellipse, and using the LOCAL emittance makes the
    result invariant under acceleration scaling.
    """
    s_ww = float(S[5, 5])
    if s_ww > 0.0 and j != 5:
        sii = S[i, i] - S[i, 5] ** 2 / s_ww
        sjj = S[j, j] - S[j, 5] ** 2 / s_ww
        sij = S[i, j] - S[i, 5] * S[j, 5] / s_ww
    else:
        sii, sjj, sij = float(S[i, i]), float(S[j, j]), float(S[i, j])
    eps_sq = sii * sjj - sij * sij
    if eps_sq <= 0.0 or sii <= 0.0:
        return None
    eps = math.sqrt(eps_sq)
    return (-sij / eps, sii / eps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _energy_change(lattice, ref, start: int, end_incl: int) -> tuple[float, float]:
    """Return (w_in, w_out) for the slice [start, end_incl] inclusive.

    Mirrors the energy-advancing logic of ``compute_transfer_matrix``
    so the two views agree.
    """
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    rc = ref.copy()
    # Replay before the period.
    for el in lattice.elements[:start]:
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc)
        else:
            rc.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc)
    w_in = float(rc.w_kin)
    for el in lattice.elements[start:end_incl + 1]:
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc)
        else:
            rc.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc)
    w_out = float(rc.w_kin)
    return w_in, w_out


# ---------------------------------------------------------------------------
# Along-s phase advance (μ as a function of position)
# ---------------------------------------------------------------------------
def structure_phase_advance_along_s(lattice, ref, period: PeriodicStructure,
                                       seed: dict | None = None,
                                       *, cache: dict | None = None,
                                       should_stop=None) -> dict:
    """Cumulative structure phase advance μ₀(s) along the whole lattice.

    Seeds the periodic Twiss (α₀, β₀) at the **period entry**, then
    propagates Courant-Snyder forward element-by-element through the
    rest of the lattice using each element's transfer matrix.  The
    accumulated phase advance ``Δμ_i = atan2(c12, c11·β - c12·α)`` is
    summed to give μ(s) at every element boundary.

    Parameters
    ----------
    cache : dict, optional
        Opt-in per-element transfer-matrix memoisation dict; see
        :func:`linac_gen.tracking.matrix_tracking.get_element_matrix`.
        Default ``None`` reproduces the original behaviour exactly.

    Returns
    -------
    dict
        ``s`` (mm), ``mu_x_deg``, ``mu_y_deg``, ``beta_x``, ``beta_y``,
        ``alpha_x``, ``alpha_y`` — all 1-D arrays of length
        ``len(lattice.elements) + 1`` (entry + each exit).  The seed
        Twiss is from ``period``.

    Notes
    -----
    *Coupled lattices* (solenoids, skew quads): we propagate the 2×2
    block independently per plane.  When coupling is present this is
    only an approximation; the per-period μ is still well-defined for
    a decoupled period that brackets the coupled section.

    *Accelerating elements*: ``compute_transfer_matrix`` already
    advances ``ref.w_kin`` through the lattice, so the per-element
    matrices are at the correct kinetic energy.  The Twiss
    propagation is symplectic only if the matrices are symplectic —
    which holds for our linear-transport library.
    """
    import numpy as np
    from linac_gen.tracking.matrix_tracking import (
        get_element_matrix, compute_twiss,
    )
    from linac_gen.elements.base import FieldMapElement, ThinKickElement

    # Seed the Twiss values from the period's one-period matrix.  If
    # the caller already computed it (popups cache), accept it directly
    # to avoid duplicating the slow per-element matrix walk.
    if seed is None:
        try:
            seed = structure_phase_advance(lattice, ref, period,
                                           cache=cache,
                                           should_stop=should_stop)
        except Exception as exc:                              # noqa: BLE001
            from linac_gen.core.cancelled import OperationCancelled
            if isinstance(exc, OperationCancelled):
                raise    # a cancel must never masquerade as "no seed"
            seed = {
                "alpha_x": None, "beta_x": None,
                "alpha_y": None, "beta_y": None,
            }
    a0_x, b0_x = seed.get("alpha_x"), seed.get("beta_x")
    a0_y, b0_y = seed.get("alpha_y"), seed.get("beta_y")
    a0_z, b0_z = seed.get("alpha_z"), seed.get("beta_z")

    n = len(lattice.elements)
    s_arr = np.zeros(n + 1)
    mu_x = np.zeros(n + 1); mu_y = np.zeros(n + 1); mu_z = np.zeros(n + 1)
    bx = np.full(n + 1, np.nan); by = np.full(n + 1, np.nan); bz = np.full(n + 1, np.nan)
    ax = np.full(n + 1, np.nan); ay = np.full(n + 1, np.nan); az = np.full(n + 1, np.nan)

    # Replay reference particle to the period entry; the seed Twiss is
    # at *that* position, so we must propagate forward from it.  For
    # element indices BEFORE the period entry the per-element transfer
    # is back-applied to obtain the Twiss at s=0 — but that requires
    # inverting the matrix, which is messy.  Simpler: only emit μ(s)
    # *from the period entry onward*; before that, leave NaN.
    rc = ref.copy()
    for el in lattice.elements[:period.start]:
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc)
        else:
            rc.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc)
    s_at_period_start = float(rc.s)

    # Pre-fill the s grid (always defined regardless of seed availability).
    rc2 = ref.copy()
    s_arr[0] = float(rc2.s)
    for i, el in enumerate(lattice.elements):
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc2)
        else:
            rc2.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc2)
        s_arr[i + 1] = float(rc2.s)

    # If a plane's seed is missing (coupled, unstable), its curves stay
    # NaN — the other planes still propagate.
    have_x = a0_x is not None and b0_x is not None
    have_y = a0_y is not None and b0_y is not None
    have_z = a0_z is not None and b0_z is not None
    if not (have_x or have_y or have_z):
        return {
            "s": s_arr,
            "mu_x_deg": mu_x, "mu_y_deg": mu_y, "mu_z_deg": mu_z,
            "beta_x": bx, "beta_y": by, "beta_z": bz,
            "alpha_x": ax, "alpha_y": ay, "alpha_z": az,
            "seed_index": period.start, "seeded": False,
            "have_x": False, "have_y": False, "have_z": False,
        }

    # Initial values at the period entry.
    if have_x:
        bx[period.start] = b0_x; ax[period.start] = a0_x
    if have_y:
        by[period.start] = b0_y; ay[period.start] = a0_y
    if have_z:
        bz[period.start] = b0_z; az[period.start] = a0_z

    a_x, b_x = (a0_x or 0.0), (b0_x or 1.0)
    a_y, b_y = (a0_y or 0.0), (b0_y or 1.0)
    a_z, b_z = (a0_z or 0.0), (b0_z or 1.0)
    mux = 0.0; muy = 0.0; muz = 0.0

    rc3 = ref.copy()
    # Replay before the period entry to set energy correctly.
    for el in lattice.elements[:period.start]:
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc3)
        else:
            rc3.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc3)

    def _propagate(c11, c12, c21, c22, alpha, beta):
        """Twiss-propagate one 2×2 block; return ``(alpha_new, beta_new, dmu)``.

        Phase increment uses ``atan2(c12, c11·β - c12·α)`` for quadrant
        correctness then takes its absolute value, so the cumulative
        μ(s) is monotonically positive — the standard MAD-X / PTC /
        TraceWin convention for "phase advance accumulated along s".

        For a stable matched cell, ``Σ |Δμ|`` over one period equals
        ``acos(½ tr(M_period))`` — i.e. it agrees with the unsigned
        period-Twiss extraction.  In our 6×6 basis the longitudinal
        block has ``c12 < 0`` for drifts (phase slip) and the bare
        ``atan2`` would accumulate a negative cumulative; the
        ``abs`` keeps the convention uniform across planes.

        Accelerating blocks (det ≠ 1): the geometric emittance scales
        by det, so β/α must be divided by it (mirrors
        ``propagate_twiss``).  The phase increment is unaffected —
        ``atan2(k·a, k·b) == atan2(a, b)`` for the conformal scale
        k = √det > 0 — but without the β/α correction every DOWNSTREAM
        increment would be biased.
        """
        g = (1.0 + alpha * alpha) / beta if beta > 0 else 0.0
        beta_new = c11 * c11 * beta - 2.0 * c11 * c12 * alpha + c12 * c12 * g
        alpha_new = (
            -c11 * c21 * beta
            + (c11 * c22 + c12 * c21) * alpha
            - c12 * c22 * g
        )
        det = c11 * c22 - c12 * c21
        if det > 0.0 and abs(det - 1.0) > 1e-9:
            beta_new /= det
            alpha_new /= det
        denom = c11 * beta - c12 * alpha
        if c12 == 0.0 and denom == 0.0:
            dmu = 0.0
        else:
            dmu = abs(math.atan2(c12, denom))
        return alpha_new, beta_new, dmu

    for i in range(period.start, n):
        _check_stop(should_stop)
        el = lattice.elements[i]
        try:
            M = get_element_matrix(el, rc3, cache=cache)
        except Exception:                                     # noqa: BLE001
            # Unsupported element — break the chain; downstream stays NaN.
            break

        if have_x:
            a_x, b_x, dmu_x = _propagate(
                M[0, 0], M[0, 1], M[1, 0], M[1, 1], a_x, b_x,
            )
            mux += dmu_x
            bx[i + 1] = b_x; ax[i + 1] = a_x
            mu_x[i + 1] = math.degrees(mux)
        if have_y:
            a_y, b_y, dmu_y = _propagate(
                M[2, 2], M[2, 3], M[3, 2], M[3, 3], a_y, b_y,
            )
            muy += dmu_y
            by[i + 1] = b_y; ay[i + 1] = a_y
            mu_y[i + 1] = math.degrees(muy)
        if have_z:
            a_z, b_z, dmu_z = _propagate(
                M[4, 4], M[4, 5], M[5, 4], M[5, 5], a_z, b_z,
            )
            muz += dmu_z
            bz[i + 1] = b_z; az[i + 1] = a_z
            mu_z[i + 1] = math.degrees(muz)

        # Advance the reference particle through the element.
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc3)
        else:
            rc3.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc3)

    # Clear pre-period samples (Twiss undefined there).
    for j in range(period.start):
        if have_x: mu_x[j] = np.nan
        if have_y: mu_y[j] = np.nan
        if have_z: mu_z[j] = np.nan
    # Planes whose seed was missing are NaN throughout.
    if not have_x: mu_x[:] = np.nan
    if not have_y: mu_y[:] = np.nan
    if not have_z: mu_z[:] = np.nan

    return {
        "s": s_arr,
        "mu_x_deg": mu_x, "mu_y_deg": mu_y, "mu_z_deg": mu_z,
        "beta_x": bx, "beta_y": by, "beta_z": bz,
        "alpha_x": ax, "alpha_y": ay, "alpha_z": az,
        "seed_index": period.start, "seeded": True,
        "have_x": have_x, "have_y": have_y, "have_z": have_z,
    }


def element_record_span(results, el_start: int, el_end_excl: int) -> tuple:
    """Translate an element-index span into record-row indices.

    Returns ``(r0, r1)`` such that record rows ``r0 .. r1`` (both
    inclusive) bracket elements ``[el_start, el_end_excl)``: ``r0`` is
    the row holding element ``el_start``'s ENTRANCE state and ``r1``
    the row holding element ``el_end_excl − 1``'s EXIT state.

    Uses ``results.element_exit_idx`` (envelope solver / MP recorder,
    one entry per element) when present — the ONLY correct mapping when
    ``record_substeps`` inserted a variable number of interior rows per
    element.  Falls back to the legacy identity mapping
    ``(el_start, el_end_excl)`` for results objects that predate the
    field (old pickles, hand-built test fixtures), which is exact for
    one-record-per-element data (row 0 = INPUT, row i = exit of
    element i−1).
    """
    exit_idx = getattr(results, "element_exit_idx", None)
    if exit_idx is None:                    # `or []` broke on ndarrays:
        exit_idx = []                       # truth value is ambiguous
    if len(exit_idx) >= el_end_excl >= 1:
        r0 = 0 if el_start <= 0 else int(exit_idx[el_start - 1])
        r1 = int(exit_idx[el_end_excl - 1])
        return r0, r1
    return el_start, el_end_excl


def _integrate_inverse_beta_split(s_vals, beta_vals, mask) -> float:
    """∑ of trapz(1/β) over each CONTIGUOUS run of valid samples.

    Invalid samples (β ≤ 0 — all particles lost at that record) split
    the integral instead of being masked out and bridged: a trapezoid
    spanning a gap fabricates phase advance across a region where β is
    unknown.  For a fully-valid span this equals a single trapz call
    bit-for-bit (one run covering the whole array).
    """
    import numpy as np
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    s_vals = np.asarray(s_vals, dtype=float)
    beta_vals = np.asarray(beta_vals, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    total = 0.0
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if j > i:
            total += float(_trapz(1.0 / beta_vals[i:j + 1],
                                  s_vals[i:j + 1]))
        i = j + 1
    return total


def _beta_z_eff_from_sigma(results) -> "np.ndarray | None":
    """EFFECTIVE longitudinal β̃_z [mm/mrad] for phase-advance integration.

    The recorded σ-matrix longitudinal block is in HELIX's native
    (Δφ [deg], ΔW [MeV]) pair, so ``σ_φφ/ε_φW`` has units deg/MeV and
    must NOT be fed to the transverse mrad→deg integrand (doing so made
    μ_z ~685× too small at PIP-II injection).  Convert per record with
    the (z, z′) Jacobian (same as recorder._convert_emit_z_to_mmmrad):

        k_φ = 360 / (β·λ)            [deg/mm]
        k_w = β²·γ·m / 1000          [MeV/mrad]   (z′ = 10³·δp/p)
        β̃_z = γ² · (k_w/k_φ) · σ_φφ/ε_φW          [mm/mrad]

    The γ² factor is the longitudinal drift coefficient (dz/ds = z′/γ²
    in slip coordinates): with it, ∫ds/β̃_z in the transverse mrad→deg
    convention yields the physical longitudinal phase advance
    dμ_z/ds = ε_zz′/(γ²σ_z²).

    Per-record β, γ, frequency come from ``results.ref_*``; the rest
    mass from ``results.mass_mev`` or, failing that, from
    ``w_kin/(γ−1)``.  Returns None when neither the σ-matrices nor the
    (σ_φ, ε_z) fallback pair is available, or the ref arrays are
    missing; entries where any factor degenerates are 0 (masked out by
    the integrators like a lost-beam record).
    """
    import numpy as np
    n = len(getattr(results, "s", []) or [])
    if n == 0:
        return None

    # Native (Δφ, ΔW) beta: σ_φφ/ε_φW per record.
    beta_phiw = _beta_z_from_sigma(results)
    if beta_phiw is None or beta_phiw.size != n:
        return None

    ref_beta = np.asarray(getattr(results, "ref_beta", []) or [], dtype=float)
    ref_gamma = np.asarray(getattr(results, "ref_gamma", []) or [], dtype=float)
    ref_freq = np.asarray(getattr(results, "ref_frequency", []) or [], dtype=float)
    if ref_beta.size != n or ref_gamma.size != n or ref_freq.size != n:
        return None

    mass = float(getattr(results, "mass_mev", 0.0) or 0.0)
    if mass <= 0.0:
        # Recover m from W = (γ−1)·m at the first usable record.
        w = np.asarray(getattr(results, "ref_w_kin", []) or [], dtype=float)
        ok = (w.size == n) & True
        if ok:
            gm1 = ref_gamma - 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                m_est = np.where(gm1 > 0, w / np.where(gm1 > 0, gm1, 1.0), 0.0)
            pos = m_est[m_est > 0]
            mass = float(pos[0]) if pos.size else 0.0
    if mass <= 0.0:
        return None

    # λ [mm] = c / f  (c = 299792458 m/s, f in MHz → 299792.458/f mm).
    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength = np.where(ref_freq > 0, 299792.458 / np.where(ref_freq > 0, ref_freq, 1.0), 0.0)
        k_phi = np.where((ref_beta > 0) & (wavelength > 0),
                         360.0 / (ref_beta * np.where(wavelength > 0, wavelength, 1.0)), 0.0)
        k_w = ref_beta * ref_beta * ref_gamma * mass * 1e-3
        factor = np.where(k_phi > 0,
                          ref_gamma * ref_gamma * k_w / np.where(k_phi > 0, k_phi, 1.0),
                          0.0)
    return np.where((beta_phiw > 0) & (factor > 0), beta_phiw * factor, 0.0)


def _beta_z_from_sigma(results) -> "np.ndarray | None":
    """Extract β_z from the recorded σ-matrices, falling back to the
    direct (σ_φ, ε_z) pair if σ-matrices weren't recorded.

    β_z is the longitudinal beta function in HELIX's NATIVE longitudinal
    units — (Δφ [deg], ΔW [MeV]), i.e. deg/MeV: σ_φ² = ε_φW · β_φW.
    NOT interchangeable with the transverse mm/mrad β — use
    :func:`_beta_z_eff_from_sigma` for anything that integrates
    ∫ds/β in the transverse convention.
    """
    import numpy as np
    sm = getattr(results, "sigma_matrix", None)
    if sm is not None and len(sm) > 0:
        try:
            stack = np.asarray(sm, dtype=float)  # (N, 6, 6)
            s11 = stack[:, 4, 4]; s22 = stack[:, 5, 5]; s12 = stack[:, 4, 5]
            emit_sq = np.clip(s11 * s22 - s12 * s12, 0.0, None)
            emit = np.sqrt(emit_sq)
            beta_z = np.where(emit > 1e-30, s11 / np.where(emit > 1e-30, emit, 1.0), 0.0)
            return beta_z
        except Exception:                                    # noqa: BLE001
            pass
    # Fallback: σ_φ² / ε_z with ε_z in deg·MeV (same native unit as σ_φ²
    # vs ΔW); only meaningful where ε_z > 0.
    sphi = np.asarray(getattr(results, "sigma_phi", []) or [], dtype=float)
    ez   = np.asarray(getattr(results, "emit_z",    []) or [], dtype=float)
    if sphi.size and ez.size and sphi.size == ez.size:
        return np.where(ez > 0, sphi * sphi / np.where(ez > 0, ez, 1.0), 0.0)
    return None


def beam_phase_advance_along_s(results, *, start_index: int = 0) -> dict:
    """Cumulative beam phase advance μ(s) from envelope results.

    ``μ(s) = ∫₀^s ds' / β_beam(s')`` — a standard cumulative-trapezoid
    integration over the envelope solver's β(s) arrays.  Computed for
    all three planes; for the z-plane the EFFECTIVE β̃_z in locally
    normalized (z, z′) coordinates is used (see
    :func:`_beta_z_eff_from_sigma` — the native deg/MeV β is not
    compatible with the transverse mrad→deg integrand).  The
    integration starts at index ``start_index`` (set this to the
    period entry to align the curve with the matched-beam Twiss).

    Returns
    -------
    dict
        ``s`` (mm), ``mu_x_deg``, ``mu_y_deg``, ``mu_z_deg`` — same
        length as the envelope's s-grid, with NaN before
        ``start_index``.  Planes for which β couldn't be extracted
        return all-NaN.
    """
    import numpy as np
    s = np.asarray(results.s, dtype=float)
    bx = np.asarray(results.beta_x, dtype=float)
    by = np.asarray(results.beta_y, dtype=float)
    bz = _beta_z_eff_from_sigma(results)
    n = s.size

    # ds in mm, β in mm/mrad → ∫ds/β has units mrad; convert mrad→deg.
    _MRAD_TO_DEG = (180.0 / math.pi) * 1e-3

    def _accum(beta):
        if beta is None or beta.size != n or n < 2:
            return np.full(n, np.nan)
        inv = np.where(beta > 0, 1.0 / np.where(beta > 0, beta, 1.0), 0.0)
        cumtrapz = getattr(np, "cumulative_trapezoid", None)
        if cumtrapz is None:
            ds = np.diff(s)
            cum = np.concatenate(([0.0], np.cumsum(0.5 * (inv[:-1] + inv[1:]) * ds)))
        else:
            cum = np.concatenate(([0.0], cumtrapz(inv, s)))
        mu = (cum - cum[max(0, min(start_index, n - 1))]) * _MRAD_TO_DEG
        if start_index > 0:
            mu[:start_index] = np.nan
        return mu

    return {
        "s": s,
        "mu_x_deg": _accum(bx),
        "mu_y_deg": _accum(by),
        "mu_z_deg": _accum(bz),
    }
