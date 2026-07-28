"""Linear transfer matrix tracking and Twiss parameter computation."""
import numpy as np
import math
from linac_gen.core.lattice import Lattice
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.base import (
    TransferMapElement, ThinKickElement, PassiveElement, FieldMapElement,
)


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------
# `get_element_matrix(element, ref)` is a deterministic function of
# (element parameters, ref state at element entry).  Callers that want
# to amortise repeated calls — phase-advance GUI panels recomputing
# σ₀(s) after every dropdown change — can pass an opt-in cache dict.
#
# The cache is **opt-in**: when ``cache is None`` (the default) we
# always recompute, preserving byte-identical behaviour for callers
# that mutate elements between calls (matching, error studies, etc.).
#
# Cache key: ``(id(element), element_fingerprint, ref_fingerprint)``.
# ``id(element)`` distinguishes physically distinct elements that happen
# to share parameter values (e.g. two FieldMap3D loaded from different
# files); ``element_fingerprint`` invalidates the entry when any
# matrix-affecting attribute is mutated in place.

def _ref_fingerprint(ref: ReferenceParticle) -> tuple:
    """Stable fingerprint of the ref state for cache keys.

    Rounds floats to suppress spurious cache misses from FP fuzz —
    12 decimal places is well below numerical-noise threshold for any
    physically meaningful ref state.
    """
    species_name = getattr(ref.species, "name", None) or repr(ref.species)
    return (
        species_name,
        round(float(ref.w_kin), 12),
        round(float(ref.frequency), 12),
        round(float(ref.phi_s), 9),
    )


def _freeze_value(v):
    """Convert any param value to a hashable form for cache keys.

    Lists/tuples of numbers become tuples of rounded floats.  Anything
    else passes through (must already be hashable; non-hashable types
    will raise at dict.get time and we'd prefer that loud failure to
    silent collisions).
    """
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, (list, tuple)):
        return tuple(_freeze_value(x) for x in v)
    return v


def _element_fingerprint(element) -> tuple | None:
    """Hashable tuple of element params that affect its transfer matrix.

    Reads attribute names listed in ``type(element)._cache_keys``
    (a ClassVar on each element class).  If the class doesn't declare
    ``_cache_keys`` we return ``None`` — that disables caching for this
    element type (a safe fallback rather than silently caching with an
    incomplete fingerprint).
    """
    keys = getattr(type(element), "_cache_keys", None)
    if keys is None:
        return None
    return tuple(_freeze_value(getattr(element, k, None)) for k in keys)


# ---------------------------------------------------------------------------
def _compute_element_matrix(element, ref_copy: ReferenceParticle) -> np.ndarray:
    """The raw element-type dispatch.  No caching."""
    if isinstance(element, TransferMapElement):
        return element.transfer_matrix(ref_copy)
    elif isinstance(element, ThinKickElement):
        return element.kick_matrix(ref_copy)
    elif isinstance(element, FieldMapElement):
        try:
            return element.fitted_matrix(ref_copy)
        except Exception as exc:
            # A SurrogateFieldMap IN the lattice (compare flows) raises
            # OutOfScopeError for OOD inputs — the registry-hook
            # contract is "caller falls back to the wrapped element",
            # so honour it here too instead of aborting the run over a
            # matrix the wrapped element can supply.
            from linac_gen.surrogates.base import OutOfScopeError
            if isinstance(exc, OutOfScopeError):
                wrapped = getattr(element, "_wrapped", None)
                if wrapped is not None:
                    return wrapped.fitted_matrix(ref_copy)
            raise
    elif isinstance(element, PassiveElement):
        # Edge is the lone PassiveElement with a real linear matrix
        # (zero-length pole-face rotation): defer to its transfer_matrix.
        if hasattr(element, "transfer_matrix"):
            return element.transfer_matrix(ref_copy)
        return np.eye(6)
    else:
        raise TypeError(
            f"Cannot compute transfer matrix for {type(element).__name__} '{element.name}': "
            f"element must be TransferMapElement, ThinKickElement, FieldMapElement, or PassiveElement"
        )


def get_element_matrix(element, ref_copy: ReferenceParticle,
                       *, cache: dict | None = None) -> np.ndarray:
    """Per-element transfer matrix.

    ``cache``: optional dict.  When provided, results are memoised keyed
    on ``(id(element), element_fingerprint, ref_fingerprint)``.  ``None``
    (the default) preserves the original recompute-every-call behaviour
    — required for matching / error-study paths that mutate elements
    in place.
    """
    if cache is None:
        return _compute_element_matrix(element, ref_copy)
    elem_fp = _element_fingerprint(element)
    if elem_fp is None:
        # Element class doesn't opt in — pass through.
        return _compute_element_matrix(element, ref_copy)
    key = (id(element), elem_fp, _ref_fingerprint(ref_copy))
    hit = cache.get(key)
    if hit is not None:
        return hit
    M = _compute_element_matrix(element, ref_copy)
    cache[key] = M
    return M


def compute_transfer_matrix(lattice: Lattice, ref: ReferenceParticle,
                            start: int = 0, end: int | None = None,
                            *, cache: dict | None = None) -> np.ndarray:
    """Product of per-element transfer matrices from ``start`` to ``end`` (inclusive).

    Pure linear transport only -- no space-charge kicks.  Both indices are
    into ``lattice.elements`` and default to the full lattice
    (``start=0``, ``end=N-1``).  The reference particle is advanced through
    each element so RF gaps / field maps use the correct energy for their
    kick_matrix calculations.

    ``cache`` (optional): dict for memoising per-element matrices across
    repeated calls — see :func:`get_element_matrix`.
    """
    ref_copy = ref.copy()
    # Replay elements before ``start`` so the reference particle's energy,
    # frequency, phase are correct at the starting boundary.
    for element in lattice.elements[:start]:
        if isinstance(element, FieldMapElement):
            # Stateful elements (slice cursor, βg≤0 resolved geometry,
            # SET_SYNC_PHASE ψ cache) must start from a clean slate — a
            # reused lattice object otherwise replays the previous run's
            # state (a βg=0 NCells reused at a different energy loses its
            # tail gaps: −11 MeV on a 45→235 MeV reuse).  Mirrors the
            # Tracker / EnvelopeSolver / backtrack resets.
            element.reset_run_state()
            element.advance_ref(ref_copy)
        else:
            ref_copy.s += element.length
            if element.length > 0:
                ref_copy.phi_s += (
                    360.0 * element.length / (ref_copy.beta * ref_copy.wavelength)
                )
            if isinstance(element, ThinKickElement):
                element.advance_ref(ref_copy)

    last = len(lattice.elements) - 1 if end is None else end
    if last < start:
        raise ValueError(f"end ({last}) must be >= start ({start})")

    M = np.eye(6)
    for element in lattice.elements[start:last + 1]:
        if isinstance(element, FieldMapElement):
            element.reset_run_state()        # see the pre-``start`` replay note
        M_elem = get_element_matrix(element, ref_copy, cache=cache)
        M = M_elem @ M
        if isinstance(element, FieldMapElement):
            element.advance_ref(ref_copy)
        else:
            ref_copy.s += element.length
            if element.length > 0:
                ref_copy.phi_s += 360.0 * element.length / (ref_copy.beta * ref_copy.wavelength)
            if isinstance(element, ThinKickElement):
                element.advance_ref(ref_copy)
    return M


def compute_twiss(
    M: np.ndarray,
    plane: str = "x",
    coupling_tol: float = 1e-8,
) -> dict:
    """Extract decoupled Twiss parameters from a one-period 6x6 transfer matrix.

    Only valid for transversely decoupled (block-diagonal) optics. Lattices
    containing solenoids or skew multipoles produce coupled 4x4 transverse
    blocks; for those, the 2x2 block trace is *not* the one-turn phase advance
    and this routine raises instead of returning a meaningless result.

    Acceleration (determinant) handling
    -----------------------------------
    Through an accelerating cell the transverse (x, x′) block is
    *conformally symplectic*: det M₂ = (βγ)_in/(βγ)_out < 1 (adiabatic
    damping of the slope coordinates), and the eigenvalues are
    λ = √det·e^{±iμ}.  The correct extraction is therefore

        cos μ = tr M₂ / (2·√det),   β = m12/(√det·sin μ),
        α = (m11 − m22)/(2·√det·sin μ)

    — the raw ``acos(tr/2)`` is biased for every accelerating cell
    (≈0.863 determinant per PIP-II HWR cavity).  The longitudinal
    (Δφ, ΔW) pair is canonical up to a fixed scale, so det = 1 through
    fixed-frequency acceleration; across a phase-coordinate FREQUENCY
    jump it equals f_out/f_in (see ``RFGap.kick_matrix``) and the same
    normalization applies.  Magnetostatic lattices take the historical
    code path bit-for-bit (``|det − 1| ≤ 1e-9`` fast path).

    Note an accelerating "cell" is not genuinely periodic: the
    normalized eigen-angle is a standard, useful linac convention but
    remains a single-pass/local quantity (endpoint coordinate scalings
    can shift the trace), not a basis-independent Floquet tune.

    Parameters
    ----------
    M : np.ndarray
        6x6 transfer matrix.
    plane : {"x", "y", "z"}
        Plane to extract.  ``"z"`` reads the longitudinal (Δφ, ΔW)
        sub-block — the coupling check then runs against the 4×4
        transverse block instead.
    coupling_tol : float
        Sum of |M_ij| across the off-plane block above which the lattice is
        deemed coupled.  Default 1e-8.

    Returns
    -------
    dict
        ``alpha``, ``beta``, ``gamma_t``, ``mu`` (deg, oriented — in
        [0, 360°) via the sign of m12), ``mu_folded`` (deg, principal
        value in [0, 180°]), ``damping`` (√det — per-period amplitude
        ratio; 1.0 on the magnetostatic fast path).
    """
    if plane == "x":
        i, j = 0, 1
        off_planes = [(2, 3)]
    elif plane == "y":
        i, j = 2, 3
        off_planes = [(0, 1)]
    elif plane == "z":
        # Longitudinal block (Δφ deg, ΔW MeV) is at indices 4..5.  The
        # x/y planes are the off-block we check for coupling — small RF
        # → transverse coupling is fine to ignore by raising the tol.
        i, j = 4, 5
        off_planes = [(0, 1), (2, 3)]
    else:
        raise ValueError(f"plane must be 'x', 'y', or 'z', got '{plane}'")

    # Reject coupled lattices -- the 2x2 block trace is only the phase advance
    # when M[i, other_*] and M[j, other_*] all vanish.
    off_diag = 0.0
    for (oi, oj) in off_planes:
        off_diag += (
            abs(M[i, oi]) + abs(M[i, oj])
            + abs(M[j, oi]) + abs(M[j, oj])
        )
    if off_diag > coupling_tol:
        raise ValueError(
            f"Lattice is coupled to plane {plane} (off-plane |M_ij| sum = "
            f"{off_diag:.2e} > coupling_tol={coupling_tol:.0e}). "
            f"compute_twiss handles decoupled optics only; remove the "
            f"coupling source (solenoid / skew quad / strong RF) before "
            f"calling, or raise coupling_tol if the coupling is "
            f"intentional and small enough to ignore."
        )

    m11 = M[i, i]
    m12 = M[i, j]
    m21 = M[j, i]
    m22 = M[j, j]
    det = m11 * m22 - m12 * m21
    if det <= 0.0:
        raise ValueError(
            f"{plane}-block determinant {det:.3e} <= 0 — not a transport "
            f"matrix (coupling leak, or a defective element map)"
        )
    if abs(det - 1.0) <= 1e-9:
        # Magnetostatic / symplectic fast path — bit-identical to the
        # historical formula (no √det rounding introduced).
        sq = 1.0
        cos_mu = 0.5 * (m11 + m22)
    else:
        # Conformally symplectic (accelerating / freq-jump) block:
        # λ = √det·e^{±iμ} ⇒ cos μ = tr/(2√det).
        sq = math.sqrt(det)
        cos_mu = 0.5 * (m11 + m22) / sq
    if abs(cos_mu) >= 1.0:
        raise ValueError(f"Unstable: cos(mu) = {cos_mu}")
    mu = math.acos(cos_mu)
    sin_mu = math.sin(mu)
    if m12 < 0:
        mu = 2 * math.pi - mu
        sin_mu = math.sin(mu)
    if sq == 1.0:
        beta = m12 / sin_mu
        alpha = (m11 - m22) / (2.0 * sin_mu)
    else:
        beta = m12 / (sq * sin_mu)
        alpha = (m11 - m22) / (2.0 * sq * sin_mu)
    gamma_t = (1.0 + alpha**2) / beta
    mu_deg = math.degrees(mu)
    return {
        "alpha": alpha, "beta": beta, "gamma_t": gamma_t, "mu": mu_deg,
        "mu_folded": mu_deg if mu_deg <= 180.0 else 360.0 - mu_deg,
        "damping": sq,
    }


def propagate_twiss(m2, alpha: float, beta: float) -> tuple[float, float]:
    """Transport Courant-Snyder ``(alpha, beta)`` through a 2x2 transfer block.

    ``m2`` is the 2x2 transfer-matrix block for one plane.  Returns
    ``(alpha, beta)`` at the block's exit.  To transport *backwards* (e.g.
    a periodic cell's matched Twiss back to the lattice entrance), pass the
    inverse block.

    For a block with det ≠ 1 (accelerating element — adiabatic damping,
    or a frequency-jump D factor) the geometric emittance scales by
    |det|, so the σ-quadratic propagation must be divided by |det| to
    return the CS parameters of the transported ellipse (β = σ²/ε with
    the *scaled* ε).  det == 1 keeps the historical bit-exact formula.
    """
    c11, c12 = float(m2[0][0]), float(m2[0][1])
    c21, c22 = float(m2[1][0]), float(m2[1][1])
    gamma = (1.0 + alpha * alpha) / beta
    beta_new = c11 * c11 * beta - 2.0 * c11 * c12 * alpha + c12 * c12 * gamma
    alpha_new = (-c11 * c21 * beta
                 + (c11 * c22 + c12 * c21) * alpha
                 - c12 * c22 * gamma)
    det = c11 * c22 - c12 * c21
    if abs(det - 1.0) > 1e-9 and det > 0.0:
        beta_new /= det
        alpha_new /= det
    return alpha_new, beta_new
