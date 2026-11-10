"""Periodic Twiss matching: find matched optics parameters for a periodic cell.

For **decoupled** lattices (no solenoids / skew quads / strong RF
coupling), :func:`find_periodic_twiss` extracts independent x and y
Twiss from the diagonal 2×2 blocks of the one-turn map -- fast and
exact when the off-diagonal x↔y blocks vanish.

For **coupled** lattices (solenoid-focused HWR cryomodules, skew-quad
correctors, strong dispersive coupling), the decoupled extraction
fails because the 2×2 block trace is no longer the one-turn phase
advance.  :func:`find_coupled_matched_twiss` implements the
eigenvector / Wolski method: eigendecompose the 4×4 transverse map,
build the matched Σ from the normalised eigenvectors, and project
back to per-plane α / β for display.

:func:`find_periodic_twiss` and :func:`find_matched_input_twiss`
auto-detect coupling and route to the coupled path transparently,
returning the same dict shape plus a ``coupled=True`` flag and the
full 4×4 Σ matrix.

**RFQs**: :func:`find_rfq_periods` finds the focusing period an RFQ's
input is matched to (its first two regular cells) and
:func:`find_rfq_matched_input_twiss` gives the matched Twiss at the RFQ
entrance, with or without space charge (a DC input beam).
"""
from dataclasses import dataclass


def _build_coupled_matched_sigma(M4: "np.ndarray",
                                 stab_tol: float = 0.15) -> "np.ndarray":
    """Build the matched 4×4 transverse Σ from a one-turn 4×4 map.

    Implements Wolski's eigenvector method (Beam Dynamics in High
    Energy Particle Accelerators, 2014, §3.3):

    1. Eigendecompose M (4×4 symplectic). For stable motion the
       eigenvalues come in unimodular complex-conjugate pairs
       (λ, λ*) per normal mode.
    2. Pick one eigenvector v_k from each pair.  Normalise via
       v_k† S v_k = i (the symplectic-form normalisation).
    3. The matched Σ is the sum over modes of the imaginary parts
       of the outer products:  Σ = Σ_k Im(v_k v_k†).

    Both modes get unit emittance (shape-only result); the absolute
    scale is set by the user's beam emittance at the call site.

    Acceleration handling
    ---------------------
    Strictly periodic lattices have eigenvalues exactly on the unit
    circle.  **Accelerating sections** (cryomodules with RF cavities)
    instead give a *conformally symplectic* map — uniform adiabatic
    damping with det(M₄) = D², eigenvalues √D·e^{±iμ}.  We
    scalar-normalize by ``det(M₄)^{1/4}`` first, which is essentially
    exact for that case (the normalized map is symplectic and its
    matched Σ is the physical normalized-coordinate solution — no
    "smooth approximation" involved).  Only a RESIDUAL deviation of the
    normalized eigenvalues from the unit circle (non-conformal damping,
    e.g. plane-dependent) is approximated; we warn to stderr above 1e-3
    so users know the matched Σ is then a smooth approximation.

    Raises ``ValueError`` if the normalized eigenvalue moduli deviate
    by more than ``stab_tol`` from unity, which indicates a truly
    unstable lattice (not just accelerating).
    """
    import numpy as np

    # 4×4 symplectic form for the transverse phase space.
    S4 = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
    ], dtype=float)

    # Scalar-normalize conformal (adiabatic) damping — exact for
    # det ≠ 1 conformally symplectic maps; magnetostatic maps take the
    # bit-identical det==1 path.
    det4 = float(np.linalg.det(M4))
    if det4 <= 0.0:
        raise ValueError(
            f"Coupled 4×4 map determinant {det4:.3e} <= 0 — not a "
            f"transport matrix"
        )
    if abs(det4 - 1.0) > 1e-9:
        M4 = M4 / det4 ** 0.25

    eigvals, eigvecs = np.linalg.eig(M4)
    moduli = np.abs(eigvals)
    deviation = np.max(np.abs(moduli - 1.0))
    if deviation > stab_tol:
        raise ValueError(
            f"Coupled 4×4 lattice is linearly unstable -- eigenvalue "
            f"moduli deviate {deviation:.3f} from unity AFTER det-"
            f"normalization (tolerance {stab_tol}); "
            f"moduli={moduli.tolist()}"
        )
    if deviation > 1e-3:
        # Residual NON-conformal deviation (plane-dependent damping):
        # the matched Σ is then a smooth approximation.  Conformal
        # damping was removed exactly by the normalization above and
        # does not reach this branch.
        import sys
        print(
            f"[matched-Σ] non-conformal deviation {deviation:.3f} "
            f"from unit circle after det-normalization "
            f"(moduli={[round(m, 4) for m in moduli]}); "
            f"matched Σ is the smooth approximation.",
            file=sys.stderr,
        )

    # Group into complex-conjugate pairs.  Sort by phase angle to
    # disambiguate which eigenvalue belongs to which "mode".
    phases = np.angle(eigvals)
    # Pick eigenvectors corresponding to eigenvalues with Im > 0
    # (the upper half of each conjugate pair).
    pos_imag = np.where(np.imag(eigvals) > 0)[0]
    if pos_imag.size != 2:
        # Degenerate case (purely real eigenvalues, or three+ on
        # the same line): not a generic stable coupled system.
        raise ValueError(
            "Cannot identify two distinct complex-conjugate pairs -- "
            "eigenvalues are degenerate or real-only.  This usually "
            "means a structural resonance is exactly hit."
        )

    Sigma = np.zeros((4, 4), dtype=float)
    for k in pos_imag:
        v = eigvecs[:, k]
        # Symplectic norm: for an antisymmetric S, v† S v is purely
        # imaginary.  Conventionally normalise so that v† S v = +i
        # for the eigenvector corresponding to the eigenvalue with
        # positive imaginary part (Wolski Eq. 3.66).  Implementation:
        #   c = -i · v† S v     ; should be a positive real
        #   v ← v / √c
        c = -1j * (v.conj() @ S4 @ v)
        if abs(c.imag) > 1e-6:
            raise ValueError(
                f"Symplectic norm of an eigenvector has a non-trivial "
                f"real part ({c}); the input map is not symplectic.")
        if c.real <= 0:
            # Sign flip convention: if c is negative real, the
            # eigenvector's symplectic norm has the wrong sign.  Flip
            # by picking v* instead of v -- the physics is the same
            # mode, just the other root in the complex-conjugate pair.
            v = v.conj()
            c = -1j * (v.conj() @ S4 @ v)
            if c.real <= 0:
                raise ValueError(
                    "Unable to symplectically normalise eigenvector "
                    "for stable coupled mode -- check that the 4×4 "
                    "transverse map is properly symplectic."
                )
        v = v / np.sqrt(c.real)
        # Wolski Eq. 3.92: the matched Σ for one normal mode is
        # 2 · Re(v v†).  Σ across modes is additive (Eq. 3.95).
        Sigma += 2.0 * np.real(np.outer(v, v.conj()))

    # Numerical residual: ensure symmetric (analytically it is; finite
    # precision can leave a 1e-15 antisymmetric part).
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


def _project_twiss_from_sigma(Sigma4: "np.ndarray") -> dict:
    """Extract per-plane projected α, β, ε from a 4×4 Σ matrix.

    The projection ignores coupling and reports the apparent x/y RMS
    optics that an x- or y-only diagnostic would see.  Useful for
    plotting but does **not** capture the full coupled physics — the
    matched Σ itself is the load-bearing object.
    """
    import math
    out = {}
    for plane, (i, j) in (("x", (0, 1)), ("y", (2, 3))):
        sii = Sigma4[i, i]
        sij = Sigma4[i, j]
        sjj = Sigma4[j, j]
        det = sii * sjj - sij * sij
        if det <= 0.0 or sii <= 0.0:
            out[f"alpha_{plane}"] = 0.0
            out[f"beta_{plane}"] = 0.0
            out[f"emit_{plane}_proj"] = 0.0
            continue
        eps = math.sqrt(det)
        out[f"alpha_{plane}"] = -sij / eps
        out[f"beta_{plane}"] = sii / eps
        out[f"emit_{plane}_proj"] = eps
    return out


def _periodic_dispersion(m6, idx=(0, 1), *, det_tol: float = 1e-9):
    """Periodic (closed-orbit) dispersion of the ``idx`` block w.r.t. ΔW.

    Solves ``η = (I − M_block)⁻¹ · d`` with ``M_block = m6[ix_(idx, idx)]``
    and ``d = m6[idx, 5]`` — the kinetic-energy column (ΔW in MeV), so the
    units are **mm/MeV** for positions and **mrad/MeV** for angles,
    identical to the ``disp_x``/``disp_xp``/``disp_y``/``disp_yp``
    convention of :func:`find_sc_matched_input_twiss` (``Σ[i,5]/Σ[5,5]``).
    Display conversion: η[m] = D[mm/MeV] × β²γ·mc²[MeV] × 1e-3.

    Returns exact zeros (no solve) when every entry of ``d`` is exactly
    0.0 — guarantees bit-exact zeros for dispersion-free lattices.  When
    ``|det(I − M_block)| < det_tol`` (near-integer tune) or the solve
    fails, prints a one-line warning to stderr and returns NaNs — never
    raises, so lattices that worked before keep working.
    """
    import sys
    import numpy as np

    idx = list(idx)
    d = np.asarray(m6[idx, 5], dtype=float)
    if not np.any(d):
        return np.zeros(len(idx))
    A = np.eye(len(idx)) - np.asarray(m6, dtype=float)[np.ix_(idx, idx)]
    det = float(np.linalg.det(A))
    if abs(det) < det_tol:
        print(
            f"[periodic-dispersion] |det(I − M)| = {abs(det):.3e} < "
            f"{det_tol:g} (near-integer tune) — periodic dispersion is "
            "undefined; returning NaN.",
            file=sys.stderr,
        )
        return np.full(len(idx), float("nan"))
    try:
        return np.linalg.solve(A, d)
    except np.linalg.LinAlgError as exc:
        print(f"[periodic-dispersion] solve failed ({exc}); returning NaN.",
              file=sys.stderr)
        return np.full(len(idx), float("nan"))


def find_coupled_matched_twiss(lattice, ref):
    """Periodic matched Twiss for a **coupled** transverse lattice.

    Uses the eigenvector / Wolski method on the 4×4 transverse
    one-turn map -- valid when solenoids, skew quads, or other x↔y
    couplers make the diagonal 2×2 blocks no longer block-diagonal.

    Returns a dict with the same shape as :func:`find_periodic_twiss`
    plus:

    * ``coupled`` (bool): always ``True`` for this path.
    * ``sigma4`` (4×4 np.ndarray): the matched transverse Σ (unit
      emittance per normal mode -- multiply rows/columns 0-1 by
      ε_1 and 2-3 by ε_2 to scale to physical beam).
    * ``mu_1``, ``mu_2`` (deg): the two normal-mode phase advances.

    Per-plane ``alpha_x``/``beta_x``/``alpha_y``/``beta_y`` are
    projections from the matched Σ (what x-only and y-only diagnostics
    would report).  ``mu_x`` / ``mu_y`` mirror ``mu_1`` / ``mu_2``
    for compatibility with :func:`find_periodic_twiss` callers.
    """
    import math
    import numpy as np
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    M6 = compute_transfer_matrix(lattice, ref.copy())
    M4 = M6[0:4, 0:4]
    Sigma4 = _build_coupled_matched_sigma(M4)
    proj = _project_twiss_from_sigma(Sigma4)

    # Periodic dispersion: full 4×4 solve — a vertical bend + solenoid
    # mixes planes, so the per-plane 2×2 solves would be wrong here.
    eta4 = _periodic_dispersion(M6, (0, 1, 2, 3))

    # Normal-mode phase advances from the eigenvalue phases.
    eigvals, _ = np.linalg.eig(M4)
    phases = np.sort(np.unique(np.abs(np.angle(eigvals))))[:2]
    # Ensure we have two finite mode advances; degenerate cases
    # (resonance) already raised inside _build_coupled_matched_sigma.
    mu_1 = math.degrees(float(phases[0])) if phases.size >= 1 else 0.0
    mu_2 = math.degrees(float(phases[1])) if phases.size >= 2 else mu_1

    return {
        "alpha_x": proj["alpha_x"], "beta_x": proj["beta_x"],
        "alpha_y": proj["alpha_y"], "beta_y": proj["beta_y"],
        "mu_x": mu_1, "mu_y": mu_2,
        "mu_1": mu_1, "mu_2": mu_2,
        "sigma4": Sigma4,
        "disp_x": float(eta4[0]), "disp_xp": float(eta4[1]),
        "disp_y": float(eta4[2]), "disp_yp": float(eta4[3]),
        "coupled": True,
    }


def find_periodic_twiss(lattice, ref):
    """Find matched Twiss parameters for a periodic lattice (one-turn map).

    Computes the one-turn transfer matrix and extracts the periodic Twiss
    (Courant-Snyder) parameters.

    Parameters
    ----------
    lattice : Lattice
        The periodic lattice cell.
    ref : ReferenceParticle
        Initial reference particle state (a copy is used internally).

    Returns
    -------
    dict with keys:
        alpha_x, beta_x, mu_x  (x-plane Twiss and phase advance in degrees)
        alpha_y, beta_y, mu_y  (y-plane Twiss and phase advance in degrees)
        disp_x, disp_xp        (periodic dispersion, mm/MeV and mrad/MeV —
                                exactly 0.0 for a dispersion-free lattice,
                                NaN near an integer tune)
        disp_y, disp_yp        (y-plane periodic dispersion)

    Raises
    ------
    ValueError
        If the lattice is linearly unstable (|cos(mu)| >= 1) in either plane.
    """
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, compute_twiss

    M = compute_transfer_matrix(lattice, ref.copy())
    # Try the cheap decoupled path first.  compute_twiss raises
    # ValueError with "coupled" in the message when the off-diagonal
    # x↔y blocks exceed tolerance (solenoids, skew quads).  In that
    # case fall through to the eigenvector / Wolski method.
    try:
        twiss_x = compute_twiss(M, "x")
        twiss_y = compute_twiss(M, "y")
    except ValueError as exc:
        if "coupled" in str(exc).lower():
            # Coupled lattice -- route to the 4×4 matched-Σ path.
            return find_coupled_matched_twiss(lattice, ref)
        # Genuinely unstable (|cos μ| ≥ 1) or otherwise -- re-raise.
        raise

    # Periodic dispersion per plane: η = (I − M₂)⁻¹ · d, with d the
    # energy column (mm/MeV, mrad/MeV).  Exact zeros for straight
    # lattices; the matched beam in a bending line (arc FODO / BTL)
    # carries this dispersion in addition to alpha/beta.
    eta_x = _periodic_dispersion(M, (0, 1))
    eta_y = _periodic_dispersion(M, (2, 3))

    return {
        "alpha_x": twiss_x["alpha"],
        "beta_x": twiss_x["beta"],
        "mu_x": twiss_x["mu"],
        "alpha_y": twiss_y["alpha"],
        "beta_y": twiss_y["beta"],
        "mu_y": twiss_y["mu"],
        # Principal [0, 180°] values — ``mu_x`` above is the ORIENTED
        # branch in [0, 360°) (sign of m12) and stays untouched for
        # backward compatibility (user-targetable via matching
        # objectives).  Form tune-depression ratios from matching
        # conventions only.
        "mu_x_folded": twiss_x["mu_folded"],
        "mu_y_folded": twiss_y["mu_folded"],
        "disp_x": float(eta_x[0]), "disp_xp": float(eta_x[1]),
        "disp_y": float(eta_y[0]), "disp_yp": float(eta_y[1]),
        "coupled": False,
    }


def find_fodo_cells(lattice):
    """Return periodic-cell candidates as ``(start, end)`` element-index pairs.

    A cell spans one full focusing period — from just after a focusing
    element to the second-next focusing element (focuser k → focuser k+2)
    — so each candidate contains one focusing and one defocusing element.
    This is the QF→QF rule the TraceWin period tools use; it needs no
    ``LATTICE`` card.

    "Focusing element" is detected as any of:

    * ``Quadrupole`` (the classical FODO case).
    * ``Solenoid``.
    * ``FieldMap`` / ``FieldMap3D`` that classifies as a **solenoid**
      via :func:`categorize_fieldmap` (kb≠0, ke=0) — this is the HWR
      cryomodule case where the focusing is delivered by a solenoid
      FieldMap rather than a separate Solenoid element.

    Mixing focusing-element types is allowed: a 5-element periodic
    structure of ``[QUAD, drift, SOL, drift, QUAD]`` would yield one
    candidate cell.  Returns ``[]`` when fewer than three focusing
    elements exist.
    """
    from linac_gen.matching.variables import categorize_fieldmap

    def _is_focuser(e):
        name = type(e).__name__
        if name in ("Quadrupole", "Solenoid"):
            return True
        if name in ("FieldMap", "FieldMap3D"):
            return categorize_fieldmap(e) == "solenoid"
        return False

    foc_idx = [i for i, e in enumerate(lattice.elements) if _is_focuser(e)]
    cells = []
    for k in range(len(foc_idx) - 2):
        start = foc_idx[k] + 1
        end = foc_idx[k + 2]
        if end > start:
            cells.append((start, end))
    return cells


@dataclass(frozen=True)
class RfqPeriod:
    """The focusing period an RFQ's input is matched to.

    ``rfq_start`` is the lattice index of the RFQ's first cell (its
    entrance); ``start`` / ``end`` (inclusive) are its first two
    consecutive regular cells of opposite sign (−2/+2: one alternating-
    gradient period), after the radial matching section (±3) and any
    transition (±4) cells — the period a TraceWin deck brackets with
    ``LATTICE 2``.  ``label`` is a short display name.
    """

    rfq_start: int
    start: int
    end: int
    label: str


def find_rfq_periods(lattice) -> list:
    """One :class:`RfqPeriod` per RFQ in ``lattice``, in lattice order.

    An RFQ is a chain of ``RfqCell`` elements; markers and other
    zero-length elements between its cells belong to it
    (:func:`linac_gen.rfq_td.chain.find_chains`).  An RFQ without two
    consecutive regular cells of opposite sign has no period and is
    skipped; a lattice without RFQ cells gives ``[]``.
    """
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.rfq_td.chain import find_chains

    els = lattice.elements
    out = []
    for first, last in find_chains(els):
        cells = [i for i in range(first, last + 1)
                 if isinstance(els[i], RfqCell)]
        for a, b in zip(cells, cells[1:]):
            ta, tb = int(els[a].cell_type), int(els[b].cell_type)
            if abs(ta) == 2 and tb == -ta:
                out.append(RfqPeriod(
                    rfq_start=first, start=a, end=b,
                    label=(f"RFQ: {els[a].name} + {els[b].name} "
                           f"(elements {a}–{b})")))
                break
    return out


def find_matched_input_twiss(lattice, ref, cell_start, cell_end):
    """Matched *input* Twiss at the lattice entrance for a transfer line.

    Computes the periodic (Courant-Snyder) Twiss of the FODO cell spanning
    element indices ``[cell_start, cell_end]`` (inclusive), then back-
    propagates it through the front section ``[0, cell_start)`` to the
    lattice entrance (s = 0).  The result is the beam to inject so that the
    periodic cell is matched — the correct "input matched Twiss" for a
    transfer line.

    Contrast :func:`find_periodic_twiss`, which is the whole-lattice
    periodic solution — correct only for a genuine ring, not a one-pass
    transfer line.

    Returns the same dict shape as :func:`find_periodic_twiss`
    (``alpha_x``/``beta_x``/``mu_x`` and the y-plane), plus ``cell_start`` /
    ``cell_end``.  ``mu_x`` / ``mu_y`` are the *per-cell* phase advance.

    Raises
    ------
    ValueError
        If the cell range is out of bounds, the cell is linearly unstable,
        or the front section's 2x2 betatron block is singular.
    """
    import numpy as np
    from linac_gen.tracking.matrix_tracking import (
        compute_transfer_matrix, compute_twiss, propagate_twiss,
    )

    n = len(lattice.elements)
    if not (0 <= cell_start <= cell_end < n):
        raise ValueError(
            f"cell range [{cell_start}, {cell_end}] out of bounds "
            f"(lattice has {n} elements)")

    # 1. periodic Twiss of the FODO cell (at the cell-start boundary).
    #    Try the cheap decoupled path; fall through to the 4×4
    #    eigenvector method (Wolski) when the cell is coupled
    #    (solenoid / skew quad / RF rotation).
    m_cell = compute_transfer_matrix(lattice, ref.copy(),
                                     start=cell_start, end=cell_end)
    try:
        tw_x = compute_twiss(m_cell, "x")
        tw_y = compute_twiss(m_cell, "y")
        alpha_x, beta_x = tw_x["alpha"], tw_x["beta"]
        alpha_y, beta_y = tw_y["alpha"], tw_y["beta"]
        coupled = False
    except ValueError as exc:
        if "coupled" not in str(exc).lower():
            raise  # genuinely unstable, propagate
        coupled = True
        # Build the matched 4×4 Σ at cell-start via the eigenvector
        # method, then back-propagate the *full* Σ through the
        # inverse 4×4 front section (not 2×2 blocks -- those would
        # discard the off-diagonal coupling information).
        Sigma_cell = _build_coupled_matched_sigma(m_cell[0:4, 0:4])

    # 2. back-propagate to lattice entrance (s = 0).
    if not coupled:
        # Periodic dispersion of the cell (mm/MeV, mrad/MeV) — the
        # matched beam in a bending cell carries it alongside α/β.
        eta_x = _periodic_dispersion(m_cell, (0, 1))
        eta_y = _periodic_dispersion(m_cell, (2, 3))
        if cell_start > 0:
            m_front = compute_transfer_matrix(lattice, ref.copy(),
                                              start=0, end=cell_start - 1)
            try:
                mx_inv = np.linalg.inv(m_front[0:2, 0:2])
                my_inv = np.linalg.inv(m_front[2:4, 2:4])
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"front section [0, {cell_start}) betatron block "
                    f"is singular — cannot back-propagate: {exc}")
            alpha_x, beta_x = propagate_twiss(mx_inv, alpha_x, beta_x)
            alpha_y, beta_y = propagate_twiss(my_inv, alpha_y, beta_y)
            # Dispersion transports affinely (η_cell = M₂ η_ent + d_front),
            # so η_ent = M₂⁻¹ (η_cell − d_front) — the exact mirror of
            # find_sc_matched_input_twiss's Stage-B seed.  NaN (integer
            # tune) propagates through the matmul as intended.
            eta_x = mx_inv @ (eta_x - m_front[[0, 1], 5])
            eta_y = my_inv @ (eta_y - m_front[[2, 3], 5])

        return {
            "alpha_x": alpha_x, "beta_x": beta_x, "mu_x": tw_x["mu"],
            "alpha_y": alpha_y, "beta_y": beta_y, "mu_y": tw_y["mu"],
            # Principal [0, 180°] values; raw mu_x/mu_y stay oriented.
            "mu_x_folded": tw_x["mu_folded"],
            "mu_y_folded": tw_y["mu_folded"],
            "disp_x": float(eta_x[0]), "disp_xp": float(eta_x[1]),
            "disp_y": float(eta_y[0]), "disp_yp": float(eta_y[1]),
            "cell_start": cell_start, "cell_end": cell_end,
            "coupled": False,
        }

    # Coupled branch: back-propagate the 4×4 Σ.
    import math
    eta4 = _periodic_dispersion(m_cell, (0, 1, 2, 3))
    if cell_start > 0:
        m_front = compute_transfer_matrix(lattice, ref.copy(),
                                          start=0, end=cell_start - 1)
        m_front_4 = m_front[0:4, 0:4]
        try:
            m_front_inv = np.linalg.inv(m_front_4)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"front section [0, {cell_start}) 4×4 block is "
                f"singular — cannot back-propagate coupled Σ: {exc}")
        # Σ propagates as: Σ_out = M Σ_in M^T  ⇒  Σ_in = M^-1 Σ_out (M^-1)^T
        Sigma_in = m_front_inv @ Sigma_cell @ m_front_inv.T
        # Dispersion transports affinely: η_ent = M₄⁻¹ (η_cell − d_front).
        eta4 = m_front_inv @ (eta4 - m_front[0:4, 5])
    else:
        Sigma_in = Sigma_cell

    proj = _project_twiss_from_sigma(Sigma_in)
    # Normal-mode phase advances from the cell's 4×4 map.
    eigvals = np.linalg.eigvals(m_cell[0:4, 0:4])
    phases = np.sort(np.unique(np.abs(np.angle(eigvals))))[:2]
    mu_1 = math.degrees(float(phases[0])) if phases.size >= 1 else 0.0
    mu_2 = math.degrees(float(phases[1])) if phases.size >= 2 else mu_1

    return {
        "alpha_x": proj["alpha_x"], "beta_x": proj["beta_x"],
        "alpha_y": proj["alpha_y"], "beta_y": proj["beta_y"],
        "mu_x": mu_1, "mu_y": mu_2,
        "mu_1": mu_1, "mu_2": mu_2,
        "sigma4": Sigma_in,
        "disp_x": float(eta4[0]), "disp_xp": float(eta4[1]),
        "disp_y": float(eta4[2]), "disp_yp": float(eta4[3]),
        "cell_start": cell_start, "cell_end": cell_end,
        "coupled": True,
    }


def _emittance_damping(lattice, ref, index, at) -> float:
    """What a geometric emittance shrinks by on the way to element ``index``, whose
    reference is ``at``: the ratio of βγ, over the cavities only.  An energy card moves
    the reference and leaves the particles' angles alone, in the tracker and the
    envelope alike."""
    from linac_gen.elements.lattice_commands import SetReferenceEnergy
    from linac_gen.tracking.matrix_tracking import reference_at
    damp = ref.bg / at.bg if at.bg > 0 else 1.0
    for i, e in enumerate(lattice.elements[:index]):
        if isinstance(e, SetReferenceEnergy):
            damp *= reference_at(lattice, ref, i + 1).bg / reference_at(lattice, ref, i).bg
    return damp


def find_sc_matched_input_twiss(lattice, ref, cell_start, cell_end,
                                current, base_initial, *,
                                max_iter: int = 50, tol: float = 1e-4,
                                sc_factor: float = 1.0):
    """Space-charge matched *input* Twiss at the lattice entrance.

    The current-carrying analogue of :func:`find_matched_input_twiss` for a
    transfer line.  Because the envelope is non-linear in the beam, it is
    done in two stages:

    * **Stage A** — iterate the envelope solver over the FODO cell
      ``[cell_start, cell_end]`` until it is space-charge-periodic (cell
      entry Twiss == cell exit Twiss), a damped fixed point;
    * **Stage B** — root-find the entrance Twiss whose envelope, after the
      front section ``[0, cell_start)``, reaches that cell match.

    The cell may bend, so the betatron Twiss AND the periodic dispersion
    are matched together (8 quantities) — the envelope solver's optional
    input-dispersion support makes that possible.

    ``base_initial`` carries the fixed envelope inputs — ``emit_x``,
    ``emit_y``, ``emit_z``, ``alpha_z``, ``beta_z``, at the entrance: the cell
    runs at the reference and with the beam the front hands it.  ``sc_factor`` is the
    space-charge compensation state at ``lattice``'s entrance (the
    multiplier a SPACE_CHARGE_COMP upstream left in force; 1.0 = full space
    charge, the default).

    Returns a dict: ``alpha_x``/``beta_x``/``alpha_y``/``beta_y`` and
    ``disp_x``/``disp_xp``/``disp_y``/``disp_yp`` at the entrance,
    ``mu_x``/``mu_y`` (zero-current per-cell phase advance),
    ``cell_start``/``cell_end``, and ``converged`` (bool).

    Raises
    ------
    ValueError
        If the cell range is out of bounds, or ``current <= 0`` (use
        :func:`find_matched_input_twiss` for the zero-current case).
    """
    import numpy as np
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at

    n = len(lattice.elements)
    if not (0 <= cell_start <= cell_end < n):
        raise ValueError(
            f"cell range [{cell_start}, {cell_end}] out of bounds "
            f"(lattice has {n} elements)")
    if current <= 0:
        raise ValueError(
            "find_sc_matched_input_twiss needs current > 0 — use "
            "find_matched_input_twiss for the zero-current case")

    elems = lattice.elements
    cell_lat = Lattice()
    for e in elems[cell_start:cell_end + 1]:
        cell_lat.add(e)
    front_lat = Lattice()
    for e in elems[:cell_start]:
        front_lat.add(e)
    step_cfg = getattr(lattice, "step_config", None)
    if step_cfg is not None:
        cell_lat.step_config = step_cfg
        front_lat.step_config = step_cfg

    def _state(sigma):
        """8-state from a 6x6 Σ: dispersion-subtracted betatron Twiss
        (alpha/beta per plane) plus the dispersion D = Σ[i,5]/Σ[5,5]."""
        s_ww = sigma[5, 5]
        tw, disp = [], []
        for i, j in ((0, 1), (2, 3)):
            if s_ww > 0.0:
                di, dj = sigma[i, 5] / s_ww, sigma[j, 5] / s_ww
                sii = sigma[i, i] - sigma[i, 5] ** 2 / s_ww
                sjj = sigma[j, j] - sigma[j, 5] ** 2 / s_ww
                sij = sigma[i, j] - sigma[i, 5] * sigma[j, 5] / s_ww
            else:
                di = dj = 0.0
                sii, sjj, sij = sigma[i, i], sigma[j, j], sigma[i, j]
            eps = (sii * sjj - sij * sij) ** 0.5
            tw.append((-sij / eps, sii / eps))
            disp.append((di, dj))
        return [tw[0][0], tw[0][1], tw[1][0], tw[1][1],
                disp[0][0], disp[0][1], disp[1][0], disp[1][1]]

    ref_cell = reference_at(lattice, ref, cell_start)
    # (a cavity damps the transverse emittances; a FREQ card puts Δφ in degrees of the
    # new clock, the same bunch)
    damp = _emittance_damping(lattice, ref, cell_start, ref_cell)
    clock_ratio = ref_cell.frequency / ref.frequency if ref.frequency > 0 else 1.0
    cell_initial = dict(base_initial)
    for key, factor in (("emit_x", damp), ("emit_y", damp),
                        ("emit_z", clock_ratio), ("beta_z", clock_ratio)):
        if key in base_initial:
            cell_initial[key] = float(base_initial[key]) * factor

    def _run(sub, st, r=ref, initial=base_initial):
        init = dict(initial)
        init.update(alpha_x=st[0], beta_x=st[1], alpha_y=st[2], beta_y=st[3],
                    disp_x=st[4], disp_xp=st[5], disp_y=st[6], disp_yp=st[7])
        env = EnvelopeSolver(sub, r.copy(), init, current=current, sc_factor=sc_factor,
                             bunch_frequency=float(ref.frequency)).run()
        return _state(np.asarray(env.sigma_matrix[-1]))

    # seed: zero-current cell betatron Twiss + periodic dispersion
    # (shared _periodic_dispersion helper; a NaN result — near-integer
    # tune — is sanitised to 0.0 here because this is only a SEED for
    # the fixed-point iteration, which can still converge from zero).
    zc = find_periodic_twiss(cell_lat, ref_cell)
    mc0 = compute_transfer_matrix(cell_lat, ref_cell.copy())
    dx0 = np.nan_to_num(_periodic_dispersion(mc0, (0, 1)), nan=0.0)
    dy0 = np.nan_to_num(_periodic_dispersion(mc0, (2, 3)), nan=0.0)
    state = [zc["alpha_x"], zc["beta_x"], zc["alpha_y"], zc["beta_y"],
             float(dx0[0]), float(dx0[1]), float(dy0[0]), float(dy0[1])]

    # --- Stage A: SC-periodic state of the cell (betatron Twiss + dispersion) ---
    damping = 0.5
    converged = False
    for _ in range(max_iter):
        out = _run(cell_lat, state, ref_cell, cell_initial)
        err = max(abs(o - s) / max(abs(s), 1e-3)
                  for o, s in zip(out, state))
        if err < tol:
            converged = True
            break
        state = [(1.0 - damping) * s + damping * o
                 for s, o in zip(state, out)]
    cell_state = state

    # --- Stage B: back-propagate the 8-state to the entrance (root-find) ---
    if cell_start == 0:
        ent = cell_state
    else:
        from scipy.optimize import least_squares
        mf0 = compute_transfer_matrix(lattice, ref.copy(),
                                      start=0, end=cell_start - 1)
        seed_bet = find_matched_input_twiss(lattice, ref, cell_start, cell_end)
        ex = np.linalg.solve(mf0[np.ix_([0, 1], [0, 1])],
                             np.array([cell_state[4], cell_state[5]])
                             - mf0[[0, 1], 5])
        ey = np.linalg.solve(mf0[np.ix_([2, 3], [2, 3])],
                             np.array([cell_state[6], cell_state[7]])
                             - mf0[[2, 3], 5])
        seed = [seed_bet["alpha_x"], seed_bet["beta_x"],
                seed_bet["alpha_y"], seed_bet["beta_y"],
                float(ex[0]), float(ex[1]), float(ey[0]), float(ey[1])]

        def _residual(v):
            out = _run(front_lat, v)
            return [(o - c) / max(abs(c), 1e-3)
                    for o, c in zip(out, cell_state)]

        sol = least_squares(_residual, seed)
        ent = [float(x) for x in sol.x]
        converged = converged and bool(sol.success)

    return {
        "alpha_x": ent[0], "beta_x": ent[1],
        "alpha_y": ent[2], "beta_y": ent[3],
        "disp_x": ent[4], "disp_xp": ent[5],
        "disp_y": ent[6], "disp_yp": ent[7],
        "mu_x": zc["mu_x"], "mu_y": zc["mu_y"],
        "cell_start": cell_start, "cell_end": cell_end,
        "converged": bool(converged),
    }


#: an RFQ whose upstream elements add up to less than this (mm) starts the
#: lattice (TraceWin decks often carry 1e-5 mm placeholder drifts before it)
RFQ_AT_ENTRANCE_MM = 1e-3
#: RF-phase offsets (deg from the synchronous phase) at which the zero-current
#: match is re-computed, and the mismatch factor above which it is flagged:
#: a DC beam enters at every phase, and only a radial matching section long
#: enough makes one input ellipse suit them all (≈2 focusing periods and up
#: give ≤ 0.04 on the demo RFQ; its own 0.5-period section gives 2.6)
RFQ_PHASE_PROBE_DEG = (-60.0, -30.0, 30.0, 60.0)
RFQ_PHASE_SENSITIVITY_LIMIT = 0.1


def _mismatch(a1, b1, a2, b2):
    """Mismatch factor of the ellipse (a1, b1) against (a2, b2) (0 = same):
    the second must grow by (1 + M) in amplitude to contain the first."""
    import math
    g1, g2 = (1.0 + a1 * a1) / b1, (1.0 + a2 * a2) / b2
    R = b1 * g2 + g1 * b2 - 2.0 * a1 * a2
    return math.sqrt(0.5 * (R + math.sqrt(max(R * R - 4.0, 0.0)))) - 1.0


def _rfq_phase_sensitivity(lattice, ref_k, period, tw0):
    """(largest mismatch factor, input-section length in focusing periods):
    the zero-current match re-computed with every cell of the RFQ's input
    section and period at RF phases :data:`RFQ_PHASE_PROBE_DEG` from the
    synchronous one, against ``tw0``.  Copies of the cells are shifted;
    the lattice is not modified.  ``inf`` when the period is unstable at
    one of those phases."""
    import copy
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.rfq_cell import RfqCell

    k, cs, ce = int(period.rfq_start), int(period.start), int(period.end)
    els = lattice.elements

    def _len(a, b):
        return sum(float(e.length) for e in els[a:b]
                   if isinstance(e, RfqCell))

    periods = _len(k, cs) / max(_len(cs, ce + 1), 1e-12)
    worst = 0.0
    for d in RFQ_PHASE_PROBE_DEG:
        sub = Lattice()
        for e in els[k:ce + 1]:
            if isinstance(e, RfqCell):
                e = copy.deepcopy(e)
                e.phi_s_deg += d
            sub.add(e)
        try:
            tw = find_matched_input_twiss(sub, ref_k, cs - k, ce - k)
        except ValueError:
            return float("inf"), periods
        for pl in "xy":
            worst = max(worst, _mismatch(tw0[f"alpha_{pl}"], tw0[f"beta_{pl}"],
                                         tw[f"alpha_{pl}"], tw[f"beta_{pl}"]))
    return worst, periods


def _rfq_entry_state(lattice, ref, k):
    """(reference, SC multiplier, at-entrance flag, upstream length mm) at
    the entrance of element ``k``: the reference replayed through the
    elements before it, FREQ cards included; the space-charge compensation
    they leave in force (the envelope solver's rule: 1 − the last
    SPACE_CHARGE_COMP factor); whether they transport the beam at all."""
    import numpy as np
    from linac_gen.elements.base import PassiveElement
    from linac_gen.tracking.matrix_tracking import (compute_transfer_matrix,
                                                   reference_at)
    before = lattice.elements[:k]
    ref_k = reference_at(lattice, ref, k)
    sc_factor = 1.0
    for e in before:
        if isinstance(e, PassiveElement) and hasattr(e, "factor"):
            sc_factor = 1.0 - float(e.factor)
    upstream = sum(float(getattr(e, "length", 0.0) or 0.0) for e in before)
    at_entrance = upstream < RFQ_AT_ENTRANCE_MM
    if at_entrance and k > 0:
        # zero-length thin lenses / edges / gaps transport the beam too
        m4 = compute_transfer_matrix(lattice, ref, start=0, end=k - 1)[:4, :4]
        at_entrance = bool(np.allclose(m4, np.eye(4), rtol=0.0, atol=1e-6))
    return ref_k, sc_factor, at_entrance, upstream


def find_rfq_matched_input_twiss(lattice, ref, period=None, *, current=0.0,
                                 emit_x=None, emit_y=None,
                                 dc_energy_spread_keV=0.0,
                                 max_iter: int = 200, tol: float = 1e-4):
    """Matched input Twiss of an RFQ, at the RFQ entrance.

    The periodic Twiss of the RFQ's first regular focusing period
    (``period``, default the first of :func:`find_rfq_periods`), carried
    back through the radial matching section to the RFQ's first cell: the
    beam the upstream line must deliver.  Elements upstream of the RFQ (a
    LEBT) are not back-propagated through; only the reference particle is
    carried through them (with the frequency of a FREQ card among them),
    and the space-charge compensation they leave in force is kept.

    * ``current == 0``: :func:`find_matched_input_twiss` (the cells'
      transfer matrices for the synchronous particle).
    * ``current > 0`` (mA): :func:`find_sc_matched_input_twiss` started
      from a DC beam, because the beam entering an RFQ is unbunched (the
      envelope becomes a phase-uniform bunch at the first RFQ cell).
      Needs ``emit_x`` / ``emit_y``: geometric rms emittances (mm mrad)
      at the LATTICE entrance, as everywhere else (``base_initial``);
      they are rescaled to the RFQ entrance energy.  ``max_iter`` is
      higher than that function's default: its relative convergence test
      is strict for α ≈ 0, which the period's Twiss are.

    Returns ``alpha_x`` / ``beta_x`` / ``alpha_y`` / ``beta_y`` (β in
    mm/mrad = m) and the zero-current phase advance per period ``mu_x`` /
    ``mu_y`` (deg), the dispersion (zero for an RFQ), ``kind`` ("rfq"),
    ``rfq_start``, ``rfq_name``, ``cell_start`` / ``cell_end`` (lattice
    indices), ``label``, ``current``, ``converged``, ``upstream_mm``
    (length of the elements before the RFQ) and ``at_lattice_entrance``
    (nothing upstream transports the beam: the result is also the beam at
    the lattice entrance).

    The match is that of the synchronous particle, while a DC beam enters
    at every RF phase; it suits the whole beam only when the radial
    matching section makes it phase-independent.  ``phase_sensitivity`` is
    the largest mismatch factor between this match and the zero-current
    match re-computed 30° and 60° either side of the synchronous phase
    (``input_periods``: length of the cells before the period, in
    focusing periods); above :data:`RFQ_PHASE_SENSITIVITY_LIMIT`,
    ``caution`` says so (else ``None``) — check such an input with a
    multiparticle run.

    Raises
    ------
    ValueError
        If the lattice has no RFQ period, the period is unstable, or
        space-charge matching is asked for without the emittances.
    """
    from linac_gen.core.lattice import Lattice

    if period is None:
        periods = find_rfq_periods(lattice)
        if not periods:
            raise ValueError(
                "no RFQ period found — the lattice needs an RFQ with two "
                "consecutive regular cells of opposite sign (−2/+2)")
        period = periods[0]
    k = int(period.rfq_start)
    if not (0 <= k <= int(period.start) <= int(period.end)
            < len(lattice.elements)):
        raise ValueError(f"invalid RFQ period {period}")
    sub = Lattice()
    for e in lattice.elements[k:]:
        sub.add(e)
    step_cfg = getattr(lattice, "step_config", None)
    if step_cfg is not None:
        sub.step_config = step_cfg
    ref_k, sc_factor, at_entrance, upstream = _rfq_entry_state(lattice, ref, k)
    cs, ce = int(period.start) - k, int(period.end) - k

    tw = find_matched_input_twiss(sub, ref_k, cs, ce)
    keys = ("alpha_x", "beta_x", "alpha_y", "beta_y",
            "disp_x", "disp_xp", "disp_y", "disp_yp")
    out = {key: float(tw[key]) for key in keys}
    for key in ("mu_x", "mu_y", "mu_x_folded", "mu_y_folded"):
        if key in tw:
            out[key] = float(tw[key])
    sens, input_periods = _rfq_phase_sensitivity(lattice, ref_k, period, tw)
    caution = None
    if not sens <= RFQ_PHASE_SENSITIVITY_LIMIT:
        how = ("the period is unstable at some of those phases"
               if sens == float("inf") else
               f"it changes by a mismatch factor of up to {sens:.2f}")
        caution = (
            "the match depends on the RF phase: re-computed 30-60° from the "
            f"synchronous phase, {how} (the radial matching section is "
            f"{input_periods:.1f} focusing periods long), so no single input "
            "ellipse suits the whole DC beam — check this input with a "
            "multiparticle run")
    converged = True
    if current > 0:
        if emit_x is None or emit_y is None:
            raise ValueError(
                "space-charge matching of an RFQ needs emit_x and emit_y "
                "(geometric rms emittances, mm mrad)")
        damp = _emittance_damping(lattice, ref, k, ref_k)
        base = {"emit_x": float(emit_x) * damp, "emit_y": float(emit_y) * damp,
                "emit_z": 0.0, "alpha_z": 0.0, "beta_z": 1.0,
                "continuous": True,
                "dc_energy_spread_keV": float(dc_energy_spread_keV)}
        sc = find_sc_matched_input_twiss(sub, ref_k, cs, ce, float(current),
                                         base, max_iter=max_iter, tol=tol,
                                         sc_factor=sc_factor)
        out.update({key: float(sc[key]) for key in keys})
        converged = bool(sc["converged"])
    out.update(kind="rfq", rfq_start=k, rfq_name=str(lattice.elements[k].name),
               cell_start=int(period.start), cell_end=int(period.end),
               label=period.label, current=float(current),
               converged=converged, upstream_mm=upstream,
               at_lattice_entrance=at_entrance, phase_sensitivity=float(sens),
               input_periods=float(input_periods), caution=caution)
    return out


# ---------------------------------------------------------------------------
# Full-Σ matched period state (for channel tunes / phase probe)
# ---------------------------------------------------------------------------
_J6 = None


def _j6():
    global _J6
    if _J6 is None:
        import numpy as np
        J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
        _J6 = np.zeros((6, 6))
        _J6[0:2, 0:2] = J2
        _J6[2:4, 2:4] = J2
        _J6[4:6, 4:6] = J2
    return _J6


def _sigma_modes(Sigma, *, im_tol: float = 1e-30):
    """Wolski mode decomposition of a covariance matrix.

    ``eig(Σ·J)`` has eigenvalues ``±i·ε_k`` with the normal-mode
    vectors as eigenvectors.  Returns ``[(ε_k, v_k), …]`` for the
    positive-imaginary set, each v normalised to ``v† J v = i``
    (mirrors ``_build_coupled_matched_sigma``).  Degenerate blocks
    (e.g. an all-zero longitudinal block of a DC beam) simply
    contribute no mode — callers must preserve those rows themselves.
    """
    import numpy as np
    J = _j6()[:Sigma.shape[0], :Sigma.shape[0]]
    eigvals, eigvecs = np.linalg.eig(np.asarray(Sigma, float) @ J)
    modes = []
    for i in np.where(eigvals.imag > im_tol)[0]:
        eps = float(eigvals[i].imag)
        v = eigvecs[:, i]
        c = -1j * (v.conj() @ J @ v)
        if c.real <= 0:
            v = v.conj()
            c = -1j * (v.conj() @ J @ v)
        if c.real <= 0:
            continue
        v = v / np.sqrt(c.real)
        modes.append((eps, v))
    return modes


def _sigma_from_modes(modes):
    """Rebuild Σ = Σ_k ε_k · 2·Re(v_k v_k†) (Wolski Eq. 3.92/3.95)."""
    import numpy as np
    if not modes:
        raise ValueError("no modes to rebuild Σ from")
    dim = modes[0][1].shape[0]
    Sigma = np.zeros((dim, dim))
    for eps, v in modes:
        Sigma += eps * 2.0 * np.real(np.outer(v, v.conj()))
    return Sigma


def _renormalize_eigenemittances(Sigma, target_modes, *, pairing: str = "overlap",
                                 keep_unspanned: bool = True):
    """Rescale Σ's normal-mode emittances back to the targets.

    Under acceleration det(M) < 1 damps every emittance per pass —
    raw covariance iteration Σ → M Σ Mᵀ then has no full-rank fixed
    point (it collapses).  Re-imposing the target eigenemittances after
    every pass is the normalized-coordinates matching: the SHAPE
    converges while the amplitudes stay physical.  Modes are paired to
    the targets by symplectic eigenvector overlap |v_t† J v|.
    Rows/columns not spanned by any mode (degenerate z block of a DC
    beam) are preserved from the input.

    ``pairing="emittance"`` pairs them in order of emittance instead (the
    largest target with the largest mode): a coupled beam overlaps both
    modes of an uncoupled cell almost equally, and the overlap pairing then
    switches between them from one call to the next.  ``keep_unspanned=
    False`` rebuilds Σ from the paired modes alone, so a subspace that no
    target spans (a plane of zero emittance) comes out zero.
    """
    import numpy as np
    Sigma = np.asarray(Sigma, float)
    modes = _sigma_modes(Sigma)
    if not modes or not target_modes:
        return Sigma.copy()
    J = _j6()[:Sigma.shape[0], :Sigma.shape[0]]
    rebuilt = []
    if pairing == "emittance":
        by_eps_t = sorted(target_modes, key=lambda m: m[0], reverse=True)
        by_eps = sorted(modes, key=lambda m: m[0], reverse=True)
        rebuilt = [(eps_t, v) for (eps_t, _), (_, v) in zip(by_eps_t, by_eps, strict=False)]
    else:
        used = set()
        for eps_t, v_t in target_modes:
            best, best_o = None, -1.0
            for k, (eps, v) in enumerate(modes):
                if k in used:
                    continue
                o = abs(v_t.conj() @ J @ v)
                if o > best_o:
                    best, best_o = k, o
            if best is None:
                continue
            used.add(best)
            rebuilt.append((eps_t, modes[best][1]))
    out = _sigma_from_modes(rebuilt)
    # Preserve any subspace the modes don't span (rank-deficient z).
    if keep_unspanned and len(rebuilt) < Sigma.shape[0] // 2:
        spanned = np.zeros(Sigma.shape[0], dtype=bool)
        for _, v in rebuilt:
            spanned |= np.abs(v) > 1e-12
        for i in range(Sigma.shape[0]):
            if not spanned[i]:
                out[i, :] = Sigma[i, :]
                out[:, i] = Sigma[:, i]
    return out


def _betatron_block(Sigma, s55: float):
    """The transverse block of Σ less the part the energy deviation accounts for, the
    dispersion's: ``Σ_T − c·cᵀ/Σ55`` with ``c = Σ[0:4, 5]``.  It is the Σ of the betatron
    motion, whose emittances a cell without RF keeps.  With no energy spread it is ``Σ_T``."""
    import numpy as np
    T = np.array(Sigma[0:4, 0:4], float)
    if s55 > 0.0:
        c = np.asarray(Sigma[0:4, 5], float)
        T = T - np.outer(c, c) / s55
    return T


def _has_negative_eigenvalue(A, rel: float = 1e-12) -> bool:
    """True when the symmetric matrix A has an eigenvalue below −rel times its largest diagonal
    entry: round-off on a singular block (a plane or an energy spread of zero) is not one."""
    import numpy as np
    A = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    scale = float(np.abs(np.diag(A)).max()) if A.size else 0.0
    return bool(A.size) and float(np.linalg.eigvalsh(A).min()) < -rel * max(scale, 1e-300)


def make_period_sigma_map(lattice, ref, period, current: float,
                          base_initial: dict):
    """Build the DISCRETE one-period Σ-map of a periodic cell, with the
    delicate prefix-state capture done once and preserved.

    Returns ``(sigma_map, state)`` where ``sigma_map(Σ_entry) -> Σ_exit``
    performs one REAL envelope pass over the FIRST repeat of ``period``, the
    exit Σ handed back on the entrance's clock when the repeat holds a change
    of clock (``state["clock_ratio"]`` = f_exit/f_entrance),
    (so the map is the solver's exact discrete dynamics, self-consistent
    SC included), and ``state`` carries ``ref_entry`` (ReferenceParticle
    at the period entry), ``sc_factor``, ``bunch_frequency`` (the
    ORIGINAL repetition frequency — a post-FREQ-jump slice must not
    re-derive it locally), ``continuous`` and ``sigma_seed`` (the
    physical Σ delivered to the period entry by ``base_initial``).

    Shared by :func:`find_matched_period_sigma` (fixed-point iteration)
    and the envelope Floquet analyzer
    (:mod:`linac_gen.analysis.envelope_floquet`), which differentiates
    this map by finite differences — one implementation of the
    prefix-state rules, two consumers.
    """
    import numpy as np
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.lattice_commands import LatticeCommand, REFERENCE_COMMANDS
    from linac_gen.tracking.envelope import EnvelopeSolver

    spans = period.spans()
    a0, b0 = spans[0]

    # --- ONE prefix pass: capture exact state at the period entry. ----
    prefix = Lattice()
    for e in lattice.elements[:a0]:
        prefix.add(e)
    step_cfg = getattr(lattice, "step_config", None)
    if step_cfg is not None:
        prefix.step_config = step_cfg
    env_pre = EnvelopeSolver(prefix, ref.copy(), base_initial,
                             current=current)
    res_pre = env_pre.run()
    ref_entry = env_pre._ref            # post-run == at period entry
    sc_factor = float(env_pre._sc_factor)
    bunch_freq = float(env_pre._bunch_freq)
    continuous = bool(env_pre._continuous)
    sigma_seed = np.asarray(res_pre.sigma_matrix[-1], float).copy()

    # (ref_entry already holds the cards that set the reference: one that shifts it would
    # count twice, and a card before a cavity would undo what the cavity did)
    prefix_cmds = [e for e in lattice.elements[:a0]
                   if isinstance(e, LatticeCommand) and not isinstance(e, REFERENCE_COMMANDS)]

    # --- Cell sub-lattice (first repeat). ------------------------------
    cell_lat = Lattice()
    for e in lattice.elements[a0:b0]:
        cell_lat.add(e)
    if step_cfg is not None:
        cell_lat.step_config = step_cfg
    # the clock at the cell's exit over the clock at its entrance (1 with no change in it)
    from linac_gen.tracking.matrix_tracking import _replay_ref
    rc = ref_entry.copy()
    _replay_ref(list(cell_lat.elements), rc, cell_lat)
    clock_ratio = (float(rc.frequency) / float(ref_entry.frequency)
                   if float(ref_entry.frequency) > 0.0 and float(rc.frequency) > 0.0 else 1.0)

    def sigma_map(Sigma_n):
        init = dict(base_initial)
        init["continuous"] = continuous
        env = EnvelopeSolver(cell_lat, ref_entry.copy(), init,
                             current=current, initial_sigma=Sigma_n,
                             bunch_frequency=bunch_freq,
                             sc_factor=sc_factor)
        for cmd in prefix_cmds:
            try:
                cmd.apply_command(env.track_state)
            except Exception:                                # noqa: BLE001
                pass
        r = env.run()
        S = np.asarray(r.sigma_matrix[-1], float)
        # a cell that holds a change of clock hands its exit on another clock: brought back
        # to the entrance's, so that the map compares like with like
        f_out = float(r.ref_frequency[-1]) if len(r.ref_frequency) else 0.0
        f_in = float(ref_entry.frequency)
        if f_in > 0.0 and f_out > 0.0 and f_out != f_in:
            k = f_in / f_out
            S = S.copy()
            S[4, :] *= k
            S[:, 4] *= k
        return S

    state = {"ref_entry": ref_entry, "sc_factor": sc_factor,
             "bunch_frequency": bunch_freq, "continuous": continuous,
             "sigma_seed": sigma_seed, "clock_ratio": clock_ratio}
    return sigma_map, state


def _no_longitudinal_focusing(lattice, period, ref_entry) -> bool:
    """True when nothing in the period's first cell changes a particle's energy with its
    phase: no RF element in it, and a zero-current map whose energy row does not depend on
    the phase (an explicit matrix element can do that without being an RF element).  The map
    is that of a copy of the cell, so no element's run state moves.  When the map cannot be
    computed the answer is False, which leaves the search as it was.  No warning filter is
    set around it: filters are process-wide, and this runs on GUI worker threads."""
    import copy

    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    from linac_gen.tracking.tracker import _is_rf_bunching_element
    a0, b0 = period.spans()[0]
    cell = lattice.elements[a0:b0]
    if any(_is_rf_bunching_element(e) for e in cell):
        return False
    try:
        sub_lat = Lattice()
        for e in copy.deepcopy(cell):
            sub_lat.add(e)
        step_cfg = getattr(lattice, "step_config", None)
        if step_cfg is not None:
            sub_lat.step_config = step_cfg
        M = compute_transfer_matrix(sub_lat, ref_entry.copy())
    except Exception:                                           # noqa: BLE001
        return False
    return float(M[5, 4]) == 0.0


def find_matched_period_sigma(lattice, ref, period, current: float,
                              base_initial: dict, *,
                              max_iter: int = 40, tol: float = 1e-6,
                              mix: float = 0.5,
                              longitudinal: str = "iterate"):
    """Matched full-6×6 entrance Σ of a (quasi-)periodic cell WITH SC.

    Fixed point of Σ → R(M_dep(Σ)·Σ·M_dep(Σ)ᵀ) over the FIRST repeat of
    ``period``, where the map application is an actual envelope pass
    (so M_dep is the solver's exact tangent map at each iterate) and R
    renormalizes the normal-mode emittances back to the seed's — the
    normalized-coordinates matching required because det(M₄) < 1 under
    acceleration damps raw covariance iteration toward a singular
    state.

    State preservation (period-scoped reruns): the prefix
    ``elements[:start]`` is walked ONCE by a real envelope pass to
    capture the reference at the period entry, the SPACE_CHARGE_COMP
    factor, the ORIGINAL bunch repetition frequency (a post-FREQ-jump
    slice must not re-derive it from the local reference), the
    DC/bunched state, and the physical seed Σ.  Sticky LatticeCommands
    from the prefix are re-applied to each iterate's track state, except
    the ones that set the reference, which the reference at the entry holds
    (idempotent set-style commands assumed — same limitation as every
    replay path in this codebase).

    ``longitudinal="frozen"`` pins the longitudinal block (and its
    cross terms) to the seed's after every iteration — an explicitly
    labelled approximation for cells whose z fixed point is not
    meaningful (strong acceleration per cell).  The result then carries
    ``sigma_model_approx="frozen-longitudinal"``.

    A bunched beam in a cell where nothing changes a particle's energy with its
    phase (no RF: a transport line's FODO or arc) has no matched bunch: the
    bunch lengthens on every pass, iterating its longitudinal block does not
    converge, and the transverse Σ handed on is wherever the search stopped.
    ``"iterate"`` therefore holds the incoming beam's bunch length, energy
    spread and their correlation there (the longitudinal 2×2 block), and
    solves for the transverse block and its correlation with the energy,
    which in a cell with bends carries the cell's periodic dispersion.  The
    phase is taken to correlate with the transverse planes only through the
    energy: in a cell with bends the path length also correlates it with x
    and x', which the held beam leaves out (that keeps Σ a covariance matrix,
    and the space-charge kick does not read it).  The betatron normal-mode
    emittances (of the transverse block less the dispersion's share) are the
    incoming beam's; a plane of zero emittance stays zero.  (An x-y coupled
    beam into an uncoupled cell has two matched beams, one for each way of
    giving its two emittances to the cell's two planes: each incoming mode
    keeps its emittance through the search, which ends at one of them.)  It
    says so: a UserWarning, ``longitudinal`` reads ``"held"``,
    ``sigma_model_approx`` ``"held-longitudinal"``, and ``longitudinal_held``
    gives the reason.  A DC beam and a cell with RF iterate as before.  A
    solve that does not converge raises a RuntimeWarning, and so does a held
    beam with a negative eigenvalue (it is then reported as not converged).

    Returns
    -------
    dict
        ``sigma_entry`` (6×6), ``converged`` (bool), ``n_iter``,
        ``residual`` (relative Frobenius), ``ref_entry``
        (ReferenceParticle at the period entry), ``bunch_frequency``,
        ``sc_factor``, ``continuous``, ``longitudinal``,
        ``sigma_model_approx`` (None | "frozen-longitudinal" |
        "held-longitudinal"), ``longitudinal_held`` (None, or why ``"iterate"``
        held the bunch length and energy spread).
    """
    import warnings

    import numpy as np

    _pass, state = make_period_sigma_map(lattice, ref, period, current,
                                         base_initial)
    ref_entry = state["ref_entry"]
    sc_factor = state["sc_factor"]
    bunch_freq = state["bunch_frequency"]
    continuous = state["continuous"]
    sigma_seed = state["sigma_seed"]

    held = None
    hold = (longitudinal == "iterate" and not continuous
            and _no_longitudinal_focusing(lattice, period, ref_entry))
    if hold:
        longitudinal = "held"
        held = ("no longitudinal focusing in the cell (no RF): the incoming beam's bunch length "
                "and energy spread are held")
        warnings.warn(f"period {getattr(period, 'label', '')!r}: a bunched beam and no RF in the "
                      "cell, so no matched bunch exists; the matched beam keeps the incoming "
                      "beam's bunch length and energy spread", UserWarning, stacklevel=2)

    n_pass = [0]

    def _counted_pass(Sigma):
        n_pass[0] += 1
        return _pass(Sigma)

    if hold:
        # The longitudinal 2×2 block is held, and so is the phase's correlation with the
        # betatron motion, at zero: Σ[0:4, 4] = Σ[0:4, 5]·Σ45/Σ55 is the part the energy accounts
        # for.  (In a cell with bends the path length correlates the phase with x and x' too; held
        # at zero, Σ = L·diag(B, zz)·Lᵀ with L unit triangular stays a covariance matrix, and the
        # space-charge kick does not read it.)  The betatron block (the transverse one less the
        # dispersion's share) and the correlation with the energy (the dispersion) are solved
        # for, the betatron normal-mode emittances renormalized to the incoming beam's.  The
        # search runs on the betatron block itself: at zero current a pass maps it to M·B·Mᵀ
        # whatever the dispersion does, so a step in the dispersion cannot make it lose a mode.
        # Renormalizing the full 6-D modes instead would mix in the longitudinal plane, whose
        # units (deg, MeV) are not canonical with the transverse ones once a bend couples them.
        s55 = float(sigma_seed[5, 5])
        zz_seed = sigma_seed[4:6, 4:6].copy()
        b_seed = _betatron_block(sigma_seed, s55)
        target_modes = _sigma_modes(b_seed)
        # A plane of zero emittance has no mode to hold its size, so the renormalized block is
        # rebuilt from the modes alone (that plane comes out zero).  Otherwise any rows the modes
        # do not span are kept from the pass, as elsewhere: of two equal emittances the mode
        # decomposition can return one, and the other plane must not be lost with it.
        eps_seed = np.sort(np.abs(np.linalg.eigvals(b_seed @ _j6()[:4, :4]).imag))
        keep_unspanned = not (eps_seed[0] <= 1e-12 * max(float(eps_seed[-1]), 1e-300))
    else:
        target_modes = _sigma_modes(sigma_seed)
        zblk_seed = sigma_seed[4:6, :].copy(), sigma_seed[:, 4:6].copy()

    def _held_complete(S, zz):
        """S with the held block zz and the phase correlated with the transverse planes only
        through the energy (its correlation with the betatron motion held at zero); symmetric.
        (zz is the block in the frame S is in.)"""
        S[4:6, 4:6] = zz
        if zz[1, 1] > 0.0:
            S[0:4, 4] = S[0:4, 5] * (zz[0, 1] / zz[1, 1])
        else:
            S[0:4, 4:6] = 0.0
        S[4:6, 0:4] = S[0:4, 4:6].T
        return S

    def _to_betatron(S):
        S = np.array(S, float)
        S[0:4, 0:4] = _betatron_block(S, s55)
        return S

    def _from_betatron(S):
        S = np.array(S, float)
        if s55 > 0.0:
            S[0:4, 0:4] += np.outer(S[0:4, 5], S[0:4, 5]) / s55
        return S

    def _step(Sigma):
        """One renormalized period map application R(M·Σ·Mᵀ).  With the bunch held, Σ in and
        out carries the betatron block in place of the transverse one."""
        if hold:
            S = _held_complete(np.array(_counted_pass(_from_betatron(Sigma)), float), zz_seed)
            S[0:4, 0:4] = _renormalize_eigenemittances(_betatron_block(S, s55), target_modes,
                                                       pairing="emittance",
                                                       keep_unspanned=keep_unspanned)
            return S
        Sigma_new = _renormalize_eigenemittances(_counted_pass(Sigma), target_modes)
        if longitudinal == "frozen":
            Sigma_new = Sigma_new.copy()
            Sigma_new[4:6, :] = zblk_seed[0]
            Sigma_new[:, 4:6] = zblk_seed[1]
        return Sigma_new

    # A cell that holds a change of clock is solved on its exit clock, where most of it runs.
    # This is conditioning only, the fixed point is the same.  The unscaled least-squares
    # polish below ends where round-off takes it on such a cell: from six starts perturbed by
    # up to 1e-11 it converged 6 and 4 times on the exit clock (one test cell switching up,
    # one switching down), against 4 and 0 on the entrance clock, where the second stalled
    # 2-3e-2 off in beta_z.  The renormalized step itself stays on the entrance clock, and
    # the result is handed back on it.
    k = float(state.get("clock_ratio", 1.0) or 1.0)
    sigma_start = sigma_seed
    if k != 1.0:
        K = np.eye(6)
        K[4, 4] = k
        Ki = np.eye(6)
        Ki[4, 4] = 1.0 / k
        _step_on_entry = _step

        def _step(S, _s=_step_on_entry):
            return K @ _s(Ki @ S @ Ki.T) @ K.T

        sigma_start = K @ sigma_seed @ K.T
    if hold:
        zz_start = np.array(sigma_start[4:6, 4:6], float)      # the held block, on this clock
        sigma_start = _to_betatron(_held_complete(np.array(sigma_start, float), zz_start))
        # the scale of the entries solved for: the held block (deg², large for a long bunch)
        # would loosen their tolerance by its own size
        solved = np.zeros((6, 6))
        solved[0:4, 0:4] = sigma_start[0:4, 0:4]
        solved[0:4, 5] = solved[5, 0:4] = sigma_start[0:4, 5]
        norm0 = max(float(np.linalg.norm(solved)), 1e-30)
    else:
        norm0 = max(float(np.linalg.norm(sigma_start)), 1e-30)
    iu = np.triu_indices(6)
    # the entries the polish varies: all of them, or with the bunch held the transverse block
    # and its correlation with the energy
    free = (np.flatnonzero((iu[1] < 4) | ((iu[1] == 5) & (iu[0] < 4))) if hold
            else np.arange(len(iu[0])))
    fixed_vec = sigma_start[iu].copy()       # only its held entries are read

    def _from_vec(v):
        full = fixed_vec.copy()
        full[free] = v
        S = np.zeros((6, 6))
        S[iu] = full
        S = S + np.triu(S, 1).T             # symmetrize
        return _held_complete(S, zz_start) if hold else S

    def _residual_vec(v):
        S = _from_vec(v)
        return (_step(S) - S)[iu][free] / norm0

    # Damped Picard warm-up (robust far from the fixed point) …
    Sigma = sigma_start.copy()
    residual = float("inf")
    n_done = 0
    warmup = min(max_iter, 8)
    for n_done in range(1, warmup + 1):
        Sigma_new = _step(Sigma)
        Sigma_mixed = (1.0 - mix) * Sigma + mix * Sigma_new
        residual = float(np.linalg.norm(Sigma_new - Sigma)) / norm0
        Sigma = Sigma_mixed
        if residual < tol:
            break
    if hold:
        Sigma = _held_complete(Sigma, zz_start)      # exactly the held block, whatever the mix

    # … then a least-squares polish (Picard converges slowly when the
    # period map's spectral radius is near 1; for accelerating cells no
    # exact fixed point exists and least_squares lands on the least-
    # residual quasi-periodic state, which is the honest answer).
    if residual >= tol:
        from scipy.optimize import least_squares
        sol = least_squares(_residual_vec, Sigma[iu][free],
                            xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=60 * len(free))
        Sigma = _from_vec(sol.x)
        residual = float(np.abs(sol.fun).max())
        n_done += int(sol.nfev)

    if hold:
        Sigma = _from_betatron(Sigma)
    if k != 1.0:
        Sigma = Ki @ Sigma @ Ki.T
    converged = residual < tol
    why = None
    if not converged:
        why = (f"did not converge (residual {residual:.2e} against {tol:.0e} after "
               f"{n_pass[0]} passes of the cell); what is computed from it is approximate")
    elif hold and (_has_negative_eigenvalue(_betatron_block(Sigma, s55))
                   or _has_negative_eigenvalue(Sigma[4:6, 4:6])):
        converged = False
        why = ("has a negative eigenvalue, so it is not a beam: it is reported as not "
               "converged")
    if why is not None:
        warnings.warn(f"period {getattr(period, 'label', '')!r}: the matched beam {why}",
                      RuntimeWarning, stacklevel=2)
    return {
        "sigma_entry": Sigma,
        "converged": converged,
        "n_iter": n_done,
        "residual": residual,
        "ref_entry": ref_entry,
        "bunch_frequency": bunch_freq,
        "sc_factor": sc_factor,
        "continuous": continuous,
        "longitudinal": longitudinal,
        "sigma_model_approx": ("frozen-longitudinal" if longitudinal == "frozen"
                               else "held-longitudinal" if hold else None),
        "longitudinal_held": held,
    }
