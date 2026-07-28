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
"""


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


def find_sc_matched_input_twiss(lattice, ref, cell_start, cell_end,
                                current, base_initial, *,
                                max_iter: int = 50, tol: float = 1e-4):
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
    ``emit_y``, ``emit_z``, ``alpha_z``, ``beta_z``.

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
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

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

    def _run(sub, st):
        init = dict(base_initial)
        init.update(alpha_x=st[0], beta_x=st[1], alpha_y=st[2], beta_y=st[3],
                    disp_x=st[4], disp_xp=st[5], disp_y=st[6], disp_yp=st[7])
        env = EnvelopeSolver(sub, ref.copy(), init, current=current).run()
        return _state(np.asarray(env.sigma_matrix[-1]))

    # seed: zero-current cell betatron Twiss + periodic dispersion
    # (shared _periodic_dispersion helper; a NaN result — near-integer
    # tune — is sanitised to 0.0 here because this is only a SEED for
    # the fixed-point iteration, which can still converge from zero).
    zc = find_periodic_twiss(cell_lat, ref)
    mc0 = compute_transfer_matrix(cell_lat, ref.copy())
    dx0 = np.nan_to_num(_periodic_dispersion(mc0, (0, 1)), nan=0.0)
    dy0 = np.nan_to_num(_periodic_dispersion(mc0, (2, 3)), nan=0.0)
    state = [zc["alpha_x"], zc["beta_x"], zc["alpha_y"], zc["beta_y"],
             float(dx0[0]), float(dx0[1]), float(dy0[0]), float(dy0[1])]

    # --- Stage A: SC-periodic state of the cell (betatron Twiss + dispersion) ---
    damping = 0.5
    converged = False
    for _ in range(max_iter):
        out = _run(cell_lat, state)
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


def _renormalize_eigenemittances(Sigma, target_modes):
    """Rescale Σ's normal-mode emittances back to the targets.

    Under acceleration det(M) < 1 damps every emittance per pass —
    raw covariance iteration Σ → M Σ Mᵀ then has no full-rank fixed
    point (it collapses).  Re-imposing the target eigenemittances after
    every pass is the normalized-coordinates matching: the SHAPE
    converges while the amplitudes stay physical.  Modes are paired to
    the targets by symplectic eigenvector overlap |v_t† J v|.
    Rows/columns not spanned by any mode (degenerate z block of a DC
    beam) are preserved from the input.
    """
    import numpy as np
    Sigma = np.asarray(Sigma, float)
    modes = _sigma_modes(Sigma)
    if not modes or not target_modes:
        return Sigma.copy()
    J = _j6()[:Sigma.shape[0], :Sigma.shape[0]]
    rebuilt = []
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
    if len(rebuilt) < Sigma.shape[0] // 2:
        spanned = np.zeros(Sigma.shape[0], dtype=bool)
        for _, v in rebuilt:
            spanned |= np.abs(v) > 1e-12
        for i in range(Sigma.shape[0]):
            if not spanned[i]:
                out[i, :] = Sigma[i, :]
                out[:, i] = Sigma[:, i]
    return out


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
    from the prefix are re-applied to each iterate's track state
    (idempotent set-style commands assumed — same limitation as every
    replay path in this codebase).

    ``longitudinal="frozen"`` pins the longitudinal block (and its
    cross terms) to the seed's after every iteration — an explicitly
    labelled approximation for cells whose z fixed point is not
    meaningful (strong acceleration per cell).  The result then carries
    ``sigma_model_approx="frozen-longitudinal"``.

    Returns
    -------
    dict
        ``sigma_entry`` (6×6), ``converged`` (bool), ``n_iter``,
        ``residual`` (relative Frobenius), ``ref_entry``
        (ReferenceParticle at the period entry), ``bunch_frequency``,
        ``sc_factor``, ``continuous``, ``longitudinal``,
        ``sigma_model_approx`` (None | "frozen-longitudinal").
    """
    import numpy as np
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.elements.lattice_commands import LatticeCommand

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

    prefix_cmds = [e for e in lattice.elements[:a0]
                   if isinstance(e, LatticeCommand)]

    # --- Cell sub-lattice (first repeat). ------------------------------
    cell_lat = Lattice()
    for e in lattice.elements[a0:b0]:
        cell_lat.add(e)
    if step_cfg is not None:
        cell_lat.step_config = step_cfg

    target_modes = _sigma_modes(sigma_seed)
    zblk_seed = sigma_seed[4:6, :].copy(), sigma_seed[:, 4:6].copy()

    def _pass(Sigma_n):
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
        return np.asarray(r.sigma_matrix[-1], float)

    def _step(Sigma):
        """One renormalized period map application R(M·Σ·Mᵀ)."""
        Sigma_new = _renormalize_eigenemittances(_pass(Sigma), target_modes)
        if longitudinal == "frozen":
            Sigma_new = Sigma_new.copy()
            Sigma_new[4:6, :] = zblk_seed[0]
            Sigma_new[:, 4:6] = zblk_seed[1]
        return Sigma_new

    norm0 = max(float(np.linalg.norm(sigma_seed)), 1e-30)
    iu = np.triu_indices(6)

    def _residual_vec(v):
        S = np.zeros((6, 6))
        S[iu] = v
        S = S + np.triu(S, 1).T          # symmetrize
        return (_step(S) - S)[iu] / norm0

    # Damped Picard warm-up (robust far from the fixed point) …
    Sigma = sigma_seed.copy()
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

    # … then a least-squares polish (Picard converges slowly when the
    # period map's spectral radius is near 1; for accelerating cells no
    # exact fixed point exists and least_squares lands on the least-
    # residual quasi-periodic state, which is the honest answer).
    if residual >= tol:
        from scipy.optimize import least_squares
        sol = least_squares(_residual_vec, Sigma[iu],
                            xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=60 * len(iu[0]))
        S = np.zeros((6, 6))
        S[iu] = sol.x
        Sigma = S + np.triu(S, 1).T
        residual = float(np.abs(sol.fun).max())
        n_done += int(sol.nfev)

    return {
        "sigma_entry": Sigma,
        "converged": residual < tol,
        "n_iter": n_done,
        "residual": residual,
        "ref_entry": ref_entry,
        "bunch_frequency": bunch_freq,
        "sc_factor": sc_factor,
        "continuous": continuous,
        "longitudinal": longitudinal,
        "sigma_model_approx": ("frozen-longitudinal"
                               if longitudinal == "frozen" else None),
    }
