"""Transfer-matrix (model) dispersion along the lattice.

Complements the *statistical* dispersion the results popup computes from
the beam Σ-matrix (``D_u = ⟨u·δ⟩/⟨δ²⟩``): here the dispersion is the
image of a unit energy-offset ray under the accumulated element transfer
matrices — pure linear machine optics, independent of the tracked beam
and of space charge.

Definition
----------
In HELIX's 6-D matrix basis (x mm, x' mrad, y mm, y' mrad, Δφ deg,
ΔW MeV) the dispersion ray is propagated as

    v(s) = M(0→s) · (η_x0, η'_x0, η_y0, η'_y0, 0, 1)ᵀ

and the reported dispersion is the renormalised ratio

    η_u(s) = v_u(s) / v_W(s)          [mm/MeV, mrad/MeV]

``v_W`` stays exactly 1 through static elements (bends, quads, drifts);
RF cavities rotate the (Δφ, ΔW) block, after which η_u(s) remains the
"per unit of *local* energy offset" derivative.  Should ``v_W`` pass
through zero (synchrotron rotation), the ratio is undefined and the
curve goes NaN from that point on.

For display next to the statistical curve, the popup converts to metres
per unit relative momentum offset δ = Δp/p exactly like the Σ-based
formula:

    D_u [m] = η_u [mm/MeV] · β²γ·mc² [MeV] · 1e-3

with β, γ of the reference particle *at that s* — so on a static,
space-charge-free line fed with a dispersion-free beam the two curves
are identical (the Σ-matrix is propagated by the very same matrices).

The seed ``eta0`` mirrors the input-dispersion seeding of the envelope
solver (``disp_x/disp_xp/disp_y/disp_yp`` in mm/MeV and mrad/MeV):
pass the same values to compare like with like.
"""
from __future__ import annotations

import numpy as np

__all__ = ["dispersion_along_s"]


def dispersion_along_s(lattice, ref, *, eta0=None,
                       cache: dict | None = None,
                       should_stop=None) -> dict:
    """Propagate the dispersion ray element-by-element over the lattice.

    Parameters
    ----------
    lattice : Lattice
        The lattice to walk (all elements, from s = 0).
    ref : ReferenceParticle
        Entrance reference particle; a copy is advanced through the
        walk so per-element matrices are evaluated at the correct
        kinetic energy (same convention as
        :func:`linac_gen.analysis.phase_advance.structure_phase_advance_along_s`).
    eta0 : sequence of 4 floats, optional
        Entrance dispersion ``(η_x, η'_x, η_y, η'_y)`` in mm/MeV and
        mrad/MeV.  Default all-zero (dispersion-free entrance).
    cache : dict, optional
        Opt-in per-element transfer-matrix memoisation dict (see
        ``matrix_tracking.get_element_matrix``).
    should_stop : callable, optional
        Cooperative-cancel poll; raising via
        ``linac_gen.core.cancelled.OperationCancelled``.

    Returns
    -------
    dict
        ``s`` (mm), ``disp_x_m`` / ``disp_y_m`` (metres, δ-normalised),
        ``eta_x`` / ``eta_xp`` / ``eta_y`` / ``eta_yp`` (mm/MeV,
        mrad/MeV), each of length ``len(lattice.elements) + 1``
        (entry + every element exit), and ``complete`` (False when an
        unsupported element broke the chain — downstream samples NaN).
    """
    from linac_gen.analysis.phase_advance import _check_stop
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    from linac_gen.tracking.matrix_tracking import get_element_matrix

    n = len(lattice.elements)
    s_arr = np.zeros(n + 1)
    eta_x = np.full(n + 1, np.nan); eta_xp = np.full(n + 1, np.nan)
    eta_y = np.full(n + 1, np.nan); eta_yp = np.full(n + 1, np.nan)
    dx_m = np.full(n + 1, np.nan); dy_m = np.full(n + 1, np.nan)

    if eta0 is None:
        eta0 = (0.0, 0.0, 0.0, 0.0)
    v = np.array([float(eta0[0]), float(eta0[1]),
                  float(eta0[2]), float(eta0[3]), 0.0, 1.0])

    rc = ref.copy()
    mass = float(rc.species.mass)

    def _emit(k: int) -> None:
        w = v[5]
        if abs(w) < 1e-12:
            return                      # renormalisation undefined → NaN
        eta_x[k] = v[0] / w; eta_xp[k] = v[1] / w
        eta_y[k] = v[2] / w; eta_yp[k] = v[3] / w
        f = (rc.beta ** 2) * rc.gamma * mass * 1e-3     # mm/MeV → m/δ
        dx_m[k] = eta_x[k] * f
        dy_m[k] = eta_y[k] * f

    s_arr[0] = float(rc.s)
    _emit(0)
    complete = True
    break_idx = n
    for i, el in enumerate(lattice.elements):
        _check_stop(should_stop)
        try:
            m6 = get_element_matrix(el, rc, cache=cache)
        except Exception:                                 # noqa: BLE001
            # Unsupported element — break the chain; downstream stays
            # NaN but the s grid is still filled below.
            complete = False
            break_idx = i
            break
        v = np.asarray(m6, dtype=float) @ v
        if isinstance(el, FieldMapElement):
            el.advance_ref(rc)
        else:
            rc.s += el.length
            if isinstance(el, ThinKickElement):
                el.advance_ref(rc)
        s_arr[i + 1] = float(rc.s)
        _emit(i + 1)

    if not complete:
        # Fill the remaining s grid (by bare lengths — energy is moot for
        # the s axis) so the curves still span the whole lattice.
        s_run = s_arr[break_idx]
        for j in range(break_idx, n):
            s_run += float(getattr(lattice.elements[j], "length", 0.0) or 0.0)
            s_arr[j + 1] = s_run

    return {
        "s": s_arr,
        "disp_x_m": dx_m, "disp_y_m": dy_m,
        "eta_x": eta_x, "eta_xp": eta_xp,
        "eta_y": eta_y, "eta_yp": eta_yp,
        "complete": complete,
    }
