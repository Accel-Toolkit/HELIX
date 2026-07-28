"""Full beam-parameter summary of one particle distribution.

Headless (no Qt) so the GUI's phase-space popup table and any script
can share it.  Operates on the standard ``(N, 6)`` particle array —
``[x mm, x' mrad, y mm, y' mrad, Δφ deg, ΔW MeV]`` — reusing the
existing moment/Twiss/emittance/halo helpers; no new physics here.

Conventions surfaced in the rows:
* longitudinal Twiss in the HELIX-internal (Δφ, ΔW) pair
  (α_z = −⟨Δφ·ΔW⟩/ε_z = −TraceWin's; β_z in deg/MeV);
* normalized emittance = geometric × βγ of the supplied reference;
* derived σ_z (mm) via the local βλ, σ_δ via β²γm — reference needed.
"""
from __future__ import annotations

import numpy as np

from linac_gen.diagnostics.eigenemittance import eigenemittances
from linac_gen.diagnostics.moments import (
    compute_halo, compute_moments, compute_twiss_from_particles,
)
from linac_gen.diagnostics.recorder import _convert_emit_z_to_mmmrad


def _f(v: float) -> str:
    return f"{float(v):.6g}"


def summarize_particles(particles, ref=None, *, species_name: str = "",
                        current_ma: float | None = None,
                        n_total: int | None = None,
                        location: str = "") -> list[tuple]:
    """Return ordered rows ``(group, name, value_str, unit)``.

    A row with an empty ``name`` marks the start of a group (the GUI
    renders it as a section header).  ``ref`` is a
    :class:`~linac_gen.core.reference.ReferenceParticle` (or None —
    reference-dependent rows are then omitted).  ``n_total`` is the
    launched macroparticle count, enabling the transmission row.
    """
    rows: list[tuple] = []

    def hdr(group):
        rows.append((group, "", "", ""))

    def add(group, name, value, unit=""):
        rows.append((group, name, value if isinstance(value, str)
                     else _f(value), unit))

    if particles is None or len(particles) == 0:
        hdr("Distribution")
        add("Distribution", "status", "no particle data", "")
        if location:
            add("Distribution", "location", location, "")
        return rows

    q = np.asarray(particles, dtype=float)
    n = len(q)

    # ---- distribution ------------------------------------------------
    hdr("Distribution")
    if location:
        add("Distribution", "location", location, "")
    if n_total:
        add("Distribution", "macroparticles",
            f"{n} / {int(n_total)}", "alive / launched")
        add("Distribution", "transmission", 100.0 * n / n_total, "%")
    else:
        add("Distribution", "macroparticles", str(n), "")
    species = getattr(ref, "species", None)
    if species is not None:
        add("Distribution", "species",
            getattr(species, "name", "") or species_name, "")
        add("Distribution", "rest mass", species.mass, "MeV/c²")
        add("Distribution", "charge state", f"{species.charge:+g}", "e")
    elif species_name:
        add("Distribution", "species", species_name, "")
    if current_ma is not None:
        add("Distribution", "beam current", current_ma, "mA")
    if ref is not None and getattr(ref, "frequency", 0.0):
        add("Distribution", "RF frequency", ref.frequency, "MHz")

    # ---- reference particle -----------------------------------------
    if ref is not None:
        hdr("Reference particle")
        add("Reference particle", "s", getattr(ref, "s", 0.0) / 1000.0, "m")
        add("Reference particle", "W_kin", ref.w_kin, "MeV")
        add("Reference particle", "β", ref.beta, "")
        add("Reference particle", "γ", ref.gamma, "")
        add("Reference particle", "βγ", ref.bg, "")
        add("Reference particle", "φ_s", getattr(ref, "phi_s", 0.0), "deg")

    # ---- centroid ----------------------------------------------------
    mom = compute_moments(q)
    mean = mom["mean"]
    hdr("Centroid")
    for i, (nm, unit) in enumerate((("⟨x⟩", "mm"), ("⟨x'⟩", "mrad"),
                                    ("⟨y⟩", "mm"), ("⟨y'⟩", "mrad"),
                                    ("⟨Δφ⟩", "deg"), ("⟨ΔW⟩", "MeV"))):
        add("Centroid", nm, mean[i], unit)
    if ref is not None:
        add("Centroid", "mean kinetic energy",
            ref.w_kin + mean[5], "MeV")

    # ---- RMS sizes ---------------------------------------------------
    hdr("RMS sizes")
    add("RMS sizes", "σ_x", mom["sigma_x"], "mm")
    add("RMS sizes", "σ_x'", mom["sigma_xp"], "mrad")
    add("RMS sizes", "σ_y", mom["sigma_y"], "mm")
    add("RMS sizes", "σ_y'", mom["sigma_yp"], "mrad")
    add("RMS sizes", "σ_φ", mom["sigma_phi"], "deg")
    add("RMS sizes", "σ_W", mom["sigma_w"], "MeV")
    if ref is not None and ref.beta > 0:
        lam_mm = getattr(ref, "wavelength", 0.0)
        if lam_mm:
            add("RMS sizes", "σ_z",
                mom["sigma_phi"] / 360.0 * ref.beta * lam_mm, "mm")
        denom = ref.beta ** 2 * ref.gamma * ref.species.mass
        if denom > 0:
            add("RMS sizes", "σ_δ (Δp/p)",
                mom["sigma_w"] / denom, "")

    # ---- Twiss -------------------------------------------------------
    hdr("Twiss")
    for plane, bu, gu, note in (("x", "mm/mrad", "mrad/mm", ""),
                                ("y", "mm/mrad", "mrad/mm", ""),
                                ("z", "deg/MeV", "MeV/deg",
                                 " (internal (Δφ,ΔW); α_z = −TraceWin)")):
        tw = compute_twiss_from_particles(q, plane)
        add("Twiss", f"α_{plane}", tw["alpha"], "" + note)
        add("Twiss", f"β_{plane}", tw["beta"], bu)
        add("Twiss", f"γ_{plane}", tw["gamma_t"], gu)

    # ---- emittance ---------------------------------------------------
    hdr("Emittance")
    ex = compute_twiss_from_particles(q, "x")["emittance"]
    ey = compute_twiss_from_particles(q, "y")["emittance"]
    ez = compute_twiss_from_particles(q, "z")["emittance"]
    add("Emittance", "ε_x (geometric)", ex, "mm·mrad")
    add("Emittance", "ε_y (geometric)", ey, "mm·mrad")
    add("Emittance", "ε_z", ez, "deg·MeV")
    if ref is not None:
        ez_mmmrad = _convert_emit_z_to_mmmrad(ez, ref)
        add("Emittance", "ε_z (geometric)", ez_mmmrad, "mm·mrad")
        add("Emittance", "ε_nx (normalized)", ex * ref.bg, "mm·mrad")
        add("Emittance", "ε_ny (normalized)", ey * ref.bg, "mm·mrad")
        add("Emittance", "ε_nz (normalized)",
            ez_mmmrad * ref.bg, "mm·mrad")
    sig = mom["sigma_matrix"]
    det4 = float(np.linalg.det(sig[:4, :4]))
    add("Emittance", "ε_4D (√det Σ₄)",
        float(np.sqrt(max(det4, 0.0))), "mm²·mrad²")
    try:
        # eigenemittances() orders by MAGNITUDE of the mixed-units Σ
        # roots, not by mode identity — the set reduces to the
        # projected {ε_x, ε_y, ε_z} only for an uncoupled beam, and
        # which slot is the longitudinal mode depends on the numbers
        # (e.g. ε_z deg·MeV can be the largest at high energy or for a
        # DC snapshot).  Label with combined units, same hedge as the
        # eigen-emittance popup — do NOT assign per-slot units.
        e1, e2, e3 = eigenemittances(sig)
        for nm, ev in (("ε₁ (eigen)", e1), ("ε₂ (eigen)", e2),
                       ("ε₃ (eigen)", e3)):
            add("Emittance", nm, ev,
                "mm·mrad / deg·MeV (magnitude-ordered)")
    except Exception:                                       # noqa: BLE001
        pass

    # ---- halo --------------------------------------------------------
    hdr("Halo (Wangler)")
    add("Halo (Wangler)", "H_x", compute_halo(q, "x"), "")
    add("Halo (Wangler)", "H_y", compute_halo(q, "y"), "")
    add("Halo (Wangler)", "H_z", compute_halo(q, "z"), "")

    # ---- extents -----------------------------------------------------
    hdr("Max extents")
    for i, (nm, unit) in enumerate((("max|x|", "mm"), ("max|x'|", "mrad"),
                                    ("max|y|", "mm"), ("max|y'|", "mrad"),
                                    ("max|Δφ|", "deg"),
                                    ("max|ΔW|", "MeV"))):
        add("Max extents", nm, float(np.max(np.abs(q[:, i]))), unit)

    return rows
