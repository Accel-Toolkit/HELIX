"""Statistical moment computation from particle arrays."""
import numpy as np

PHASE_COL = 4          # Δφ [deg] — the only 2π-periodic coordinate


def wrap_phase_column(particles: np.ndarray) -> tuple[np.ndarray, int]:
    """Fold Δφ into one RF period about the bunch centroid — VIEW ONLY.

    Returns ``(particles_or_folded_copy, n_folded)``; the input array is
    returned UNCHANGED (same object, no copy) when no particle lies more
    than 180° from the median, and the fold is anchored on the median
    because the mean of a multi-bucket sample sits between buckets.

    WHAT IT IS FOR.  An RFQ turns a DC beam into a *bunch train*: one
    bunch per RF period.  HELIX seeds one RF period, and during bunching
    space charge pushes ~20 % of the particles across a bucket boundary,
    so a raw φ–ΔW plot shows a row of stripes 360° apart instead of one
    bunch.  Folding gives the single-bucket view TraceWin/Toutatis draw.
    The GUI phase-space popup calls this behind its "fold φ" checkbox.

    WHY IT IS **NOT** USED FOR REPORTED STATISTICS (σ_φ, ε_z, z-Twiss).
    Two adversarial reviews measured that on the real PXIE deck
    (2026-07-30) and it does not survive:

    * BIASED ESTIMATOR — the ±1 buckets are not periodic images of the
      core.  They differ at 8-52 σ, with a space-charge-generated
      −35 keV/bucket energy chirp (the satellite population goes
      0.6 % → 23.8 % from 0 → 5 mA, and the bias vanishes at 0 mA), so
      folding superposes statistically distinct populations: ε_z +123 %,
      σ_W +69 %, and α_z comes out with the WRONG SIGN (−0.150 folded
      vs +0.130 for the main bunch alone).  σ_W is not periodic at all,
      so no phase treatment can fix its half of the ε_z error.
    * NO ROBUST TRAIN TEST EXISTS AT THE STATISTICS LEVEL — a
      compactness threshold (the obvious gate) flips on shot noise:
      bootstrap gives 15 %/71 % fold probability on adjacent steps, and
      a 0.9 % change in the folded σ moved the reported σ_φ by 271 %,
      putting cliffs into σ_φ(s)/ε_z(s) and discontinuities into the
      matching objective.  It also silently under-reports genuine
      debunching (a mismatched MEBT+HWR beam: 162° raw → 15° folded)
      and turns the exactly-constant ε_z of a drift into a staircase.

    THE REAL FIX for the reported numbers is to give DC-injected beams
    periodic phase coordinates during TRACKING (the Toutatis
    convention), so the train never forms and the moments need no
    treatment.  That now exists: ``BeamConfig.periodic_phase``
    (opt-in, default off) folds Δφ into one bunch spacing in
    ``Tracker._fold_phase``, and with it on the moments below are
    single-bunch values with no statistics code involved.  This helper
    stays as the DISPLAY fold for runs made WITHOUT the flag — where
    σ_φ / ε_z remain train-wide values, honestly wrong rather than
    subtly wrong, and the single-bunch view is available in the plot.

    Note the two folds differ deliberately: this one is anchored on the
    MEDIAN with a hard 360° period (it has only the array to work
    from), while the tracker anchors on Δφ = 0 and uses the true bunch
    spacing 360·f_local/f_bunch, which is 720° after a frequency jump.
    KNOWN LIMITATION of that asymmetry: downstream of a frequency jump
    a beam may legitimately span more than ±180 LOCAL degrees inside a
    single bunch, and this helper would fold its wings.  It bites only
    a beam that is both post-jump and nearly debunched; passing the
    period through is the fix if that ever matters.
    """
    if len(particles) == 0:
        return particles, 0
    dphi = particles[:, PHASE_COL]
    med = np.median(dphi)
    dev = dphi - med
    n_wrapped = int(np.count_nonzero(np.abs(dev) > 180.0))
    if n_wrapped == 0:
        return particles, 0
    out = particles.copy()
    out[:, PHASE_COL] = med + ((dev + 180.0) % 360.0 - 180.0)
    return out, n_wrapped


def compute_moments(particles: np.ndarray) -> dict:
    if len(particles) == 0:
        return {
            "mean": np.zeros(6), "sigma_matrix": np.zeros((6, 6)),
            "sigma_x": 0.0, "sigma_xp": 0.0, "sigma_y": 0.0,
            "sigma_yp": 0.0, "sigma_phi": 0.0, "sigma_w": 0.0,
        }
    mean = np.mean(particles, axis=0)
    centered = particles - mean
    sigma = (centered.T @ centered) / len(particles)
    return {
        "mean": mean, "sigma_matrix": sigma,
        "sigma_x": np.sqrt(sigma[0, 0]), "sigma_xp": np.sqrt(sigma[1, 1]),
        "sigma_y": np.sqrt(sigma[2, 2]), "sigma_yp": np.sqrt(sigma[3, 3]),
        "sigma_phi": np.sqrt(sigma[4, 4]), "sigma_w": np.sqrt(sigma[5, 5]),
    }


def compute_emittance(particles: np.ndarray, plane: str = "x") -> float:
    if len(particles) == 0:
        return 0.0
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    up = particles[:, j] - np.mean(particles[:, j])
    uu = np.mean(u * u)
    upup = np.mean(up * up)
    uup = np.mean(u * up)
    emit_sq = uu * upup - uup * uup
    # emit^2 is mathematically >= 0. Round-off can make it slightly negative
    # for very small samples or near-singular distributions; clamp to 0 rather
    # than taking abs() so a genuinely large negative would surface as zero
    # (loud via downstream "emit == 0" guards) instead of being silently flipped.
    return float(np.sqrt(max(emit_sq, 0.0)))


def compute_twiss_from_particles(particles: np.ndarray, plane: str = "x") -> dict:
    if len(particles) == 0:
        return {"alpha": 0.0, "beta": 0.0, "gamma_t": 0.0, "emittance": 0.0}
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    up = particles[:, j] - np.mean(particles[:, j])
    emit = compute_emittance(particles, plane)
    if emit < 1e-30:
        return {"alpha": 0.0, "beta": 0.0, "gamma_t": 0.0, "emittance": 0.0}
    beta = np.mean(u * u) / emit
    alpha = -np.mean(u * up) / emit
    gamma_t = np.mean(up * up) / emit
    return {"alpha": float(alpha), "beta": float(beta), "gamma_t": float(gamma_t), "emittance": float(emit)}


def compute_halo(particles: np.ndarray, plane: str = "x") -> float:
    if len(particles) == 0:
        return 0.0
    col_map = {"x": 0, "y": 2, "z": 4}
    i = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    u2 = np.mean(u**2)
    u4 = np.mean(u**4)
    if u2 < 1e-30:
        return 0.0
    return float(u4 / u2**2 - 1.0)
