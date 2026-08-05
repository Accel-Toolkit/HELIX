"""TraceWin-compatible output writers.

Two ASCII formats are implemented, both fully specified in the TraceWin
manual:

1. ``write_partran_out`` — the per-element Partran/Toutatis output schema
   documented on PDF page 43-44 ("Partran and Toutatis output").  One line
   per element exit, ~30 columns.  Same schema for envelope and tracking.

2. ``write_envelope_txt`` — the 26-column tab-separated form produced by
   TraceWin's GUI "Save data" button (the format of every reference file
   in ``Tracewin_code/*.txt``).  Header is preserved verbatim.

Both use signed 3-digit-exponent scientific notation (e.g. ``+1.234e+000``)
to match TraceWin's C++ ``%+.6e`` output.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    """``%+.6e`` with a 3-digit exponent (TraceWin convention).

    Python's default exponent is 2-digit (``1.234e+05``).  TraceWin emits
    3-digit (``+1.234567e+005``).  We pad if needed.
    """
    s = f"{float(x):+.6e}"
    mantissa, exp = s.split("e")
    sign, digits = exp[0], exp[1:]
    if len(digits) < 3:
        digits = digits.zfill(3)
    return f"{mantissa}e{sign}{digits}"


def _row(values: Sequence[float], sep: str = "\t") -> str:
    return sep.join(_fmt(v) for v in values)


def _safe(seq: Sequence, idx: int, default: float = 0.0) -> float:
    """Read ``seq[idx]`` defensively — returns ``default`` if out-of-range
    or the value isn't a finite float."""
    try:
        v = seq[idx]
    except (IndexError, TypeError):
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


# --------------------------------------------------------------------------- #
# 1. Partran/Toutatis output (PDF p. 43-44)                                   #
# --------------------------------------------------------------------------- #
_PARTRAN_HEADER = (
    "# Partran/Toutatis output (TraceWin manual p. 43-44; column layout\n"
    "# audited against a genuine TraceWin partran1.out — 50 columns)\n"
    "# Element# Position(m) gam-1 "
    "x(mm) y(mm) phi(deg) xp(mrad) yp(mrad) W(MeV) "
    "sigx(mm) sigy(mm) sigp(deg) "
    "sxxp(mm.mrad) syyp(mm.mrad) spW(deg.MeV) "
    "exx_n(pi.mm.mrad) eyy_n(pi.mm.mrad) eW_n(pi.deg.MeV) "
    "Hxx Hyy Hzpp "
    "Nparticles "
    "sigmax(deg/mm) sigmay(deg/mm) sigmaz(deg/mm) "
    "e99_xx e99_yy e99_zpp "
    "phi_s(deg) W_s(MeV) Ibeam(mA) Ap(mm) "
    "e4D_n(pi.mm.mrad)^2 err_n(pi.mm.mrad) sigmar(deg/mm) Plost(W) "
    "Xmax(mm) Ymax(mm) "
    "ezpp_n(pi.mm.mrad) sigz(mm) zpp(mm.mrad) "
    "Dh(mm) Dv(mm) Dhp(mrad) Dvp(mrad) E6D "
    "sigxy sigxpyp sigxyp sigxpy\n"
)


def read_partran_out(path: str | Path) -> dict:
    """Parse a TraceWin/partran .out file into per-step arrays.

    Inverse of :func:`write_partran_out` — returns the column subset
    used by the GUI's overlay-comparison popup.  Only numeric rows are
    kept; comment lines (``#`` prefix) and blank lines are skipped.

    Returns
    -------
    dict with arrays:
        ``s_m, sigma_x_mm, sigma_y_mm, sigma_phi_deg,
          emit_x_mm_mrad, emit_y_mm_mrad, emit_z_deg_MeV,
          ref_W_MeV, n_alive`` plus ``aperture_mm`` if present.
    """
    rows: list[list[float]] = []
    mc2_MeV = 0.0
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                vals = [float(tok) for tok in stripped.split()]
            except ValueError:
                continue
            # Genuine TW files (and this module's writer) carry a short
            # numeric parameter line after the column header:
            # "mc2 freq charge current npart".  It used to be swallowed
            # as a bogus data row (s = 162.5 m, NaN-padded, garbage
            # n_alive).  Capture mc2 from it instead — it lets us
            # reconstruct the absolute energy from gam-1.
            if len(vals) < 20:
                if not rows and 3 <= len(vals) <= 8 and vals[0] > 0:
                    mc2_MeV = vals[0]
                continue
            rows.append(vals)
    if not rows:
        return {
            "s_m": np.array([]),
            "sigma_x_mm": np.array([]), "sigma_y_mm": np.array([]),
            "sigma_phi_deg": np.array([]),
            "emit_x_mm_mrad": np.array([]),
            "emit_y_mm_mrad": np.array([]),
            "emit_z_deg_MeV": np.array([]),
            "ref_W_MeV": np.array([]),
            "n_alive": np.array([], dtype=int),
            "aperture_mm": np.array([]),
        }
    # Pad short rows so np.asarray succeeds even with the occasional
    # truncated trailing row.
    n_cols_max = max(len(r) for r in rows)
    arr = np.full((len(rows), n_cols_max), np.nan, dtype=float)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    # Column semantics per the GENUINE TraceWin schema (verified against
    # tests/analysis/fixtures/partran1_subset.out): cols 12-14 are the
    # covariances <xx'>, <yy'>, <phiW>; cols 15-17 are the NORMALIZED
    # emittances.  (This reader previously mislabeled cols 12-14 as
    # geometric emittances — real TW covariances were read as ε.)
    # Geometric emittance is derived: ε_geo = ε_n / (βγ), with βγ from
    # gam-1 (col 2).
    n_rows = arr.shape[0]
    gm1 = arr[:, 2] if arr.shape[1] > 2 else np.zeros(n_rows)
    gamma = gm1 + 1.0
    with np.errstate(invalid="ignore"):
        bg = np.sqrt(np.clip(gamma * gamma - 1.0, 0.0, None))
    bg_safe = np.where(bg > 0, bg, 1.0)

    def _col(j):
        return arr[:, j] if arr.shape[1] > j else np.array([])

    def _geo(j):
        c = _col(j)
        return c / bg_safe[: c.size] if c.size else c

    # Absolute beam energy: col 8 (W0) is the centroid ΔW OFFSET in
    # genuine TW files — the absolute energy lives in gam-1 (col 2),
    # convertible when the parameter line supplied mc².  Old HELIX-written
    # files (no parameter line) put the absolute energy in col 8 — keep
    # that as the fallback so they still read correctly.
    if mc2_MeV > 0:
        ref_W = gm1 * mc2_MeV
    else:
        ref_W = _col(8)

    n_alive_col = _col(21)
    n_alive = (np.nan_to_num(n_alive_col, nan=0.0).astype(int)
               if n_alive_col.size else np.array([], dtype=int))

    out = {
        "s_m":            _col(1),
        "ref_W_MeV":      ref_W,
        "sigma_x_mm":     _col(9),
        "sigma_y_mm":     _col(10),
        "sigma_phi_deg":  _col(11),
        "emit_x_mm_mrad": _geo(15),
        "emit_y_mm_mrad": _geo(16),
        # TW's ep (col 17) is the RAW (φ,W) emittance — no βγ division
        # (fixture-verified; only the transverse pair is normalized).
        "emit_z_deg_MeV": _col(17),
        "n_alive":        n_alive,
        # Aperture is col 31 (the old 32 read TW's e4D as aperture).
        "aperture_mm":    _col(31),
    }
    return out


def write_partran_out(
    results: Any,
    lattice: Any,
    beam_cfg: Any,
    path: str | Path,
    *,
    aperture_mm: float = 0.0,
) -> Path:
    """Emit one line per element following the manual's "Partran and
    Toutatis output" schema (PDF p. 43-44).

    The first line is the input beam parameters (s=0); subsequent lines
    correspond to each element exit.

    Parameters
    ----------
    results
        ``EnvelopeResults`` or any object exposing the same per-step
        sequences (``s``, ``sigma_x``, ``sigma_y``, ``sigma_phi``,
        ``emit_x``, ``emit_y``, ``emit_z``, ``alpha_x/y``, ``beta_x/y``,
        ``ref_*``, ``element_names``, optionally ``transmission``,
        ``centroid``, ``halo_x/y``, ``x_max``, ``y_max``, ``emit_4d``).
    lattice
        Used for element ordering and aperture lookup.  May be ``None``
        for envelope-only runs that don't need element-level info.
    beam_cfg
        ``BeamConfig`` — used for the beam current column and species
        mass.  May be ``None`` if ``results`` already exposes the same.
    path
        Destination file.
    aperture_mm
        Default aperture written if no per-element aperture is available.
    """
    path = Path(path)
    s_arr     = list(getattr(results, "s", []))
    n         = len(s_arr)
    if n == 0:
        path.write_text(_PARTRAN_HEADER, encoding="latin-1", errors="replace")
        return path

    sig_x   = list(getattr(results, "sigma_x",   []))
    sig_y   = list(getattr(results, "sigma_y",   []))
    sig_phi = list(getattr(results, "sigma_phi", []))
    sig_w   = list(getattr(results, "sigma_w",   []))
    emit_x  = list(getattr(results, "emit_x",    []))   # mm.mrad geometric
    emit_y  = list(getattr(results, "emit_y",    []))
    emit_z  = list(getattr(results, "emit_z",    []))   # deg.MeV native
    emit_zmm = list(getattr(results, "emit_z_mmmrad", []))
    ref_w   = list(getattr(results, "ref_w_kin", []))
    ref_phi = list(getattr(results, "ref_phi_s", []))
    ref_bg  = list(getattr(results, "ref_bg",    []))
    if not ref_bg:
        ref_beta  = list(getattr(results, "ref_beta",  []))
        ref_gamma = list(getattr(results, "ref_gamma", []))
        ref_bg = [b * g for b, g in zip(ref_beta, ref_gamma)]
    element_names = list(getattr(results, "element_names", []))

    centroid = list(getattr(results, "centroid", []))
    halo_x   = list(getattr(results, "halo_x", []))
    halo_y   = list(getattr(results, "halo_y", []))
    x_max    = list(getattr(results, "x_max", []))
    y_max    = list(getattr(results, "y_max", []))
    emit_4d  = list(getattr(results, "emit_4d", []))
    transmission = list(getattr(results, "transmission", []))
    sigma_mats = list(getattr(results, "sigma_matrix", []))
    ref_freq = list(getattr(results, "ref_frequency", []))
    ref_beta_l  = list(getattr(results, "ref_beta",  []))
    ref_gamma_l = list(getattr(results, "ref_gamma", []))
    mass_mev = float(getattr(results, "mass_mev", 0.0) or 0.0)
    if mass_mev <= 0.0 and beam_cfg is not None:
        mass_mev = float(getattr(getattr(beam_cfg, "species", None),
                                 "mass", 0.0) or 0.0)

    # TraceWin's file is one row per ELEMENT.  With substep recording
    # the results carry interior rows too — select the INPUT row + each
    # element's exit row via element_exit_idx (identity for legacy
    # one-record-per-element results).
    exit_idx = list(getattr(results, "element_exit_idx", []) or [])
    if exit_idx and max(exit_idx) < n:
        row_indices = [0] + [int(r) for r in exit_idx]
    else:
        row_indices = list(range(n))

    # Phase advance per metre (TW columns kx/ky/kz, header unit deg/mm):
    # the ELEMENT-AVERAGE Δμ/Δs (verified against a genuine partran1.out
    # — endpoint densities average to TW's value), from the cumulative
    # beam μ(s).  Thin/zero-length rows carry TW's 1e-5 sentinel.
    _mu_curves = None
    try:
        from linac_gen.analysis.phase_advance import (
            beam_phase_advance_along_s,
        )
        _mu_curves = beam_phase_advance_along_s(results)
    except Exception:                                        # noqa: BLE001
        _mu_curves = None

    def _k_pm(prev_r: int, r: int, plane: str) -> float:
        if r == prev_r:
            return 0.0          # the s=0 INPUT row — genuine TW writes 0.0
        if _mu_curves is None:
            return 1e-5
        mu = _mu_curves.get(f"mu_{plane}_deg")
        if mu is None:
            return 1e-5
        ds = float(s_arr[r]) - float(s_arr[prev_r])
        if ds <= 0 or r >= len(mu) or prev_r >= len(mu):
            return 1e-5
        dmu = float(mu[r]) - float(mu[prev_r])
        if not np.isfinite(dmu):
            return 1e-5
        return max(dmu / ds, 1e-5)

    # Element-number lookup: tracker records one pre-lattice INPUT record
    # at index 0 ("INPUT"), then one record per element exit.  Match by
    # name when possible, otherwise by sequential index.
    elements = list(getattr(lattice, "elements", []) or [])
    name_to_index = {getattr(e, "name", None): i + 1 for i, e in enumerate(elements)}

    current_mA = float(getattr(beam_cfg, "current", 0.0)
                       if beam_cfg is not None else
                       getattr(results, "current_mA", 0.0))
    n_particles_total = int(getattr(beam_cfg, "n_particles", 0) or 0)

    with path.open("w", encoding="latin-1", errors="replace") as fh:
        fh.write(_PARTRAN_HEADER)
        # Genuine TW files carry a numeric parameter line after the
        # column header: mc² [MeV], RF frequency [MHz], charge sign,
        # beam current [mA], macro-particle count.  Emitting it keeps
        # HELIX output byte-compatible with TW tooling and gives the
        # reader mc² so it can reconstruct the absolute energy from
        # gam-1 (col 8 is the centroid ΔW OFFSET, not absolute W).
        f0_MHz = _safe(ref_freq, 0)
        if f0_MHz <= 0.0 and beam_cfg is not None:
            f0_MHz = float(getattr(beam_cfg, "frequency", 0.0) or 0.0)
        charge_sign = -1.0
        sp = getattr(beam_cfg, "species", None) if beam_cfg is not None else None
        if sp is not None:
            q = float(getattr(sp, "charge", -1.0) or -1.0)
            charge_sign = math.copysign(1.0, q)
        fh.write(f"{mass_mev:.6f} {f0_MHz:.7E} {charge_sign:.0f}. "
                 f"{current_mA:.5E} {n_particles_total}\n")
        prev_r = row_indices[0] if row_indices else 0
        for row_pos, i in enumerate(row_indices):
            name = element_names[i] if i < len(element_names) else ""
            # Element# convention: 0 for the pre-lattice input record,
            # 1-based otherwise.  With the exit-row selection above the
            # ordinal position IS the element number.
            if name == "INPUT":
                elem_no = 0
            elif exit_idx and max(exit_idx) < n:
                elem_no = row_pos
            else:
                elem_no = name_to_index.get(name, i)

            bg = ref_bg[i] if i < len(ref_bg) else 1.0
            gam_minus_1 = (np.sqrt(1.0 + bg * bg) - 1.0) if bg else 0.0

            # Centroid — 6-vector when present (x, x', y, y', z/φ, dW).
            if i < len(centroid):
                c = np.asarray(centroid[i], dtype=float).flatten()
                if c.size >= 6:
                    cx, cxp, cy, cyp, cphi, cdW = c[:6]
                else:
                    cx = cxp = cy = cyp = cphi = cdW = 0.0
            else:
                cx = cxp = cy = cyp = cphi = cdW = 0.0
            # Centroid units: the recorder stores particle means in the
            # tracking coordinates, which ARE mm / mrad / deg / MeV —
            # no conversion.  (A historical ×1e3 here assumed metres and
            # inflated every MP centroid — and Xmax/Ymax below — 1000×.)
            x_mm   = float(cx)
            y_mm   = float(cy)
            phi_dg = float(cphi)
            xp_mr  = float(cxp)
            yp_mr  = float(cyp)
            # Col 8 (W0) is the centroid energy OFFSET ΔW [MeV] in
            # genuine TW files (fixture values ~1e-6 MeV with gam-1
            # carrying the absolute energy in col 2) — the old absolute
            # ref_w + cdW here disagreed with TW by the full beam energy.
            W_MeV  = float(cdW)

            # σ_x / σ_y stored in mm internally; σ_φ in deg.
            sx_mm  = _safe(sig_x,   i)
            sy_mm  = _safe(sig_y,   i)
            sphi_d = _safe(sig_phi, i)
            sW_MeV = _safe(sig_w,   i)
            # σ_z (mm) reconstructed from σ_φ (deg) and the synchrotron
            # constant: λ = c / f.  c expressed in mm/s so wavelength
            # comes out in mm directly.  When the run is DC (continuous),
            # σ_z is not physical; emit zero.  The frequency is the
            # PER-RECORD one (FREQ-jump lattices change it mid-line —
            # a fixed beam_cfg.frequency mis-scales σ_z downstream).
            beta = bg / np.sqrt(1.0 + bg * bg) if bg else 0.0
            f_MHz = _safe(ref_freq, i)
            if f_MHz <= 0.0 and beam_cfg is not None:
                f_MHz = float(getattr(beam_cfg, "frequency", 0.0) or 0.0)
            freq_Hz = f_MHz * 1e6
            if freq_Hz > 0 and beta > 0 and not getattr(results, "continuous", False):
                wavelength_mm = 299792458.0e3 / freq_Hz       # mm
                sz_mm = sphi_d * beta * wavelength_mm / 360.0
            else:
                sz_mm = 0.0

            # Covariances <xx'>, <yy'>, <φW> (TW cols 12-14 — verified
            # against a genuine partran1.out; the old writer put
            # geometric emittances here) + normalized emittances.
            S = None
            if i < len(sigma_mats):
                S_arr = np.asarray(sigma_mats[i], dtype=float)
                if S_arr.shape == (6, 6) and np.all(np.isfinite(S_arr)):
                    S = S_arr
            sxxp = float(S[0, 1]) if S is not None else 0.0
            syyp = float(S[2, 3]) if S is not None else 0.0
            spW  = float(S[4, 5]) if S is not None else 0.0

            ex_geo = _safe(emit_x, i)
            ey_geo = _safe(emit_y, i)
            ezphi_geo = _safe(emit_z, i)             # deg.MeV
            ex_n  = ex_geo * bg
            ey_n  = ey_geo * bg
            # TW's ep (col 17) is the RAW (φ, W) rms emittance — the
            # fixture satisfies ep = (360·mc²[GeV]/λ[mm])·ezdp to seven
            # digits, i.e. NO ×βγ (only the transverse cols 15-16 are
            # normalized).  HELIX's native emit_z IS that quantity.
            ezphi_tw = ezphi_geo                      # π.deg.MeV, raw
            ezpp_geo  = _safe(emit_zmm, i)            # mm.mrad equivalent
            ezpp_n    = ezpp_geo * bg

            # Halo
            hx = _safe(halo_x, i)
            hy = _safe(halo_y, i)
            hzpp = 0.0  # not currently tracked — left zero per manual

            # Number of particles (alive)
            if i < len(transmission):
                n_alive = int(round(n_particles_total * transmission[i] / 100.0))
            else:
                n_alive = n_particles_total

            # Phase advances (TW header unit deg/mm): the ELEMENT-
            # AVERAGE Δμ/Δs across this row's element, from the
            # cumulative beam μ(s).  1e-5 sentinel for the INPUT row /
            # thin elements (TW convention).
            sigma_x_pm = _k_pm(prev_r, i, "x")
            sigma_y_pm = _k_pm(prev_r, i, "y")
            sigma_z_pm = _k_pm(prev_r, i, "z")
            # TW's kr is NOT the x/y average (fixture kr / mean(kx,ky)
            # varies 7.06-7.29) — leave honestly absent rather than
            # writing a plausible-looking wrong number.
            sigma_r_pm = 0.0

            # ε at 99 % — not tracked; leave zero.
            e99_x = e99_y = e99_zpp = 0.0

            # Synchronous-phase / energy difference: φ_s vs reference φ
            # and W_s vs reference W.  Recorder stores ref_phi_s; for an
            # envelope run the synchronous and reference particles are
            # identical, so the difference is zero.  For tracking, the
            # centroid c[4]/c[5] above already captures it.
            phi_s_diff = float(cphi)
            W_s_diff   = float(cdW)

            # Aperture — pull from the matching element, fallback default.
            if elem_no >= 1 and elem_no <= len(elements):
                ap = float(getattr(elements[elem_no - 1], "aperture", aperture_mm)
                           or aperture_mm)
            else:
                ap = aperture_mm

            # 4-D normalised emittance squared (π.mm.mrad)²
            e4d_geo = _safe(emit_4d, i)
            # TW's e4D = eps_nx*eps_ny exactly (fixture-verified);
            # the header's (pi.mm.mrad)^2 is the UNIT of the product,
            # not an instruction to square.  Normalize with bg^2.
            e4d_n_sq = e4d_geo * bg * bg
            err_n = 0.5 * (ex_n + ey_n)

            # Lost power: transmission gives % alive, beam power = I·V
            P_lost_W = 0.0
            if i < len(transmission) and current_mA > 0:
                lost_frac = max(0.0, 1.0 - transmission[i] / 100.0)
                w_kin = ref_w[i] if i < len(ref_w) else 0.0
                P_lost_W = current_mA * 1e-3 * w_kin * 1e6 * lost_frac

            # Recorder stores particle extrema in mm already (see the
            # centroid note above — the old ×1e3 inflated these 1000×).
            xmax_mm = _safe(x_max, i)
            ymax_mm = _safe(y_max, i)

            # zp/p in mm.mrad — not directly stored; leave 0.
            zpp = 0.0

            # Beam dispersion from the recorded correlations:
            # D[mm] = <x·δ>/<δ²> with δ = ΔW/(β²γ·m) — expressed via the
            # native (x, ΔW) covariances as Σ[0,5]/Σ[5,5]·β²γm.
            Dh = Dv = Dhp = Dvp = 0.0
            if S is not None and S[5, 5] > 0 and mass_mev > 0:
                beta_i = _safe(ref_beta_l, i, default=beta)
                gamma_i = _safe(ref_gamma_l, i,
                                default=(1.0 + gam_minus_1))
                conv = beta_i * beta_i * gamma_i * mass_mev
                Dh  = float(S[0, 5] / S[5, 5] * conv)
                Dv  = float(S[2, 5] / S[5, 5] * conv)
                Dhp = float(S[1, 5] / S[5, 5] * conv)
                Dvp = float(S[3, 5] / S[5, 5] * conv)
            # TW's E6D = eps_nx*eps_ny*eps_n,zdp exactly (fixture:
            # 0.2052723*0.1972638*0.3556927 = 0.01440299) — NOT
            # sqrt(det Sigma) in native mixed units.
            e6d = ex_n * ey_n * ezpp_n
            # σ-cross terms <xy>, <x'y'>, <xy'>, <x'y>.
            if S is not None:
                sxy, sxpyp = float(S[0, 2]), float(S[1, 3])
                sxypp, sxpyy = float(S[0, 3]), float(S[1, 2])
            else:
                sxy = sxpyp = sxypp = sxpyy = 0.0

            # ref.s is recorded in mm internally; manual specifies metres.
            position_m = _safe(s_arr, i) * 1e-3
            row = [
                elem_no,
                position_m,
                gam_minus_1,
                x_mm, y_mm, phi_dg, xp_mr, yp_mr, W_MeV,        # 6 centroid
                sx_mm, sy_mm, sphi_d,                           # 3 rms
                sxxp, syyp, spW,                                # 3 covariances
                ex_n, ey_n, ezphi_tw,                           # eps_nx eps_ny ep
                hx, hy, hzpp,                                   # 3 halo
                n_alive,
                sigma_x_pm, sigma_y_pm, sigma_z_pm,             # phase advances
                e99_x, e99_y, e99_zpp,                          # 99 % emit
                phi_s_diff, W_s_diff, current_mA, ap,
                e4d_n_sq, err_n, sigma_r_pm, P_lost_W,
                xmax_mm, ymax_mm,
                ezpp_n, sz_mm, zpp,
                Dh, Dv, Dhp, Dvp, e6d,
                sxy, sxpyp, sxypp, sxpyy,
            ]
            fh.write(_row(row) + "\n")
            prev_r = i
    return path


# --------------------------------------------------------------------------- #
# 2. 26-column "Save data" tab format (Tracewin_code/*.txt fixtures)          #
# --------------------------------------------------------------------------- #
_ENV_HEADER_TAB = (
    "position\tgam-1"
    "\tcentroid position(x,x',y,y',z,dp/p,z',phase,time,energy)"
    "\trms_size(x,x',y,y',z,dp/p,z',phase,time,energy)"
    "\t(dispX,dispY,betX,betY)"
    "\tunit(m,rad,deg,s,MeV)\n"
)


def write_envelope_txt(
    results: Any,
    beam_cfg: Any,
    path: str | Path,
) -> Path:
    """Emit the 26-column tab format used in ``Tracewin_code/*.txt``.

    Layout: ``position  gam-1  centroid(10 cols) rms_size(10 cols)
    (dispX,dispY,betX,betY)``.  Centroid and dispersion are zeros for an
    on-axis envelope run.
    """
    path = Path(path)
    s_arr = list(getattr(results, "s", []))
    n = len(s_arr)
    sig_x   = list(getattr(results, "sigma_x",   []))
    sig_y   = list(getattr(results, "sigma_y",   []))
    sig_phi = list(getattr(results, "sigma_phi", []))
    sig_w   = list(getattr(results, "sigma_w",   []))
    ref_bg  = list(getattr(results, "ref_bg",    []))
    ref_w   = list(getattr(results, "ref_w_kin", []))
    if not ref_bg:
        ref_beta  = list(getattr(results, "ref_beta",  []))
        ref_gamma = list(getattr(results, "ref_gamma", []))
        ref_bg = [b * g for b, g in zip(ref_beta, ref_gamma)]
    beta_x  = list(getattr(results, "beta_x", []))
    beta_y  = list(getattr(results, "beta_y", []))

    freq_Hz = (float(getattr(beam_cfg, "frequency", 0.0)) * 1e6
               if beam_cfg is not None else 0.0)

    with path.open("w", encoding="latin-1", errors="replace") as fh:
        fh.write(_ENV_HEADER_TAB)
        fh.write("\n")  # blank line as in the reference fixtures
        for i in range(n):
            bg = ref_bg[i] if i < len(ref_bg) else 0.0
            gam_minus_1 = (np.sqrt(1.0 + bg * bg) - 1.0) if bg else 0.0
            beta = bg / np.sqrt(1.0 + bg * bg) if bg else 0.0
            # σ in our internals: σ_x/σ_y in mm, σ_φ in deg, σ_W in MeV.
            # Reference layout uses metres / radians / deg / s / MeV.
            sx_m  = _safe(sig_x,   i) * 1e-3
            sy_m  = _safe(sig_y,   i) * 1e-3
            sphi_d = _safe(sig_phi, i)
            sW_MeV = _safe(sig_w,   i)
            # σ_x' from σ_x and Twiss β_x: σ_x' = σ_x / β_x  (in rad);
            # geometry only — falls back to zero if β unavailable.
            bx = _safe(beta_x, i)
            by = _safe(beta_y, i)
            sxp = (sx_m / bx) if bx > 0 else 0.0
            syp = (sy_m / by) if by > 0 else 0.0
            # σ_z (m) from σ_φ
            wavelength_m = (299792458.0 / freq_Hz) if freq_Hz > 0 else 0.0
            sz_m = (sphi_d * beta * wavelength_m / 360.0) if wavelength_m else 0.0
            # σ_dpp from σ_W: E·dE = c²·p·dp ⇒ dp/p = dW·γ/(βγ)²/mc²
            #                               = dW/(β²·γ·mc²)
            mc2 = float(getattr(results, "mass_mev", 0.0) or 0.0)
            gamma = gam_minus_1 + 1.0
            sdpp = (sW_MeV / (beta * beta * gamma * mc2)
                    if (beta > 0 and mc2 > 0) else 0.0)

            # 10-vector ordering per header:
            # (x, x', y, y', z, dp/p, z', phase, time, energy)
            #          z'   = sz/βλ in rad (placeholder zero)
            #          time = z / (βc) in s (placeholder zero)
            # Envelope results now carry a first moment — emit it with
            # the same unit conversions as the rms block (mm→m,
            # mrad→rad, Δφ→z and dp/p via the kinematics above).
            cvec = getattr(results, "centroid", None)
            if cvec is not None and i < len(cvec) and cvec[i] is not None:
                cx, cxp, cy, cyp, cphi, cdw = (float(v) for v in cvec[i])
                # Δφ is in degrees of the LOCAL machine clock — use the
                # per-record frequency (ref_frequency) so z stays
                # physically invariant across FREQ jumps (the fixed
                # entrance-frequency wavelength above mis-scales it; the
                # legacy rms σ_z column shares that old convention).
                f_row = _safe(getattr(results, "ref_frequency", []), i)
                wl_row = (299792458.0 / (f_row * 1e6) if f_row > 0
                          else wavelength_m)
                cz_m = (-cphi * beta * wl_row / 360.0
                        if wl_row else 0.0)
                cdpp = (cdw / (beta * beta * gamma * mc2)
                        if (beta > 0 and mc2 > 0) else 0.0)
                cent10 = [cx * 1e-3, cxp * 1e-3, cy * 1e-3, cyp * 1e-3,
                          cz_m, cdpp, 0.0, cphi, 0.0, 0.0]
            else:
                cent10 = [0.0] * 10                 # on-axis envelope
            rms10  = [sx_m, sxp, sy_m, syp, sz_m, sdpp, 0.0, sphi_d, 0.0,
                      _safe(ref_w, i)]
            disp4 = [0.0, 0.0, bx, by]              # only β reported

            # ref.s is recorded in mm internally; manual format is metres.
            position_m = _safe(s_arr, i) * 1e-3
            row = [position_m, gam_minus_1] + cent10 + rms10 + disp4
            fh.write(_row(row) + "\n")
    return path
