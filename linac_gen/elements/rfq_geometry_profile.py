"""Vane-geometry gradient profile for the RFQ card tracker.

The two-term card model applies a constant electric-quadrupole strength
V/r0² per cell.  The real vane geometry deviates from that in ways the
cards cannot express: the radial-matcher entrance ramp, the intra-cell
gradient breathing of modulated cells, the exit flare, and a small x/y
asymmetry.  On the PXIE LEBT+RFQ benchmark those differences are worth
~20 % in transverse phase advance and 10+ points of transmission at
5 mA (2026-08-02 identical-beam study vs TraceWin/Toutatis; TraceWin's
own card-based envelope mode shows the same deficit, i.e. this is a
limitation of the card model class, not of any one implementation).

This module measures the real profile once from a TraceWin ``.vane``
file and returns it as normalised per-plane linear gradients

    gx(z) =  (dE_x/dx)|axis / G_plateau      (≈ +1 on the plateau)
    gy(z) =  (dE_y/dy)|axis / G_plateau      (≈ −1 on the plateau)

obtained from windowed 3-D Laplace solves of the vane potential
(:class:`~linac_gen.elements.vane_rfq_laplace3d.Laplace3DCache`) with
per-slice linear least-squares fits over the trusted core region.
The first and last windows are padded with a field-free extension of
the end rows so the open-boundary artefact of the solver lands in the
fake pad instead of contaminating the radial matcher / exit cells
(validated against the local-2D tip model (r0/a)², 2026-08-02).

The profile is consumed by :class:`~linac_gen.elements.rfq_cell.RfqCell`
via :func:`linac_gen.io.rfq_geometry_helper.apply_rfq_geometry`; the
solve costs about a minute for a 200-cell RFQ, so results are cached on
disk next to the vane file.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from linac_gen.io.tracewin_vane import VaneGeometry, parse_vane_file

_log = logging.getLogger(__name__)

_CACHE_VERSION = 1


@dataclass
class RfqGeometryProfile:
    """Normalised per-plane linear gradients of the real vane field.

    ``z_mm`` is measured from the start of the first RFQ cell (the vane
    file's own longitudinal origin).  ``gx``/``gy`` are the on-axis
    linear gradients dE_x/dx and dE_y/dy divided by the plateau quad
    gradient ``g_plateau`` (V/m² per m); on an unmodulated plateau cell
    gx ≈ +1 and gy ≈ −1.
    """

    z_mm: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    g_plateau: float
    vane_path: str
    params: dict
    #: unpadded longitudinal span of the vane table itself, mm
    vane_z_mm: tuple | None = None
    #: median vane-tip aperture of the table, mm (wall sanity checks)
    median_aperture_mm: float | None = None


def _pad_field_free(arrs: dict, fields: Sequence[str], dz: float,
                    pad_mm: float, lo: bool) -> dict:
    """Extend every vane column by ``pad_mm`` holding the end row.

    The end rows of a real vane table are wide-open flares, so the held
    extension is essentially field-free on axis; solver boundary
    artefacts decay within a few aperture radii and stay in the pad.
    """
    out = dict(arrs)
    z = arrs["z"]
    if lo:
        zp = np.arange(z[0] - pad_mm * 1e-3, z[0] - 1e-12, dz)
        take = 0
    else:
        zp = np.arange(z[-1] + dz, z[-1] + pad_mm * 1e-3, dz)
        take = -1
    for n in fields:
        col = arrs[n]
        ext = zp if n == "z" else np.full(len(zp), col[take])
        out[n] = (np.concatenate([ext, col]) if lo
                  else np.concatenate([col, ext]))
    return out


def _solve_windows(vane: VaneGeometry, *, nx: int, z_subsample: int,
                   window_mm: float, pad_mm: float, solver: str,
                   verbose: bool):
    from linac_gen.elements.vane_rfq_laplace3d import Laplace3DCache

    fields = [f.name for f in dataclasses.fields(vane)]
    z_all = vane.z * 1e3
    dz_native = float(np.median(np.diff(np.unique(z_all)))) * 1e-3

    plan = []
    start = z_all[0]
    while start < z_all[-1] - 1.0:
        stop = min(start + window_mm, z_all[-1])
        plan.append((start, stop))
        start = stop

    rows = []
    for i, (z0, z1) in enumerate(plan):
        first, last = i == 0, i == len(plan) - 1
        lo = z0 - (0.0 if first else pad_mm / 1.5)
        hi = z1 + (0.0 if last else pad_mm / 1.5)
        sel = np.where((z_all >= lo) & (z_all <= hi))[0]
        arrs = {n: getattr(vane, n)[sel] for n in fields}
        if first:
            arrs = _pad_field_free(arrs, fields, dz_native, pad_mm, lo=True)
        if last:
            arrs = _pad_field_free(arrs, fields, dz_native, pad_mm, lo=False)
        win = dataclasses.replace(vane, **arrs)
        cache = Laplace3DCache(win, nx=nx, ny=nx,
                               z_subsample=z_subsample, solver=solver)
        zg = cache.z_mm
        keep = (zg >= (z0 - pad_mm if first else z0)) & \
               (zg < (z1 + pad_mm if last else z1))
        rows.extend(_fit_slices(cache, np.where(keep)[0], vane))
        if verbose:
            _log.info("rfq geometry profile: window %d/%d solved",
                      i + 1, len(plan))
    rows = np.array(rows)
    order = np.argsort(rows[:, 0])
    return rows[order]


def _fit_slices(cache, iz_keep: np.ndarray, vane: VaneGeometry):
    """Per z-slice linear fits E_x ≈ ax·x, E_y ≈ ay·y over the core.

    The fit region is limited to 80 % of the local vane aperture (and
    4.5 mm) so points inside the electrode metal — where the gridded
    potential is boundary fill, not field — never enter the fit.
    """
    zv = vane.z * 1e3
    a_h = 0.5 * (vane.aperture_v1 + vane.aperture_v3) * 1e3
    a_v = 0.5 * (vane.aperture_v2 + vane.aperture_v4) * 1e3
    order = np.argsort(zv)
    zv, a_h, a_v = zv[order], a_h[order], a_v[order]

    xg = cache._x_mm
    yg = cache._y_mm
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    rows = []
    for iz in iz_keep:
        zz = float(cache.z_mm[iz])
        ah = float(np.interp(zz, zv, a_h))
        av = float(np.interp(zz, zv, a_v))
        phi = cache.phi_static[iz]
        Ex = -np.gradient(phi, xg * 1e-3, axis=0)
        Ey = -np.gradient(phi, yg * 1e-3, axis=1)
        m = (np.abs(XX) < min(0.8 * ah, 4.5)) & \
            (np.abs(YY) < min(0.8 * av, 4.5))
        x = XX[m]
        y = YY[m]
        sxx = float(np.dot(x, x))
        syy = float(np.dot(y, y))
        if sxx <= 0 or syy <= 0:
            continue
        rows.append([zz, float(np.dot(Ex[m], x)) / sxx,
                     float(np.dot(Ey[m], y)) / syy])
    return rows


def _cache_path(vane_path: Path) -> Path:
    return vane_path.with_name(vane_path.name + ".geomprofile.npz")


def _fingerprint(vane_path: Path, params: dict) -> str:
    st = vane_path.stat()
    items = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return f"v{_CACHE_VERSION};{st.st_size};{int(st.st_mtime)};{items}"


def build_rfq_geometry_profile(
        vane: Union[str, Path, VaneGeometry],
        *,
        plateau_z_mm: tuple[float, float] | None = None,
        nx: int = 61,
        z_subsample: int = 4,
        window_mm: float = 340.0,
        pad_mm: float = 60.0,
        solver: str = "auto",
        use_cache: bool = True,
        verbose: bool = False) -> RfqGeometryProfile:
    """Measure gx(z), gy(z) from a vane file (cached on disk).

    Parameters
    ----------
    plateau_z_mm : (z0, z1) or None
        Longitudinal range used to normalise the profile — it should
        cover unmodulated (m ≈ 1) cells away from the ends.  ``None``
        falls back to the middle of the first third of the structure.
    solver : "auto" | "pyamg" | "spsolve"
        ``auto`` prefers pyamg (needed for the default nx=61 grids)
        and degrades to nx=31 + spsolve with a warning if pyamg is
        not installed.
    """
    # Resolve the solver BEFORE the cache lookup so the fingerprint is
    # identical on read and write — otherwise a pyamg-less machine could
    # never hit its own cache (adversarial review 2026-08-02, finding 1).
    if solver == "auto":
        try:
            import pyamg                             # noqa: F401
            solver = "pyamg"
        except ImportError:
            _log.warning(
                "pyamg not installed -- rfq geometry profile falls back "
                "to spsolve on a coarser nx=31 grid (slower, ~2x noisier "
                "fits).  `pip install pyamg` for the full-quality solve.")
            solver, nx = "spsolve", min(nx, 31)

    params = dict(nx=nx, z_subsample=z_subsample, window_mm=window_mm,
                  pad_mm=pad_mm, solver=solver,
                  plateau=None if plateau_z_mm is None
                  else (round(plateau_z_mm[0], 3),
                        round(plateau_z_mm[1], 3)))

    vane_path = None
    if not isinstance(vane, VaneGeometry):
        vane_path = Path(vane)
        if use_cache:
            cpath = _cache_path(vane_path)
            if cpath.exists():
                try:
                    d = np.load(cpath, allow_pickle=False)
                    if str(d["fingerprint"]) == _fingerprint(vane_path,
                                                             params):
                        vz = (tuple(d["vane_z_mm"])
                              if "vane_z_mm" in d.files else None)
                        med = (float(d["median_aperture_mm"])
                               if "median_aperture_mm" in d.files else None)
                        return RfqGeometryProfile(
                            z_mm=d["z_mm"], gx=d["gx"], gy=d["gy"],
                            g_plateau=float(d["g_plateau"]),
                            vane_path=str(vane_path), params=params,
                            vane_z_mm=vz, median_aperture_mm=med)
                except Exception as exc:            # corrupt cache
                    _log.warning("rfq geometry profile cache unreadable "
                                 "(%s) -- rebuilding", exc)
        vane = parse_vane_file(vane_path)

    rows = _solve_windows(vane, nx=nx, z_subsample=z_subsample,
                          window_mm=window_mm, pad_mm=pad_mm,
                          solver=solver, verbose=verbose)
    z_mm, ax, ay = rows[:, 0], rows[:, 1], rows[:, 2]

    if plateau_z_mm is None:
        # fallback window over the UNPADDED vane span -- the padded ends
        # are field-free and would drag the normalisation toward zero
        # (adversarial review 2026-08-02, finding 4)
        v0, v1 = float(np.min(vane.z) * 1e3), float(np.max(vane.z) * 1e3)
        span = v1 - v0
        plateau_z_mm = (v0 + 0.10 * span, v0 + 0.35 * span)
    pm = (z_mm >= plateau_z_mm[0]) & (z_mm < plateau_z_mm[1])
    if pm.sum() < 10:
        raise ValueError(
            f"rfq geometry profile: plateau range {plateau_z_mm} covers "
            f"only {int(pm.sum())} solved slices -- pass a wider "
            "plateau_z_mm over unmodulated cells")
    g_pl = float(ax[pm].mean())
    # normalisation sanity: the plateau must carry a healthy fraction of
    # the structure's peak gradient -- a pad/flare-dominated window
    # would silently mis-scale every cell (and could even flip the
    # global sign).  Fail loudly instead.
    peak = float(np.max(np.abs(ax)))
    if peak <= 0 or abs(g_pl) < 0.3 * peak:
        raise ValueError(
            f"rfq geometry profile: plateau normalisation failed "
            f"(|mean gradient| {abs(g_pl):.3g} < 30% of peak {peak:.3g} "
            f"over z = {plateau_z_mm}) -- pass plateau_z_mm covering "
            "unmodulated cells away from the ends")
    sgn = np.sign(g_pl) or 1.0
    g_plateau = abs(g_pl)
    gx = sgn * ax / g_plateau
    gy = sgn * ay / g_plateau

    vane_span = (float(np.min(vane.z) * 1e3), float(np.max(vane.z) * 1e3))
    med_ap = float(np.median(
        0.25 * (vane.aperture_v1 + vane.aperture_v2
                + vane.aperture_v3 + vane.aperture_v4)) * 1e3)
    prof = RfqGeometryProfile(z_mm=z_mm, gx=gx, gy=gy,
                              g_plateau=g_plateau,
                              vane_path=str(vane_path or ""),
                              params=params, vane_z_mm=vane_span,
                              median_aperture_mm=med_ap)
    if use_cache and vane_path is not None:
        try:
            # temp-then-rename: concurrent builders (e.g. parallel scan
            # workers on a cold cache) can never leave a torn file
            cpath = _cache_path(vane_path)
            # (suffix must stay .npz — np.savez appends it otherwise)
            tmp = cpath.with_name(cpath.name + f".tmp{os.getpid()}.npz")
            np.savez_compressed(
                tmp, z_mm=z_mm, gx=gx, gy=gy,
                g_plateau=g_plateau, vane_z_mm=np.array(vane_span),
                median_aperture_mm=med_ap,
                fingerprint=_fingerprint(vane_path, params))
            os.replace(tmp, cpath)
        except OSError as exc:
            _log.warning("rfq geometry profile cache not written: %s", exc)
    return prof
