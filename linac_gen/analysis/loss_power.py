"""Beam-loss power accounting and plane power-density maps.

Turns the per-particle loss record (``Beam.record_loss`` → structured
``loss_table``: particle_id, s [mm], x/y [mm], energy [MeV], element
name) into the quantities machine protection and absorber engineering
actually use:

* a per-element loss table in **watts** (the BLM/MPS view — the
  universal hands-on limit for a proton linac is ~1 W/m);
* a lineal loss-density profile **W/m vs s**;
* a transverse **power-density map [W/cm²]** at any plane (dump face,
  window, foil) from a particle distribution.

Power convention
----------------
``current_mA`` is the PEAK (in-pulse) beam current — the number a
BeamConfig carries — and ``duty_pct`` converts it to an average:
``I_avg = current_mA·1e-3 · duty_pct/100``.  Each of the ``n_macro``
LAUNCHED macroparticles carries ``I_avg/n_macro`` of beam current, so a
macroparticle lost (or arriving) at kinetic energy ``E`` [MeV] carries

    P = (I_avg / n_macro) · E·1e6   [W]

(the elementary charge cancels for any ±1 species; H⁻ included).
Sanity anchor: 2 mA × 550 µs × 20 Hz (duty 1.1 %) at 800 MeV is
17.6 kW — the PIP-II Booster-bound beam against the 25 kW absorber
rating.
"""
from __future__ import annotations

import numpy as np

LOSS_POWER_DTYPE = np.dtype([
    ("element_name", "U32"),
    ("n_lost", np.int64),
    ("s_min_mm", np.float64),
    ("s_max_mm", np.float64),
    ("e_mean_mev", np.float64),
    ("watts", np.float64),
])


def average_current_A(current_mA: float, duty_pct: float = 100.0) -> float:
    """Average beam current [A] from peak current [mA] and duty [%]."""
    return float(current_mA) * 1e-3 * float(duty_pct) / 100.0


def watts_per_macroparticle(energy_mev, current_mA: float,
                            duty_pct: float, n_macro: int):
    """Beam power carried by one macroparticle at ``energy_mev`` [W].

    ``energy_mev`` may be a scalar or an array (per-particle energies).
    """
    if n_macro <= 0:
        raise ValueError("n_macro must be positive")
    i_avg = average_current_A(current_mA, duty_pct)
    return (i_avg / float(n_macro)) * np.asarray(energy_mev,
                                                dtype=float) * 1e6


def loss_power_table(loss_table, *, current_mA: float,
                     duty_pct: float = 100.0,
                     n_macro: int) -> np.ndarray:
    """Aggregate the loss record per element, sorted by watts (desc).

    Returns a structured array (:data:`LOSS_POWER_DTYPE`); empty when
    nothing was lost.  Energies at the LOSS POINT are used — a particle
    scraped in the MEBT costs 2 MeV-scale power, not 800.
    """
    lt = np.asarray(loss_table)
    if lt.size == 0:
        return np.array([], dtype=LOSS_POWER_DTYPE)
    w_each = watts_per_macroparticle(lt["energy"], current_mA,
                                     duty_pct, n_macro)
    rows = []
    for name in np.unique(lt["element_name"]):
        m = lt["element_name"] == name
        rows.append((str(name), int(np.count_nonzero(m)),
                     float(lt["s"][m].min()), float(lt["s"][m].max()),
                     float(lt["energy"][m].mean()),
                     float(w_each[m].sum())))
    out = np.array(rows, dtype=LOSS_POWER_DTYPE)
    return out[np.argsort(out["watts"])[::-1]]


def loss_power_profile(loss_table, *, current_mA: float,
                       duty_pct: float = 100.0, n_macro: int,
                       s_end_mm: float, bin_mm: float = 1000.0):
    """Lineal loss density: ``(s_centers_mm, watts_per_m)``.

    ``bin_mm`` defaults to 1 m bins so the numbers read directly
    against the 1 W/m hands-on-maintenance criterion.
    """
    lt = np.asarray(loss_table)
    n_bins = max(1, int(np.ceil(float(s_end_mm) / float(bin_mm))))
    edges = np.linspace(0.0, n_bins * float(bin_mm), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if lt.size == 0:
        return centers, np.zeros(n_bins)
    w_each = watts_per_macroparticle(lt["energy"], current_mA,
                                     duty_pct, n_macro)
    hist, _ = np.histogram(lt["s"], bins=edges, weights=w_each)
    return centers, hist / (float(bin_mm) * 1e-3)     # W per metre


def plane_power_density(x_mm, y_mm, *, energy_mev, current_mA: float,
                        duty_pct: float = 100.0, n_macro: int,
                        bins: int = 64, extent=None):
    """Transverse beam power density at a plane, in **W/cm²**.

    ``x_mm``/``y_mm``: particle coordinates at the plane (e.g. the
    surviving beam at the dump face, or the lost particles recorded ON
    one aperture).  ``energy_mev`` is a scalar or per-particle array.
    Returns ``(H, xedges_mm, yedges_mm)`` with ``H`` shaped
    ``(bins, bins)``, x along axis 0 (numpy histogram2d convention).
    ``extent=(xmin, xmax, ymin, ymax)`` in mm; default = data bounds
    padded 5 %.
    """
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)
    if x.size == 0:
        raise ValueError("no particles at the plane")
    w = np.broadcast_to(
        watts_per_macroparticle(energy_mev, current_mA, duty_pct,
                                n_macro), x.shape)
    if extent is None:
        pad_x = 0.05 * max(np.ptp(x), 1e-6)
        pad_y = 0.05 * max(np.ptp(y), 1e-6)
        extent = (x.min() - pad_x, x.max() + pad_x,
                  y.min() - pad_y, y.max() + pad_y)
    rng = [[extent[0], extent[1]], [extent[2], extent[3]]]
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng, weights=w)
    area_cm2 = (np.diff(xe)[0] * 1e-1) * (np.diff(ye)[0] * 1e-1)
    return H / area_cm2, xe, ye


def loss_power_summary(loss_table, *, current_mA: float,
                       duty_pct: float = 100.0, n_macro: int,
                       w_exit_mev: float | None = None,
                       n_alive: int | None = None, top: int = 5) -> dict:
    """Scalar overview for reports.

    Keys: ``i_avg_ma``, ``n_lost``, ``lost_w`` (total), ``top``
    (list of (element, watts, n) for the worst offenders), and — when
    ``w_exit_mev``/``n_alive`` are given — ``delivered_w`` (power in
    the surviving beam at the exit plane).
    """
    tab = loss_power_table(loss_table, current_mA=current_mA,
                           duty_pct=duty_pct, n_macro=n_macro)
    out = {
        "i_avg_ma": average_current_A(current_mA, duty_pct) * 1e3,
        "n_lost": int(tab["n_lost"].sum()) if tab.size else 0,
        "lost_w": float(tab["watts"].sum()) if tab.size else 0.0,
        "top": [(str(r["element_name"]), float(r["watts"]),
                 int(r["n_lost"])) for r in tab[:top]],
    }
    if w_exit_mev is not None and n_alive is not None:
        out["delivered_w"] = float(
            n_alive * watts_per_macroparticle(w_exit_mev, current_mA,
                                              duty_pct, n_macro))
    return out
