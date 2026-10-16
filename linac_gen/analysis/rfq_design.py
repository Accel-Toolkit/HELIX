"""RFQ design and run quantities for display (the Results tab's RFQ panels).

Pure functions, no Qt:

* :func:`find_rfq` -- the RFQ cell chain of a lattice and where it sits;
* :func:`cell_table` -- per-cell card values and derived quantities, and the
  run's values at every cell exit when results are given;
  :func:`rfq_frequency` -- the RF frequency at the RFQ;
* :func:`solver_electrodes`, :func:`cards_electrodes` -- the vanes as the
  time-domain solver uses them (vane file, else the cards) and as the cards
  describe them;
* :func:`cross_section` -- a transverse cut through the electrodes;
* :func:`kilpatrick_field`, :func:`peak_surface_field` -- the Kilpatrick
  limit and the peak surface field on the vane tips;
* :func:`cached_field_map`, :func:`cards_field_map`, :func:`axis_fields` --
  the 3-D field (time-domain map when it is cached, else the cards' two-term
  field) and the on-axis E_z and quadrupole gradient along z;
* :func:`cached_vane_profile` -- the per-cell model's vane-field profile from
  its disk cache;
* :func:`cell_losses` -- the losses of a run, cell by cell.

Nothing here builds a field map or solves a vane profile: plots use what is
cached and say so otherwise.

Formulas -- smooth approximation of the two-term potential in HELIX's
convention (A01 = 1), U = V/2 [(y² − x²)/r0² ∓ A10·I0(kr)·cos kz], k = π/L,
with V the inter-vane voltage of the cards, q the charge and m c² the rest
energy (eV):

* energy gain per cell q·(π/4)·A10·V·cos φs, i.e. E0T = π·A10·V / (2βλ);
* focusing strength B = q·V·λ² / (m c² r0²);
* RF defocusing Δ_rf = π²·q·A10·V·sin φs / (2 m c² β² γ) (negative below
  the crest), from the cycle-averaged radial force of the A10 term;
* zero-current phase advances per focusing period βλ:
  σ0t² = B²/(8π²) + Δ_rf and σ0l² = π²·q·A10·V·|sin φs| / (m c² β² γ³);
* current limits of a uniform ellipsoidal bunch (semi-axes a, a, b) that
  fills the cell's minimum aperture a and the separatrix (half-length
  b = βλ·1.5|φs|/(2π)), from the linear space-charge forces of the
  ellipsoid (form factor f of the exact spheroid, p = γb/a):
  I_t = 8π (m c²/q) γ³ a² b σ0t² / (3 Z0 λ³ (1 − f)) and
  I_l = 4π (m c²/q) γ³ a² b σ0l² / (3 Z0 λ³ f).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: free-space impedance (Ω) and the speed of light (m/s)
Z0_OHM = 376.730313668
C_M_S = 299_792_458.0


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------
@dataclass
class RfqChain:
    """The (first) contiguous chain of RFQ cells of a lattice."""
    cells: list                  # RfqCell, in order
    index: list                  # their element indices in the lattice
    s0: np.ndarray               # (C,) lattice s of each cell's entrance, mm
    s1: np.ndarray               # (C,) exit, mm
    edges: np.ndarray            # (C+1,) chain coordinates, 0 at the entrance
    n_chains: int = 1            # chains in the lattice (only the first used)

    @property
    def s_start(self) -> float:
        return float(self.s0[0])

    @property
    def names(self) -> list:
        return [str(getattr(c, "name", "")) for c in self.cells]

    def to_s(self, z_chain):
        """Lattice s (mm) of chain coordinates z (mm)."""
        return np.asarray(z_chain, float) + self.s_start


def find_rfq(lattice) -> RfqChain | None:
    """The first contiguous chain of :class:`RfqCell` elements, or None."""
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.rfq_td.chain import find_chains
    elements = list(getattr(lattice, "elements", None) or [])
    chains = find_chains(elements)
    if not chains:
        return None
    s_entry = np.concatenate([[0.0], np.cumsum([
        float(getattr(e, "length", 0.0) or 0.0) for e in elements])])
    i0, i1 = chains[0]
    idx = [k for k in range(i0, i1 + 1) if isinstance(elements[k], RfqCell)]
    cells = [elements[k] for k in idx]
    s0 = np.array([s_entry[k] for k in idx])
    s1 = np.array([s_entry[k + 1] for k in idx])
    edges = np.concatenate([[0.0], np.cumsum([float(c.length) for c in cells])])
    return RfqChain(cells=cells, index=idx, s0=s0, s1=s1, edges=edges,
                    n_chains=len(chains))


def species_constants(species: str | None):
    """(rest energy in eV, |charge| in units of e) of a species name."""
    from linac_gen.core.particle import PROTON
    from linac_gen.distributions.factory import SPECIES_MAP
    p = SPECIES_MAP.get(str(species or ""), PROTON)
    return float(p.mass) * 1e6, abs(float(p.charge)) or 1.0


def wavelength_mm(frequency_MHz: float) -> float:
    return C_M_S / (float(frequency_MHz) * 1e6) * 1e3


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def cell_tips(cell, z_local):
    """(R_x, R_y) tip distances (mm) of one cell from its card, at cell-local
    positions (mm) -- the per-cell model's and the card-built vanes' model."""
    from linac_gen.rfq_td.cards_geometry import _cell_tips
    return _cell_tips(cell, np.atleast_1d(np.asarray(z_local, float)))


def min_aperture(cell, n: int = 9) -> float:
    """Smallest tip distance of a cell from its card (mm)."""
    L = float(cell.length)
    if L <= 0:
        return float(cell.r0_mm)
    rx, ry = cell_tips(cell, np.linspace(0.0, L, n))
    return float(min(np.min(rx), np.min(ry)))


def solver_electrodes(lattice, chain: RfqChain):
    """The electrodes the time-domain solver uses: ``(electrodes, scale,
    source, note)``; ``electrodes`` is None (and ``note`` says why) when the
    vanes cannot be used.  ``scale`` multiplies the field map's field.
    Without the solver's log notes (those belong to the run); the
    lattice-level refusals that make its vanes or their scale ambiguous
    (several chains, several vane voltages) are the solver's own."""
    from linac_gen.rfq_td.chain import RfqTdError, chain_electrodes
    if chain.n_chains > 1:
        return None, None, "", (f"{chain.n_chains} separate RFQ cell "
                                "chains; the time-domain solver tracks one")
    if len({float(c.voltage_V) for c in chain.cells}) > 1:
        return None, None, "", ("the RFQ cells carry different vane "
                                "voltages; the time-domain solver needs one "
                                "voltage")
    try:
        el, scale = chain_electrodes(lattice, chain.cells,
                                     float(chain.edges[-1]), notes=False)
    except (RfqTdError, ValueError, OSError) as exc:
        return None, None, "", str(exc)
    source = ("vane file" if getattr(lattice, "rfq_geom_file", None)
              else "RFQ_CELL cards")
    return el, float(scale), source, ""


def cards_electrodes(lattice, chain: RfqChain):
    """The electrodes the cards describe (at the cards' voltage), or None."""
    from linac_gen.rfq_td.cards_geometry import electrodes_from_cards
    try:
        return electrodes_from_cards(
            chain.cells, gaps=getattr(lattice, "rfq_gaps_mm", None))
    except ValueError:
        return None


def tips_along(el, z=None):
    """Tip distances (mm) along chain coordinates: dict with z, R_x, R_y,
    R_mx, R_my, rho_x, rho_y (the electrodes' own table when ``z`` is None)."""
    zz = np.asarray(el.z if z is None else z, float)
    out = {"z": zz}
    for k in ("R_x", "R_y", "R_mx", "R_my", "rho_x", "rho_y"):
        out[k] = np.interp(zz, el.z, np.asarray(getattr(el, k), float))
    return out


def cross_section(el, z: float, half_width_mm: float, n: int = 241):
    """Metal (True) on an n×n grid over ±half_width (mm) at chain z (mm):
    ``(xs, ys, metal)`` with ``metal[iy, ix]``, plus the tip distances there."""
    xs = np.linspace(-half_width_mm, half_width_mm, int(n))
    X, Y = np.meshgrid(xs, xs)
    metal = el.inside(X, Y, np.full_like(X, float(z)))
    tips = {k: float(v[0]) for k, v in tips_along(el, [float(z)]).items()
            if k != "z"}
    return xs, xs.copy(), np.asarray(metal, bool), tips


# ---------------------------------------------------------------------------
# per-cell table
# ---------------------------------------------------------------------------
def _spheroid_form_factor(p: float) -> float:
    """Form factor f of a uniformly charged spheroid with axis ratio
    p = (longitudinal semi-axis)/(transverse semi-axis); f(1) = 1/3."""
    if not np.isfinite(p) or p <= 0:
        return float("nan")
    if abs(p - 1.0) < 1e-9:
        return 1.0 / 3.0
    if p > 1.0:
        s = math.sqrt(p * p - 1.0)
        return (p / s * math.log(p + s) - 1.0) / (p * p - 1.0)
    s = math.sqrt(1.0 - p * p)
    return (1.0 - p / s * math.acos(p)) / (1.0 - p * p)


def figures_of_merit(V, r0_mm, A10, phi_s_deg, beta, a_mm, *, mc2_eV, q,
                     lam_mm):
    """Smooth-approximation figures of one cell (module docstring): dict
    with B, delta_rf, sigma0t_deg, sigma0l_deg, E0T_MV_m, I_t_mA, I_l_mA
    (NaN where undefined: no acceleration, unstable, unknown β)."""
    out = dict(B=float("nan"), delta_rf=float("nan"),
               sigma0t_deg=float("nan"), sigma0l_deg=float("nan"),
               E0T_MV_m=float("nan"), I_t_mA=float("nan"),
               I_l_mA=float("nan"))
    lam = lam_mm * 1e-3
    r0 = r0_mm * 1e-3
    if r0 <= 0 or lam <= 0 or mc2_eV <= 0:
        return out
    B = q * V * lam ** 2 / (mc2_eV * r0 ** 2)
    out["B"] = B
    if not (np.isfinite(beta) and 0.0 < beta < 1.0):
        return out
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    phi = math.radians(phi_s_deg)
    d_rf = math.pi ** 2 * q * A10 * V * math.sin(phi) / (
        2.0 * mc2_eV * beta ** 2 * gamma)
    out["delta_rf"] = d_rf
    out["E0T_MV_m"] = math.pi * A10 * V / (2.0 * beta * lam) * 1e-6
    s2t = B * B / (8.0 * math.pi ** 2) + d_rf
    s2l = math.pi ** 2 * q * A10 * V * abs(math.sin(phi)) / (
        mc2_eV * beta ** 2 * gamma ** 3)
    st = math.sqrt(s2t) if s2t > 0 else float("nan")
    sl = math.sqrt(s2l) if s2l > 0 else float("nan")
    out["sigma0t_deg"] = math.degrees(st)
    out["sigma0l_deg"] = math.degrees(sl)
    a = a_mm * 1e-3
    b = beta * lam * 1.5 * abs(phi) / (2.0 * math.pi)
    if a > 0 and b > 0 and A10 > 0 and phi < 0:
        f = _spheroid_form_factor(gamma * b / a)
        common = (mc2_eV / q) * gamma ** 3 * a * a * b / (3.0 * Z0_OHM *
                                                          lam ** 3)
        if np.isfinite(st) and 0.0 < f < 1.0:
            out["I_t_mA"] = 8.0 * math.pi * common * st * st / (1.0 - f) * 1e3
        if np.isfinite(sl) and f > 0.0:
            out["I_l_mA"] = 4.0 * math.pi * common * sl * sl / f * 1e3
    return out


def _run_rows(results, chain: RfqChain):
    """Recorder row of every cell's exit, or None with a reason."""
    if results is None:
        return None, "no run"
    names = list(getattr(results, "element_names", None) or [])
    exit_idx = getattr(results, "element_exit_idx", None)
    exit_idx = list(exit_idx) if exit_idx is not None else []
    rows = []
    for ei, cell in zip(chain.index, chain.cells):
        r = int(exit_idx[ei]) if ei < len(exit_idx) else ei + 1
        if r >= len(names) or str(names[r]) != str(cell.name):
            return None, ("the results do not belong to this lattice "
                          "(cell names differ)")
        rows.append(r)
    return np.asarray(rows, int), ""


def _col(results, name, rows, j=None):
    v = getattr(results, name, None)
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim == 0 or a.shape[0] <= int(np.max(rows)):
        return None
    a = a[rows]
    if j is not None:
        if a.ndim < 2 or a.shape[1] <= j:
            return None
        a = a[:, j]
    return a


def rfq_frequency(lattice, chain: RfqChain, results=None, default_MHz=None):
    """The RF frequency (MHz) at the RFQ: the run's reference frequency at
    the RFQ's first cell exit when the results belong to the lattice, else
    the last ``FREQ`` card before the RFQ, else ``default_MHz`` (the beam's).
    Not the end of the run: a linac after the RFQ may run at a multiple."""
    from linac_gen.elements.lattice_commands import Freq
    rows, _note = _run_rows(results, chain)
    if rows is not None:
        f = _col(results, "ref_frequency", rows)
        if f is not None and np.isfinite(f[0]) and f[0] > 0:
            return float(f[0])
    f_card = None
    for e in list(lattice.elements)[:chain.index[0]]:
        if isinstance(e, Freq) and float(e.frequency_mhz) > 0:
            f_card = float(e.frequency_mhz)
    if f_card is not None:
        return f_card
    return float(default_MHz) if default_MHz else None


def cell_table(lattice, *, species="proton", frequency_MHz=None,
               results=None):
    """Every per-cell quantity, one array per key (``None`` when the lattice
    has no RFQ).  ``beta`` is the reference particle's at each cell exit
    when ``results`` belong to this lattice, else the design value 2L/λ of
    accelerating cells.  Run keys (``trans``, ``sigma_x`` …) are present only
    with matching results; ``run_note`` says why they are missing."""
    from linac_gen.elements.rfq_coefficients import (
        modulation_consistency, synchronous_phase_from_lengths)
    chain = find_rfq(lattice)
    if chain is None:
        return None
    mc2_eV, q = species_constants(species)
    f_MHz = float(frequency_MHz or 0.0)
    cells = chain.cells
    C = len(cells)
    t = {
        "cell": np.arange(1, C + 1),
        "name": chain.names,
        "s0": chain.s0, "s1": chain.s1,
        "z0": chain.edges[:-1], "z1": chain.edges[1:],
        "type": np.array([int(c.cell_type) for c in cells]),
        "L": np.array([float(c.length) for c in cells]),
        "V": np.array([float(c.voltage_V) for c in cells]),
        "r0": np.array([float(c.r0_mm) for c in cells]),
        "m": np.array([float(c.modulation) for c in cells]),
        "A10": np.array([float(c.A10) for c in cells]),
        "phi_s": np.array([float(c.phi_s_deg) for c in cells]),
        "Tc": np.array([float(getattr(c, "Tc_mm", 0.0) or 0.0)
                        or 0.75 * float(c.r0_mm) for c in cells]),
        "a_min": np.array([min_aperture(c) for c in cells]),
    }
    t["A10_theory"] = np.array([modulation_consistency(
        float(c.r0_mm), float(c.A10), float(c.modulation),
        float(c.length))[0] for c in cells])
    lam = wavelength_mm(f_MHz) if f_MHz > 0 else float("nan")
    t["lambda_mm"] = lam
    if np.isfinite(lam):
        phi_len, _ok = synchronous_phase_from_lengths(
            t["L"], t["A10"], t["V"], mc2_eV * 1e-6, q, lam)
        beta_design = np.where(np.abs(t["type"]) == 2, 2.0 * t["L"] / lam,
                               np.nan)
    else:
        phi_len = np.full(C, np.nan)
        beta_design = np.full(C, np.nan)
    t["phi_s_from_L"] = phi_len
    t["beta"] = beta_design
    t["beta_from_run"] = False
    t["beta_source"] = "the design β = 2L/λ (accelerating cells only)"
    rows, note = _run_rows(results, chain)
    t["run_note"] = note
    if rows is not None:
        def put(key, name, j=None):
            a = _col(results, name, rows, j)
            if a is not None:
                t[key] = a
        put("beta", "ref_beta")
        if "beta" in t and np.all(np.isfinite(t["beta"])):
            t["beta_from_run"] = True
            t["beta_source"] = ("the β of the run's reference particle at "
                                "each cell exit")
        put("W", "ref_w_kin")
        put("trans", "transmission")
        put("sigma_x", "sigma_x")
        put("sigma_y", "sigma_y")
        put("x_max", "x_max")
        put("y_max", "y_max")
        put("sigma_phi", "sigma_phi")
        put("sigma_w", "sigma_w")
        put("phi_offset", "centroid", 4)
        put("w_offset", "centroid", 5)
        if "W" in t and "w_offset" in t:
            t["W_beam"] = t["W"] + t["w_offset"]
    t["beta_lambda_half"] = 0.5 * np.asarray(t["beta"], float) * lam
    fom = [figures_of_merit(t["V"][k], t["r0"][k], t["A10"][k],
                            t["phi_s"][k], float(t["beta"][k]),
                            t["a_min"][k], mc2_eV=mc2_eV, q=q,
                            lam_mm=lam if np.isfinite(lam) else 0.0)
           for k in range(C)]
    for key in ("B", "delta_rf", "sigma0t_deg", "sigma0l_deg", "E0T_MV_m",
                "I_t_mA", "I_l_mA"):
        t[key] = np.array([d[key] for d in fom])
    return t


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def cell_losses(results, chain: RfqChain, *, current_mA=None,
                duty_pct=100.0):
    """The run's losses inside the RFQ: dict with per-cell ``count`` and
    ``watts`` (when the current is known) and the individual losses
    ``s``, ``x``, ``y``, ``energy``, ``cell`` (0-based); None without a
    loss table."""
    lt = getattr(results, "loss_table", None) if results is not None else None
    if lt is None:
        return None
    lt = np.asarray(lt)
    names = chain.names
    C = len(names)
    if lt.size == 0 or "element_name" not in (lt.dtype.names or ()):
        empty = np.zeros(0)
        return dict(count=np.zeros(C, int), watts=np.zeros(C), s=empty,
                    x=empty, y=empty, energy=empty,
                    cell=np.zeros(0, int), n_macro=0)
    pos = {n: k for k, n in enumerate(names)}
    en = np.array([str(n.decode() if isinstance(n, bytes) else n)
                   for n in lt["element_name"]])
    cell = np.array([pos.get(n, -1) for n in en])
    m = cell >= 0
    count = np.bincount(cell[m], minlength=C)
    n_macro = int(getattr(results, "n_macro", 0) or 0)
    watts = np.full(C, np.nan)
    if current_mA is not None and n_macro > 0:
        from linac_gen.analysis.loss_power import watts_per_macroparticle
        w = watts_per_macroparticle(np.asarray(lt["energy"], float)[m],
                                    float(current_mA), float(duty_pct),
                                    n_macro)
        watts = np.bincount(cell[m], weights=w, minlength=C).astype(float)
    return dict(count=count, watts=watts,
                s=np.asarray(lt["s"], float)[m],
                x=np.asarray(lt["x"], float)[m],
                y=np.asarray(lt["y"], float)[m],
                energy=np.asarray(lt["energy"], float)[m],
                cell=cell[m], n_macro=n_macro)


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------
def kilpatrick_field(frequency_MHz: float) -> float:
    """Kilpatrick limit E_K (MV/m) at a frequency (MHz): the root of
    f = 1.643·E_K²·exp(−8.5/E_K)."""
    f = float(frequency_MHz)
    if f <= 0:
        return float("nan")
    lo, hi = 1e-3, 1e4
    for _ in range(200):                      # bisection: monotonic in E
        mid = 0.5 * (lo + hi)
        if 1.643 * mid * mid * math.exp(-8.5 / mid) < f:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cached_field_map(el, chain: RfqChain):
    """The time-domain solver's field map of these electrodes if it is
    already in memory or on disk, else None (never builds one)."""
    from linac_gen.rfq_td.build import get_field_map, map_cached
    if el is None or not map_cached(el, chain.edges):
        return None
    return get_field_map(el, chain.edges)


def cards_field_map(chain: RfqChain):
    """The cards' analytic two-term field as a field map at 1 V (scale by
    the cells' voltage)."""
    from linac_gen.rfq_td.cards_geometry import two_term_field_map
    return two_term_field_map(chain.cells, voltage=1.0)


def cards_axis_fields(chain: RfqChain, z, *, dx_mm=0.2):
    """On-axis E_z (V/mm) and ∂E_x/∂x (V/mm²) of the cards' two-term field
    along chain z (mm), each cell at its own voltage (the field is linear in
    V and local to its cell)."""
    zz = np.asarray(z, float)
    ez, g = axis_fields(cards_field_map(chain), zz, dx_mm=dx_mm)
    k = np.clip(np.searchsorted(chain.edges, zz, side="right") - 1, 0,
                len(chain.cells) - 1)
    v = np.array([float(c.voltage_V) for c in chain.cells])[k]
    return ez * v, g * v


def _field_fn(fmap):
    def field(x, y, z):
        ex, ey, ez, _n = fmap.field(np.ravel(x), np.ravel(y), np.ravel(z))
        return ex, ey, ez
    return field


def axis_fields(fmap, z, *, scale=1.0, dx_mm=None):
    """On-axis accelerating field E_z (V/mm, at the RF crest) and the
    transverse quadrupole gradient ∂E_x/∂x (V/mm²) along chain z (mm)."""
    zz = np.asarray(z, float)
    dx = float(dx_mm or getattr(fmap, "h", 0.1))
    zero = np.zeros_like(zz)
    _ex, _ey, ez, _n = fmap.field(zero, zero, zz)
    ex, _ey2, _ez2, _n2 = fmap.field(np.full_like(zz, dx), zero, zz)
    return np.asarray(ez) * scale, np.asarray(ex) * scale / dx


#: steepest tip slope |dR/dz| where the surface-field estimate applies (the
#: tip is treated as straight along z; PXIE's modulated cells reach 0.25)
MAX_TIP_SLOPE = 0.35


def tip_slope(el, z, *, window_mm=3.0):
    """Largest |dR/dz| of the ±x and ±y tips within ±window of each chain z
    (catches flares, vane ends and steps between cards; the window is about
    r0, the distance over which a step's corner raises the field)."""
    zt = np.asarray(el.z, float)
    sl = np.zeros_like(zt)
    for R in (el.R_x, el.R_y, el.R_mx, el.R_my):
        sl = np.maximum(sl, np.abs(np.gradient(np.asarray(R, float), zt)))
    zz = np.atleast_1d(np.asarray(z, float))
    lo = np.searchsorted(zt, zz - window_mm, side="left")
    hi = np.searchsorted(zt, zz + window_mm, side="right")
    out = np.empty(zz.size)
    for i, (a, b) in enumerate(zip(lo, hi)):
        out[i] = sl[max(a - 1, 0):min(b + 1, zt.size)].max()
    return out


def peak_surface_field(field, el, z, *, offsets_mm, x_store_mm, scale=1.0,
                       n_theta=46, max_slope=MAX_TIP_SLOPE):
    """Peak field magnitude on the ±x and ±y vane tips at chain z (V/mm, RF
    crest): ``(E_x_tip, E_y_tip)``.

    The tip of a pole is a circular arc (radius ρ) around (R + ρ) on the
    pole axis.  ``|E|`` is evaluated on arcs at the two distances
    ``offsets_mm`` from the tip surface, over the side that faces the bore
    and the neighbouring pole (where the peak is: towards the pole of the
    other polarity), and extrapolated to the surface with the fall-off of a
    curved conductor, E(δ) = E_s·ρ_e/(ρ_e + δ), fitted to the two samples
    (linear when that fit does not apply); the maximum over the arc is
    returned.  On a stored potential the offsets must clear the metal by
    the interpolation stencil (≥ 3 grid steps on a field map).  Points
    beyond the stored box (``x_store_mm``) are left out; NaN when none is
    left, and where a tip is steeper than ``max_slope`` along z (flares,
    vane ends, steps between cards), where the tip is not straight along z.
    ``field(x, y, z) -> (Ex, Ey, Ez)`` is any field function (V/mm)."""
    d1, d2 = (float(v) for v in offsets_mm)
    zz = np.atleast_1d(np.asarray(z, float))
    tips = tips_along(el, zz)
    th = np.radians(np.linspace(0.0, 90.0, int(n_theta)))
    out = []
    for R, rho, axis in ((tips["R_x"], tips["rho_x"], 0),
                         (tips["R_y"], tips["rho_y"], 1)):
        Z = np.repeat(zz, th.size)
        T = np.tile(th, zz.size)
        c = np.repeat(R + rho, th.size)
        r = np.repeat(rho, th.size)
        mags = []
        for d in (d1, d2):
            u = c - (r + d) * np.cos(T)          # along the pole
            w = (r + d) * np.sin(T)              # towards the next pole
            X, Y = (u, w) if axis == 0 else (w, u)
            ex, ey, ez = field(X, Y, Z)
            mag = np.sqrt(np.asarray(ex) ** 2 + np.asarray(ey) ** 2
                          + np.asarray(ez) ** 2) * scale
            mag[(np.abs(X) > x_store_mm) | (np.abs(Y) > x_store_mm)] = np.nan
            mags.append(mag.reshape(zz.size, th.size))
        e1, e2 = mags
        with np.errstate(all="ignore"):
            rho_e = (e2 * d2 - e1 * d1) / (e1 - e2)
            e_fit = e1 * (rho_e + d1) / rho_e
        e_lin = e1 + (e1 - e2) * d1 / (d2 - d1)
        e_s = np.where((e1 > e2) & (rho_e > 0), e_fit, e_lin)
        with np.errstate(all="ignore"):
            peak = np.where(np.all(np.isnan(e_s), axis=1), np.nan,
                            np.nanmax(np.where(np.isnan(e_s), -np.inf, e_s),
                                      axis=1))
        out.append(peak)
    steep = tip_slope(el, zz) > max_slope
    return (np.where(steep, np.nan, out[0]), np.where(steep, np.nan, out[1]))


# ---------------------------------------------------------------------------
# the per-cell model's vane-field profile
# ---------------------------------------------------------------------------
def cached_vane_profile(vane_path):
    """(z_mm, gx, gy) of the per-cell model's vane-field profile from its
    disk cache when the cache belongs to this vane file (same size and
    modification time), else None -- never solves it."""
    from pathlib import Path
    from linac_gen.elements.rfq_geometry_profile import (_CACHE_VERSION,
                                                          _cache_path)
    if not vane_path:
        return None
    vp = Path(vane_path)
    cp = _cache_path(vp)
    if not (vp.is_file() and cp.is_file()):
        return None
    try:
        d = np.load(cp, allow_pickle=False)
        fp = str(d["fingerprint"]).split(";")
        st = vp.stat()
        if (len(fp) < 3 or fp[0] != f"v{_CACHE_VERSION}"
                or fp[1] != str(st.st_size)
                or fp[2] != str(int(st.st_mtime))):
            return None
        return (np.asarray(d["z_mm"], float), np.asarray(d["gx"], float),
                np.asarray(d["gy"], float))
    except Exception:                                   # noqa: BLE001
        return None
