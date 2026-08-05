"""HELIX driver for the LEBT space-charge-compensation analysis.

Feeds the ported SCC physics from a HELIX DC (continuous-beam) run: the
actual envelope sigma(z), per-element apertures, transmission-attenuated
local current and beam energy come from the results/lattice; the user
supplies the residual-gas description.  Outputs per-z profiles
(eta_ss, f_c, tau_scc, beam potential, gas-loss survival) plus suggested
:class:`~linac_gen.elements.space_charge_comp.SpaceChargeComp` factors.

Formulas are ported 1:1 from the source tool's ``SimulationEngine``
(``scc/lebt_scc/simulation.py``): gas-mixture assembly, ionisation-
weighted mean trapped-ion mass, pressure-profile multiplier, the exact
per-z potential ``phi = phi_uncomp(z) (1 - f_c(z))``, the cumulative
optical depth, the end-taper phenomenology (three disclosed magic
numbers standing in for extractor/RFQ-repeller axial escape), and the
per-slice steady-state balance (:class:`SelfConsistentSCC`).

Guard: refuses (in-band ``reason``) on bunched results — this analysis
is the DC/LEBT mirror of the Hofmann diagnostic's bunched-only guard.
"""
from __future__ import annotations

import math

import numpy as np

from linac_gen.analysis.scc import physics as P
from linac_gen.analysis.scc.balance import SCCNotConverged, SelfConsistentSCC
from linac_gen.analysis.scc.constants import (
    M_U,
    BEAM_SPECIES,
    C_LIGHT,
    EPS0,
    GAS_LIBRARY,
    M_E,
    characteristic_current,
    sigma_ionization,
)
from linac_gen.analysis.scc.radial import RadialGrid

__all__ = [
    "GAS_CHOICES",
    "SPECIES_CHOICES",
    "resolve_species",
    "scc_analysis",
    "suggested_scc_deck_lines",
]

GAS_CHOICES = tuple(GAS_LIBRARY)
SPECIES_CHOICES = tuple(BEAM_SPECIES)


def is_continuous_results(results) -> bool:
    """True when the results describe DC/continuous transport.

    Envelope results carry a scalar ``continuous``; MP recorder results
    carry only the per-record ``continuous_at`` list — a genuinely DC MP
    run must not be refused as bunched (adversarial finding F1).
    """
    flag = getattr(results, "continuous", None)
    if flag is not None:
        return bool(flag)
    per = getattr(results, "continuous_at", None)
    if per is not None and len(per):
        return bool(per[0])
    return False


def _arr(results, name):
    v = getattr(results, name, None)
    if v is None:
        return np.zeros(0)
    try:
        return np.asarray(v, float)
    except (TypeError, ValueError):
        return np.zeros(0)

#: End-taper phenomenology (ported verbatim, incl. the magic numbers —
#: stands in for trapped-ion axial escape at the extractor/RFQ repeller).
_ETA_TAPER_FRACTION = 0.15
_ETA_TAPER_FLOOR = 0.25

_SPECIES_ALIASES = {
    "h-": "H-", "hminus": "H-", "h_minus": "H-",
    "h+": "H+", "proton": "H+", "p": "H+",
    "d-": "D-", "dminus": "D-",
    "d+": "D+", "deuteron": "D+",
    "he+": "He+", "helium+": "He+",
}


def resolve_species(name: str):
    """Map a HELIX species string onto the SCC species registry."""
    key = _SPECIES_ALIASES.get(str(name).strip().lower())
    if key is None and name in BEAM_SPECIES:
        key = name
    if key is None:
        raise ValueError(
            f"species {name!r} not supported by the SCC gas model; "
            f"supported: {sorted(BEAM_SPECIES)} (H/D/He, singly charged)")
    return BEAM_SPECIES[key]


def _gas_mix(gas, pressure_mbar, baseline_h2_mbar, temperature_K):
    """Assemble the GasMix exactly as the source tool does."""
    if isinstance(gas, dict):
        bad = sorted(set(gas) - set(GAS_LIBRARY))
        if bad:
            raise ValueError(f"unknown gas(es) {bad} in mixture; "
                             f"choices: {sorted(GAS_LIBRARY)}")
        components = dict(gas)
        components["H2"] = components.get("H2", 0.0) + baseline_h2_mbar
    else:
        components = {"H2": baseline_h2_mbar
                      + (pressure_mbar if gas == "H2" else 0.0)}
        if gas != "H2":
            if gas not in GAS_LIBRARY:
                raise ValueError(f"unknown gas {gas!r}; "
                                 f"choices: {sorted(GAS_LIBRARY)}")
            components[gas] = pressure_mbar
    components = {g: v for g, v in components.items() if v > 0}
    return P.GasMix(components=components, temperature_K=temperature_K)


def _pipe_radius_m(results, lattice, pipe_radius_mm, n):
    """Per-record pipe radius: explicit value > lattice apertures > 60 mm."""
    if pipe_radius_mm is not None:
        return np.full(n, float(pipe_radius_mm) * 1e-3)
    default = 60.0e-3                       # source-tool default pipe
    if lattice is None:
        return np.full(n, default)
    aper = np.full(n, np.nan)
    raw = getattr(results, "element_exit_idx", None)
    exit_idx = ([int(v) for v in raw]
                if raw is not None and len(raw) else [])
    prev = 0
    for k, el in enumerate(getattr(lattice, "elements", [])):
        if k >= len(exit_idx):
            break
        a_mm = float(getattr(el, "aperture", 0.0) or 0.0)
        r1 = int(exit_idx[k])
        if a_mm > 0 and r1 >= prev:
            aper[prev:min(r1 + 1, n)] = a_mm * 1e-3
        prev = r1 + 1
    if not np.isfinite(aper).any():
        return np.full(n, default)
    fill = float(np.nanmax(aper))
    return np.where(np.isfinite(aper), aper, fill)


def _cleared_window(z, z_a, z_b):
    """Smooth flat-top membership window for a CLEARED interval.

    Same tanh flat-top shape as :func:`_taper_profile`, but over an
    arbitrary [z_a, z_b] instead of the whole line — the physical edge
    (clearing-field fringe) is soft, not a step.  w=1 deep inside the
    interval, w=0 well outside.
    """
    L = float(z_b - z_a)
    if L <= 0.0:
        return (np.abs(z - float(z_a)) < 1e-12).astype(float)
    edge = _ETA_TAPER_FRACTION * L
    w = P.solenoid_field(
        z - float(z_a),
        z_center=0.5 * L,
        length=max(L - 2.0 * edge, 0.0),
        peak_B=1.0,
        edge_soft=max(edge / 3.0, 1e-4),
    )
    return np.clip(w, 0.0, 1.0)


def _taper_profile(z):
    """C-infinity tanh flat-top taper (verbatim port of the source tool)."""
    L = float(z[-1] - z[0])
    if L <= 0:
        return np.ones_like(z)
    taper_len = _ETA_TAPER_FRACTION * L
    frac = P.solenoid_field(
        z - z[0],
        z_center=0.5 * L,
        length=max(L - 2.0 * taper_len, 0.0),
        peak_B=1.0,
        edge_soft=max(taper_len / 3.0, 1e-4),
    )
    return _ETA_TAPER_FLOOR + (1.0 - _ETA_TAPER_FLOOR) * np.clip(frac, 0.0, 1.0)


def scc_analysis(
    results, lattice=None, *,
    species: str = "H-",
    gas="H2",
    pressure_mbar: float = 8.0e-6,
    baseline_h2_mbar: float = 1.0e-6,
    temperature_K: float = 300.0,
    mode: str = "computed",
    eta_assumed: float = 0.92,
    trapped_temp_eV: float = 3.0,
    taper: bool = True,
    build_up_us: float | None = None,
    pipe_radius_mm: float | None = None,
    n_slices: int = 9,
    n_radial: int = 240,
    n_cards: int = 8,
    pressure_profile=None,
    current_mA: float | None = None,
    cleared_regions=None,
    cleared_residual: float = 0.0,
    advise_iterate: bool = True,
    should_stop=None,
) -> dict:
    """LEBT gas-neutralisation analysis of a HELIX DC run.

    Parameters mirror the source tool's controls: ``gas`` is a gas name
    with ``pressure_mbar`` (a ``baseline_h2_mbar`` H2 background is always
    added, as in the tool) or a ``{gas: partial_mbar}`` mixture dict;
    ``mode`` = ``"computed"`` (self-consistent Poisson–Boltzmann balance;
    ``eta_assumed`` ignored) or ``"assumed"`` (``eta_assumed`` used
    directly); ``build_up_us`` evaluates the exact f_c relaxation at that
    time (None = steady state); ``pressure_profile`` is an optional list
    of ``(z_metres, multiplier)`` anchors.

    Returns a dict of per-record arrays over the run's ``s`` grid
    (``z_m``, ``sigma_r_mm``, ``eta_ss``, ``fc``, ``tau_scc_us``,
    ``phi_V``, ``survival_gas``, ``loss_frac_per_m``, ``pipe_mm``),
    per-slice balance provenance (``slice_z_m``, ``slice_converged``,
    ``slice_over_neutralised``), scalars (``tau_scc_global_us``,
    ``mean_fc``, ``phi_min_V``, ``strip_mfp_m``, ``mean_ion_mass_amu``,
    ``transmission_gas_pct``, ``perveance``, ``fc_source``), suggested
    ``scc_cards`` (list of ``{z_m, factor, element_index}``), ``notes``,
    and ``reason`` (None when usable).
    """
    out: dict = {"reason": None, "notes": [], "fc_source": None}

    if build_up_us is not None and (
            not math.isfinite(float(build_up_us)) or build_up_us < 0):
        raise ValueError(f"build_up_us must be a non-negative finite time, "
                         f"got {build_up_us!r}")
    if not is_continuous_results(results):
        if (getattr(results, "continuous", None) is None
                and getattr(results, "continuous_at", None) is None):
            out["reason"] = (
                "results carry no continuous-beam marker (e.g. a loaded "
                "archive) — re-run the DC envelope/MP simulation in this "
                "session.")
        else:
            out["reason"] = (
                "bunched-beam results: the SCC gas-neutralisation analysis "
                "applies to DC/continuous (LEBT) transport — for bunched "
                "beams use the Hofmann stability diagnostic instead.")
        return out

    s_mm = np.asarray(getattr(results, "s", []), float)
    sx = np.asarray(getattr(results, "sigma_x", []), float)
    sy = np.asarray(getattr(results, "sigma_y", []), float)
    n = s_mm.size
    if n < 2 or sx.size != n or sy.size != n:
        out["reason"] = "results carry no usable envelope (s/sigma_x/sigma_y)"
        return out

    sp = resolve_species(species)
    w_kin = np.asarray(getattr(results, "ref_w_kin", []), float)
    if w_kin.size == 0:
        out["reason"] = "results carry no reference energy (ref_w_kin)"
        return out
    T_keV = float(w_kin[0]) * 1e3
    beta, _, _ = P.kinematic(sp, T_keV)
    v_beam = beta * C_LIGHT

    z = (s_mm - s_mm[0]) * 1e-3                        # m, from line start
    sigma_r = np.sqrt(np.maximum(sx, 1e-9) * np.maximum(sy, 1e-9)) * 1e-3
    pipe = _pipe_radius_m(results, lattice, pipe_radius_mm, n)
    if n < 20:
        # Every profile here is resolved on the RUN's record grid, so a
        # one-record-per-element run gives a coarse f_c(z) — and the
        # suggested card factors are averages over those coarse bins.
        out["notes"].append(
            f"only {n} records: enable record_substeps (Numerics tab) for a "
            "smoothly resolved f_c(z) and finer card factors")

    # Beam current: the RUN's own record wins (results.current_mA on
    # envelope and loaded-HDF5 results, then results.beam.current on MP
    # results with the beam attached) — the beam sizes being analysed
    # were computed at that current, so analysing them at any other
    # current is silently inconsistent.  The explicit parameter (the
    # GUI/tool pass the Beam-tab value) is the fallback for MP recorder
    # results, which carry NO current attribute — found live: an MP run
    # otherwise computed I=0 and the balance returned f_c=0 everywhere.
    cleared_z: list[tuple[float, float]] = []
    if cleared_regions:
        # Ion-cleared stretches (e.g. a biased chopper: the PXIE-style
        # un-neutralised section) — element ranges, INCLUSIVE, indexed
        # against the lattice the results were tracked with.
        if lattice is None:
            out["reason"] = ("cleared_regions are element ranges — pass "
                             "the lattice alongside the results")
            return out
        if not (0.0 <= float(cleared_residual) <= 1.0):
            raise ValueError(f"cleared_residual must be in [0, 1], "
                             f"got {cleared_residual!r}")
        from linac_gen.analysis.phase_advance import element_record_span
        n_el = len(lattice.elements)
        exit_map = getattr(results, "element_exit_idx", None)
        if ((exit_map is None or len(exit_map) == 0) and n > n_el + 1):
            # identity fallback would mislocate regions on a substep
            # grid (verified: [[2,2]] landed at z=0.04-0.06 m) — refuse
            out["reason"] = (
                "cleared_regions need the results' element_exit_idx map "
                "to locate elements on the record grid — these results "
                "carry none (re-run with a current HELIX)")
            return out
        for reg in cleared_regions:
            try:
                e0f, e1f = float(reg[0]), float(reg[1])
            except (TypeError, ValueError, IndexError, KeyError):
                raise ValueError("each cleared region must be a "
                                 f"[start, end] element pair, got {reg!r}")
            if e0f != int(e0f) or e1f != int(e1f):
                raise ValueError("cleared region bounds must be whole "
                                 f"element indices, got {reg!r}")
            e0, e1 = int(e0f), int(e1f)
            if not (0 <= e0 <= e1 < n_el):
                raise ValueError(f"cleared region [{e0}, {e1}] outside "
                                 f"the lattice (0..{n_el - 1})")
            r0, r1 = element_record_span(results, e0, e1 + 1)
            r0 = max(0, min(int(r0), n - 1))
            r1 = max(0, min(int(r1), n - 1))
            cleared_z.append((float(z[r0]), float(z[r1])))
    run_mA = getattr(results, "current_mA", None)
    if run_mA is None:
        run_mA = getattr(getattr(results, "beam", None), "current", None)
    if run_mA is not None and math.isfinite(float(run_mA)):
        if (current_mA is not None and math.isfinite(float(current_mA))
                and abs(float(current_mA) - float(run_mA)) > 1e-9):
            out["notes"].append(
                f"analysis uses the run's recorded current "
                f"{float(run_mA):g} mA; the supplied value "
                f"{float(current_mA):g} mA differs and was ignored")
        current_mA = run_mA
    if current_mA is None or not math.isfinite(float(current_mA)):
        out["reason"] = (
            "no beam current available — results carry neither current_mA "
            "nor an attached beam, and no current_mA= was supplied.")
        return out
    if float(current_mA) == 0.0:
        # a genuine 0 mA run is a valid degenerate case, not missing data
        out["notes"].append(
            "beam current is 0 mA — space-charge compensation is "
            "trivially zero at zero current")
    I0_A = abs(float(current_mA)) * 1e-3
    trans = _arr(results, "transmission")
    if trans.size == n and trans[0] > 0:
        i_frac = np.clip(trans / trans[0], 0.0, 1.0)
    else:
        i_frac = np.ones(n)

    mix = _gas_mix(gas, pressure_mbar, baseline_h2_mbar, temperature_K)
    tau_global = P.compensation_time(mix, sp, T_keV)
    inv_mfp = P.beam_loss_rate_per_length(mix, sp, T_keV)

    # pressure-profile multiplier (z anchors in METRES; explicit by design)
    if pressure_profile:
        pz = np.asarray([p_[0] for p_ in pressure_profile], float)
        pm = np.asarray([p_[1] for p_ in pressure_profile], float)
        mult = np.interp(z, pz, pm, left=pm[0], right=pm[-1])
    else:
        mult = np.ones(n)
    tau_z = (tau_global / np.maximum(mult, 1e-9)
             if math.isfinite(tau_global) else np.full(n, math.inf))
    inv_mfp_z = inv_mfp * mult

    # cumulative optical depth -> gas survival and local current
    seg = 0.5 * (inv_mfp_z[1:] + inv_mfp_z[:-1]) * np.diff(z)
    od = np.concatenate(([0.0], np.cumsum(seg)))
    survival = np.exp(-od)
    I_local = I0_A * i_frac * survival

    # ionisation-weighted mean trapped-ion mass (mixture-correct)
    nd = mix.number_densities()
    w_sum = m_sum = 0.0
    for gname, n_g in nd.items():
        w = n_g * sigma_ionization(gname, sp, T_keV)
        w_sum += w
        m_sum += w * GAS_LIBRARY[gname].ion_mass_kg()
    mean_ion_mass = (m_sum / w_sum) if w_sum > 0 else GAS_LIBRARY["H2"].ion_mass_kg()

    # ---- steady-state neutralisation eta_ss(z) ----
    slice_z = np.linspace(z[0], z[-1], max(int(n_slices), 2))
    slice_conv = np.ones(slice_z.size, dtype=bool)
    slice_over = np.zeros(slice_z.size, dtype=bool)
    if mode == "assumed":
        eta_slices = np.full(slice_z.size, float(eta_assumed))
        out["fc_source"] = f"assumed eta = {eta_assumed:g}"
    elif mode == "computed":
        beam_sign = math.copysign(1.0, sp.charge)
        trapped_mass = mean_ion_mass if sp.charge < 0 else M_E
        eta_slices = np.empty(slice_z.size)
        L = float(z[-1] - z[0]) if z[-1] > z[0] else 1.0
        for i, zi in enumerate(slice_z):
            if should_stop is not None and should_stop():
                raise RuntimeError("SCC analysis aborted")
            sig_i = float(np.interp(zi, z, sigma_r))
            pipe_i = float(np.interp(zi, z, pipe))
            I_i = float(np.interp(zi, z, I_local))
            mult_i = float(np.interp(zi, z, mult))
            a = float(np.clip(P.rms_to_edge_radius(sig_i), 1e-4,
                              0.95 * pipe_i))
            grid = RadialGrid(n=int(n_radial), r_pipe_m=pipe_i)
            lam = math.copysign(I_i / max(v_beam, 1.0), sp.charge)
            rho = np.where(grid.r <= a, lam / (math.pi * a * a), 0.0)
            total = float((rho * grid.cell_area).sum())
            if total != 0.0:
                rho = rho * (lam / total)   # exact discrete line charge
            rate = sum(n_g * sigma_ionization(g, sp, T_keV)
                       for g, n_g in nd.items()) * I_i / abs(sp.charge)
            try:
                res = SelfConsistentSCC(
                    grid, T_trapped_eV=float(trapped_temp_eV)).solve(
                    rho, beam_sign=beam_sign,
                    production_per_m_per_s=rate * mult_i,
                    lebt_length_m=L, trapped_mass_kg=trapped_mass)
                eta_slices[i] = res.fc
                slice_over[i] = bool(res.over_neutralised)
            except SCCNotConverged:
                eta_slices[i] = float(eta_assumed)
                slice_conv[i] = False
        out["fc_source"] = "computed balance (radial Poisson–Boltzmann)"
        if not slice_conv.all():
            out["notes"].append(
                f"{int((~slice_conv).sum())}/{slice_conv.size} slices did "
                f"not converge — assumed eta = {eta_assumed:g} substituted "
                "there")
        if slice_over.any():
            out["notes"].append(
                f"{int(slice_over.sum())}/{slice_over.size} slices "
                "over-neutralised (f_c clamped at 1; model caveat above "
                "~1e-5 mbar)")
    else:
        raise ValueError(f"mode must be 'computed' or 'assumed', got {mode!r}")

    eta_ss = np.interp(z, slice_z, eta_slices)
    if taper:
        eta_ss = eta_ss * _taper_profile(z)
        out["notes"].append(
            "end taper applied (phenomenological extractor/repeller escape: "
            f"floor {_ETA_TAPER_FLOOR}, first/last "
            f"{_ETA_TAPER_FRACTION:.0%} of the line)")
    if cleared_z:
        # AFTER the end-taper, BEFORE build-up: clearing overrides the
        # natural balance wherever the window is up, taper included.
        w = np.zeros_like(z)
        unresolved = []
        for (z_a, z_b), reg in zip(cleared_z, cleared_regions):
            wi = _cleared_window(z, z_a, z_b)
            if float(wi.max()) < 0.9:
                # the record grid never samples the window's core (few
                # records per element, or a sub-mm region): the clearing
                # is a near-no-op and must not claim otherwise
                unresolved.append([int(reg[0]), int(reg[1])])
            w = np.maximum(w, wi)
        eta_ss = eta_ss * (1.0 - w) + float(cleared_residual) * w
        out["cleared_regions"] = [[int(r[0]), int(r[1])]
                                  for r in cleared_regions]
        out["cleared_residual"] = float(cleared_residual)
        out["cleared_resolved"] = not unresolved
        out["notes"].append(
            f"cleared region(s) {out['cleared_regions']} forced toward "
            f"residual f_c = {float(cleared_residual):g} (ion clearing — "
            "PXIE-style un-neutralised section)")
        if unresolved:
            out["notes"].append(
                f"cleared region(s) {unresolved} UNRESOLVED on this "
                "record grid — the window never reaches its core, so the "
                "clearing had little effect; enable record_substeps "
                "(Numerics tab) or use Iterate")

    # ---- f_c: steady state, or exact build-up at t = build_up_us ----
    # NOTE: the build-up starts from f_c(0) = 0; the source tool's live
    # engine warm-starts at 0.6*eta, so its transient trajectory differs
    # even though the relaxation ODE is identical.
    if build_up_us is None:
        fc = eta_ss.copy()
    else:
        t_s = float(build_up_us) * 1e-6
        with np.errstate(over="ignore"):
            relax = np.where(np.isfinite(tau_z) & (tau_z > 0),
                             1.0 - np.exp(-t_s / tau_z), 0.0)
        fc = eta_ss * relax
    fc = np.clip(fc, 0.0, 1.0)

    # ---- on-axis potential with local compensation (verbatim port) ----
    r_edge = np.maximum(P.rms_to_edge_radius(sigma_r), 1e-9)
    log_ratio = np.log(np.maximum(pipe / r_edge, 1.0))
    sign = math.copysign(1.0, sp.charge)
    phi_uncomp = (sign * I_local / (4.0 * math.pi * EPS0 * beta * C_LIGHT)
                  * (1.0 + 2.0 * log_ratio))
    phi = phi_uncomp * (1.0 - fc)

    # ---- suggested SpaceChargeComp cards ----
    cards = []
    edges = np.linspace(z[0], z[-1], max(int(n_cards), 1) + 1)
    _raw_idx = getattr(results, "element_exit_idx", None)
    exit_idx = ([int(v) for v in _raw_idx]
                if _raw_idx is not None and len(_raw_idx) else [])
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m_ = (z >= a_) & (z <= b_)
        if not m_.any():
            continue
        z_card = float(a_)
        elem = None
        if exit_idx:
            r0 = int(np.searchsorted(z, z_card))
            elem = int(np.searchsorted(np.asarray(exit_idx), r0))
        cards.append({"z_m": z_card,
                      "factor": round(float(np.clip(np.mean(fc[m_]),
                                                    0.0, 1.0)), 4),
                      "element_index": elem})

    out.update({
        "z_m": z, "sigma_r_mm": sigma_r * 1e3, "pipe_mm": pipe * 1e3,
        "eta_ss": eta_ss, "fc": fc,
        "tau_scc_us": tau_z * 1e6, "phi_V": phi,
        "survival_gas": survival,
        "loss_frac_per_m": inv_mfp_z,
        "slice_z_m": slice_z, "slice_eta": eta_slices,
        "slice_converged": slice_conv, "slice_over_neutralised": slice_over,
        "tau_scc_global_us": (tau_global * 1e6
                              if math.isfinite(tau_global) else float("inf")),
        "mean_fc": float(np.mean(fc)),
        "phi_min_V": float(np.min(phi)), "phi_max_V": float(np.max(phi)),
        "strip_mfp_m": (1.0 / inv_mfp) if inv_mfp > 0 else float("inf"),
        "mean_ion_mass_amu": mean_ion_mass / M_U,
        "transmission_gas_pct": float(survival[-1] * 100.0),
        "perveance": P.generalized_perveance(sp, T_keV, I0_A),
        "I0_char_A": characteristic_current(sp),
        "beta": beta, "T_keV": T_keV,
        "species": sp.name, "gas_mix_mbar": dict(mix.components),
        "build_up_us": build_up_us,
        "scc_cards": cards,
    })
    if advise_iterate:
        # Appended LAST: the GUI surfaces notes[0] and must keep showing
        # the more actionable warnings (coarse grid, current mismatch).
        out["notes"].append(
            "one-shot analysis: the beam sizes were tracked at the run's "
            "own space-charge state, not at the suggested factors — "
            "iterate to self-consistency (Iterate button or "
            "scc_self_consistent) for final card values")
    return out


def suggested_scc_deck_lines(analysis: dict) -> list[str]:
    """Render the suggested cards as TraceWin-dialect deck lines."""
    lines = [f"; SCC factors from linac_gen.analysis.scc "
             f"({analysis.get('fc_source')})"]
    for c in analysis.get("scc_cards", []):
        lines.append(f"SPACE_CHARGE_COMP {c['factor']:.4f}   "
                     f"; at z = {c['z_m']:.3f} m")
    return lines
