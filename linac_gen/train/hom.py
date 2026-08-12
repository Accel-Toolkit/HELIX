"""Dipole-HOM long-range wakes / cumulative BBU — ONE implementation,
two drivers (M4).

`HomManager` owns the per-cavity per-mode transverse phasor state (which
lives in the M2 `CavityStateRegistry`, `_CavityState.hom`) and exposes the
element entry hook consumed by the tracked driver (TrainRunner); the fast
full-pulse driver advances the SAME state through `hom_passage` — the
shared per-passage primitive — exactly as M3 reuses
`BeamLoadingManager.bunch_passage`.  `delayen_bbu_reference` is the
module's independent test anchor: the standard cumulative-BBU recursion
(Delayen's formulation) with the wake sum written out explicitly, no
phasor recursion, so tests/train/test_hom.py can arbitrate the
conventions below against an independent code path.

TRANSVERSE R/Q CONVENTION (pinned; the Delayen anchor is the arbiter)
---------------------------------------------------------------------
Each dipole mode carries a transverse R/Q in **Ohm, per cavity**, in the
LINAC (accelerator) convention

    (R/Q)_t = |V_par(r)|^2 / (omega U (k r)^2),        k = omega / c,

i.e. the mode's longitudinal voltage grows linearly off axis,
V_par(x) = V' x, with (R/Q)_t = V'^2 c^2 / (omega^3 U).  Panofsky-Wenzel
then ties the transverse kick voltage to the same amplitude,
V_t = (c/omega) V', 90 degrees out of RF phase with V_par.  A point bunch
of SIGNED charge q [C] crossing the cavity at offset u along the mode's
polarization axis leaves, for a test charge dt later,

    V_t(dt) = q u kappa e^{-dt/tau} sin(omega dt),
    kappa   = omega^2 (R/Q)_t / (2 c)   [V / (C m)],
    tau     = 2 Q_L / omega,

directed ALONG the polarization axis, in the direction of the source
offset for 0 < omega dt < pi — the destabilizing first lobe of the
dipole wake, which is what makes cumulative BBU GROW (anchored by
test_kick_sign_growth).  The momentum kick on a bunch of charge state Z:

    Delta u' [mrad] = 1e3 * Z * V_t[MV] / (betagamma * m [MeV]).

With the excitation computed from the SIGNED source charge (q = Z |q_b|)
the deflection scales as Z^2 — species-sign independent, as any
two-particle wake must be.  The kick voltage a bunch sees is the SUM
over all modes of the cavity.

No transverse self-kick: W_t(0+) = 0 (Panofsky-Wenzel), so — unlike the
fundamental mode's half-self-kick — the exciting bunch receives nothing
from its own passage.  `hom_passage` kicks from the stored phasor first
and adds the excitation after; the order is provably indifferent because
a bunch's own excitation enters purely real (Im = 0) in its own arrival
frame.

Bookkeeping (shared with M2/M3): phasors advance between bunch arrivals
on the train slot clock (T_slot = 1/f_bunch); arrival-time differences
at EVERY cavity equal slot differences x T_slot (rigid time-of-flight —
the M3 no-TOF-feedback approximation).  Each mode advances at its OWN
(i omega - 1/tau): there is no rotating frame here — `w` is the complex
envelope in absolute time, V_t(now) = Im[w].

Offsets are the lab-frame beam centroid at cavity ENTRY (tracked mode:
mean over alive particles; fast mode: the per-bunch centroid state).
Cavity transverse misalignment is NOT subtracted from the offset in v1
(the mode axis is taken as the lab axis; deferred with the
random-misalignment Delayen forms).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from linac_gen.core.constants import C_LIGHT
from linac_gen.pic.macrocharge import macro_charge_for
from linac_gen.train.cavity_state import CavityStateRegistry


@dataclass
class HOMMode:
    """One dipole higher-order mode of a cavity (sidecar entry).

    ``r_over_q_t`` is in Ohm per cavity, transverse linac convention —
    see the module docstring for the exact definition and the induced
    kick-voltage formula it implies.  ``polarization_deg`` is the mode's
    deflection axis in the transverse plane (0 = x, 90 = y).
    """
    f_MHz: float
    r_over_q_t: float                  # Ohm (transverse linac convention)
    q_loaded: float
    polarization_deg: float = 0.0

    def __post_init__(self):
        if self.f_MHz <= 0.0:
            raise ValueError(f"HOM mode f_MHz must be > 0, got {self.f_MHz}")
        if self.q_loaded <= 0.0:
            raise ValueError(
                f"HOM mode q_loaded must be > 0, got {self.q_loaded}")
        if self.r_over_q_t < 0.0:
            raise ValueError(
                f"HOM mode r_over_q_t must be >= 0, got {self.r_over_q_t}")

    @classmethod
    def from_dict(cls, d: dict) -> "HOMMode":
        missing = [k for k in ("f_MHz", "r_over_q_t", "q_loaded")
                   if k not in d]
        if missing:
            raise ValueError(
                "hom mode entry missing required key(s): "
                + ", ".join(missing))
        return cls(f_MHz=float(d["f_MHz"]),
                   r_over_q_t=float(d["r_over_q_t"]),
                   q_loaded=float(d["q_loaded"]),
                   polarization_deg=float(d.get("polarization_deg", 0.0)))

    # ---- derived constants (single source for both drivers + tests) ----
    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.f_MHz * 1e6

    @property
    def tau_s(self) -> float:
        return 2.0 * self.q_loaded / self.omega

    @property
    def kappa_V_per_C_m(self) -> float:
        """kappa = omega^2 (R/Q)_t / (2 c)  [V / (C m)]."""
        return self.omega ** 2 * self.r_over_q_t / (2.0 * C_LIGHT)


@dataclass
class _HomState:
    """Mutable per-mode wake phasor (one per mode per cavity; a mode's
    polarization axis IS its plane, so orthogonal polarizations are two
    modes).  ``w`` is the transverse kick voltage phasor in MV: the kick
    voltage a test bunch arriving NOW would see is Im[w]."""
    mode: HOMMode
    w: complex = 0j                    # MV, absolute-time complex envelope


class HomManager:
    """Dipole-HOM physics over the shared cavity registry.

    Mirrors BeamLoadingManager's lifecycle (design pass, per-slot clock,
    entry hook) but touches NO element attributes — kicks go straight to
    the beam (tracked) or the centroid state (fast), so there is nothing
    to compose with or restore at teardown.  Slot bookkeeping is separate
    from the fundamental mode's (``_CavityState.hom_last_slot``) so both
    managers can share one registry without clobbering each other's decay
    clocks.
    """

    def __init__(self, registry: CavityStateRegistry,
                 bunch_frequency_MHz: float):
        self.reg = registry
        self.T_slot_s = 1.0 / (bunch_frequency_MHz * 1e6)
        self.current_slot = 0
        self._design_mode = False
        # Tracked-mode ledger of APPLIED kicks (M6 replay):
        # (slot, element_index, name) -> (dxp_mrad, dyp_mrad) as applied
        # to the alive particles by entry_hook.  Pure bookkeeping — feeds
        # TrainResults.applied_hom for the lossless replay construction.
        self.applied: dict = {}
        # Tracked-mode per-mode wake history (M7 persistence): (slot,
        # element_index, name) -> tuple of per-mode ``w`` AFTER this
        # bunch's excitation (complex MV envelopes, mode order =
        # ``_CavityState.hom`` order).  Hook-only bookkeeping — the fast
        # driver keeps its own stride-decimated histories.
        self.w_after: dict = {}

    # ------------------------------------------------------------ design
    def begin_design_pass(self):
        self._design_mode = True

    def end_design_pass(self, lattice):
        """Validate the design records: every HOM-bound cavity must have
        been reached by a live design beam (fast mode transports the
        per-bunch centroid on the design trajectory; a train whose design
        beam dies upstream of a HOM cavity is a broken study — refuse
        loudly rather than silently skip the cavity)."""
        self._design_mode = False
        problems = []
        for (idx, name), st in self.reg.items():
            if not st.hom:
                continue
            if st.centroid_design is None or not math.isfinite(st.bg_design):
                problems.append(
                    f"{name}: design pass never reached this cavity with "
                    "live particles — HOM excitation offset undefined")
        if problems:
            raise ValueError(
                "hom: design-pass records missing:\n  "
                + "\n  ".join(problems))

    # ------------------------------------------------------------- hooks
    def entry_hook(self, element, index, beam):
        name = getattr(element, "name", "")
        st = self.reg.get(index, name)
        if st is None or not st.hom:
            return
        alive = beam.alive_mask
        n_alive = int(beam.n_alive)
        if self._design_mode:
            st.s_design_mm = float(beam.ref.s)
            st.bg_design = float(beam.ref.bg)
            if n_alive:
                p = beam.particles
                st.centroid_design = np.array(
                    [float(np.mean(p[alive, 0])), float(np.mean(p[alive, 1])),
                     float(np.mean(p[alive, 2])), float(np.mean(p[alive, 3]))])
            return
        if n_alive == 0:
            return                      # no charge: no kick, no excitation
        p = beam.particles
        x_mm = float(np.mean(p[alive, 0]))
        y_mm = float(np.mean(p[alive, 2]))
        z = float(beam.ref.species.charge)
        q_signed = macro_charge_for(beam) * n_alive * z
        dxp, dyp = self.hom_passage(
            st, self.current_slot, q_signed, x_mm, y_mm,
            float(beam.ref.bg), float(beam.ref.species.mass), z)
        self.applied[(self.current_slot, index, name)] = (dxp, dyp)
        self.w_after[(self.current_slot, index, name)] = \
            tuple(h.w for h in st.hom)
        if dxp != 0.0:
            p[alive, 1] += dxp
        if dyp != 0.0:
            p[alive, 3] += dyp

    # ------------------------------------------------- per-slot advance
    def begin_bunch(self, slot: int):
        self.current_slot = int(slot)

    # ------------------------------------------------ shared primitive
    def hom_passage(self, st, slot: int, q_signed_C: float,
                    x_mm: float, y_mm: float, bg: float, mass_MeV: float,
                    z_charge: float):
        """THE per-passage primitive (both drivers): decay/rotate every
        mode phasor to this bunch's arrival, read the transverse kick,
        then add this bunch's excitation.  Returns (dxp, dyp) in mrad —
        the caller applies them (tracked: to all alive particles' slopes;
        fast: to the centroid perturbation state)."""
        if st.hom_last_slot is not None:
            dt = (slot - st.hom_last_slot) * self.T_slot_s
            if dt > 0.0:
                for h in st.hom:
                    if h.w != 0j:
                        m = h.mode
                        rot = m.omega * dt
                        h.w *= math.exp(-dt / m.tau_s) * complex(
                            math.cos(rot), math.sin(rot))
        st.hom_last_slot = slot
        pc = bg * mass_MeV              # MeV
        dxp = dyp = 0.0
        for h in st.hom:
            m = h.mode
            a = math.radians(m.polarization_deg)
            ca, sa = math.cos(a), math.sin(a)
            kick_mrad = 1e3 * z_charge * h.w.imag / pc
            dxp += kick_mrad * ca
            dyp += kick_mrad * sa
            u_m = (x_mm * ca + y_mm * sa) * 1e-3
            # Purely real in the bunch's own frame (no transverse
            # self-kick, W_t(0+) = 0) — see module docstring.
            h.w += q_signed_C * u_m * m.kappa_V_per_C_m * 1e-6
        return dxp, dyp


# ---------------------------------------------------------------------------
# Independent reference: cumulative BBU a la Delayen (explicit wake sums)
# ---------------------------------------------------------------------------
def delayen_bbu_reference(slots, T_slot_s: float, q_signed_C: float,
                          z_charge: float, modes, cavity_s_mm,
                          entry_centroid, bg: float,
                          mass_MeV: float) -> np.ndarray:
    """Cumulative-BBU recursion in Delayen's formulation (PRST-AB 6,
    084402: transverse displacement of an arbitrary bunch-time profile
    driven by explicit dipole-wake sums over the train, chained through a
    focusing-free line of identical thin cavities separated by drifts).

    INDEPENDENT code path from `HomManager.hom_passage`, for the test
    anchor: the wake seen by bunch n is evaluated as the explicit sum

        V_t(n) = sum_{k < n} q u_k kappa e^{-(t_n-t_k)/tau}
                                        sin(omega (t_n - t_k))   [per mode]

    — no complex phasor recursion anywhere — so agreement with the fast
    driver at ~1e-9 (tests/train/test_hom.py) validates the phasor
    decay/rotation algebra, the excitation and kick constants, and the
    centroid transport bookkeeping against the same physical inputs.

    Model contract (the reduced BBU lattice): N thin cavities at
    ``cavity_s_mm`` (entry positions, mm), pure drifts between them, no
    focusing, no acceleration (``bg`` constant), every bunch injected on
    the same ``entry_centroid`` = (x mm, xp mrad, y mm, yp mrad) at the
    FIRST cavity.  ``slots`` are the filled slot indices (arbitrary
    profile), ``q_signed_C`` the signed bunch charge, ``modes`` a list of
    HOMMode shared by all cavities.

    Returns (n_cav, n_bunch, 4): post-kick (x, xp, y, yp) centroids at
    every cavity, matching the fast driver's per-cavity records.
    """
    slots = np.asarray(slots, dtype=float)
    times = slots * T_slot_s
    nb = times.size
    ncav = len(cavity_s_mm)
    x = np.full(nb, float(entry_centroid[0]))
    xp = np.full(nb, float(entry_centroid[1]))
    y = np.full(nb, float(entry_centroid[2]))
    yp = np.full(nb, float(entry_centroid[3]))
    pc = bg * mass_MeV
    om = [m.omega for m in modes]
    tau = [m.tau_s for m in modes]
    kap = [m.kappa_V_per_C_m for m in modes]
    ca = [math.cos(math.radians(m.polarization_deg)) for m in modes]
    sa = [math.sin(math.radians(m.polarization_deg)) for m in modes]
    out = np.empty((ncav, nb, 4))
    for mcav in range(ncav):
        if mcav:
            L_m = (float(cavity_s_mm[mcav])
                   - float(cavity_s_mm[mcav - 1])) * 1e-3
            x = x + xp * L_m            # mm += mrad * m
            y = y + yp * L_m
        u_hist = np.empty((len(modes), nb))
        for n in range(nb):
            for j in range(len(modes)):
                if n:
                    dt = times[n] - times[:n]
                    vt_MV = (q_signed_C * kap[j] * 1e-6) * float(
                        np.sum(u_hist[j, :n] * np.exp(-dt / tau[j])
                               * np.sin(om[j] * dt)))
                    kick = 1e3 * z_charge * vt_MV / pc
                    xp[n] += kick * ca[j]
                    yp[n] += kick * sa[j]
                # Source offset in metres; position is untouched by the
                # kick, so recording before/after is equivalent.
                u_hist[j, n] = (x[n] * ca[j] + y[n] * sa[j]) * 1e-3
        out[mcav, :, 0] = x
        out[mcav, :, 1] = xp
        out[mcav, :, 2] = y
        out[mcav, :, 3] = yp
    return out
