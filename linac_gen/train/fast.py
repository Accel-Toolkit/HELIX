"""FastPulseRunner — the fast full-pulse driver (M3): a per-slot phasor
recursion over the FULL bunch pattern (~89k slots in seconds) plus a
per-bunch centroid exit-energy ledger.

One-physics-implementation invariant (plan §4): the per-cavity phasor
advance is ``BeamLoadingManager.bunch_passage`` — the same code the
tracked driver's element hooks are built from — so the fast and tracked
modes cannot drift apart.  tests/train/test_fast.py anchors the
equivalence: phasor histories, applied slot values and per-bunch exit
energies of a lossless tracked train match to float tolerance.

The DESIGN PASS is one tracked pass through the M2 machinery
(``begin_design_pass``/``end_design_pass``, psi-pinning, prior
snapshot): it measures per-cavity dW_design and derives V_design/phi_s
exactly as the tracked mode does.  Fast mode runs it even with no bound
cavities — the ledger needs W_design_exit.

Model scope (the documented approximations):

* **Centroid energy ledger** — per-bunch exit energy
      W_exit(k) = W_design_exit
                + sum_cav [ A_cav (1+vr_k) cos(phi_s + po_k) - dW_design ]
  with (vr_k, po_k) the cavity's registry-frame loading for bunch k
  (``bunch_passage`` output, i.e. q_e|V_tot,k|cos(phi_eff,k) minus the
  design term) and A_cav = dW_design / cos(phi_s) the measured gain
  amplitude — exact for elements whose energy gain is separable as
  V_eff cos(phi_eff) (thin RFGap), first order for field maps / NCells.
  When |cos(phi_s)| < 1e-6 (pure buncher — only reachable with a
  sidecar ``v_design_MV``) the ledger falls back to
  A_cav = q_species * v_design_MV, reading the sidecar voltage as the
  effective (q V T) voltage.
* **No time-of-flight feedback**: a bunch's energy error at one cavity
  does not shift its arrival phase at downstream cavities.  (For
  p_flag=0 RFGaps the tracked mode has no such coupling either — the
  phase is the prescribed synchronous phase — so equivalence there is
  exact; see the FREQ-jump anchor test.)
* **Loss-free**: every bunch carries charge I/f_bunch (the tracked mode
  uses macro_charge_for(beam) * n_alive — identical while nothing is
  lost).  Lossy trains need the tracked or (M6) hybrid mode.
* **Slot clock vs cavity clock**: the recursion advances on the bunch
  slot clock (T_slot = 1/f_bunch from TrainConfig.bunch_frequency_MHz);
  each cavity's decay constant tau = 2 Q_L/omega and induced-voltage
  magnitude use its OWN omega (bound from element.frequency at sidecar
  match time).  The rotating frame absorbs the carrier: for f_cav an
  integer multiple of f_bunch the frame advance between slots is a
  whole number of RF turns — identical to the tracked M2 bookkeeping,
  where only ``detuning_Hz`` rotates the phasor between arrivals.
* **Transverse centroid (M4, hom channel)**: each bunch carries a
  4-vector perturbation (dx, dxp, dy, dyp) about the DESIGN-PASS
  centroid trajectory; the design pass measures (mean of alive
  particles) the centroid, the entry s and betagamma at every HOM-bound
  cavity.  HOM kicks (``HomManager.hom_passage`` — the same primitive
  the tracked entry hook applies to real particles) land on the
  perturbation slopes; between consecutive HOM cavities the
  perturbation is transported as a PURE DRIFT over the design-pass
  entry-to-entry distance (documented v1 choice: the design part of the
  centroid is exact by construction, only the PERTURBATION ignores
  focusing / adiabatic damping / RF defocusing between and inside
  cavities — exact for the drift-separated thin-cavity reduced lattice
  of the Delayen anchor, first order elsewhere; per-section transfer
  maps are the M6+ upgrade path).  Kick conversion uses the design-pass
  betagamma at each cavity (per-bunch loading energy shifts are small
  and ignored here; with hom on and loading off the tracked mode has
  exactly the design betagamma too).
* ``beam_loading=False`` with ``mode="fast"`` and hom off runs TRIVIALLY
  (documented choice, tested): the recursion has no cavities, every
  bunch exits at the design energy — a cheap pattern/machinery sanity
  path rather than a refusal.  With hom ON and loading off the
  transverse recursion runs while every bunch still exits at the design
  energy (the dipole-wake energy loss is second order in offset and not
  modelled).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from linac_gen.pic.macrocharge import macro_charge_coulombs
from linac_gen.train.driver import TrainRunner


@dataclass(eq=False)     # ndarray fields: identity compare, not elementwise
class FastCavityRecord:
    """Per-cavity fast-mode product: design constants + per-bunch
    histories (decimated by ``history_stride``; see FastPulseSummary)."""
    element_index: int
    name: str
    frequency_MHz: float
    v_design_MV: float
    phi_s_deg: float
    dw_design_MeV: float
    r_over_q: float
    q_loaded: float
    detuning_Hz: float
    # Applied slot values per recorded bunch — PRIOR-COMPOSED, i.e.
    # exactly what the tracked entry hook writes onto the element.
    # All-zero (and v_beam_MV all-0j, v_design/phi_s/dw inert) when the
    # beam_loading channel is off (hom-only cavity): the tracked hook
    # then writes nothing and any user-set prior simply stays in place.
    voltage_rel: np.ndarray = field(repr=False)
    phase_offset_deg: np.ndarray = field(repr=False)
    # Beam-induced phasor (MV, rotating frame) AFTER the bunch's own
    # full kick — the state the next arrival will decay from.
    v_beam_MV: np.ndarray = field(repr=False)
    # ---- dipole-HOM channel (M4); None/() when the cavity carries no
    # hom_modes or the hom channel is off.  ``centroid`` is the full
    # per-recorded-bunch centroid (x mm, xp mrad, y mm, yp mrad) at
    # cavity ENTRY, post-kick (design + accumulated perturbation);
    # ``hom_w`` is the (n_modes, n_rec) complex kick-voltage phasor
    # AFTER the bunch's own excitation — the state the next arrival
    # decays from (mirrors v_beam_MV).
    centroid: np.ndarray = field(repr=False, default=None)
    hom_w: np.ndarray = field(repr=False, default=None)
    hom_modes: tuple = ()


@dataclass(eq=False)     # ndarray fields: identity compare, not elementwise
class FastPulseSummary:
    """Fast-mode summary stored on TrainResults.fast (HDF5: train/fast/).

    ``slot``/``w_exit_MeV`` cover EVERY processed bunch; the per-cavity
    histories are stored every ``history_stride``-th bunch (the slots
    they belong to are in ``history_slot``) so full-pulse runs on
    many-cavity lattices can bound their footprint.
    """
    slot: np.ndarray
    w_exit_MeV: np.ndarray
    w_design_exit_MeV: float
    charge_per_bunch_C: float
    history_stride: int
    history_slot: np.ndarray
    cavities: list[FastCavityRecord]
    # Abort truncation: True when the recursion stopped (should_abort)
    # before the last filled slot — ``slot``/``w_exit_MeV`` then cover
    # only the processed prefix (still a loadable partial result).
    truncated: bool = False

    def write_hdf5(self, grp) -> None:
        grp.attrs["w_design_exit_MeV"] = float(self.w_design_exit_MeV)
        grp.attrs["charge_per_bunch_C"] = float(self.charge_per_bunch_C)
        grp.attrs["history_stride"] = int(self.history_stride)
        grp.attrs["n_bunches"] = int(self.slot.size)
        grp.attrs["truncated"] = bool(self.truncated)
        grp.create_dataset("slot", data=np.asarray(self.slot, np.int64))
        grp.create_dataset("w_exit_MeV", data=np.asarray(self.w_exit_MeV,
                                                         float))
        grp.create_dataset("history_slot",
                           data=np.asarray(self.history_slot, np.int64))
        cg = grp.create_group("cavities")
        for rec in self.cavities:
            g = cg.create_group(f"c_{rec.element_index:04d}")
            for a in ("name", "element_index", "frequency_MHz",
                      "v_design_MV", "phi_s_deg", "dw_design_MeV",
                      "r_over_q", "q_loaded", "detuning_Hz"):
                g.attrs[a] = getattr(rec, a)
            g.create_dataset("voltage_rel", data=rec.voltage_rel)
            g.create_dataset("phase_offset_deg", data=rec.phase_offset_deg)
            g.create_dataset("v_beam_re_MV", data=rec.v_beam_MV.real)
            g.create_dataset("v_beam_im_MV", data=rec.v_beam_MV.imag)
            if rec.centroid is not None:
                g.create_dataset("centroid", data=rec.centroid)
            if rec.hom_w is not None:
                g.create_dataset("hom_w_re_MV", data=rec.hom_w.real)
                g.create_dataset("hom_w_im_MV", data=rec.hom_w.imag)
            if rec.hom_modes:
                hg = g.create_group("hom_modes")
                for key in ("f_MHz", "r_over_q_t", "q_loaded",
                            "polarization_deg"):
                    hg.create_dataset(key, data=np.array(
                        [getattr(m, key) for m in rec.hom_modes]))


class FastPulseRunner(TrainRunner):
    """Per-slot recursion over the full pattern (mode="fast").

    Reuses the whole TrainRunner lifecycle — construction validation,
    prior snapshot, design pass with psi-pinning, teardown restore —
    and replaces only the per-bunch stage: no tracking, one
    ``bunch_passage`` per cavity per filled slot.
    """

    def __init__(self, lattice, beam_config, train_config, sc_config=None,
                 history_stride: int = 1, record_slots=None, **kwargs):
        mode = getattr(train_config, "mode", None)
        if mode != "fast":
            raise ValueError(
                f"FastPulseRunner requires train_config.mode='fast' "
                f"(got {mode!r}); tracked modes run through TrainRunner")
        if int(history_stride) < 1:
            raise ValueError("history_stride must be >= 1")
        self.history_stride = int(history_stride)
        # M6 hybrid replay channel: absolute slot indices whose per-cavity
        # applied state must be captured regardless of history_stride.
        # After run(), ``slot_records[slot]`` is
        #   {"loading": {(idx, name): (voltage_rel, phase_offset_deg)},
        #    "hom":     {(idx, name): (dxp_mrad, dyp_mrad)}}
        # — exactly what the tracked entry hooks would write/apply for
        # that bunch.  Empty frozenset (default) = zero change.
        self._record_slots = (frozenset(int(s) for s in record_slots)
                              if record_slots else frozenset())
        self.slot_records: dict[int, dict] = {}
        super().__init__(lattice, beam_config, train_config,
                         sc_config=sc_config, **kwargs)

    # ------------------------------------------------------------------
    def _run_design_pass(self, results) -> None:
        # Fast mode ALWAYS needs the design pass: W_design_exit anchors
        # the ledger even when no cavity is bound (beam_loading off).
        super()._run_design_pass(results, force=True)

    # ------------------------------------------------------------------
    def _ledger_amplitudes(self, items):
        """Per-cavity gain amplitude A_cav [MeV] with A cos(phi_s) ==
        dW_design (measured), so the ledger reproduces the tracked
        element response exactly where the gain is separable."""
        from linac_gen.cli.common import build_ref
        q_sign = float(build_ref(self.beam_config).species.charge)
        amp = np.empty(len(items))
        for j, ((idx, _name), st) in enumerate(items):
            c = math.cos(math.radians(st.phi_s_deg))
            if abs(c) >= 1e-6 and st.dw_design_MeV != 0.0:
                amp[j] = st.dw_design_MeV / c
            else:
                # Pure buncher (sidecar-pinned v_design at a zero-crossing
                # phase), where the measured-gain route is singular.  The
                # effective SIGN depends on how the phase was established:
                #  - psi-CALIBRATED (SET_SYNC_PHASE -> sync_phase_pin from
                #    the design pass): the calibration absorbs the species
                #    charge -- the cavity delivers phi_s as specified for
                #    the ACTUAL beam, so the effective amplitude is
                #    +v_design regardless of species (the same convention
                #    that makes measured dW_design/cos(phi_s) species-
                #    free).  q*V here flips dW(po) for negative species:
                #    the tracked PIP-II MEBT bunchers (H-) respond
                #    +A sin(po) at phi_s=-90 while q*V gave -A sin(po).
                #  - PRESCRIBED phase (p_flag=0 RFGap, uncalibrated
                #    maps): the raw law dW = q V T cos(phi) applies
                #    verbatim, so the species sign stays.
                el = self.lattice.elements[idx]
                calibrated = getattr(el, "sync_phase_pin", None) is not None
                if not calibrated:
                    calibrated = any(
                        getattr(ch, "sync_phase_pin", None) is not None
                        for ch in (getattr(el, "children", None) or ()))
                amp[j] = (st.v_design_MV if calibrated
                          else q_sign * st.v_design_MV)
        return amp

    def _run_bunches(self, results) -> None:
        slots = self.train.pattern.filled_slots
        n = len(slots)
        w_design_exit = float(results.design_result.ref_w_kin[-1])
        mgr = self._loading
        hom_mgr = self._hom
        items = (list(self._registry.items())
                 if self._registry is not None else [])
        ncav = len(items)
        keys = [key for key, _st in items]
        states = [st for _key, st in items]
        priors = ([mgr._priors.get(key, (0.0, 0.0)) for key, _st in items]
                  if mgr is not None else [(0.0, 0.0)] * ncav)
        amp = (self._ledger_amplitudes(items)
               if (ncav and mgr is not None) else np.zeros(ncav))
        dwd = np.array([st.dw_design_MeV for st in states])
        phis = np.array([st.phi_s_deg for st in states])
        # Loss-free fast-mode bunch charge = I/f_bunch (single source of
        # truth for the convention; tracked mode: macro_charge * n_alive).
        q_bunch = macro_charge_coulombs(self.beam_config.current,
                                        self._f_bunch, 1)
        stride = self.history_stride
        n_hist = (n + stride - 1) // stride
        vr_hist = np.empty((ncav, n_hist))
        po_hist = np.empty((ncav, n_hist))
        vb_hist = np.empty((ncav, n_hist), complex)
        hist_slot = np.empty(n_hist, np.int64)
        w_exit = np.full(n, w_design_exit)
        cos, rad = math.cos, math.radians

        # ---- dipole-HOM centroid model (M4; see module docstring) ----
        hom_idx = ([j for j in range(ncav) if states[j].hom]
                   if hom_mgr is not None else [])
        n_hom = len(hom_idx)
        hom_pos = {j: k for k, j in enumerate(hom_idx)}
        if n_hom:
            from linac_gen.cli.common import build_ref
            sp = build_ref(self.beam_config).species
            z_q = float(sp.charge)
            mass = float(sp.mass)
            hom_states = [states[j] for j in hom_idx]
            hom_bg = [float(st.bg_design) for st in hom_states]
            hom_cdes = [st.centroid_design for st in hom_states]
            # Drift lengths between consecutive HOM-cavity ENTRIES (m),
            # from the design-pass s records; 0.0 for the first cavity
            # (every bunch enters ON the design centroid there).
            hom_dL_m = [0.0] + [
                (hom_states[k].s_design_mm
                 - hom_states[k - 1].s_design_mm) * 1e-3
                for k in range(1, n_hom)]
            # SIGNED source charge: excitation and kick both carry the
            # species sign, so the wake deflection scales as Z^2
            # (species-sign independent) — hom.py convention.
            q_hom = q_bunch * z_q
            cent_hist = np.empty((n_hom, n_hist, 4))
            homw_hist = [np.empty((len(st.hom), n_hist), complex)
                         for st in hom_states]

        n_done = 0
        for i in range(n):
            if self.should_abort is not None and self.should_abort():
                warnings.warn(
                    f"fast train aborted after {i}/{n} bunches; partial "
                    "results returned", stacklevel=2)
                break
            slot = int(slots[i])
            record = (i % stride == 0)
            if record:
                hist_slot[i // stride] = slot
            # M6: per-selected-slot capture for the hybrid replay (None
            # for every slot when record_slots was not given).
            srec = (self.slot_records.setdefault(
                        slot, {"loading": {}, "hom": {}})
                    if self._record_slots and slot in self._record_slots
                    else None)
            w = w_design_exit
            if n_hom:
                dx = dxp = dy = dyp = 0.0
                k_hom = 0
            for j in range(ncav):
                st = states[j]
                if mgr is not None:
                    r_vr, r_po = mgr.bunch_passage(st, slot, q_bunch)
                    w += amp[j] * (1.0 + r_vr) * cos(rad(phis[j] + r_po)) \
                        - dwd[j]
                    if record or srec is not None:
                        vr0, po0 = priors[j]
                        # Prior composition — the same law as the tracked
                        # entry hook (compose, never clobber).
                        vr_app = (1.0 + r_vr) * (1.0 + vr0) - 1.0
                        po_app = po0 + r_po
                        if record:
                            hi = i // stride
                            vr_hist[j, hi] = vr_app
                            po_hist[j, hi] = po_app
                            vb_hist[j, hi] = st.v_beam
                        if srec is not None:
                            srec["loading"][keys[j]] = (vr_app, po_app)
                elif record:
                    hi = i // stride
                    vr_hist[j, hi] = 0.0
                    po_hist[j, hi] = 0.0
                    vb_hist[j, hi] = 0j
                if n_hom and st.hom:
                    L = hom_dL_m[k_hom]
                    if L:
                        dx += dxp * L           # mm += mrad * m
                        dy += dyp * L
                    cd = hom_cdes[k_hom]
                    ddxp, ddyp = hom_mgr.hom_passage(
                        st, slot, q_hom, cd[0] + dx, cd[2] + dy,
                        hom_bg[k_hom], mass, z_q)
                    if srec is not None:
                        srec["hom"][keys[j]] = (ddxp, ddyp)
                    dxp += ddxp
                    dyp += ddyp
                    if record:
                        hi = i // stride
                        cent_hist[k_hom, hi, 0] = cd[0] + dx
                        cent_hist[k_hom, hi, 1] = cd[1] + dxp
                        cent_hist[k_hom, hi, 2] = cd[2] + dy
                        cent_hist[k_hom, hi, 3] = cd[3] + dyp
                        hw = homw_hist[k_hom]
                        for mi, h in enumerate(st.hom):
                            hw[mi, hi] = h.w
                    k_hom += 1
            w_exit[i] = w
            n_done = i + 1
            if self.progress_callback is not None and \
                    (n_done % 4096 == 0 or n_done == n):
                try:
                    self.progress_callback(n_done, n)
                except Exception:                             # noqa: BLE001
                    pass

        n_rec = (n_done + stride - 1) // stride
        cavities = [
            FastCavityRecord(
                element_index=int(idx), name=str(name),
                frequency_MHz=float(st.frequency_MHz),
                v_design_MV=float(st.v_design_MV),
                phi_s_deg=float(st.phi_s_deg),
                dw_design_MeV=float(st.dw_design_MeV),
                r_over_q=float(st.mode.r_over_q),
                q_loaded=float(st.mode.q_loaded),
                detuning_Hz=float(st.mode.detuning_Hz),
                voltage_rel=vr_hist[j, :n_rec].copy(),
                phase_offset_deg=po_hist[j, :n_rec].copy(),
                v_beam_MV=vb_hist[j, :n_rec].copy(),
                centroid=(cent_hist[hom_pos[j], :n_rec].copy()
                          if j in hom_pos else None),
                hom_w=(homw_hist[hom_pos[j]][:, :n_rec].copy()
                       if j in hom_pos else None),
                hom_modes=(tuple(h.mode for h in st.hom)
                           if j in hom_pos else ()),
            )
            for j, ((idx, name), st) in enumerate(items)
        ]
        results.fast = FastPulseSummary(
            slot=slots[:n_done].astype(np.int64),
            w_exit_MeV=w_exit[:n_done],
            w_design_exit_MeV=w_design_exit,
            charge_per_bunch_C=q_bunch,
            history_stride=stride,
            history_slot=hist_slot[:n_rec].copy(),
            cavities=cavities,
            truncated=(n_done < n),
        )
