"""Fundamental-mode beam loading — ONE implementation, two drivers.

`BeamLoadingManager` owns the per-cavity phasor state and exposes the
element entry/exit hooks consumed by the tracked driver (TrainRunner);
the fast full-pulse driver (M3) advances the same registry slot-by-slot
through `advance_empty_slot` / `bunch_passage`.  Conventions and the
slot-mapping of (voltage_rel, phase_offset) are documented in
cavity_state.py and anchored by tests/train/test_beam_loading.py.

Self-kick: the entry hook includes HALF of the bunch's own predicted
induced voltage (fundamental theorem of beam loading), predicted from
the ENTRY charge; the exit hook then adds the FULL induced phasor of
the charge that actually traversed.  Generator model v1: the generator
holds V_design rigidly (no LLRF dynamics beyond that; plan-scoped).
"""
from __future__ import annotations

import math

from linac_gen.pic.macrocharge import macro_charge_for
from linac_gen.train.cavity_state import CavityStateRegistry


class BeamLoadingManager:
    def __init__(self, registry: CavityStateRegistry,
                 bunch_frequency_MHz: float):
        self.reg = registry
        self.T_slot_s = 1.0 / (bunch_frequency_MHz * 1e6)
        self.current_slot = 0
        self._design_mode = False   # True during the design pass (measure)
        # Pre-existing slot values (error studies, manual settings) are
        # PRIORS: the design pass runs with them active, so the derived
        # V_design/phi_s already describe the erred cavity; the loading
        # kick then COMPOSES on the prior and teardown RESTORES it
        # (adversarial check 2: zeroing clobbered user state).
        self._priors: dict = {}
        # Tracked-mode ledger of APPLIED slot values (M6 replay):
        # (slot, element_index, name) -> (voltage_rel, phase_offset) as
        # written onto the element by entry_hook.  Pure bookkeeping —
        # feeds TrainResults.applied_loading so a tracked bunch can be
        # replayed losslessly from its own recorded state.
        self.applied: dict = {}
        # Tracked-mode phasor history (M7 persistence): (slot,
        # element_index, name) -> v_beam AFTER the bunch's full induced
        # kick (complex MV, rotating frame) — the state the next arrival
        # decays from, mirroring FastCavityRecord.v_beam_MV.  Hook-only
        # bookkeeping: ``bunch_passage`` (the fast-mode primitive, run
        # per slot over ~89k-slot pulses with its own stride-decimated
        # storage) records nothing here.
        self.v_beam_after: dict = {}

    def snapshot_priors(self, lattice) -> None:
        for (idx, name), st in self.reg.items():
            el = lattice.elements[idx]
            self._priors[(idx, name)] = (
                float(getattr(el, "voltage_rel", 0.0) or 0.0),
                float(getattr(el, "phase_offset", 0.0) or 0.0))

    # ------------------------------------------------------------ design
    def begin_design_pass(self):
        self._design_mode = True

    def end_design_pass(self, lattice):
        """Derive V_design (and phi_s where possible) for every bound
        cavity from the design pass; refuse loudly where underivable."""
        self._design_mode = False
        problems = []
        for (idx, name), st in self.reg.items():
            el = lattice.elements[idx]
            if st.mode.phi_s_deg is not None:
                st.phi_s_deg = float(st.mode.phi_s_deg)
            else:
                phi = self._design_phi_s(el)
                if phi is None:
                    problems.append(
                        f"{name}: synchronous phase not derivable "
                        "(relative-phase field map?) — set phi_s_deg in "
                        "the sidecar")
                    continue
                st.phi_s_deg = float(phi)
            if st.mode.v_design_MV is not None:
                st.v_design_MV = float(st.mode.v_design_MV)
            else:
                cphi = math.cos(math.radians(st.phi_s_deg))
                if abs(cphi) < 1e-6 or st.dw_design_MeV == 0.0:
                    problems.append(
                        f"{name}: V_design not derivable (dW="
                        f"{st.dw_design_MeV:.4g} MeV, phi_s="
                        f"{st.phi_s_deg:.4g} deg) — set v_design_MV in "
                        "the sidecar")
                    continue
                st.v_design_MV = abs(st.dw_design_MeV / cphi)
        if problems:
            raise ValueError(
                "beam loading: missing design quantities:\n  "
                + "\n  ".join(problems))

    @staticmethod
    def _design_phi_s(el) -> float | None:
        if getattr(el, "p_flag", 0) == 1:
            return float(el.effective_phase)          # SET_SYNC_PHASE theta_s
        if getattr(el, "sync_phase", False):
            return float(el.theta_s_deg)              # NCells
        from linac_gen.elements.rf_gap import RFGap
        if isinstance(el, RFGap):
            return float(el.effective_phase)
        return None

    # ------------------------------------------------------------- hooks
    def entry_hook(self, element, index, beam):
        name = getattr(element, "name", "")
        st = self.reg.get(index, name)
        if st is None:
            return
        if self._design_mode:
            st._w_at_entry = float(beam.ref.w_kin)
            return
        # decay to this bunch's arrival
        if st.last_slot is not None:
            dt = (self.current_slot - st.last_slot) * self.T_slot_s
            CavityStateRegistry.decay(st, dt)
        st.last_slot = self.current_slot
        q_entry = macro_charge_for(beam) * beam.n_alive
        half_self = 0.5 * CavityStateRegistry.induced_dv_MV(st, q_entry)
        v_tot = st.v_design_MV + st.v_beam + half_self
        vr0, po0 = self._priors.get((index, name), (0.0, 0.0))
        # Compose on the prior: V_design already IS the erred voltage
        # (measured by the design pass with priors active), so the slot
        # values reproduce the prior exactly when v_beam == 0.
        element.voltage_rel = \
            (abs(v_tot) / st.v_design_MV) * (1.0 + vr0) - 1.0
        element.phase_offset = po0 + math.degrees(
            math.atan2(v_tot.imag, v_tot.real))
        self.applied[(self.current_slot, index, name)] = (
            element.voltage_rel, element.phase_offset)

    def exit_hook(self, element, index, beam):
        name = getattr(element, "name", "")
        st = self.reg.get(index, name)
        if st is None:
            return
        if self._design_mode:
            st.dw_design_MeV = float(beam.ref.w_kin) - st._w_at_entry
            return
        q_traversed = macro_charge_for(beam) * beam.n_alive
        st.v_beam += CavityStateRegistry.induced_dv_MV(st, q_traversed)
        self.v_beam_after[(self.current_slot, index, name)] = st.v_beam

    # ------------------------------------------------- per-slot advance
    def begin_bunch(self, slot: int):
        self.current_slot = int(slot)

    def restore_design(self, lattice):
        """Restore the PRIOR slot values on every cavity this manager
        perturbed (train teardown; zero-coupling contract)."""
        for (idx, name), st in self.reg.items():
            el = lattice.elements[idx]
            vr0, po0 = self._priors.get((idx, name), (0.0, 0.0))
            el.voltage_rel = vr0
            el.phase_offset = po0

    # Fast-mode primitive (M3): a bunch of charge q at slot k passing the
    # cavity WITHOUT tracking — returns (voltage_rel, phase_offset_deg).
    def bunch_passage(self, st, slot: int, charge_C: float):
        if st.last_slot is not None:
            CavityStateRegistry.decay(
                st, (slot - st.last_slot) * self.T_slot_s)
        st.last_slot = slot
        half_self = 0.5 * CavityStateRegistry.induced_dv_MV(st, charge_C)
        v_tot = st.v_design_MV + st.v_beam + half_self
        vr = abs(v_tot) / st.v_design_MV - 1.0
        po = math.degrees(math.atan2(v_tot.imag, v_tot.real))
        st.v_beam += CavityStateRegistry.induced_dv_MV(st, charge_C)
        return vr, po
