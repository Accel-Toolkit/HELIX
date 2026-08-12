"""TrainRunner — the sequential N-pass multibunch driver (opt-in study).

Architecture (plan 2026-08-10): one process, ONE shared lattice, one full
tracking pass per filled bunch slot.  Elements reset their per-pass
integrator state at each traversal (Tracker calls reset_run_state), while
the FieldError slots (voltage_rel / phase_offset) survive — that is the
channel through which bunch k+1 will see the cavity state left behind by
bunches 1..k once the physics modules (M2+) are enabled.  With all physics
off (M1), the train is BIT-IDENTICAL to N independent single-bunch runs —
the zero-coupling contract.
"""
from __future__ import annotations

import warnings

from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.train.config import TrainConfig
from linac_gen.train.results import TrainResults


def _chain_hooks(hooks):
    """Compose several (element, index, beam) hooks into one, preserving
    order.  Module-level function (not a lambda) so the tracked driver's
    hook plumbing stays picklable/debuggable."""
    def chained(element, index, beam):
        for h in hooks:
            h(element, index, beam)
    return chained


def pattern_image_factors(pattern, slot: int) -> tuple:
    """Per-image charge factors from the pulse pattern (M5 convention).

    Convention (measured, tests/train/test_direct_sc.py): the LEADING
    image is the neighbour injected one slot EARLIER — it travels one
    train period AHEAD (dphi −360·h → z > 0); the TRAILING image is the
    slot-later neighbour, one period behind.  A bunch right after a
    chopped gap therefore has f_lead = 0 (and feels a net forward push
    from its surviving trailing neighbour); the last bunch before a gap
    has f_trail = 0.  The first/last filled slots of the pulse lose the
    corresponding image too.  Shared by the tracked driver and the M6
    hybrid replay so the two cannot drift apart.
    """
    filled = pattern.filled
    f_lead = 1.0 if slot > 0 and filled[slot - 1] else 0.0
    f_trail = (1.0 if slot + 1 < filled.size and filled[slot + 1]
               else 0.0)
    return (f_lead, f_trail)


class TrainRunner:
    def __init__(self, lattice, beam_config, train_config: TrainConfig,
                 sc_config=None, progress_callback=None, should_abort=None,
                 lattice_path=None, **sim_kwargs):
        if not isinstance(train_config, TrainConfig):
            raise TypeError("train_config must be a TrainConfig")
        if train_config.mode == "fast" and type(self) is TrainRunner:
            raise ValueError(
                "mode='fast' runs through FastPulseRunner "
                "(linac_gen.train.fast) or run_train(); the tracked "
                "TrainRunner would launch a full tracking pass per slot")
        if train_config.mode == "hybrid" and type(self) is TrainRunner:
            raise ValueError(
                "mode='hybrid' runs through HybridReplayRunner "
                "(linac_gen.train.replay) or run_train(); the tracked "
                "TrainRunner has no two-pass replay stage")
        if bool(getattr(beam_config, "periodic_phase", False)):
            raise ValueError(
                "multibunch train mode requires periodic_phase=False: the "
                "periodic-phase fold collapses everything into one bunch "
                "spacing and would silently destroy the train structure")
        self.lattice = lattice
        self.beam_config = beam_config
        self.train = train_config
        self.sc_config = sc_config
        self.progress_callback = progress_callback
        self.should_abort = should_abort
        # Optional path of the deck the live lattice was parsed from —
        # consumed by the M6 hybrid replay's parallel branch (scan_pool
        # workers re-parse it); inert for the tracked/fast drivers.
        self.lattice_path = lattice_path
        self.sim_kwargs = dict(sim_kwargs)
        self._pins_set: list = []
        self._loading = None
        self._hom = None
        self._registry = None
        need_loading = train_config.physics.beam_loading
        need_hom = train_config.physics.hom
        if need_loading or need_hom:
            from linac_gen.train.cavity_state import CavityStateRegistry
            reg = CavityStateRegistry()
            modes = CavityStateRegistry.load_sidecar(
                train_config.cavity_params,
                need_fundamental=need_loading, need_hom=need_hom)
            n_bound = reg.bind(lattice, modes)
            if n_bound == 0:
                channels = "+".join(
                    n for n, on in (("beam_loading", need_loading),
                                    ("hom", need_hom)) if on)
                raise ValueError(
                    f"{channels} enabled but the sidecar matched no "
                    "lattice cavity by name — check the name patterns in "
                    f"{train_config.cavity_params!r}")
            self._registry = reg
            if need_loading:
                from linac_gen.train.beam_loading import BeamLoadingManager
                self._loading = BeamLoadingManager(
                    reg, train_config.bunch_frequency_MHz)
                self._loading.snapshot_priors(lattice)
            if need_hom:
                from linac_gen.train.hom import HomManager
                self._hom = HomManager(
                    reg, train_config.bunch_frequency_MHz)
        # Reconcile the two bunch-rate declarations (adversarial F6): the
        # train's rate must actually reach the physics (macrocharge).
        cfg_f = float(getattr(beam_config, "bunch_frequency_MHz", 0.0) or 0.0)
        if cfg_f > 0.0 and abs(cfg_f - train_config.bunch_frequency_MHz) > 1e-9:
            raise ValueError(
                f"beam_config.bunch_frequency_MHz={cfg_f} disagrees with "
                f"train_config.bunch_frequency_MHz="
                f"{train_config.bunch_frequency_MHz}; set one (or make "
                "them equal)")
        if (train_config.physics.beam_loading or train_config.physics.hom) \
                and train_config.mode == "envelope":
            raise NotImplementedError(
                "beam_loading/hom with mode='envelope' is not wired (M3 "
                "note): EnvelopeSolver has no element entry/exit hooks, "
                "so neither the per-cavity design ledger nor the HOM "
                "centroid kicks can act on an envelope pass.  Use "
                "mode='mp' (tracked), mode='fast' (per-slot recursion) "
                "or mode='hybrid' (fast pass + full-MP replay of "
                "selected bunches)")
        self._f_bunch = train_config.bunch_frequency_MHz
        if cfg_f == 0.0:
            import dataclasses
            self.beam_config = dataclasses.replace(
                beam_config, bunch_frequency_MHz=self._f_bunch)
        # ---- direct bunch-to-bunch space charge (M5) -----------------
        self._direct_sc = bool(train_config.physics.direct_sc)
        self._dsc_factors = None       # (f_lead, f_trail) for current bunch
        self._dsc_recorder = None      # buffer being written (this bunch)
        self._dsc_provider = None      # previous bunch's buffer (read)
        self._dsc_engaged_any = False
        self._last_pic = None
        if self._direct_sc:
            self._check_direct_sc_config(sc_config)
        self._warn_if_freq_jumps_without_cards()

    # ------------------------------------------------------------------
    @staticmethod
    def _check_direct_sc_config(sc_config) -> None:
        """physics.direct_sc needs the numpy 3-D PIC — refuse loudly.

        Silent inertness is the one failure mode a rarely-run study must
        never have: factors that never reach a solver, or a backend that
        quietly runs isolated physics, would produce a 'train study'
        with no train physics in it.
        """
        if sc_config is None or isinstance(sc_config, str):
            raise ValueError(
                f"physics.direct_sc=True but sc_config={sc_config!r}: "
                "direct bunch-to-bunch space charge acts through the "
                "3-D PIC bunch-train images — pass a SpaceChargeConfig "
                "(numpy backend)")
        backend = getattr(sc_config, "sc_backend", "numpy")
        if backend == "torch":
            raise NotImplementedError(
                "physics.direct_sc is not implemented on the torch PIC "
                "backend (it has no bunch-train path); use "
                "sc_backend='numpy'")
        if backend != "numpy":
            raise ValueError(
                "physics.direct_sc supports only the numpy 3-D PIC "
                f"(sc_backend={backend!r})")
        if getattr(sc_config, "train_images", None) is False:
            raise ValueError(
                "physics.direct_sc=True conflicts with "
                "sc_config.train_images=False (bunch-train images "
                "explicitly disabled); leave train_images=None (auto) "
                "or set it True")

    def _direct_sc_factors(self, slot: int) -> tuple:
        """Per-image charge factors — see ``pattern_image_factors``."""
        return pattern_image_factors(self.train.pattern, slot)

    def _pic_setup(self, pic) -> None:
        """Simulation ``pic_setup_hook`` (bound method, no lambdas):
        configure THIS bunch's freshly built PIC solver."""
        from linac_gen.pic.pic_solver import PicSolver
        if not isinstance(pic, PicSolver):
            raise NotImplementedError(
                "physics.direct_sc: the built space-charge solver is "
                f"{type(pic).__name__}, not the numpy PicSolver — no "
                "bunch-train path")
        pic.train_image_factors = self._dsc_factors
        pic.train_force_engage = bool(self.train.direct_sc_force_engage)
        if self._dsc_recorder is not None:
            pic.train_snapshot_recorder = self._dsc_recorder.record
        if self._dsc_provider is not None:
            pic.train_neighbor_provider = self._dsc_provider.snapshot
        self._last_pic = pic

    # ------------------------------------------------------------------
    def _warn_if_freq_jumps_without_cards(self) -> None:
        """Arrival-time observables need FREQ-card-driven clock rescales
        (lattice_commands.Freq); the per-element fallback leaves ref.phi_s
        in old-frequency degrees.  Warn once if the deck relies on it."""
        from linac_gen.elements.lattice_commands import Freq

        f_run = None
        fallback_jump = False
        for el in self.lattice.elements:
            if isinstance(el, Freq):
                f_run = float(el.frequency_mhz)
                continue
            f_el = float(getattr(el, "frequency", 0.0) or
                         getattr(el, "frequency_mhz", 0.0) or 0.0)
            if f_el <= 0.0:
                continue
            # Magnetic-only field maps carry the parse-time frequency as
            # inert metadata — only elements with an electric channel can
            # move the clock.
            has_e = getattr(el, "_has_electric_channel", None)
            if has_e is not None and not has_e():
                continue
            if f_run is None:
                f_run = f_el
            elif abs(f_el - f_run) > 1e-9:
                fallback_jump = True
                f_run = f_el
        if fallback_jump:
            warnings.warn(
                "train mode: this lattice changes RF frequency without a "
                "FREQ card; the reference phase clock (ref.phi_s) will not "
                "be rescaled there and bunch ARRIVAL-TIME observables "
                "across that boundary are unreliable (per-bunch dynamics "
                "are unaffected)", stacklevel=2)

    # ------------------------------------------------------------------
    def _pinnable_cavities(self):
        def _walk(elems):
            for el in elems:
                kids = getattr(el, "children", None)
                if kids:
                    # SuperposedFieldMap: (z, child) pairs
                    _walk_children = [c for _z, c in kids]
                    yield from _walk(_walk_children)
                if (getattr(el, "p_flag", 0) == 1
                        or getattr(el, "sync_phase", False)):
                    if hasattr(el, "sync_phase_pin"):
                        yield el
        yield from _walk(self.lattice.elements)

    def _run_design_pass(self, results: TrainResults,
                         force: bool = False) -> None:
        """One nominal pass to let lazy SET_SYNC_PHASE calibration run,
        then pin psi on every calibrated cavity so bunch passes reuse the
        DESIGN operating point (and skip the iterative probe).  Pins are
        RUNNER-SCOPED: run() clears every pin it set in a finally block,
        so a train can never contaminate later normal runs on the shared
        lattice (a stale pin at a different energy was measured to
        sign-flip a cavity's energy gain — adversarial review K2).

        ``force`` runs the pass even with nothing to pin or load — the
        fast driver (M3) always needs W_design_exit for its ledger."""
        cavs = list(self._pinnable_cavities())
        if not force and not cavs and self._loading is None \
                and self._hom is None:
            return
        if self._loading is not None:
            self._loading.begin_design_pass()
        if self._hom is not None:
            self._hom.begin_design_pass()
        res = self._run_single(seed=self.train.seed)
        if self._loading is not None:
            self._loading.end_design_pass(self.lattice)
        if self._hom is not None:
            self._hom.end_design_pass(self.lattice)
        results.design_result = res
        n_pinned = 0
        index_of = {id(e): i for i, e in enumerate(self.lattice.elements)}
        for el in cavs:
            psi = getattr(el, "_sync_offset_deg", None)
            if psi is not None and el.sync_phase_pin is None:
                el.sync_phase_pin = float(psi)
                self._pins_set.append(el)
                name = getattr(el, "name", repr(el))
                results.pins[name] = float(psi)
                idx = index_of.get(id(el))
                if idx is None:
                    # SuperposedFieldMap CHILD — not addressable by the
                    # top-level @index/NAME override selectors; the M6
                    # replay refuses loudly when this list is non-empty.
                    results.pins_unindexed.append(name)
                else:
                    results.pin_by_index[idx] = float(psi)
                n_pinned += 1
        if cavs and n_pinned == 0:
            warnings.warn(
                "train mode: design pass calibrated no SET_SYNC_PHASE "
                "cavities (none reached?) — pins not set", stacklevel=2)

    # ------------------------------------------------------------------
    def _make_beam(self, slot: int):
        # M1: identical bunches — same seed, no jitter.  charge_scale and
        # jitter engage with the physics milestones.
        return create_beam(self.beam_config, seed=self.train.seed)

    def _hook_kwargs(self):
        # Both physics managers ride the SAME two Tracker hooks; with two
        # active they are chained in registration order (loading first,
        # hom second — physically independent: loading writes the
        # longitudinal FieldError slots, hom kicks transverse slopes).
        entry = []
        exit_ = []
        if self._loading is not None:
            entry.append(self._loading.entry_hook)
            exit_.append(self._loading.exit_hook)
        if self._hom is not None:
            entry.append(self._hom.entry_hook)
        out = {}
        if entry:
            out["element_entry_hook"] = (entry[0] if len(entry) == 1
                                         else _chain_hooks(entry))
        if exit_:
            out["element_exit_hook"] = (exit_[0] if len(exit_) == 1
                                        else _chain_hooks(exit_))
        return out

    def _run_single(self, seed: int):
        if self.train.mode == "envelope":
            return self._run_envelope_once()
        beam = create_beam(self.beam_config, seed=seed)
        sim = Simulation(self.lattice, beam, space_charge=self.sc_config,
                         **self._hook_kwargs(), **self.sim_kwargs)
        return sim.run()

    def _run_envelope_once(self):
        # Canonical seed helpers (NOT hand-rolled Twiss): the envelope of
        # a mismatched / DC / off-axis config must be the same beam as in
        # every other mode (factory.geometric_emittances contract).
        from linac_gen.cli.common import _envelope_initial, build_ref
        from linac_gen.tracking.envelope import EnvelopeSolver

        cfg = self.beam_config
        ref = build_ref(cfg)
        initial = _envelope_initial(cfg, ref)
        return EnvelopeSolver(self.lattice, ref, initial,
                              current=cfg.current,
                              bunch_frequency=self._f_bunch).run()

    # ------------------------------------------------------------------
    def run(self) -> TrainResults:
        results = TrainResults(self.train, self.train.mode)
        # Save-time context (M7): TrainResults.save_hdf5 writes the same
        # provenance/ group as single-bunch files, which needs the run's
        # beam/SC configuration and deck identity.  References only —
        # nothing is copied.
        results.beam_config = self.beam_config
        results.sc_config = self.sc_config
        results.lattice = self.lattice
        results.lattice_path = self.lattice_path
        try:
            self._run_design_pass(results)
            self._run_bunches(results)
        finally:
            # Applied-value ledgers (M6 replay) + phasor/wake histories
            # (M7 persistence): copy out before teardown so a tracked
            # run's own recorded state can reconstruct any of its
            # bunches losslessly (also populated on abort paths).
            if self._loading is not None:
                results.applied_loading = dict(self._loading.applied)
                results.v_beam_loading = dict(self._loading.v_beam_after)
            if self._hom is not None:
                results.applied_hom = dict(self._hom.applied)
                results.hom_w = dict(self._hom.w_after)
            if self._registry is not None:
                results.cavity_table = [
                    {"element_index": int(idx), "name": str(name),
                     "frequency_MHz": float(st.frequency_MHz),
                     "v_design_MV": float(st.v_design_MV),
                     "phi_s_deg": float(st.phi_s_deg),
                     "dw_design_MeV": float(st.dw_design_MeV),
                     "r_over_q": float(st.mode.r_over_q),
                     "q_loaded": float(st.mode.q_loaded),
                     "detuning_Hz": float(st.mode.detuning_Hz),
                     "hom_modes": tuple(h.mode for h in st.hom)}
                    for (idx, name), st in self._registry.items()]
            # Zero-coupling contract (plan 3b): the shared lattice must
            # leave a train run exactly as it entered — clear every pin
            # THIS runner set (and only those), even on abort/exception,
            # and zero any beam-loading perturbation on the slots.
            for el in self._pins_set:
                el.sync_phase_pin = None
            self._pins_set.clear()
            if self._loading is not None:
                self._loading.restore_design(self.lattice)
            # Direct-SC teardown (M5): drop factors/snapshot buffers and
            # disarm the last per-bunch solver so nothing train-scoped
            # can leak into later use of the objects (the caller's
            # sc_config was never mutated — factors ride on the solver
            # instance via the pic_setup_hook, not on the config).
            self._dsc_factors = None
            self._dsc_recorder = None
            self._dsc_provider = None
            if self._last_pic is not None:
                self._last_pic.train_image_factors = None
                self._last_pic.train_force_engage = False
                self._last_pic.train_snapshot_recorder = None
                self._last_pic.train_neighbor_provider = None
                self._last_pic = None
        if self._direct_sc and results.slots and not self._dsc_engaged_any:
            warnings.warn(
                "physics.direct_sc was enabled but the bunch-train "
                "images never engaged on any bunch (beam never "
                "train-like, or its core σφ stayed below the 35° gate) "
                "— the pattern factors were inert.  Set "
                "TrainConfig.direct_sc_force_engage=True to model "
                "neighbours regardless of bunch length.", stacklevel=2)
        return results

    def _run_bunches(self, results: TrainResults) -> None:
        slots = self.train.pattern.filled_slots
        n = len(slots)
        for i, slot in enumerate(slots):
            if self.should_abort is not None and self.should_abort():
                warnings.warn(
                    f"train aborted after {i}/{n} bunches; partial results "
                    "returned", stacklevel=2)
                # The tracked-mode mirror of FastPulseSummary.truncated
                # (M8): the partial TrainResults stays fully saveable /
                # loadable and carries an explicit abort marker.
                results.truncated = True
                break
            if self._loading is not None:
                self._loading.begin_bunch(int(slot))
            if self._hom is not None:
                self._hom.begin_bunch(int(slot))
            dsc_kwargs = {}
            if self._direct_sc:
                self._dsc_factors = self._direct_sc_factors(int(slot))
                results.direct_sc[int(slot)] = self._dsc_factors
                if self.train.direct_sc_neighbors == "distinct":
                    from linac_gen.train.direct_sc import (
                        NeighborSnapshotBuffer)
                    # ring swap: this bunch reads the PREVIOUS bunch's
                    # buffer and records its own for the next one —
                    # bounded to two live buffers.
                    self._dsc_provider = self._dsc_recorder
                    self._dsc_recorder = NeighborSnapshotBuffer(
                        n_sub=self.train.direct_sc_subsample)
                dsc_kwargs["pic_setup_hook"] = self._pic_setup
            if self.train.mode == "envelope":
                res = self._run_envelope_once()
            else:
                beam = self._make_beam(int(slot))
                sim = Simulation(self.lattice, beam,
                                 space_charge=self.sc_config,
                                 **self._hook_kwargs(),
                                 **dsc_kwargs,
                                 **self.sim_kwargs)
                res = sim.run()
                if self._last_pic is not None \
                        and self._last_pic._train_ever_engaged:
                    self._dsc_engaged_any = True
            # append computes the summary row from the FULL result before
            # keep_full_results=False drops it — summary-only trains keep
            # a real summary table (M7; was all-NaN before).
            results.append(int(slot), res,
                           keep=self.train.keep_full_results)
            if self.progress_callback is not None:
                try:
                    self.progress_callback(i + 1, n)
                except Exception:                             # noqa: BLE001
                    pass


def run_train(lattice, beam_config, train_config, sc_config=None,
              **kwargs) -> TrainResults:
    """Convenience one-shot entry point for the multibunch study.

    Dispatches on ``train_config.mode``: "fast" runs the per-slot
    FastPulseRunner (M3); "hybrid" runs the two-pass HybridReplayRunner
    (M6); "mp"/"envelope" run the tracked TrainRunner.
    """
    mode = getattr(train_config, "mode", None)
    if mode == "fast":
        from linac_gen.train.fast import FastPulseRunner
        return FastPulseRunner(lattice, beam_config, train_config,
                               sc_config=sc_config, **kwargs).run()
    if mode == "hybrid":
        from linac_gen.train.replay import HybridReplayRunner
        return HybridReplayRunner(lattice, beam_config, train_config,
                                  sc_config=sc_config, **kwargs).run()
    return TrainRunner(lattice, beam_config, train_config,
                       sc_config=sc_config, **kwargs).run()
