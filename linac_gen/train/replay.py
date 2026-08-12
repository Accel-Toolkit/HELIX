"""Hybrid two-pass replay (M6) — full-MP fidelity for SELECTED bunches of
a full pulse, at fast-mode cost for the rest.

Pass 1 — ``FastPulseRunner`` over the FULL pattern (design pass + per-slot
phasor recursion, M3/M4 physics verbatim), recording at every SELECTED
slot the per-cavity state a tracked bunch would have seen there:

  * applied ``voltage_rel`` / ``phase_offset`` (prior-composed — exactly
    what the tracked entry hook writes onto the element),
  * the design-pass ``sync_phase_pin`` values (the LLRF-fixed operating
    point; without them a fresh lattice would re-calibrate psi against
    the LOADED voltage — the SET_SYNC_PHASE trap), and
  * (hom channel on) the per-cavity dipole-HOM kick (dxp, dyp) that
    bunch receives at entry.

Pass 2 — each selected bunch is replayed INDEPENDENTLY as a full MP run
with that state applied as STATIC element attributes through
``linac_gen.cli.common.apply_element_override`` ``(selector, value)``
pairs ("@N.voltage_rel", "@N.sync_phase_pin", "@N.hom_kick_x", ...).
The HOM kick rides on the new ``hom_kick_x``/``hom_kick_y`` element
attributes (mrad, class default 0.0 = inert), consumed once at element
entry by the Tracker with the same guarded array ops as
``HomManager.entry_hook`` — chosen over a hook-based design because the
scan_pool worker builds a plain ``Simulation`` with no hook plumbing,
and a default-0.0 attribute keeps every single-bunch run bit-identical.

Replay branches:
  * IN-PROCESS (default): sequential replays on the live lattice; the
    touched attributes are snapshotted before and restored (exactly,
    instance-dict level) after every bunch — the lattice leaves the run
    as it entered (zero-coupling contract).
  * PARALLEL (``replay_parallel=True``, needs ``lattice_path``): one
    scan_pool ``ScanPoint`` per bunch; workers re-parse the deck and
    apply the same override pairs.  A LATTICE FINGERPRINT (element count
    + name-sequence hash) of the live lattice is checked against a
    driver-side parse of ``lattice_path`` with the WORKER'S OWN parser
    (``_parse_lattice_for_scan``) at construction — a deck that does not
    describe the live lattice is refused before anything runs.

Documented v1 approximations / contracts:

* **No pass-2 → pass-1 feedback**: the fast recursion's loss-free,
  rigid-time-of-flight state is final; replayed bunches do not update
  the phasor histories (a replay that loses particles would, in a
  coupled world, load the cavities slightly less for later bunches).
* Replays are MP only (the deliverable of the hybrid mode is full-MP
  fidelity; envelope replays add nothing over mode="envelope").
* ``physics.direct_sc`` composes into the replays with the pattern
  image factors of each slot ("images" neighbours only — an independent
  replay has no previously tracked bunch to snapshot — and in-process
  only: the per-bunch PIC hooks do not transport through scan_pool).
* Parallel replay reproduces the FILE lattice + transported overrides:
  manual in-memory edits to elements OUTSIDE the transported attributes
  are not carried (the fingerprint checks structure/names, not every
  parameter) — keep the deck authoritative or replay in-process.
* Hybrid with all coupling physics off is REFUSED at TrainConfig
  construction (beam_loading is the minimum) — a two-pass replay of
  bit-identical bunches is a mislabelled single-bunch run.

The lossless-construction contract (tests/train/test_replay.py): a
replay fed a TRACKED run's own recorded state (``applied_loading`` /
``applied_hom`` / ``pin_by_index`` via ``overrides_for_tracked_slot``)
reproduces that tracked bunch BIT-IDENTICALLY.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np

from linac_gen.train.config import TrainConfig
from linac_gen.train.results import TrainResults

_MISSING = object()

#: Element attributes the replay override transport may touch (single
#: source of truth for application, snapshot and teardown).
REPLAY_ATTRS = ("voltage_rel", "phase_offset", "sync_phase_pin",
                "hom_kick_x", "hom_kick_y")


# ---------------------------------------------------------------------------
# selection / fingerprint / override construction
# ---------------------------------------------------------------------------
def auto_select_bunches(pattern, cap: int = 24) -> list[int]:
    """The ``select_bunches="auto"`` policy: every PATTERN EDGE (first and
    last bunch of each filled run — where beam-loading transients start
    and peak) plus log-spaced interior bunches (the loading build-up is
    ~geometric, so early bunches get denser coverage), capped at ~``cap``
    total.  Edges are never dropped: a pattern with more than ``cap``
    edges keeps them all (the cap bounds only the interior fill).
    Returns sorted absolute slot indices (all filled)."""
    filled = pattern.filled
    starts = np.flatnonzero(filled & ~np.r_[False, filled[:-1]])
    ends = np.flatnonzero(filled & ~np.r_[filled[1:], False])
    out = {int(s) for s in starts} | {int(e) for e in ends}
    slots = pattern.filled_slots
    n_extra = int(cap) - len(out)
    if n_extra > 0 and slots.size > len(out):
        # log-spaced ORDINALS over the filled-bunch axis (1..n_bunches),
        # mapped back to absolute slots; endpoint duplicates of the
        # pulse-edge slots collapse into the set.
        ords = np.unique(np.rint(
            np.geomspace(1.0, float(slots.size), n_extra + 2)
        ).astype(np.int64)) - 1
        out.update(int(slots[o]) for o in ords)
    return sorted(out)


def lattice_fingerprint(lattice) -> str:
    """Cheap structural identity: element count + SHA-256 of the
    top-level name sequence (the same identity the @index/NAME override
    selectors address by)."""
    names = [str(getattr(el, "name", type(el).__name__))
             for el in lattice.elements]
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return f"{len(names)}:{digest[:16]}"


def build_overrides(loading: dict, hom: dict, pins: dict) -> list:
    """Serialise one bunch's recorded per-cavity state into
    ``(selector, value)`` pairs for ``apply_element_override``.

    ``loading``: {(element_index, name): (voltage_rel, phase_offset)};
    ``hom``:     {(element_index, name): (dxp_mrad, dyp_mrad)};
    ``pins``:    {element_index: psi_deg}.

    Selectors use the unambiguous 1-based ``@index`` form (deck names
    need not be unique).  Zero HOM kicks are skipped — the tracked hook
    guards ``!= 0.0`` before touching the particles, so omitting them is
    exactly equivalent (and keeps a fresh lattice's attributes at their
    inert class default).  Values are floats; ``apply_element_override``
    round-trips them through ``str`` (repr) losslessly.
    """
    out = []
    for (idx, _name), (vr, po) in sorted(loading.items()):
        out.append((f"@{idx + 1}.voltage_rel", float(vr)))
        out.append((f"@{idx + 1}.phase_offset", float(po)))
    for (idx, _name), (kx, ky) in sorted(hom.items()):
        if float(kx) != 0.0:
            out.append((f"@{idx + 1}.hom_kick_x", float(kx)))
        if float(ky) != 0.0:
            out.append((f"@{idx + 1}.hom_kick_y", float(ky)))
    for idx, psi in sorted(pins.items()):
        out.append((f"@{idx + 1}.sync_phase_pin", float(psi)))
    return out


def overrides_for_tracked_slot(results: TrainResults, slot: int) -> list:
    """Override list reconstructing bunch ``slot`` of a TRACKED
    TrainRunner run from its own recorded applied values — the
    lossless-construction seam (anchor test 1)."""
    if results.pins_unindexed:
        raise NotImplementedError(
            "replay override transport cannot address pinned "
            "SuperposedFieldMap children (not top-level elements): "
            + ", ".join(results.pins_unindexed))
    slot = int(slot)
    loading = {(i, n): v for (s, i, n), v in results.applied_loading.items()
               if s == slot}
    hom = {(i, n): v for (s, i, n), v in results.applied_hom.items()
           if s == slot}
    return build_overrides(loading, hom, results.pin_by_index)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
class HybridReplayRunner:
    """mode="hybrid": fast pass over the full pattern, then full-MP
    replay of the selected bunches (module docstring has the model).

    Composition, not inheritance: pass 1 IS a FastPulseRunner (which
    validates the whole TrainRunner lifecycle — sidecar, periodic_phase,
    bunch-rate reconciliation, FREQ-jump warning — on construction);
    pass 2 is plain ``Simulation`` runs over transported overrides.
    """

    def __init__(self, lattice, beam_config, train_config: TrainConfig,
                 sc_config=None, lattice_path=None, replay_parallel=False,
                 max_workers=None, history_stride: int = 1,
                 progress_callback=None, should_abort=None, **sim_kwargs):
        if not isinstance(train_config, TrainConfig):
            raise TypeError("train_config must be a TrainConfig")
        if train_config.mode != "hybrid":
            raise ValueError(
                f"HybridReplayRunner requires train_config.mode='hybrid' "
                f"(got {train_config.mode!r})")
        self.lattice = lattice
        self.train = train_config
        self.sc_config = sc_config
        self.lattice_path = lattice_path
        self.replay_parallel = bool(replay_parallel)
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.should_abort = should_abort
        self.sim_kwargs = dict(sim_kwargs)
        # ---- selection --------------------------------------------------
        sel = train_config.select_bunches
        self._selected = (auto_select_bunches(train_config.pattern)
                          if sel == "auto" else list(sel))
        # ---- direct-SC composition (in-process only) --------------------
        self._direct_sc = bool(train_config.physics.direct_sc)
        self._dsc_factors = None
        self._dsc_engaged_any = False
        self._last_pic = None
        if self._direct_sc:
            from linac_gen.train.driver import TrainRunner
            TrainRunner._check_direct_sc_config(sc_config)
        # ---- parallel-branch prerequisites (checked BEFORE any work) ----
        if self.replay_parallel:
            self._validate_parallel_branch(beam_config)
        # ---- pass 1 (constructs = validates everything else) ------------
        import dataclasses
        # select_bunches is reset for the derived fast config: pass 1
        # receives the selection through ``record_slots`` below, and an
        # explicit selection on a non-hybrid TrainConfig is refused
        # (config M8 honesty guard).
        cfg_fast = dataclasses.replace(
            train_config, mode="fast", select_bunches="auto",
            physics=dataclasses.replace(train_config.physics,
                                        direct_sc=False))
        from linac_gen.train.fast import FastPulseRunner
        self._p1 = FastPulseRunner(
            lattice, beam_config, cfg_fast, sc_config=sc_config,
            history_stride=history_stride, record_slots=self._selected,
            should_abort=should_abort, **sim_kwargs)
        # Post-replace beam config (bunch_frequency_MHz reconciled) — the
        # single beam definition for every replay branch.
        self.beam_config = self._p1.beam_config

    # ------------------------------------------------------------------
    def _validate_parallel_branch(self, beam_config) -> None:
        if not self.lattice_path:
            raise ValueError(
                "replay_parallel=True requires lattice_path (the deck "
                "the scan_pool workers re-parse); omit it to replay "
                "in-process on the live lattice")
        if self._direct_sc:
            raise ValueError(
                "physics.direct_sc replays run in-process only: the "
                "per-bunch PIC image factors ride on a pic_setup_hook, "
                "which does not transport through scan_pool workers — "
                "drop replay_parallel or disable direct_sc")
        if self.sim_kwargs:
            raise ValueError(
                "replay_parallel=True cannot transport extra Simulation "
                f"kwargs {sorted(self.sim_kwargs)} through scan_pool "
                "workers — drop them or replay in-process")
        current = float(getattr(beam_config, "current", 0.0) or 0.0)
        sc = self.sc_config
        if current > 0.0:
            from linac_gen.core.config import SpaceChargeConfig
            if not isinstance(sc, SpaceChargeConfig):
                raise ValueError(
                    "replay_parallel=True with beam current > 0 needs a "
                    "SpaceChargeConfig: the scan_pool worker always "
                    "builds space charge for current-carrying beams, so "
                    f"sc_config={sc!r} could not be reproduced — pass "
                    "the config or replay in-process")
            if not (sc.ny == sc.nx and sc.nz == sc.nx):
                raise ValueError(
                    "replay_parallel=True supports only cubic SC grids "
                    f"(ScanPoint carries one nx; got {sc.nx}x{sc.ny}x"
                    f"{sc.nz}) — use nx=ny=nz or replay in-process")
        # THE fingerprint check: parse the deck with the WORKER'S OWN
        # parser and compare against the live lattice.
        from linac_gen.parallel.scan_pool import _parse_lattice_for_scan
        fp_live = lattice_fingerprint(self.lattice)
        fp_file = lattice_fingerprint(_parse_lattice_for_scan(
            str(self.lattice_path)))
        if fp_live != fp_file:
            raise ValueError(
                f"lattice fingerprint mismatch: live lattice is "
                f"{fp_live} but {self.lattice_path!r} parses to "
                f"{fp_file} (element count : name-sequence hash) — the "
                "deck does not describe the lattice pass 1 will run on; "
                "refusing the parallel replay")

    # ------------------------------------------------------------------
    def run(self) -> TrainResults:
        res1 = self._p1.run()
        results = TrainResults(self.train, "hybrid")
        # Save-time context (M7) — same references the fast pass carried.
        results.beam_config = self.beam_config
        results.sc_config = self.sc_config
        results.lattice = self.lattice
        results.lattice_path = self.lattice_path
        results.design_result = res1.design_result
        results.pins = dict(res1.pins)
        results.pin_by_index = dict(res1.pin_by_index)
        results.pins_unindexed = list(res1.pins_unindexed)
        results.fast = res1.fast
        if results.pins_unindexed:
            raise NotImplementedError(
                "replay override transport cannot address pinned "
                "SuperposedFieldMap children (not top-level elements): "
                + ", ".join(results.pins_unindexed))
        recs = self._p1.slot_records
        missing = [s for s in self._selected if s not in recs]
        if missing:
            warnings.warn(
                f"hybrid replay: pass 1 recorded no state for slot(s) "
                f"{missing} (aborted pass?); replaying only the "
                f"{len(recs)} recorded bunches", stacklevel=2)
        slots = [s for s in self._selected if s in recs]
        for s in slots:
            results.replay_overrides[s] = build_overrides(
                recs[s]["loading"], recs[s]["hom"], results.pin_by_index)
        if self.replay_parallel:
            self._replay_parallel(slots, results)
        else:
            self._replay_serial(slots, results)
        if self._direct_sc and results.replay_bunches \
                and not self._dsc_engaged_any:
            warnings.warn(
                "physics.direct_sc was enabled but the bunch-train "
                "images never engaged on any replayed bunch — the "
                "pattern factors were inert.  Set "
                "TrainConfig.direct_sc_force_engage=True to model "
                "neighbours regardless of bunch length.", stacklevel=2)
        return results

    # ------------------------------------------------------------ serial
    def _replay_serial(self, slots, results: TrainResults) -> None:
        n = len(slots)
        for i, slot in enumerate(slots):
            if self.should_abort is not None and self.should_abort():
                warnings.warn(
                    f"hybrid replay aborted after {i}/{n} bunches; "
                    "partial results returned", stacklevel=2)
                results.truncated = True
                break
            results.replay_bunches[slot] = self._replay_one(
                slot, results.replay_overrides[slot], results)
            if self.progress_callback is not None:
                try:
                    self.progress_callback(i + 1, n)
                except Exception:                             # noqa: BLE001
                    pass

    def _replay_one(self, slot: int, overrides, results: TrainResults):
        """One in-process replay: apply the overrides to the LIVE
        lattice, run a fresh Simulation (no hooks — the state is
        static), restore the exact prior attribute state in a finally
        block (instance-dict level, so an attribute that was a class
        default returns to being a class default)."""
        from linac_gen.cli.common import apply_element_override
        from linac_gen.core.simulation import Simulation
        from linac_gen.distributions.factory import create_beam

        lat = self.lattice
        saved = []
        for sel_str, _val in overrides:
            target, attr = sel_str.rsplit(".", 1)
            el = lat.elements[int(target[1:]) - 1]
            saved.append((el, attr, el.__dict__.get(attr, _MISSING)))
        try:
            for sel_str, val in overrides:
                apply_element_override(lat, sel_str, val)
            beam = create_beam(self.beam_config, seed=self.train.seed)
            kw = dict(self.sim_kwargs)
            if self._direct_sc:
                from linac_gen.train.driver import pattern_image_factors
                self._dsc_factors = pattern_image_factors(
                    self.train.pattern, slot)
                results.direct_sc[slot] = self._dsc_factors
                kw["pic_setup_hook"] = self._pic_setup
            res = Simulation(lat, beam, space_charge=self.sc_config,
                             **kw).run()
            if self._last_pic is not None \
                    and self._last_pic._train_ever_engaged:
                self._dsc_engaged_any = True
            return res
        finally:
            for el, attr, old in saved:
                if old is _MISSING:
                    el.__dict__.pop(attr, None)
                else:
                    setattr(el, attr, old)
            self._dsc_factors = None
            if self._last_pic is not None:
                self._last_pic.train_image_factors = None
                self._last_pic.train_force_engage = False
                self._last_pic = None

    def _pic_setup(self, pic) -> None:
        """Simulation ``pic_setup_hook`` for direct-SC replays (bound
        method, no lambdas): arm THIS bunch's pattern image factors on
        the freshly built numpy PIC ("images" neighbours only)."""
        from linac_gen.pic.pic_solver import PicSolver
        if not isinstance(pic, PicSolver):
            raise NotImplementedError(
                "physics.direct_sc: the built space-charge solver is "
                f"{type(pic).__name__}, not the numpy PicSolver — no "
                "bunch-train path")
        pic.train_image_factors = self._dsc_factors
        pic.train_force_engage = bool(self.train.direct_sc_force_engage)
        self._last_pic = pic

    # ---------------------------------------------------------- parallel
    def _replay_parallel(self, slots, results: TrainResults) -> None:
        """One scan_pool ScanPoint per selected bunch; workers re-parse
        ``lattice_path``, apply the override pairs, run full MP and
        write full results HDF5s into a scratch dir, which the driver
        loads back (arrays are float64 → exact round-trip) and removes.
        """
        import shutil
        import tempfile
        from dataclasses import asdict, fields as dc_fields
        from pathlib import Path
        from types import SimpleNamespace

        from linac_gen.io.hdf5_output import load_results_hdf5
        from linac_gen.parallel.scan_pool import ScanPoint, run_scan_points

        if not slots:
            return
        step = self.lattice.step_config
        sc = self.sc_config
        current = float(getattr(self.beam_config, "current", 0.0) or 0.0)
        if current > 0.0:
            nx = int(sc.nx)
            grid_extent = float(sc.grid_extent)
            use_gpu = str(sc.use_gpu)
            sc_overrides = tuple(sorted(
                (f.name, getattr(sc, f.name)) for f in dc_fields(sc)
                if f.name not in ("nx", "ny", "nz", "grid_extent",
                                  "use_gpu")))
        else:
            from linac_gen.core.config import SpaceChargeConfig
            d = SpaceChargeConfig()          # never built by the worker
            nx, grid_extent, use_gpu = d.nx, d.grid_extent, "cpu"
            sc_overrides = ()
        beam_dict = asdict(self.beam_config)
        tmpdir = tempfile.mkdtemp(prefix="helix_hybrid_replay_")
        try:
            points = [
                ScanPoint(
                    lattice_path=str(self.lattice_path),
                    beam_config=beam_dict,
                    nx=nx, grid_extent=grid_extent,
                    step1=float(step.integration_steps_per_metre),
                    step2=float(step.sc_steps_per_metre),
                    seed=int(self.train.seed), use_gpu=use_gpu,
                    element_overrides=tuple(results.replay_overrides[s]),
                    sc_overrides=sc_overrides,
                    out_path=str(Path(tmpdir) / f"replay_{s:06d}.h5"),
                )
                for s in slots
            ]
            rows = run_scan_points(points, max_workers=self.max_workers,
                                   should_stop=self.should_abort)
            if len(rows) < len(points):
                warnings.warn(
                    f"hybrid replay aborted: {len(rows)}/{len(points)} "
                    "bunches completed; partial results returned",
                    stacklevel=2)
                results.truncated = True
            for row in rows:
                path = row.get("results_path")
                if not path:                 # pragma: no cover - guard
                    raise RuntimeError(
                        f"scan_pool row carried no results_path: {row}")
                slot = int(Path(path).stem.split("_")[-1])
                results.replay_bunches[slot] = SimpleNamespace(
                    **load_results_hdf5(path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
