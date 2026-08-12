"""Results container + HDF5 persistence for the multibunch / pulse study.

On-disk schema (M7, ``schema_version`` = 1) — written by
:meth:`TrainResults.save_hdf5`, read by :func:`load_train_results`::

    results.h5
    ├── provenance/       same attr group as single-bunch files
    │                     (io/hdf5_output._write_provenance) PLUS
    │                     run_type = "train" — single-bunch files carry
    │                     no run_type and their schema is untouched
    ├── beam_config/      scalar beam-config attrs (single-bunch convention)
    ├── train/
    │   ├── attrs         schema_version, mode, bunch_frequency_MHz,
    │   │                 n_slots, n_bunches_pattern, n_bunches_tracked,
    │   │                 truncated (True when the run was aborted
    │   │                 mid-train — tracked/replay break, or the fast
    │   │                 recursion's own flag; M8, optional attr:
    │   │                 pre-M8 files load as truncated=False),
    │   │                 pattern_rle, seed, keep_full_results,
    │   │                 physics_{beam_loading,hom,direct_sc},
    │   │                 charge_scale ("none"; "custom" unreachable in
    │   │                 v1 — TrainConfig refuses the inert knob),
    │   │                 cavity_params,
    │   │                 select_bunches ("auto"|"explicit"),
    │   │                 direct_sc_* knobs, jitter_* (when set),
    │   │                 w_design_exit_MeV (when a design pass ran),
    │   │                 n_bunches_replayed (hybrid)
    │   ├── pattern       uint8[n_slots] (1 = filled slot)
    │   ├── selected_slots int64[]           (explicit hybrid selection)
    │   ├── summary/      slot,ordinal int64[] + per-bunch end-of-lattice
    │   │                 float64 columns (_SUMMARY_KEYS + mean_* centroid)
    │   ├── pins/         name/psi_deg (by cavity name), element_index/
    │   │                 psi_by_index (top-level pins), unindexed_name
    │   ├── cavity_state/ tracked-mode beam loading (physics.beam_loading,
    │   │   └── CAV_%03d/ mode="mp"): static attrs (name, element_index,
    │   │                 frequency_MHz, v_design_MV, phi_s_deg,
    │   │                 dw_design_MeV, r_over_q, q_loaded, detuning_Hz)
    │   │                 + slot/voltage_rel/phase_offset_deg (applied,
    │   │                 prior-composed — the entry-hook ledger) +
    │   │                 v_beam_slot/v_beam_re_MV/v_beam_im_MV (phasor
    │   │                 AFTER each bunch's full kick — the exit-hook
    │   │                 ledger; a separate slot axis because a bunch
    │   │                 dying inside a cavity records entry, not exit)
    │   ├── hom/          tracked-mode dipole HOMs (physics.hom, "mp"):
    │   │   └── CAV_%03d/ attrs name/element_index; modes/{f_MHz,
    │   │                 r_over_q_t,q_loaded,polarization_deg} float64[
    │   │                 n_modes]; slot int64[] + kick_xp_mrad/
    │   │                 kick_yp_mrad (per-bunch applied kicks) +
    │   │                 w_re_MV/w_im_MV float64[n_modes, n_hits]
    │   │                 (per-mode wake AFTER each bunch's excitation)
    │   ├── fast/         FastPulseSummary.write_hdf5 (M3 layout + the
    │   │                 M7 ``truncated`` abort flag): slot/w_exit_MeV/
    │   │                 history_slot + cavities/c_%04d stride-decimated
    │   │                 phasor histories & design ledger
    │   ├── direct_sc/    slot/f_lead/f_trail pattern image factors
    │   ├── replay_slots  int64[] (completed hybrid replays — pre-M7
    │   │                 name, kept)
    │   └── replay/
    │       └── b_%04d/   selector (utf-8 str[]) + value float64[] — the
    │                     exact ``apply_element_override`` pairs the
    │                     replay ran with
    └── bunches/
        └── b_%04d/       per-bunch full results, FLAT single-bunch
                          content (same quantities as the single-bunch
                          envelope/ + reference/ groups, reference keys
                          prefixed ref_*, plus the (n,6) ``centroid``
                          history and int64 ``element_exit_idx``); attrs
                          continuous/current_mA when known.  Written for
                          tracked bunches only when keep_full_results
                          (else summary-only) and for every completed
                          hybrid replay.  Tracked and replayed bunches
                          never coexist (the modes are exclusive);
                          tracked slots are ``train/summary/slot``,
                          replayed slots ``train/replay_slots``.

SINGLE-BUNCH FILES ARE UNCHANGED — io/hdf5_output.py is not touched by
the train schema; old readers keep working (product requirement).

run_type: train files are stamped ``provenance/run_type = "train"``.
The GUI auto-dump seam (gui/linac_gen_gui/interphase/app.py,
``_auto_dump_results``) uses run_type only as the FILENAME token
(``<ts>_<run_type>.h5``) and calls ``save_results_hdf5`` on a recorder;
TrainResults is not a recorder, so the GUI dispatch (TrainResults →
``TrainResults.save_hdf5`` under the "train" filename token) lands with
the M8 GUI surface — core takes no GUI dependency here.
"""
from __future__ import annotations

import numpy as np

from linac_gen.train.config import PulsePattern, TrainJitter, TrainPhysics

#: On-disk train schema version (train.attrs["schema_version"]).
TRAIN_SCHEMA_VERSION = 1

_SUMMARY_KEYS = ("sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                 "emit_x", "emit_y", "emit_z", "transmission",
                 "ref_w_kin")
#: Final-centroid summary columns (recorder/EnvelopeResults ``centroid``
#: convention: x mm, x' mrad, y mm, y' mrad, Δφ deg, ΔW MeV).
_CENTROID_COLS = ("mean_x", "mean_xp", "mean_y", "mean_yp",
                  "mean_phi", "mean_w")

#: Per-bunch float64 series persisted under bunches/b_%04d — the flat
#: union of the single-bunch envelope/ + reference/ (ref_*) quantities.
#: Absent attributes (e.g. envelope-mode results have no transmission)
#: are simply skipped.
_BUNCH_FLOAT_KEYS = (
    "s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
    "emit_x", "emit_y", "emit_z", "emit_nx", "emit_ny",
    "alpha_x", "beta_x", "alpha_y", "beta_y", "alpha_z", "beta_z",
    "halo_x", "halo_y", "transmission",
    "ref_w_kin", "ref_phi_s", "ref_beta", "ref_gamma", "ref_bg")


def _summary_row(result) -> dict:
    """End-of-lattice scalar row for one bunch (NaN where unavailable).

    Computed at append time so summary-only trains
    (keep_full_results=False) keep a REAL summary table after the full
    per-bunch result is dropped.
    """
    row = {}
    for k in _SUMMARY_KEYS:
        arr = getattr(result, k, None) if result is not None else None
        val = float("nan")
        if arr is not None and len(arr):
            try:
                val = float(arr[-1])
            except (TypeError, ValueError):
                val = float("nan")
        row[k] = val
    cent = getattr(result, "centroid", None) if result is not None else None
    c = None
    if cent is not None and len(cent):
        c = np.asarray(cent[-1], dtype=np.float64).reshape(-1)
        if c.size != 6:
            c = None
    if c is None:
        c = np.full(6, np.nan)
    for j, k in enumerate(_CENTROID_COLS):
        row[k] = float(c[j])
    return row


class TrainResults:
    """Per-bunch results of a train run.

    ``bunch_results`` holds the per-bunch recorder/results objects (None
    when keep_full_results=False); ``summary()`` gives one row per bunch
    of end-of-lattice scalars, per-bunch moments ONLY (train-wide moments
    are biased; see diagnostics/moments.py docstring).  Rows are captured
    by :meth:`append` from the full result BEFORE it may be dropped.
    """

    def __init__(self, train_config, mode: str):
        self.config = train_config
        self.mode = mode
        self.slots: list[int] = []
        self.bunch_results: list = []
        self._summary_rows: list[dict] = []
        self.design_result = None
        # Design-pass SET_SYNC_PHASE pins by element name (the lattice
        # itself is scrubbed after the run; M6 replay re-applies these
        # via element_overrides).
        self.pins: dict[str, float] = {}
        # ---- M6 hybrid-replay bookkeeping -----------------------------
        # Pins again, keyed by TOP-LEVEL element index (the @index
        # override selector); names in ``pins_unindexed`` are pinned
        # elements that are NOT top-level (SuperposedFieldMap children)
        # and therefore not transportable — replay refuses when any
        # exist.
        self.pin_by_index: dict[int, float] = {}
        self.pins_unindexed: list[str] = []
        # Tracked-mode applied ledgers (see BeamLoadingManager.applied /
        # HomManager.applied): (slot, element_index, name) -> values as
        # written/applied by the entry hooks.  Feed
        # replay.overrides_for_tracked_slot for lossless reconstruction.
        self.applied_loading: dict = {}
        self.applied_hom: dict = {}
        # Tracked-mode histories (M7 persistence): beam-induced phasor
        # after each bunch's full kick, and per-mode HOM wake envelopes
        # after each bunch's excitation — same (slot, element_index,
        # name) keys as the applied ledgers.
        self.v_beam_loading: dict = {}
        self.hom_w: dict = {}
        # Static per-cavity table from the registry (element_index, name,
        # frequency/design/mode parameters, HOMMode tuple) — the attrs of
        # the cavity_state/ and hom/ groups.
        self.cavity_table: list[dict] = []
        # Hybrid results: slot -> full-MP replay results for the selected
        # bunches (DiagnosticRecorder in-process; a namespace of loaded
        # arrays from the parallel scan_pool branch) + the override lists
        # each replay ran with (provenance).
        self.replay_bunches: dict[int, object] = {}
        self.replay_overrides: dict[int, list] = {}
        # M3 fast-mode product (fast.FastPulseSummary): per-bunch W_exit
        # + per-cavity applied/phasor histories.  None for tracked modes
        # (fast mode conversely tracks no bunches: slots/bunch_results
        # stay empty and n_bunches lives under train/fast/).
        self.fast = None
        # M5 direct-SC provenance: slot -> (f_lead, f_trail) pattern
        # image factors the driver applied to that bunch (empty unless
        # physics.direct_sc).
        self.direct_sc: dict[int, tuple] = {}
        # M8 abort marker (tracked/hybrid mirror of the fast summary's
        # ``truncated``): True when the run stopped between bunches on
        # should_abort — the partial result stays fully loadable.
        self.truncated = False
        # Save-time context (set by the runners; overridable per call in
        # save_hdf5) — provenance needs the run's configuration.
        self.beam_config = None
        self.sc_config = None
        self.lattice = None
        self.lattice_path = None

    def append(self, slot: int, result, keep: bool = True) -> None:
        """Record one tracked bunch: summary row always, full result only
        when ``keep`` (keep_full_results)."""
        self.slots.append(int(slot))
        self._summary_rows.append(_summary_row(result))
        self.bunch_results.append(result if keep else None)

    def summary(self) -> dict:
        slots = np.asarray(self.slots, dtype=np.int64)
        out = {"slot": slots}
        # Filled ordinal: position of each tracked slot along the
        # pattern's filled-bunch axis (0-based).
        filled = self.config.pattern.filled_slots
        out["ordinal"] = np.searchsorted(filled, slots).astype(np.int64)
        for k in _SUMMARY_KEYS + _CENTROID_COLS:
            out[k] = np.asarray([row[k] for row in self._summary_rows],
                                dtype=np.float64)
        return out

    # ---- persistence ---------------------------------------------------
    def save_hdf5(self, path: str, *, beam_config=None, lattice=None,
                  lattice_path=None, sc_config=None) -> None:
        """Write the full train schema (module docstring has the tree).

        The keyword arguments override the save-time context the runner
        attached (``results.beam_config`` etc.); they feed the same
        ``provenance/`` group single-bunch files carry, plus
        ``run_type = "train"``.
        """
        import h5py

        from linac_gen.io.hdf5_output import _write_provenance

        beam_config = beam_config if beam_config is not None \
            else self.beam_config
        lattice = lattice if lattice is not None else self.lattice
        lattice_path = lattice_path if lattice_path is not None \
            else self.lattice_path
        sc_config = sc_config if sc_config is not None else self.sc_config
        str_dt = h5py.string_dtype(encoding="utf-8")

        def _write_str(grp, name, values):
            ds = grp.create_dataset(name, shape=(len(values),),
                                    dtype=str_dt)
            if len(values):
                ds[:] = [str(v) for v in values]

        with h5py.File(path, "w") as f:
            # ---- provenance (single-bunch convention + run_type) -----
            _write_provenance(f, lattice_path=lattice_path,
                              seed=int(self.config.seed),
                              sc_config=sc_config, lattice=lattice)
            f["provenance"].attrs["run_type"] = "train"
            if beam_config is not None:
                cfg = f.create_group("beam_config")
                for key, val in beam_config.__dict__.items():
                    if val is not None:
                        try:
                            cfg.attrs[key] = val
                        except TypeError:
                            pass        # h5py-unencodable — skip
            # ---- train/ ----------------------------------------------
            tc = self.config
            tr = f.create_group("train")
            tr.attrs["schema_version"] = int(TRAIN_SCHEMA_VERSION)
            tr.attrs["mode"] = str(self.mode)
            tr.attrs["bunch_frequency_MHz"] = float(tc.bunch_frequency_MHz)
            tr.attrs["n_slots"] = int(tc.pattern.n_slots)
            tr.attrs["n_bunches_pattern"] = int(tc.pattern.n_bunches)
            tr.attrs["n_bunches_tracked"] = len(self.slots)
            tr.attrs["truncated"] = bool(
                getattr(self, "truncated", False)
                or (self.fast is not None
                    and getattr(self.fast, "truncated", False)))
            tr.attrs["pattern_rle"] = tc.pattern.to_rle()
            tr.attrs["seed"] = int(tc.seed)
            tr.attrs["keep_full_results"] = bool(tc.keep_full_results)
            tr.attrs["physics_beam_loading"] = bool(tc.physics.beam_loading)
            tr.attrs["physics_hom"] = bool(tc.physics.hom)
            tr.attrs["physics_direct_sc"] = bool(tc.physics.direct_sc)
            tr.attrs["charge_scale"] = ("none" if tc.charge_scale is None
                                        else "custom")
            tr.attrs["cavity_params"] = str(tc.cavity_params or "")
            tr.attrs["direct_sc_neighbors"] = str(tc.direct_sc_neighbors)
            tr.attrs["direct_sc_force_engage"] = \
                bool(tc.direct_sc_force_engage)
            tr.attrs["direct_sc_subsample"] = int(tc.direct_sc_subsample)
            if tc.jitter is not None:
                tr.attrs["jitter_phase_deg_rms"] = \
                    float(tc.jitter.phase_deg_rms)
                tr.attrs["jitter_amplitude_rel_rms"] = \
                    float(tc.jitter.amplitude_rel_rms)
                tr.attrs["jitter_charge_rel_rms"] = \
                    float(tc.jitter.charge_rel_rms)
                tr.attrs["jitter_seed"] = int(tc.jitter.seed)
            sb = tc.select_bunches
            if isinstance(sb, str):
                tr.attrs["select_bunches"] = sb
            else:
                tr.attrs["select_bunches"] = "explicit"
                tr.create_dataset(
                    "selected_slots",
                    data=np.asarray(sorted(int(s) for s in sb), np.int64))
            dr = self.design_result
            if dr is not None:
                wk = getattr(dr, "ref_w_kin", None)
                if wk is not None and len(wk):
                    tr.attrs["w_design_exit_MeV"] = float(wk[-1])
            tr.create_dataset("pattern",
                              data=tc.pattern.filled.astype(np.uint8))
            sg = tr.create_group("summary")
            for k, v in self.summary().items():
                sg.create_dataset(k, data=v)
            # ---- pins ------------------------------------------------
            if self.pins or self.pin_by_index or self.pins_unindexed:
                pg = tr.create_group("pins")
                names = sorted(self.pins)
                _write_str(pg, "name", names)
                pg.create_dataset(
                    "psi_deg",
                    data=np.asarray([self.pins[n] for n in names],
                                    np.float64))
                idxs = sorted(self.pin_by_index)
                pg.create_dataset("element_index",
                                  data=np.asarray(idxs, np.int64))
                pg.create_dataset(
                    "psi_by_index",
                    data=np.asarray([self.pin_by_index[i] for i in idxs],
                                    np.float64))
                _write_str(pg, "unindexed_name", self.pins_unindexed)
            # ---- cavity_state/ (tracked beam loading) ----------------
            if self.cavity_table and tc.physics.beam_loading \
                    and self.mode == "mp":
                cs = tr.create_group("cavity_state")
                for rec in self.cavity_table:
                    key = (rec["element_index"], rec["name"])
                    g = cs.create_group(f"CAV_{key[0]:03d}")
                    for a in ("name", "element_index", "frequency_MHz",
                              "v_design_MV", "phi_s_deg", "dw_design_MeV",
                              "r_over_q", "q_loaded", "detuning_Hz"):
                        g.attrs[a] = rec[a]
                    s_app = sorted(s for (s, i, n) in self.applied_loading
                                   if (i, n) == key)
                    g.create_dataset("slot",
                                     data=np.asarray(s_app, np.int64))
                    for j, dsname in ((0, "voltage_rel"),
                                      (1, "phase_offset_deg")):
                        g.create_dataset(dsname, data=np.asarray(
                            [self.applied_loading[(s, *key)][j]
                             for s in s_app], np.float64))
                    s_vb = sorted(s for (s, i, n) in self.v_beam_loading
                                  if (i, n) == key)
                    vb = np.asarray(
                        [self.v_beam_loading[(s, *key)] for s in s_vb],
                        np.complex128)
                    g.create_dataset("v_beam_slot",
                                     data=np.asarray(s_vb, np.int64))
                    g.create_dataset("v_beam_re_MV",
                                     data=vb.real.astype(np.float64))
                    g.create_dataset("v_beam_im_MV",
                                     data=vb.imag.astype(np.float64))
            # ---- hom/ (tracked dipole HOMs) --------------------------
            if self.cavity_table and tc.physics.hom and self.mode == "mp":
                hg_root = tr.create_group("hom")
                for rec in self.cavity_table:
                    modes = rec["hom_modes"]
                    if not modes:
                        continue
                    key = (rec["element_index"], rec["name"])
                    g = hg_root.create_group(f"CAV_{key[0]:03d}")
                    g.attrs["name"] = rec["name"]
                    g.attrs["element_index"] = rec["element_index"]
                    mg = g.create_group("modes")
                    for a in ("f_MHz", "r_over_q_t", "q_loaded",
                              "polarization_deg"):
                        mg.create_dataset(a, data=np.asarray(
                            [getattr(m, a) for m in modes], np.float64))
                    s_hom = sorted(s for (s, i, n) in self.applied_hom
                                   if (i, n) == key)
                    g.create_dataset("slot",
                                     data=np.asarray(s_hom, np.int64))
                    for j, dsname in ((0, "kick_xp_mrad"),
                                      (1, "kick_yp_mrad")):
                        g.create_dataset(dsname, data=np.asarray(
                            [self.applied_hom[(s, *key)][j]
                             for s in s_hom], np.float64))
                    w = np.asarray(
                        [self.hom_w[(s, *key)] for s in s_hom],
                        np.complex128).reshape(len(s_hom), len(modes)).T
                    g.create_dataset("w_re_MV",
                                     data=w.real.astype(np.float64))
                    g.create_dataset("w_im_MV",
                                     data=w.imag.astype(np.float64))
            # ---- fast/ ------------------------------------------------
            if self.fast is not None:
                self.fast.write_hdf5(tr.create_group("fast"))
            # ---- direct_sc/ ------------------------------------------
            if self.direct_sc:
                dg = tr.create_group("direct_sc")
                s_dsc = sorted(self.direct_sc)
                dg.create_dataset("slot", data=np.asarray(s_dsc, np.int64))
                dg.create_dataset("f_lead", data=np.asarray(
                    [self.direct_sc[s][0] for s in s_dsc], np.float64))
                dg.create_dataset("f_trail", data=np.asarray(
                    [self.direct_sc[s][1] for s in s_dsc], np.float64))
            # ---- tracked bunches -------------------------------------
            for slot, r in zip(self.slots, self.bunch_results):
                if r is None:
                    continue
                _write_bunch(f, slot, r)
            # ---- hybrid replays --------------------------------------
            if self.replay_bunches:
                tr.attrs["n_bunches_replayed"] = len(self.replay_bunches)
                tr.create_dataset(
                    "replay_slots",
                    data=np.asarray(sorted(self.replay_bunches), np.int64))
                for slot in sorted(self.replay_bunches):
                    _write_bunch(f, slot, self.replay_bunches[slot])
            if self.replay_overrides:
                rg = tr.create_group("replay")
                for slot in sorted(self.replay_overrides):
                    bg = rg.create_group(f"b_{slot:04d}")
                    pairs = self.replay_overrides[slot]
                    _write_str(bg, "selector", [p[0] for p in pairs])
                    bg.create_dataset("value", data=np.asarray(
                        [float(p[1]) for p in pairs], np.float64))


def _write_bunch(f, slot: int, r) -> None:
    """One bunches/b_%04d group: flat single-bunch content (module
    docstring), from a recorder/EnvelopeResults or a loaded namespace."""
    bg = f.create_group(f"bunches/b_{slot:04d}")
    for k in _BUNCH_FLOAT_KEYS:
        arr = getattr(r, k, None)
        if arr is not None and len(arr):
            bg.create_dataset(k, data=np.asarray(arr, np.float64))
    cent = getattr(r, "centroid", None)
    if cent is not None and len(cent):
        c = np.asarray(cent, np.float64)
        if c.ndim == 2 and c.shape[1] == 6:
            bg.create_dataset("centroid", data=c)
    idx = getattr(r, "element_exit_idx", None)
    if idx is not None and len(idx):
        bg.create_dataset("element_exit_idx",
                          data=np.asarray(idx, np.int64))
    cont = getattr(r, "continuous", None)
    if cont is not None:
        bg.attrs["continuous"] = bool(cont)
    cur = getattr(r, "current_mA", None)
    if cur is not None:
        bg.attrs["current_mA"] = float(cur)


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------
class LoadedTrainResults:
    """TrainResults-like namespace produced by :func:`load_train_results`.

    Field-compatible with TrainResults where the data exists on disk
    (slots / bunch_results / pins / applied_* / replay_* / fast / mode /
    direct_sc ...), plus the loaded config scalars (pattern,
    bunch_frequency_MHz, physics, seed, ...) and ``provenance``.
    ``summary`` is the loaded TABLE (dict of arrays), not a method;
    ``design_result`` is always None (the design pass is not persisted —
    only ``w_design_exit_MeV``).
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"LoadedTrainResults(mode={self.mode!r}, "
                f"n_bunches_tracked={len(self.slots)}, "
                f"n_replayed={len(self.replay_bunches)}, "
                f"fast={'yes' if self.fast is not None else 'no'})")


def _as_str(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _load_group_ns(g):
    """SimpleNamespace of every dataset (arrays) + attr in a group."""
    from types import SimpleNamespace
    d = {k: g[k][:] for k in g}
    for k, v in g.attrs.items():
        d[k] = v.item() if hasattr(v, "item") else v
    return SimpleNamespace(**d)


def _load_fast_group(g):
    """Reconstruct a FastPulseSummary (real dataclasses) from train/fast."""
    from linac_gen.train.fast import FastCavityRecord, FastPulseSummary
    from linac_gen.train.hom import HOMMode

    cavities = []
    cg = g.get("cavities")
    if cg is not None:
        for key in sorted(cg, key=lambda k: int(k.rsplit("_", 1)[-1])):
            c = cg[key]
            hom_modes = ()
            if "hom_modes" in c:
                hm = c["hom_modes"]
                cols = {a: hm[a][:] for a in ("f_MHz", "r_over_q_t",
                                              "q_loaded",
                                              "polarization_deg")}
                hom_modes = tuple(
                    HOMMode(f_MHz=float(cols["f_MHz"][i]),
                            r_over_q_t=float(cols["r_over_q_t"][i]),
                            q_loaded=float(cols["q_loaded"][i]),
                            polarization_deg=float(
                                cols["polarization_deg"][i]))
                    for i in range(len(cols["f_MHz"])))
            cavities.append(FastCavityRecord(
                element_index=int(c.attrs["element_index"]),
                name=_as_str(c.attrs["name"]),
                frequency_MHz=float(c.attrs["frequency_MHz"]),
                v_design_MV=float(c.attrs["v_design_MV"]),
                phi_s_deg=float(c.attrs["phi_s_deg"]),
                dw_design_MeV=float(c.attrs["dw_design_MeV"]),
                r_over_q=float(c.attrs["r_over_q"]),
                q_loaded=float(c.attrs["q_loaded"]),
                detuning_Hz=float(c.attrs["detuning_Hz"]),
                voltage_rel=c["voltage_rel"][:],
                phase_offset_deg=c["phase_offset_deg"][:],
                v_beam_MV=(c["v_beam_re_MV"][:]
                           + 1j * c["v_beam_im_MV"][:]),
                centroid=(c["centroid"][:] if "centroid" in c else None),
                hom_w=((c["hom_w_re_MV"][:] + 1j * c["hom_w_im_MV"][:])
                       if "hom_w_re_MV" in c else None),
                hom_modes=hom_modes,
            ))
    return FastPulseSummary(
        slot=g["slot"][:].astype(np.int64),
        w_exit_MeV=g["w_exit_MeV"][:],
        w_design_exit_MeV=float(g.attrs["w_design_exit_MeV"]),
        charge_per_bunch_C=float(g.attrs["charge_per_bunch_C"]),
        history_stride=int(g.attrs["history_stride"]),
        history_slot=g["history_slot"][:].astype(np.int64),
        cavities=cavities,
        truncated=bool(g.attrs["truncated"]),
    )


def load_train_results(path: str) -> LoadedTrainResults:
    """Load a multibunch train results file written by
    :meth:`TrainResults.save_hdf5` (any mode: mp / envelope / fast /
    hybrid; full or summary-only).

    Refuses loudly on non-train files — single-bunch results load through
    :func:`linac_gen.io.hdf5_output.load_results_hdf5` (that path is
    untouched by the train schema).
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "train" not in f:
            raise ValueError(
                f"{path}: not a multibunch train results file (no "
                "'train' group) — single-bunch results load through "
                "linac_gen.io.hdf5_output.load_results_hdf5")
        tr = f["train"]
        if "schema_version" not in tr.attrs:
            raise ValueError(
                f"{path}: train results file predates the M7 schema "
                "(train group without schema_version) — re-run the study "
                "with this HELIX to produce a loadable file")
        a = tr.attrs
        physics = TrainPhysics(
            direct_sc=bool(a["physics_direct_sc"]),
            beam_loading=bool(a["physics_beam_loading"]),
            hom=bool(a["physics_hom"]))
        jitter = None
        if "jitter_phase_deg_rms" in a:
            jitter = TrainJitter(
                phase_deg_rms=float(a["jitter_phase_deg_rms"]),
                amplitude_rel_rms=float(a["jitter_amplitude_rel_rms"]),
                charge_rel_rms=float(a["jitter_charge_rel_rms"]),
                seed=int(a["jitter_seed"]))
        select_bunches: object = _as_str(a["select_bunches"])
        if select_bunches == "explicit":
            select_bunches = [int(s) for s in tr["selected_slots"][:]]
        summary = {k: tr["summary"][k][:] for k in tr["summary"]}
        if "slot" not in summary:
            raise ValueError(
                f"{path}: train/summary has no 'slot' column — corrupt "
                "or truncated train results file")
        slots = [int(s) for s in summary["slot"]]
        # ---- pins ----------------------------------------------------
        pins: dict = {}
        pin_by_index: dict = {}
        pins_unindexed: list = []
        if "pins" in tr:
            pg = tr["pins"]
            names = [_as_str(n) for n in pg["name"].asstr()[:]] \
                if pg["name"].shape[0] else []
            for n, psi in zip(names, pg["psi_deg"][:]):
                pins[n] = float(psi)
            for i, psi in zip(pg["element_index"][:],
                              pg["psi_by_index"][:]):
                pin_by_index[int(i)] = float(psi)
            if pg["unindexed_name"].shape[0]:
                pins_unindexed = [_as_str(n)
                                  for n in pg["unindexed_name"].asstr()[:]]
        # ---- cavity_state / hom (tracked ledgers) --------------------
        applied_loading: dict = {}
        v_beam_loading: dict = {}
        applied_hom: dict = {}
        hom_w: dict = {}
        cavity_state: dict = {}
        hom: dict = {}
        if "cavity_state" in tr:
            for key in sorted(tr["cavity_state"]):
                g = tr["cavity_state"][key]
                idx = int(g.attrs["element_index"])
                name = _as_str(g.attrs["name"])
                cavity_state[(idx, name)] = _load_group_ns(g)
                for s, vr, po in zip(g["slot"][:], g["voltage_rel"][:],
                                     g["phase_offset_deg"][:]):
                    applied_loading[(int(s), idx, name)] = (float(vr),
                                                            float(po))
                for s, re_, im_ in zip(g["v_beam_slot"][:],
                                       g["v_beam_re_MV"][:],
                                       g["v_beam_im_MV"][:]):
                    v_beam_loading[(int(s), idx, name)] = complex(re_, im_)
        if "hom" in tr:
            from types import SimpleNamespace

            from linac_gen.train.hom import HOMMode
            for key in sorted(tr["hom"]):
                g = tr["hom"][key]
                idx = int(g.attrs["element_index"])
                name = _as_str(g.attrs["name"])
                mg = g["modes"]
                modes = tuple(
                    HOMMode(f_MHz=float(mg["f_MHz"][i]),
                            r_over_q_t=float(mg["r_over_q_t"][i]),
                            q_loaded=float(mg["q_loaded"][i]),
                            polarization_deg=float(
                                mg["polarization_deg"][i]))
                    for i in range(mg["f_MHz"].shape[0]))
                w = g["w_re_MV"][:] + 1j * g["w_im_MV"][:]
                s_arr = g["slot"][:]
                hom[(idx, name)] = SimpleNamespace(
                    name=name, element_index=idx, modes=modes,
                    slot=s_arr.astype(np.int64),
                    kick_xp_mrad=g["kick_xp_mrad"][:],
                    kick_yp_mrad=g["kick_yp_mrad"][:], w=w)
                for j, s in enumerate(s_arr):
                    applied_hom[(int(s), idx, name)] = (
                        float(g["kick_xp_mrad"][j]),
                        float(g["kick_yp_mrad"][j]))
                    hom_w[(int(s), idx, name)] = tuple(
                        complex(w[m, j]) for m in range(w.shape[0]))
        # ---- bunches -------------------------------------------------
        bunches: dict = {}
        if "bunches" in f:
            for key in f["bunches"]:
                bunches[int(key.rsplit("_", 1)[-1])] = \
                    _load_group_ns(f["bunches"][key])
        replay_slots = ([int(s) for s in tr["replay_slots"][:]]
                        if "replay_slots" in tr else [])
        replay_overrides: dict = {}
        if "replay" in tr:
            for key in tr["replay"]:
                g = tr["replay"][key]
                sels = ([_as_str(s) for s in g["selector"].asstr()[:]]
                        if g["selector"].shape[0] else [])
                replay_overrides[int(key.rsplit("_", 1)[-1])] = \
                    list(zip(sels, [float(v) for v in g["value"][:]]))
        direct_sc: dict = {}
        if "direct_sc" in tr:
            dg = tr["direct_sc"]
            for s, fl, ft in zip(dg["slot"][:], dg["f_lead"][:],
                                 dg["f_trail"][:]):
                direct_sc[int(s)] = (float(fl), float(ft))
        prov = {}
        if "provenance" in f:
            for k, v in f["provenance"].attrs.items():
                prov[k] = v.item() if hasattr(v, "item") else v
        beam_config = {}
        if "beam_config" in f:
            for k, v in f["beam_config"].attrs.items():
                beam_config[k] = v.item() if hasattr(v, "item") else v
        return LoadedTrainResults(
            schema_version=int(a["schema_version"]),
            run_type=_as_str(prov.get("run_type", "train")),
            mode=_as_str(a["mode"]),
            bunch_frequency_MHz=float(a["bunch_frequency_MHz"]),
            n_slots=int(a["n_slots"]),
            n_bunches_pattern=int(a["n_bunches_pattern"]),
            n_bunches_tracked=int(a["n_bunches_tracked"]),
            n_bunches_replayed=int(a.get("n_bunches_replayed", 0)),
            truncated=bool(a.get("truncated", False)),
            pattern=PulsePattern(tr["pattern"][:].astype(bool)),
            pattern_rle=_as_str(a["pattern_rle"]),
            seed=int(a["seed"]),
            keep_full_results=bool(a["keep_full_results"]),
            physics=physics,
            charge_scale=_as_str(a["charge_scale"]),
            cavity_params=(_as_str(a["cavity_params"]) or None),
            select_bunches=select_bunches,
            jitter=jitter,
            w_design_exit_MeV=(float(a["w_design_exit_MeV"])
                               if "w_design_exit_MeV" in a else None),
            summary=summary,
            slots=slots,
            bunch_results=[bunches.get(s) for s in slots],
            design_result=None,
            pins=pins, pin_by_index=pin_by_index,
            pins_unindexed=pins_unindexed,
            applied_loading=applied_loading,
            v_beam_loading=v_beam_loading,
            applied_hom=applied_hom, hom_w=hom_w,
            cavity_state=cavity_state, hom=hom,
            fast=(_load_fast_group(tr["fast"]) if "fast" in tr else None),
            replay_slots=replay_slots,
            replay_bunches={s: bunches[s] for s in replay_slots
                            if s in bunches},
            replay_overrides=replay_overrides,
            direct_sc=direct_sc,
            provenance=prov,
            beam_config=beam_config,
        )
