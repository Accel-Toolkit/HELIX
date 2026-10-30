"""Run a chain of ``RfqCell`` elements in the time domain, replay it per cell.

The multiparticle tracker walks the lattice one element at a time.  For
an RFQ in time-domain mode, the first cell of a contiguous chain runs the
whole chain at once (:class:`~linac_gen.rfq_td.tracker.TimeDomainTracker`)
and keeps every particle's crossing of every cell exit; each cell then
loads its own exit state into the beam.  Everything around the element
call — hooks, the bunch-train marker, the aperture check, the recorder
row, snapshots, progress — runs as for any other element, so a
time-domain RFQ keeps one result row, name and loss bookkeeping per cell.

Reference particle: advanced through the chain by the cells' own code on
an empty beam with the multiparticle sub-step sequence, so ``ref`` is
exactly what a per-cell run produces.  Particle phases are relative to
it; the RF phase origin is free for a DC-injected beam (a time shift), so
the bunch core is centred at the chain exit and the same constant is used
at every cell.  Each particle is assigned to the main bunch by whole RF
periods (exact for a steady DC-injected train — the time-domain space
charge assumes the same periodicity).
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

_log = logging.getLogger(__name__)

#: rows per cell when the run records sub-steps (``record_substeps``): a
#: subset of the per-cell model's own sub-step record points, the last at
#: the cell exit (the per-cell model records every 0.1 mm -- tens of
#: thousands of planes over an RFQ, too many to keep for every particle)
SUBPLANES_PER_CELL = 4


class RfqTdError(RuntimeError):
    """The time-domain RFQ solver cannot run this configuration."""


_SCALED: dict = {}        # the last voltage-scaled map (repeated runs)


def _scaled(fmap, factor):
    key = (fmap.key, float(factor))
    hit = _SCALED.get(key)
    if hit is None:
        _SCALED.clear()
        hit = _SCALED[key] = fmap.scaled(factor)
    return hit


def find_chains(elements):
    """Index ranges ``(first, last)`` of contiguous ``RfqCell`` chains;
    zero-length elements between cells belong to the chain."""
    from linac_gen.elements.rfq_cell import RfqCell
    chains = []
    i, n = 0, len(elements)
    while i < n:
        if isinstance(elements[i], RfqCell):
            j = last = i
            while j + 1 < n:
                el = elements[j + 1]
                if isinstance(el, RfqCell):
                    j += 1
                    last = j
                elif float(getattr(el, "length", 0.0) or 0.0) <= 1e-9:
                    j += 1
                else:
                    break
            chains.append((i, last))
            i = last + 1
        else:
            i += 1
    return chains


def _note_once(lattice, level, msg, *args):
    """Log ``msg % args`` once per lattice: "auto" checks the RFQ before
    the run and the tracker again, and repeated runs on one lattice (the
    GUI) need not repeat it."""
    seen = getattr(lattice, "_rfq_td_notes", None)
    if seen is None:
        seen = set()
        try:
            lattice._rfq_td_notes = seen
        except AttributeError:
            pass
    key = (msg, args)
    if key in seen:
        return
    seen.add(key)
    _log.log(level, msg, *args)


class _Probe:
    """What :class:`RfqTdDriver`'s up-front checks read from a tracker."""

    def __init__(self, lattice, beam, entry_hook, exit_hook):
        self.lattice = lattice
        self.beam = beam
        self._element_entry_hook = entry_hook
        self._element_exit_hook = exit_hook


def time_domain_refusal(lattice, beam, entry_hook=None, exit_hook=None):
    """Why the time-domain solver cannot run this lattice and beam -- the
    message of the :class:`RfqTdError` its up-front checks raise -- or None
    when it can.  Tracks nothing and builds no field map."""
    try:
        RfqTdDriver(_Probe(lattice, beam, entry_hook, exit_hook))
    except RfqTdError as exc:
        return str(exc)
    return None


def resolve_rfq_solver(lattice, beam, entry_hook=None, exit_hook=None):
    """The RFQ solver one multiparticle run uses, as ``(solver, note)``.

    ``solver`` is None for a lattice without RFQ cells, else
    ``"time_domain"`` or ``"cells"``.  ``StepConfig.rfq_solver = "auto"``
    gives ``"time_domain"`` when the time-domain solver can run the lattice
    and beam, else ``"cells"`` with ``note`` saying why (also logged, once
    per lattice and reason).  RFQ cells that carry a vane-geometry profile
    armed for the per-cell model (``apply_rfq_geometry``) are an explicit
    per-cell set-up: "auto" keeps them.  An explicit choice is returned as
    it is: an explicit ``"time_domain"`` that cannot run fails in the
    tracker."""
    from linac_gen.elements.rfq_cell import RfqCell
    cells = [e for e in lattice.elements if isinstance(e, RfqCell)]
    if not cells:
        return None, None
    choice = getattr(getattr(lattice, "step_config", None), "rfq_solver",
                     "auto")
    if choice != "auto":
        return choice, None
    if any(getattr(c, "_geom_z", None) is not None for c in cells):
        return keep_cells(lattice, "the RFQ cells carry a vane-geometry "
                          "profile armed for the per-cell model", quiet=True)
    why = time_domain_refusal(lattice, beam, entry_hook, exit_hook)
    if why is None:
        return "time_domain", None
    return keep_cells(lattice, why)


def keep_cells(lattice, why, quiet=False):
    """``("cells", why)`` for ``rfq_solver = "auto"``, noted once per
    lattice and reason: a warning when the time-domain solver cannot run,
    information (``quiet``) when the caller set up the per-cell model."""
    _note_once(lattice, logging.INFO if quiet else logging.WARNING,
               "RFQ solver auto: the RFQ is tracked cell by cell (per-cell "
               "model): %s", why)
    return "cells", why


@dataclass
class _ChainRun:
    cells: list                      # RfqCell elements in order
    entry_s: float                   # mm
    edges: np.ndarray                # (C+1,) chain-local cell edges, mm
    ref_tab: np.ndarray              # (C, 4) w_kin, phi_s, s, frequency at exits
    alive_in: np.ndarray             # (N_alive,) particle ids tracked
    states: np.ndarray               # (C, N_alive, 6) float32 HELIX coords
    crossed: np.ndarray              # (C, N_alive) bool
    loss_cell: np.ndarray            # (N_alive,) cell index of the loss, -1 none
    loss_z: np.ndarray
    loss_xy: np.ndarray
    loss_W: np.ndarray
    exit_idx: np.ndarray             # (C,) record-plane index of each cell exit
    plane_z: np.ndarray              # (P,) chain-local record planes, mm
    plane_ref: np.ndarray            # (P, 4) ref at each record plane
    train_gate: bool = False         # PIC train-image gate at the chain exit
    info: dict = field(default_factory=dict)
    # particles that leave the chain unaccelerated (rfq_stats_accelerated):
    # their beam ids and the record plane from which the statistics leave
    # them out
    behind_pid: np.ndarray = field(
        default_factory=lambda: np.zeros(0, np.int64))
    behind_plane: np.ndarray = field(
        default_factory=lambda: np.zeros(0, np.int64))


def chain_electrodes(lattice, cells, chain_len, notes=True):
    """The electrodes of one RFQ cell chain as the time-domain solver uses
    them, and the factor its field map is scaled by: ``(electrodes,
    voltage_scale)``.  From the deck's ``RFQ_GEOM`` vane file (scale = the
    cells' voltage over the file's inter-vane voltage), else from the
    ``RFQ_CELL`` cards at unit voltage (scale = the cells' voltage).
    Raises :class:`RfqTdError` when the vanes cannot be used.  Also used by
    the GUI's RFQ panels, so they show the solver's own geometry; they pass
    ``notes=False``, which leaves the solver's once-per-lattice log notes to
    the run."""
    from linac_gen.io.tracewin_vane import parse_vane_file
    from linac_gen.rfq_td.geometry import EndPlate, RfqElectrodes
    lat = lattice
    if not lat.rfq_geom_file:
        from linac_gen.rfq_td.cards_geometry import electrodes_from_cards
        if notes:
            _note_once(lat, logging.INFO,
                       "time-domain RFQ: no RFQ_GEOM vane file -- the "
                       "electrodes are built from the RFQ_CELL cards")
        try:
            # at unit voltage: one field map serves every vane voltage
            el = electrodes_from_cards(
                cells, gaps=getattr(lat, "rfq_gaps_mm", None),
                voltage=1.0)
        except ValueError as exc:
            raise RfqTdError(f"RFQ electrodes from the cards: {exc}") \
                from exc
        return el, float(cells[0].voltage_V)
    from linac_gen.rfq_td.geometry import end_plate_hole
    vane = parse_vane_file(lat.rfq_geom_file)
    plates = [EndPlate(z_face=0.0, side=-1,
                       hole_radius=end_plate_hole(cells[0].r0_mm)),
              EndPlate(z_face=chain_len, side=+1,
                       hole_radius=end_plate_hole(cells[-1].r0_mm))]
    el = RfqElectrodes.from_vane(vane, plates=plates,
                                 source=str(lat.rfq_geom_file))
    el.check_symmetric()
    zs, ze = el.z_span
    if zs < -1e-6 or ze > chain_len + 1e-3:
        raise RfqTdError(
            f"vane table z {zs:.3f}..{ze:.3f} mm does not fit inside the "
            f"RFQ cell chain 0..{chain_len:.3f} mm -- wrong vane file?")
    gaps = getattr(lat, "rfq_gaps_mm", None)
    if gaps is not None and notes:
        if abs(zs - gaps[0]) > 0.05 or abs(chain_len - ze - gaps[1]) > 0.05:
            _note_once(
                lat, logging.WARNING,
                "RFQ_GAP_RMS_FFS (%.4f, %.4f mm) disagrees with the vane "
                "table (vanes %.4f..%.4f mm in a %.4f mm chain); the "
                "vane table is used", gaps[0], gaps[1], zs, ze, chain_len)
    # one voltage for all cells (validated); scale the vane field to it
    v_cells = float(cells[0].voltage_V)
    v_vane = abs(el.V_y - el.V_x)
    if v_vane <= 0:
        raise RfqTdError("vane file has no inter-vane voltage")
    return el, v_cells / v_vane


class RfqTdDriver:
    """Owns the RFQ chains of one tracker run in time-domain mode."""

    def __init__(self, tracker):
        from linac_gen.elements.rfq_cell import RfqCell
        from linac_gen.elements.marker import Marker
        self.tr = tracker
        lat = tracker.lattice
        cfg = getattr(lat, "step_config", None)
        self.steps = int(getattr(cfg, "rfq_td_steps_per_period", 48))
        self.sc_every = int(getattr(cfg, "rfq_td_sc_every", 4))
        self.stats_accelerated = bool(getattr(cfg, "rfq_stats_accelerated",
                                              True))
        # the particles this run leaves out of the statistics are marked as
        # its chains are replayed; nothing is left over from an earlier run
        tracker.beam.stats_excluded = None
        self.chains = find_chains(lat.elements)
        self._owner = {}
        for ci, (i0, i1) in enumerate(self.chains):
            for k in range(i0, i1 + 1):
                self._owner[id(lat.elements[k])] = ci
        self._runs: dict = {}
        self._entry_dc: dict = {}
        # ---- validate before anything is tracked ------------------------
        if tracker._element_entry_hook is not None or \
                tracker._element_exit_hook is not None:
            raise RfqTdError("the time-domain RFQ solver does not support "
                             "element hooks (bunch-train runs) yet")
        if len(self.chains) > 1:
            raise RfqTdError(f"{len(self.chains)} separate RFQ cell chains; "
                             "the time-domain solver handles one RFQ per "
                             "lattice")
        for (i0, i1) in self.chains:
            els = lat.elements[i0:i1 + 1]
            cells = [e for e in els if isinstance(e, RfqCell)]
            for e in els:
                if not isinstance(e, (RfqCell, Marker)):
                    raise RfqTdError(
                        f"'{e.name}' ({type(e).__name__}) sits between RFQ "
                        "cells; the time-domain solver allows only markers "
                        "there")
            align = {(float(getattr(c, "dx", 0.0) or 0.0),
                      float(getattr(c, "dy", 0.0) or 0.0),
                      float(getattr(c, "tilt_deg", 0.0) or 0.0))
                     for c in cells}
            if len(align) > 1:
                raise RfqTdError("the RFQ cells carry different misalignments; "
                                 "the time-domain solver moves the RFQ as one "
                                 "body (equal dx, dy, tilt on every cell)")
            volts = {float(c.voltage_V) for c in cells}
            if len(volts) > 1:
                raise RfqTdError("the RFQ cells carry different vane voltages; "
                                 "the time-domain solver needs one voltage")
            if lat.rfq_geom_file and not os.path.isfile(lat.rfq_geom_file):
                raise RfqTdError(
                    "the time-domain RFQ solver cannot read the vane "
                    f"geometry {lat.rfq_geom_file!r} named by RFQ_GEOM "
                    "(without an RFQ_GEOM card the electrodes are built "
                    "from the RFQ_CELL cards)")
            if len(cells) < 3:
                raise RfqTdError("the time-domain RFQ solver needs at least "
                                 f"3 RFQ cells (this chain has {len(cells)})")
        # the beam must reach the RFQ as DC: injected DC, and no RF buncher
        # upstream (a static electric element -- einzel lens, extraction
        # column -- clears Beam.continuous without bunching anything)
        from linac_gen.tracking.tracker import (_is_rf_bunching_element,
                                                _is_time_varying_rf_buncher)
        self._dc_upstream: dict = {}
        for ci, (i0, _i1) in enumerate(self.chains):
            if not getattr(tracker.beam, "continuous", False):
                raise RfqTdError(
                    "the time-domain RFQ solver needs a DC (continuous) beam "
                    "at the RFQ entrance; this beam starts bunched")
            for e in lat.elements[:i0]:
                if _is_time_varying_rf_buncher(e) or (
                        type(e).__name__ == "VaneRFQ"
                        and _is_rf_bunching_element(e)):
                    raise RfqTdError(
                        "the time-domain RFQ solver needs a DC beam at the "
                        f"RFQ entrance; '{e.name}' ({type(e).__name__}) "
                        "bunches it upstream")
            self._dc_upstream[ci] = True
        # the electrodes and the field map's availability, now
        self._geom: dict = {}
        for ci, (i0, i1) in enumerate(self.chains):
            cells = [e for e in lat.elements[i0:i1 + 1]
                     if isinstance(e, RfqCell)]
            edges = np.concatenate([[0.0], np.cumsum([c.length
                                                      for c in cells])])
            try:
                self._geom[ci] = self._electrodes(cells, float(edges[-1]))
            except RfqTdError:
                raise
            except (ValueError, OSError) as exc:
                raise RfqTdError(f"the RFQ vanes cannot be used: {exc}") \
                    from exc
            from linac_gen.rfq_td.build import cache_writable, map_cached
            if not map_cached(self._geom[ci][0], edges):
                try:
                    import pyamg  # noqa: F401
                except ImportError as exc:
                    raise RfqTdError(
                        "building the time-domain RFQ field map needs pyamg "
                        "(pip install pyamg); a map already cached loads "
                        "without it") from exc
                why = cache_writable()
                if why:
                    raise RfqTdError(
                        "the time-domain RFQ field map cannot be cached ("
                        f"{why}); point HELIX_CACHE_DIR at a writable "
                        "directory")

    def _geometry(self, ci, cells, chain_len):
        """The chain's electrodes and voltage scale (built at start)."""
        got = self._geom.get(ci)
        return got if got is not None else self._electrodes(cells, chain_len)

    # ------------------------------------------------------------------
    def owns(self, element) -> bool:
        return id(element) in self._owner

    def note_entry(self, element, continuous: bool) -> None:
        """Called by the tracker before the DC→bunched flip at an owned
        element: the beam's state at the chain entrance."""
        ci = self._owner.get(id(element))
        if ci is not None and ci not in self._entry_dc:
            self._entry_dc[ci] = bool(continuous)

    def track(self, element) -> None:
        """Stand-in for the element's own tracking."""
        ci = self._owner[id(element)]
        if ci not in self._runs:
            self._runs[ci] = self._run_chain(ci)
        self._load(self._runs[ci], element)

    # ------------------------------------------------------------------
    def _reference_table(self, cells, n_sub: int = 0):
        """``ref`` (w_kin, phi_s, s, frequency) at every cell exit, advanced
        by the cells' own code on an empty beam with the multiparticle
        sub-step sequence.  With ``n_sub`` > 0 also, per cell, at ``n_sub``
        of the points where a per-cell run with ``record_substeps`` records
        a row (the last one is the exit): a list of (n, 4) arrays."""
        from linac_gen.core.beam import Beam
        from linac_gen.tracking.tracker import field_map_bundle_sizes, field_map_step_counts
        beam = self.tr.beam
        ref = beam.ref.copy()
        ghost = Beam(ref=ref, n_particles=1, current=0.0)
        ghost.lost[:] = True
        tab, sub = [], []

        def point():
            return (ref.w_kin, ref.phi_s, ref.s,
                    float(getattr(ref, "frequency", 0.0)))

        for c in cells:
            c.reset_run_state()
            n_int, _n_sc, ds, sc_every, n_bundles = field_map_step_counts(
                self.tr.lattice, c)
            pts = []
            # the per-cell walk records after every second half-step of a
            # bundle, counted through the kick, the short last bundle
            # included (Tracker._track_field_map)
            for k_steps in field_map_bundle_sizes(n_int, sc_every, n_bundles):
                for _k in range(2 * k_steps):
                    c.track_rk4(ghost, ds / 2)
                    if n_sub and _k % 2 == 1:
                        pts.append(point())
            c.reset_run_state()
            tab.append(point())
            if n_sub:
                m = min(n_sub, len(pts))
                pick = [int(round((j + 1) * len(pts) / m)) - 1
                        for j in range(m)]
                sub.append(np.array([pts[j] for j in pick]))
        return np.array(tab), sub

    def _electrodes(self, cells, chain_len):
        return chain_electrodes(self.tr.lattice, cells, chain_len)

    def _run_chain(self, ci):
        from linac_gen.elements.rfq_cell import RfqCell
        from linac_gen.pic.macrocharge import macro_charge_for
        from linac_gen.rfq_td.build import get_field_map
        from linac_gen.rfq_td.space_charge import PeriodicTrainSpaceCharge
        from linac_gen.rfq_td.tracker import (REASON_OUTSIDE, Aborted,
                                              TimeDomainTracker)
        tr = self.tr
        beam = tr.beam
        i0, i1 = self.chains[ci]
        els = tr.lattice.elements[i0:i1 + 1]
        cells = [e for e in els if isinstance(e, RfqCell)]
        if not (self._entry_dc.get(ci, False)
                or self._dc_upstream.get(ci, False)):
            raise RfqTdError(
                "the time-domain RFQ solver needs a DC (continuous) beam at "
                "the RFQ entrance; this beam arrives bunched")
        t_start = time.time()
        edges = np.concatenate([[0.0], np.cumsum([c.length for c in cells])])
        entry_s = float(beam.ref.s)
        n_sub = SUBPLANES_PER_CELL if tr._record_substeps else 0
        ref_tab, sub_tab = self._reference_table(cells, n_sub)
        if n_sub:
            # record planes: each cell's sub-step points; its last is the exit
            plane_ref = np.concatenate(sub_tab)
            exit_idx = np.cumsum([len(a) for a in sub_tab]) - 1
            plane_z = plane_ref[:, 2] - entry_s
            plane_z[exit_idx] = edges[1:]
            plane_ref[exit_idx] = ref_tab
        else:
            plane_ref = ref_tab
            exit_idx = np.arange(len(cells))
            plane_z = edges[1:].copy()
        alive = np.flatnonzero(beam.alive_mask)
        if alive.size == 0:
            # nothing reaches the RFQ: no map needed, the cells only move ref
            nP = plane_z.size
            info = dict(steps=0, wall_s=round(time.time() - t_start, 2),
                        track_s=0.0, sc=False, sc_solves=0, map_key="",
                        outside_map_steps=0, n_tracked=0, n_lost=0,
                        n_stalled=0, n_left_map=0,
                        stats_accelerated=self.stats_accelerated,
                        n_unaccelerated=0, n_left_out=0,
                        phase_origin_deg=0.0,
                        exit_w_offset_kev=0.0,
                        steps_per_period=self.steps, sc_every=self.sc_every,
                        geometry="", voltage_scale=1.0)
            tr.recorder.rfq_td = info
            return _ChainRun(
                cells=cells, entry_s=entry_s, edges=edges, ref_tab=ref_tab,
                alive_in=alive, states=np.zeros((nP, 0, 6), np.float32),
                crossed=np.zeros((nP, 0), bool),
                loss_cell=np.zeros(0, np.int64), loss_z=np.zeros(0),
                loss_xy=np.zeros((0, 2)), loss_W=np.zeros(0),
                exit_idx=exit_idx, plane_z=plane_z, plane_ref=plane_ref,
                train_gate=False, info=info)
        el, vscale = self._geometry(ci, cells, float(edges[-1]))
        fmap = get_field_map(el, edges, should_abort=tr._should_abort)
        if abs(vscale - 1.0) > 1e-12:
            fmap = _scaled(fmap, vscale)
        # ---- particles at the chain entrance -> common start time ---------
        ref = beam.ref
        mass = float(ref.species.mass)
        charge = float(ref.species.charge)
        f_mhz = float(ref.frequency)
        f_ghz = f_mhz * 1e-3
        T = 1.0 / f_ghz
        P = beam.particles[alive]
        W = ref.w_kin + P[:, 5]
        g = 1.0 + W / mass
        bg = np.sqrt(np.maximum(g * g - 1.0, 0.0))
        xp, yp = P[:, 1] * 1e-3, P[:, 3] * 1e-3
        uz = bg / np.sqrt(1.0 + xp * xp + yp * yp)
        ux, uy = uz * xp, uz * yp
        vz = 299.792458 * uz / g
        t_arr = P[:, 4] / (360.0 * f_ghz)              # ns after ref
        t0 = float(t_arr.min()) - 0.25 * T
        Z = -vz * (t_arr - t0)
        X = P[:, 0] + xp * Z
        Y = P[:, 2] + yp * Z
        # ---- space charge as the tracker would apply it --------------------
        sc = None
        if tr._space_charge_active():
            q = macro_charge_for(beam) * float(tr._sc_factor)
            sc = PeriodicTrainSpaceCharge(q, float(ref.wavelength))
        planes = plane_z
        t_ref = (plane_ref[:, 1] - ref.phi_s) / (360.0 * f_ghz)   # ns
        z_len = float(edges[-1])
        n_el = len(tr.lattice.elements)

        def progress(z_med, _z_end):
            if tr._progress_callback is not None:
                try:
                    tr._progress_callback(entry_s + max(0.0, z_med), i0 + 1,
                                          n_el)
                except Exception:
                    pass

        # stragglers: three times the transit time at the entrance velocity
        v_in = float(np.median(vz))
        t_max = 3.0 * (z_len + 20.0) / max(v_in, 1e-9) + (0.0 - t0)
        tdt = TimeDomainTracker(fmap, el, freq_mhz=f_mhz, mass_mev=mass,
                                charge=charge,
                                steps_per_period=self.steps, space_charge=sc,
                                sc_every=self.sc_every, t_max_ns=t_max)
        res = tdt.run(X, Y, Z, ux, uy, uz, t0, planes, t_ref=t_ref,
                      w_ref=plane_ref[:, 0], progress=progress,
                      should_abort=tr._should_abort)
        # ---- phase origin and buckets ---------------------------------------
        # rec holds x, x', y, y', t - t_ref[k], W - w_ref[k] (float32).  A
        # DC-injected beam is one period of a steady train with no phase
        # origin of its own: the bunch core is centred at the chain exit
        # (circular mean of the exit phases).  The time-domain bucket need
        # not line up with the injection window, so the captured particles
        # can fill two neighbouring buckets; which one is a step in the
        # injection phase.  Each particle therefore gets a whole number of
        # periods from its INJECTION phase (one boundary fitted on the
        # captured particles; in the periodic-train model this equals
        # re-wrapping its injection by whole periods): the captured
        # particles leave as one bunch, a straggler keeps its lag, and every
        # particle's phase is continuous along the RFQ.
        rec = res.rec
        nP = planes.size
        w = 2.0 * math.pi * f_ghz
        ex = rec[-1]
        out = np.isfinite(ex[:, 4])
        if out.any():
            th = w * ex[out, 4].astype(np.float64)
            core = math.atan2(float(np.mean(np.sin(th))),
                              float(np.mean(np.cos(th))))
        else:
            core = 0.0
        d_exit = np.full(alive.size, np.nan)
        d_exit[out] = np.degrees(w * ex[out, 4].astype(np.float64) - core)
        periods = self._bucket_periods(P[:, 4], d_exit)
        crossed = np.isfinite(rec[:, :, 4])
        # to HELIX coordinates, in place: x' and y' in mrad, phase in deg
        shift = np.degrees(core) + 360.0 * periods
        for k in range(nP):
            r = rec[k]
            r[:, 1] *= 1e3
            r[:, 3] *= 1e3
            r[:, 4] = np.degrees(w * r[:, 4].astype(np.float64)) - shift
        states = rec
        loss_cell = np.full(alive.size, -1, np.int64)
        lz = res.loss_z
        lost = res.lost
        n_cells = len(cells)
        loss_cell[lost] = np.clip(np.searchsorted(edges, lz[lost],
                                                  side="right") - 1,
                                  0, n_cells - 1)
        gate = self._train_gate(states, crossed, exit_idx)
        behind = (self._behind_planes(states, crossed, plane_ref[:, 0],
                                      float(ref.w_kin))
                  if self.stats_accelerated else np.full(alive.size, -1))
        b_sel = np.flatnonzero(behind >= 0)
        n_unacc = int(np.count_nonzero(crossed[-1, b_sel]))
        n_left_map = int((res.loss_reason == REASON_OUTSIDE).sum())
        # where the bunch leaves relative to the reference particle, which
        # follows the RFQ_CELL cards (as TraceWin's does)
        w_off = (float(np.median(states[-1][out, 5].astype(np.float64)))
                 if out.any() else 0.0)
        w_ref_exit = float(plane_ref[-1, 0])
        if abs(w_off) > 0.005 * w_ref_exit:
            _log.warning(
                "time-domain RFQ: the bunch leaves the RFQ %+.1f keV from "
                "the reference particle (%.2f %% of %.4f MeV); the "
                "reference follows the RFQ_CELL cards, as in TraceWin, and "
                "downstream RF elements are phased on it", 1e3 * w_off,
                100.0 * w_off / w_ref_exit, w_ref_exit)
        info = dict(steps=res.steps, wall_s=round(time.time() - t_start, 2),
                    track_s=round(res.wall_s, 2), sc=sc is not None,
                    sc_solves=res.sc_solves, map_key=fmap.key,
                    outside_map_steps=res.outside_map_steps,
                    n_tracked=int(alive.size), n_lost=int(lost.sum()),
                    n_stalled=int((res.loss_reason == 3).sum()),
                    n_left_map=n_left_map,
                    stats_accelerated=self.stats_accelerated,
                    n_unaccelerated=n_unacc, n_left_out=int(b_sel.size),
                    phase_origin_deg=math.degrees(core),
                    bucket_shifted=int(np.count_nonzero(
                        periods != np.median(periods))) if periods.size
                    else 0,
                    exit_w_offset_kev=round(1e3 * w_off, 3),
                    steps_per_period=self.steps, sc_every=self.sc_every,
                    geometry=str(el.source), voltage_scale=vscale)
        if n_left_map:
            _log.warning("time-domain RFQ: %d particle(s) left the stored field "
                         "region (%.1f mm from the axis, between the vanes) "
                         "and are counted as lost there", n_left_map,
                         float(getattr(fmap, "box_half_width", float("nan"))))
        # the step on which a particle leaves is counted as well
        if res.outside_map_steps > n_left_map:
            _log.warning("time-domain RFQ: %d particle-steps fell outside the "
                         "stored field region (zero field there)",
                         res.outside_map_steps - n_left_map)
        if b_sel.size:
            _log.info("time-domain RFQ: the statistics leave out %d "
                      "particle(s) from where they fall behind the bunch: %d "
                      "leave the RFQ unaccelerated (they stay in the beam and "
                      "the transmission), %d are lost on the vanes later",
                      b_sel.size, n_unacc, b_sel.size - n_unacc)
        _log.info("time-domain RFQ: %d particles, %d lost, %d steps, %.1f s",
                  alive.size, int(lost.sum()), res.steps, info["wall_s"])
        tr.recorder.rfq_td = info
        return _ChainRun(cells=cells, entry_s=entry_s, edges=edges,
                         ref_tab=ref_tab, alive_in=alive, states=states,
                         crossed=crossed, loss_cell=loss_cell, loss_z=lz,
                         loss_xy=res.loss_xy, loss_W=res.loss_W,
                         exit_idx=exit_idx, plane_z=plane_z,
                         plane_ref=plane_ref, train_gate=gate, info=info,
                         behind_pid=alive[b_sel].astype(np.int64),
                         behind_plane=behind[b_sel].astype(np.int64))

    @staticmethod
    def _bucket_periods(phi_in, d_exit):
        """Whole RF periods to take off each particle's phase (int array):
        ``phi_in`` the injection phases (deg) of the tracked particles,
        ``d_exit`` their exit phases less the core (deg; NaN if lost).

        A captured particle (exit phase within 120 deg of the centre of the
        main bucket or a neighbour) takes the bucket it ends in: in the
        periodic-train model a particle captured one period later is the
        main bunch's particle from the neighbouring injection window.  The
        others -- stragglers, lost particles -- take the period their
        injection phase implies, from the boundary b between injection
        phases that end in the main bucket and those that end in a
        neighbour (fitted on the captured): a straggler keeps its lag."""
        n = np.zeros(phi_in.size, np.int64)
        out = np.isfinite(d_exit)
        if not out.any():
            return n
        r = np.round(d_exit[out] / 360.0)
        vals, counts = np.unique(r, return_counts=True)
        r_main = int(vals[np.argmax(counts)])
        resid = d_exit[out] - 360.0 * r
        cap = (np.abs(resid) < 120.0) & (np.abs(r - r_main) <= 1)
        k = (r[cap] - r_main).astype(np.int64)
        f = np.asarray(phi_in, float)[out][cap]
        n[:] = r_main
        if k.size == 0 or not np.any(k):
            return n
        order = np.argsort(f, kind="stable")
        f, k = f[order], k[order]
        m = f.size
        # split after the first i particles (i = 0..m): case "up" -- main
        # below, the next bucket (+1) above; case "down" -- the previous
        # bucket (-1) below, main above.  Keep the split that classifies
        # the most captured particles as observed.
        c0 = np.concatenate([[0], np.cumsum(k == 0)])
        cp = np.concatenate([[0], np.cumsum(k == 1)])
        cm = np.concatenate([[0], np.cumsum(k == -1)])
        up = c0 + (cp[-1] - cp)
        down = cm + (c0[-1] - c0)
        i_up, i_down = int(np.argmax(up)), int(np.argmax(down))

        def cut(i):
            if i <= 0:
                return -np.inf
            if i >= m:
                return np.inf
            return 0.5 * (f[i - 1] + f[i])

        phi = np.asarray(phi_in, float)
        if up[i_up] >= down[i_down]:
            n[phi >= cut(i_up)] += 1
        else:
            n[phi < cut(i_down)] -= 1
        idx = np.flatnonzero(out)[cap]           # captured: their bucket
        n[idx] = r_main + (r[cap] - r_main).astype(np.int64)
        return n

    @staticmethod
    def _behind_planes(states, crossed, w_ref, w_in) -> np.ndarray:
        """Per tracked particle, the record plane from which the statistics
        leave it out (-1: never).  A particle that fell behind the bunch --
        at its last record plane (the exit, or the last before its loss on
        the vanes) below the reference energy by more than half the reference
        particle's gain since the entrance, once that gain exceeds a tenth
        of the RFQ's -- is left out from the plane after the last one where
        it lagged the reference by less than a tenth of the gain.  Left out
        only once half the gain behind, a particle that slips out of the
        bucket slowly would inflate the energy spread until then; one lost
        later still counts in the transmission until its loss.
        ``states`` (P, N, 6) hold W - ``w_ref`` (MeV) in column 5.  An RFQ
        that hardly accelerates (gain under 10 % of the entrance energy)
        has no unaccelerated particles in this sense: none is left out."""
        n = states.shape[1]
        out = np.full(n, -1, np.int64)
        gain = np.asarray(w_ref, float) - float(w_in)
        if gain.size == 0 or not gain[-1] > 0.1 * float(w_in):
            return out
        dw = states[:, :, 5].astype(np.float64)
        seen = np.flatnonzero(crossed.any(axis=0))
        last = gain.size - 1 - np.argmax(crossed[::-1, seen], axis=0)
        g_last = gain[last]
        with np.errstate(invalid="ignore"):
            fell = ((g_last > 0.1 * gain[-1])
                    & (dw[last, seen] < -0.5 * g_last))
            lag = crossed & (dw < -0.1 * gain[:, None])
        for i in seen[fell]:
            ok = np.flatnonzero(crossed[:, i] & ~lag[:, i])
            out[i] = int(ok[-1]) + 1 if ok.size else 0
        return out

    @staticmethod
    def _train_gate(states, crossed, exit_idx) -> bool:
        """The PIC solver's bunch-train image gate as a per-cell run leaves
        it at the chain exit.  There the solver is consulted at every kick
        once the beam is a bunch train (from the second cell on): it engages
        when the core phase spread (half the 16-84 percentile span) reaches
        35 deg and releases at 25 deg (PicSolver.kick).  Evaluated here at
        the cell exits."""
        engaged = False
        for k in exit_idx[1:]:
            ph = states[k, crossed[k], 4]
            if ph.size < 2:
                continue
            p16, p84 = np.percentile(ph.astype(np.float64), (16.0, 84.0))
            core = 0.5 * float(p84 - p16)
            engaged = core > 25.0 if engaged else core >= 35.0
        return engaged

    def _load(self, run: _ChainRun, element) -> None:
        """Put the beam into the state at this element's exit (with
        ``record_substeps``, first through its interior record planes,
        one recorder row each, as a per-cell run records them)."""
        from linac_gen.elements.rfq_cell import RfqCell
        if not isinstance(element, RfqCell):
            return                       # markers inside the chain: unchanged
        tr = self.tr
        k = run.cells.index(element)
        # losses inside this cell, in z order, at their own position
        here = np.flatnonzero(run.loss_cell == k)
        here = here[np.argsort(run.loss_z[here], kind="stable")]
        p0 = int(run.exit_idx[k - 1]) + 1 if k > 0 else 0
        p1 = int(run.exit_idx[k])
        nxt = 0
        for p in range(p0, p1 + 1):
            if p == p1:
                stop = here.size         # every loss of the cell by its exit
            else:
                stop = int(np.searchsorted(run.loss_z[here], run.plane_z[p],
                                           side="right"))
            self._apply_losses(run, element, here[nxt:stop])
            nxt = max(nxt, stop)
            self._load_plane(run, p)
            if tr._record_substeps:
                tr._record_substep(element)
        if k == len(run.cells) - 1:
            ps = tr.pic_solver
            if ps is not None and hasattr(ps, "_train_engaged") \
                    and not getattr(ps, "train_force_engage", False):
                ps._train_engaged = bool(run.train_gate)

    def _apply_losses(self, run: _ChainRun, element, idx) -> None:
        beam = self.tr.beam
        for j in idx:
            pid = int(run.alive_in[j])
            if beam.lost[pid]:
                continue
            beam.particles[pid, 0] = run.loss_xy[j, 0]
            beam.particles[pid, 2] = run.loss_xy[j, 1]
            beam.particles[pid, 5] = run.loss_W[j] - beam.ref.w_kin
            s_loss = run.entry_s + float(np.clip(run.loss_z[j], 0.0,
                                                 run.edges[-1]))
            beam.record_loss(pid, s_loss, element.name)

    def _load_plane(self, run: _ChainRun, p: int) -> None:
        beam = self.tr.beam
        ok = run.crossed[p]
        ids = run.alive_in[ok]
        keep = ~beam.lost[ids]
        beam.particles[ids[keep]] = run.states[p, ok][keep].astype(np.float64)
        out = run.behind_plane <= p
        if out.any():
            # left out of the statistics from here on (the recorder reads
            # beam.stats_excluded); still in the beam and its transmission
            ex = getattr(beam, "stats_excluded", None)
            if ex is None or ex.shape != beam.lost.shape:
                ex = beam.stats_excluded = np.zeros(beam.lost.shape, bool)
            ex[run.behind_pid[out]] = True
        if p <= int(run.exit_idx[0]) and getattr(self.tr, "_periodic_phase",
                                                  False):
            # the tracker folds a bunch train from the element after the
            # one that bunched it; fold the first cell's planes here, with
            # its rule (one bunch spacing = 360 deg at the RFQ frequency)
            sel = ids[keep]
            ph = beam.particles[sel, 4]
            far = np.abs(ph) > 180.0
            if far.any():
                ph[far] = (ph[far] + 180.0) % 360.0 - 180.0
                beam.particles[sel, 4] = ph
        w, phi_s, s, _f = run.plane_ref[p]
        beam.ref.w_kin = float(w)
        beam.ref.phi_s = float(phi_s)
        beam.ref.s = float(s)
