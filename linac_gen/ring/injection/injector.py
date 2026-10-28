"""A ring run's injection, turn by turn: the foil's pass over the circulating
beam and the particles an injection turn brings.

:class:`Injection` holds what a run needs — the resolved foil
(:mod:`.foil`), the source pool (:mod:`.source`), the magnet programs
(:mod:`.programs`) — and the turn-by-turn record.  The engine calls, in each
turn, :meth:`Injection.apply_programs` before tracking, then — when the beam
reaches the foil — :meth:`Injection.foil_pass` (the circulating particles
that cross it; on a CUDA GPU :meth:`Injection.foil_cross` with the
particles the GPU found inside the foil) and :meth:`Injection.inject` (the
turn's new particles: their first crossing strips them; outside the foil
they miss it), and tracks on.

The macroparticles stand for the tracked bucket's real particles (their
weights, the ledger); the record's real particles and energies are the
ring's — every filled bucket is the same, so B times the tracked one's.

Without injection settings, a ring with a foil element still has its foil
act on the circulating beam (no programs, no source).
"""
from __future__ import annotations

import numpy as np

from linac_gen.ring import particles as P
from linac_gen.ring.injection import chopper as CH
from linac_gen.ring.injection import foil as F
from linac_gen.ring.injection import source as S
from linac_gen.ring.injection.config import InjectionConfig, InjectionError

E_CHARGE = 1.602176634e-19
#: loss cause of an injected particle that missed the foil (particles.LOSS_CAUSES)
CAUSE_MISSED = 7
CAUSE_UNSTRIPPED = 4
#: the turn-by-turn record (one entry per turn of the run); the real particles
#: and energies are the ring's (all its filled buckets)
RECORD_FIELDS = ("injected", "stripped", "h0", "hminus", "missed", "injected_charge",
                 "foil_hits", "foil_hits_injected", "deposit_mev", "electron_mev")


def _energy(delta, beta0):
    """(ptau, rpp, rvv) — the kernels' expressions (optics._energy)."""
    from linac_gen.ring.optics import _energy as e
    return e(delta, beta0)


def _hex(v):
    return None if v is None else [float(x).hex() for x in np.ravel(v)]


def _unhex(v):
    return None if v is None else np.array([float.fromhex(x) for x in v])


class Injection:
    """The injection (and the foil) of one ring run.

    ``cfg`` — the :class:`InjectionConfig` (None: the lattice's foil acts
    alone); ``rl`` — the compiled ring; ``lattice`` — the HELIX lattice (for
    its foil element); ``beam_config`` — the Beam tab's settings (the "beam"
    source); ``seed`` — the run's seed; ``rf_on`` — the RF; ``period_m`` —
    the bucket's length in zeta (:func:`bucket_length`; None: the RF is not at
    a harmonic of the revolution frequency); ``n_turns`` — the run's
    turns."""

    def __init__(self, cfg, rl, *, lattice, beam_config, seed: int, rf_on: bool,
                 period_m, n_turns: int, programs=None):
        if cfg is not None and not isinstance(cfg, InjectionConfig):
            cfg = InjectionConfig.from_dict(cfg)
        self.cfg = cfg
        self.seed = int(seed)
        self.rf_on = bool(rf_on)
        self.n_turns = int(n_turns)
        self.notes: list[str] = []
        ref = rl.reference
        self.pool = None
        self.perm = None
        if cfg is not None:
            if period_m is None:
                raise InjectionError("injection: the RF is not at a harmonic of the revolution "
                                     "frequency — the injected particles cannot be placed in "
                                     "its buckets")
            self.pool = S.load_source(cfg.source, beam_config=beam_config,
                                      n_needed=cfg.particles_total, seed=self.seed,
                                      constants=ref.constants)
            if self.pool.n < cfg.particles_total:
                self.notes.append(
                    f"the source's {self.pool.n} particles are drawn "
                    f"{cfg.particles_total / self.pool.n:.3g} times over the injection "
                    f"({cfg.particles_total} macroparticles)")
            self.perm = self.pool.order(self.seed, cfg.source.draw)
        self.period = float(period_m) if period_m is not None else float(rl.length)
        settings = F.resolve(None if cfg is None else cfg.foil, rl, lattice=lattice,
                             species_injected=None if self.pool is None else self.pool.species)
        self.foil = F.RingFoil(settings, mass_mev=ref.mass)
        self.node = int(settings.node)
        self.programs = programs
        if programs is None and cfg is not None and cfg.magnets:
            from linac_gen.ring.injection.programs import MagnetPrograms
            self.programs = MagnetPrograms(cfg.magnets, rl)
        self.buckets = 1
        self.bunch_mhz = 0.0
        if cfg is not None:
            if self.rf_on:
                # (the RF buckets: the circumference over the RF period)
                buckets = int(round(float(rl.length) / self.period)) - cfg.notch_buckets
                if buckets < 1:
                    raise InjectionError(
                        f"injection: notch_buckets={cfg.notch_buckets} leaves no bucket filled "
                        f"({int(round(float(rl.length) / self.period))} RF buckets)")
                self.buckets = buckets
            self.bunch_mhz = float(cfg.source.bunch_frequency_mhz or self.pool.frequency_mhz
                                   or 0.0)
            if not self.bunch_mhz > 0.0 and self.rf_on and cfg.chopper_window_deg < 180.0:
                raise InjectionError("injection: the chopper needs the linac's bunch rate — the "
                                     "source gives no bunch frequency (give its frequency_mhz "
                                     "or bunch_frequency_mhz)")
            if (self.pool.species == "H-" and settings.strip_model == "off"
                    and cfg.foil.strip_model is None):
                self.notes.append(
                    f"the foil strips every H- ion: its strip model is off (the foil element "
                    f"{settings.name}'s) — strip_model two_step gives the H0 and H- waste")
            if cfg.start_turn >= self.n_turns:
                self.notes.append(f"the injection starts at turn {cfg.start_turn}, after the "
                                  f"run's {self.n_turns} turn(s): nothing is injected")
            elif cfg.end_turn > self.n_turns:
                self.notes.append(f"the run ends before the injection: "
                                  f"{self.n_turns - cfg.start_turn} of its {cfg.turns} turns run")
        self.t_inj0 = None if cfg is None else (0.0 if cfg.start_turn == 0 else None)
        self.rec = {k: np.zeros(self.n_turns) for k in RECORD_FIELDS}
        self.closed_orbit = np.full((self.n_turns + 1, 4), np.nan)
        self.wave = np.full((self.n_turns + 1, len(self.programs.groups) if self.programs
                             else 0), np.nan)
        self._co_last = None
        self._co_key = None
        self._z_prev = None
        self._no_bunch: list[int] = []        # injection turns the chopper kept no bunch of
        self.kinetic_ref = float(ref.kinetic0)      # the ring's reference at the foil, last seen
        self._e_el = None

    # -- the programs ---------------------------------------------------------
    @property
    def has_programs(self) -> bool:
        return self.programs is not None

    def injecting(self, turn: int) -> bool:
        c = self.cfg
        return c is not None and c.start_turn <= turn < c.end_turn

    def pending(self, turn: int) -> bool:
        """Whether turns that inject are still to come (from ``turn`` on)."""
        return self.cfg is not None and turn < self.cfg.end_turn

    def apply_programs(self, turn: int, t_run: float, params) -> None:
        """The magnets at the start of turn ``turn`` (``t_run`` seconds from the
        start of the run), written into ``params``."""
        if self.programs is None:
            return
        w = self.programs.apply(t_run, params)
        if turn <= self.n_turns:
            self.wave[turn] = w

    def record_closed_orbit(self, turn: int, rl) -> None:
        """The 4-D closed orbit (delta 0, RF off) at the foil with the
        magnets of turn ``turn`` — computed again only when they changed."""
        if turn > self.n_turns:
            return
        key = None if self.programs is None else tuple(np.round(self.wave[turn], 15))
        if key is not None and key == self._co_key and self._co_last is not None:
            self.closed_orbit[turn] = self._co_last
            return
        from linac_gen.ring.optics import ClosedOrbitError, closed_orbit
        try:
            rl_off = rl.with_rf(False)
            try:
                z = closed_orbit(rl_off, delta=0.0, guess=self._z_prev)
            except ClosedOrbitError:
                if self._z_prev is None:
                    raise
                z = closed_orbit(rl_off, delta=0.0)
            co = self._to_foil(rl_off, z)
            self._z_prev = z
        except (ClosedOrbitError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            co = np.full(4, np.nan)
        self.closed_orbit[turn] = co
        self._co_last, self._co_key = co, key

    def _to_foil(self, rl, z) -> np.ndarray:
        from linac_gen.ring.kernels.cpu import backend
        X = np.array(z, dtype=float).reshape(6, 1).copy(order="C")
        pt, rp, rv = _energy(X[5], float(rl.ref_beta0[0]))
        pt, rp, rv = (np.ascontiguousarray(a, dtype=float) for a in (pt, rp, rv))
        out = np.empty((0, 6, 1))
        backend().track_probe(X, pt, rp, rv, *rl.node_table(), 0, self.node, out, False)
        return X[:4, 0].copy()

    # -- the foil over the circulating beam ---------------------------------------
    def foil_pass(self, arrays, weight, n_used: int, turn: int, *, p0c: float, beta0: float,
                  kinetic_mev: float) -> None:
        """The circulating particles inside the foil cross it (in place):
        ``arrays`` = (coords, ptau, rpp, rvv, state, foil_hits) over the slots."""
        coords, ptau, rpp, rvv, state, hits = arrays
        n = int(n_used)
        self.kinetic_ref = float(kinetic_mev)
        circ = np.flatnonzero(state[:n] == P.CIRCULATING)
        if not circ.size:
            return
        m = self.foil.inside(coords[0, circ], coords[2, circ])
        idx = circ[m]
        if not idx.size:
            return
        X = np.ascontiguousarray(coords[:, idx])
        pt, rp, rv = self.foil_cross(X, weight[idx], turn, p0c=p0c, beta0=beta0,
                                     kinetic_mev=kinetic_mev, crossings=hits[idx])
        coords[:, idx] = X
        ptau[idx], rpp[idx], rvv[idx] = pt, rp, rv
        hits[idx] += 1

    def foil_cross(self, X, w, turn: int, *, p0c: float, beta0: float, kinetic_mev: float,
                   crossings=None):
        """The circulating particles inside the foil — ``X`` (6, k), in the
        order of their slots, weights ``w``, the crossings of the foil each has
        made so far ``crossings`` — through it (X in place); returns their
        (ptau, rpp, rvv), None when there are none.  (foil_pass over the slots;
        a CUDA run gives the particles its GPU found inside.)"""
        self.kinetic_ref = float(kinetic_mev)
        if not X.shape[1]:
            return None
        rng = np.random.default_rng([self.seed, S.PURPOSE_FOIL, int(turn)])
        loss = self.foil.cross(X, rng, p0c=p0c, beta0=beta0, kinetic_mev=kinetic_mev,
                               crossings=crossings)
        pt, rp, rv = _energy(X[5], beta0)
        # (the ring's real particles: every filled bucket's)
        w = w * self.buckets
        self.foil.record(X[0], X[2], w, loss)
        if turn < self.n_turns:
            self.rec["foil_hits"][turn] += X.shape[1]
            self.rec["deposit_mev"][turn] += float(np.sum(w * loss))
        return pt, rp, rv

    # -- an injection turn ---------------------------------------------------------
    def weight(self, turn_time_s: float) -> float:
        """Real particles per injected macroparticle in a turn of
        ``turn_time_s``: I T / e over the filled buckets and the macroparticles
        (the tracked bucket's)."""
        c = self.cfg
        return (c.current_ma * 1e-3 * turn_time_s / E_CHARGE / self.buckets
                / c.particles_per_turn)

    def inject(self, parts, turn: int, *, t_turn: float, turn_time_s: float, ref) -> slice | None:
        """The particles of injection turn ``turn`` at the foil (``ref``: the
        ring's reference there, a RingReference): drawn from the pool, put on
        the turn's linac bunches, through the foil — stripped, scattered and
        slowed down; outside the foil, or left H⁰ / H⁻, waste.  ``t_turn`` —
        the turn's start, seconds from the start of the run.  Returns the
        slots filled (None: no particle this turn)."""
        if not self.injecting(turn):
            return None
        c = self.cfg
        if self.t_inj0 is None:
            self.t_inj0 = float(t_turn)
        k = int(turn - c.start_turn)
        n = c.particles_per_turn
        idx = self.pool.indices(k, n, self.perm)
        X = S.to_ring(self.pool, idx, ring_ref=ref,
                      centroid=(c.source.x_m, c.source.px, c.source.y_m, c.source.py))
        # (the linac bunches of the turn, and the chopper)
        t0 = float(t_turn) - self.t_inj0
        rng_b = np.random.default_rng([self.seed, S.PURPOSE_BUNCH, int(turn)])
        if self.bunch_mhz > 0.0:
            z = CH.bunch_zeta(t0, t0 + turn_time_s, f_linac_hz=self.bunch_mhz * 1e6,
                              beta0=ref.beta0, period_m=self.period,
                              phase_deg=c.chopper_phase_deg)
            z = CH.kept(z, period_m=self.period, window_deg=c.chopper_window_deg,
                        rf_on=self.rf_on)
            if not z.size:
                self._no_bunch.append(int(turn))
                return None
            X[4] += z[rng_b.integers(0, z.size, size=n)]
        else:
            # (no bunch frequency: a linac beam without bunches, spread over the period)
            X[4] += (rng_b.random(n) - 0.5) * self.period
        X[4] -= self.period * np.floor(X[4] / self.period + 0.5)
        inside = self.foil.inside(X[0], X[2])
        hit = np.flatnonzero(inside)
        code = np.full(n, F.STRIPPED, dtype=np.int8)
        loss = np.zeros(n)
        kin = float(ref.kinetic0)
        self.kinetic_ref = kin
        if hit.size:
            rng_f = np.random.default_rng([self.seed, S.PURPOSE_FOIL_INJECTED, int(turn)])
            Xh = np.ascontiguousarray(X[:, hit])
            loss[hit] = self.foil.cross(Xh, rng_f, p0c=ref.p0c, beta0=ref.beta0,
                                        kinetic_mev=kin)
            X[:, hit] = Xh
            code[hit] = self.foil.strip(hit.size, rng_f, kinetic_mev=self._ion_kinetic(ref),
                                        species=self.pool.species)
        w = self.weight(turn_time_s)
        sl = parts.add(X, w, ref=ref, turn=int(turn), state=P.CIRCULATING)
        slots = np.arange(sl.start, sl.stop)
        missed = slots[~inside]
        waste = slots[inside & (code != F.STRIPPED)]
        for ids, cause in ((missed, CAUSE_MISSED), (waste, CAUSE_UNSTRIPPED)):
            if ids.size:
                parts.state[ids] = P.WASTE
                parts.lost_turn[ids] = int(turn)
                parts.lost_node[ids] = self.node
                parts.lost_cause[ids] = cause
        parts.foil_hits[slots[inside]] += 1
        # (the ring's real particles: every filled bucket's)
        w_ring = w * self.buckets
        self.foil.record(X[0, inside], X[2, inside], np.full(int(inside.sum()), w_ring),
                         loss[inside])
        if turn < self.n_turns:
            r = self.rec
            r["injected"][turn] = n
            r["stripped"][turn] = int(np.sum(inside & (code == F.STRIPPED)))
            r["h0"][turn] = int(np.sum(inside & (code == F.H0)))
            r["hminus"][turn] = int(np.sum(inside & (code == F.HMINUS)))
            r["missed"][turn] = int(np.sum(~inside))
            r["injected_charge"][turn] = n * w_ring
            r["foil_hits_injected"][turn] = int(inside.sum())
            r["deposit_mev"][turn] += float(np.sum(w_ring * loss[inside]))
            if self.pool.species == "H-":
                # (the electrons stripped: two from each ion left a proton, one from each H0)
                n_el = 2 * r["stripped"][turn] + r["h0"][turn]
                r["electron_mev"][turn] = n_el * w_ring * self._electron_mev()
        return sl

    def _ion_kinetic(self, ref) -> float:
        """The kinetic energy of the injected ion of the pool's reference
        (what the stripping cross sections scale with: its velocity)."""
        return float(self.pool.reference.kinetic0) if self.pool is not None \
            else float(ref.kinetic0)

    def _electron_mev(self) -> float:
        """A stripped electron's kinetic energy (its ion's velocity)."""
        if self._e_el is None:
            self._e_el = self.foil.electron_mev(self._ion_kinetic(None), self.pool.species)
        return self._e_el

    # -- results ----------------------------------------------------------------------
    def results(self, turns_done: int, parts) -> dict:
        """The injection's record for the results."""
        n = int(turns_done)
        out = {k: v[:n].copy() for k, v in self.rec.items()}
        out["closed_orbit_foil"] = self.closed_orbit[:n + 1].copy()
        if self.programs is not None:
            out["waveforms"] = self.wave[:n + 1].copy()
            out["waveform_names"] = list(self.programs.names)
        out["foil"] = self.foil.s.to_dict()
        ph = self.foil.physics(self.kinetic_ref)
        out["foil"].update(theta0_rad=ph["theta0"], mean_loss_mev=ph["mean_loss_mev"],
                           straggling_used=ph["straggling"], at_kinetic_mev=self.kinetic_ref)
        if self.pool is not None:
            f_p, f_h0, f_hm = self.foil.stripping_fractions(self._ion_kinetic(None))
            out["foil"].update(stripped_fraction=f_p, h0_fraction=f_h0, hminus_fraction=f_hm)
            if self.pool.species == "H-":
                out["foil"]["electron_mev_each"] = self._electron_mev()
        if self.foil.hit_map is not None:
            out["hit_map"] = self.foil.hit_map.copy()
            out["deposit_map_mev"] = self.foil.deposit_map.copy()
            out["map_extent_m"] = self.foil.map_extent()
        nu = parts.n_used
        hits = parts.foil_hits[:nu]
        out["hits_histogram"] = np.bincount(hits[parts.state[:nu] != P.WASTE]).astype(np.int64) \
            if nu else np.zeros(1, dtype=np.int64)
        circ = parts.state[:nu] == P.CIRCULATING
        out["mean_hits_circulating"] = float(hits[circ].mean()) if circ.any() else float("nan")
        waste = parts.state[:nu] == P.WASTE
        cause = parts.lost_cause[:nu]
        w = parts.weight[:nu]
        out["waste"] = {}
        for name, code in (("missed", CAUSE_MISSED), ("unstripped", CAUSE_UNSTRIPPED)):
            m = waste & (cause == code)
            out["waste"][name] = int(m.sum())
            out["waste"][name + "_charge"] = float(w[m].sum())
        notes = list(self.notes)
        if self.cfg is not None:
            out["settings"] = self.cfg.to_dict()
            out["source"] = {"origin": self.pool.origin, "species": self.pool.species,
                             "kinetic_mev": float(self.pool.reference.kinetic0),
                             "frequency_mhz": float(self.pool.frequency_mhz),
                             "bunch_frequency_mhz": float(self.bunch_mhz),
                             "pool": int(self.pool.n)}
            out["buckets"] = int(self.buckets)
            if self._no_bunch:
                notes.append(f"the chopper kept no linac bunch in {len(self._no_bunch)} "
                             f"injection turn(s) (the first: turn {self._no_bunch[0]}) — "
                             "nothing was injected then")
        out["notes"] = notes
        return out

    # -- checkpoints ----------------------------------------------------------------------
    def state_arrays(self) -> dict:
        d = {f"inj_{k}": v for k, v in self.rec.items()}
        d["inj_closed_orbit"] = self.closed_orbit
        d["inj_wave"] = self.wave
        if self.foil.hit_map is not None:
            d["inj_hit_map"] = self.foil.hit_map
            d["inj_deposit_map"] = self.foil.deposit_map
        return d

    def state_meta(self) -> dict:
        """What a restart needs besides the arrays — the closed orbit's warm
        start and its cache (hex: bit for bit) among them (the notes of the
        settings are made again)."""
        return {"t_inj0": None if self.t_inj0 is None else float(self.t_inj0).hex(),
                "no_bunch": list(self._no_bunch),
                "co_z_prev": _hex(self._z_prev), "co_last": _hex(self._co_last),
                "co_key": _hex(self._co_key)}

    def restore(self, arrays: dict, meta: dict, turn0: int) -> None:
        for k in RECORD_FIELDS:
            v = arrays.get(f"inj_{k}")
            if v is not None:
                m = min(len(v), self.n_turns)
                self.rec[k][:m] = v[:m]
        for name, dst in (("inj_closed_orbit", self.closed_orbit), ("inj_wave", self.wave)):
            v = arrays.get(name)
            if v is not None and v.shape[1:] == dst.shape[1:]:
                m = min(len(v), len(dst))
                dst[:m] = v[:m]
        if self.foil.hit_map is not None and "inj_hit_map" in arrays:
            self.foil.hit_map[...] = arrays["inj_hit_map"]
            self.foil.deposit_map[...] = arrays["inj_deposit_map"]
        meta = meta or {}
        t = meta.get("t_inj0")
        self.t_inj0 = None if t is None else float.fromhex(t)
        self._no_bunch = [int(k) for k in meta.get("no_bunch") or []]
        self._z_prev = _unhex(meta.get("co_z_prev"))
        self._co_last = _unhex(meta.get("co_last"))
        key = _unhex(meta.get("co_key"))
        self._co_key = None if key is None else tuple(key)


def summary_text(inj: Injection) -> str:
    """One line on the foil and the injection (for the run's notes)."""
    s = inj.foil.s
    rect = ("open" if not s.closed else
            f"x {s.x_min_m * 1e3:.3g}..{s.x_max_m * 1e3:.3g} mm, "
            f"y {s.y_min_m * 1e3:.3g}..{s.y_max_m * 1e3:.3g} mm")
    out = (f"foil at {s.name}: {s.material} {s.thickness_ug_cm2:g} ug/cm2, {rect}, "
           f"stripping {s.strip_model}")
    if inj.cfg is not None:
        c = inj.cfg
        out += (f"; injection over turns {c.start_turn}..{c.end_turn - 1}, "
                f"{c.particles_per_turn} macroparticles a turn, {c.current_ma:g} mA"
                + (f" into {inj.buckets} filled buckets" if inj.rf_on else ""))
    return out


def bucket_length(rl, rf_on: bool) -> float | None:
    """The bucket's length in zeta the injected particles are placed in: the
    RF period C/h, h the common harmonic of the cavities with a voltage
    (whatever zeta's wrapping), the circumference without RF; None when the
    RF is not at a harmonic of the revolution frequency."""
    from linac_gen.ring.engine import wrap_length
    if not rf_on:
        return float(rl.length)
    return wrap_length(rl)
