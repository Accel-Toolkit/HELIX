"""Ring-mode turn loop — a storage ring at fixed energy (milestone M2) or a
ramp (M3).

:class:`RingSimulation` tracks a bunch turn after turn through the compiled
ring (:mod:`linac_gen.ring.compile`) with the ring kernels
(:mod:`linac_gen.ring.kernels`) — numba on every CPU core, or a CUDA GPU —
at the ring's fixed reference energy, or, with a ramp
(:mod:`linac_gen.ring.programs`), with the reference energy, the RF voltage,
phase and frequency set turn by turn (each cavity followed by a reference
change; one turn per launch, the node table written before each):

* the particles come from the Beam tab's distribution (HELIX's generators,
  converted exactly to ring coordinates) or are generated matched to the
  ring's periodic optics (6-D with the RF on; without RF, 4-D and a coasting
  beam);
* each turn is one pass through every node — an aperture loses a particle
  with its turn, element, position and cause — after which zeta is wrapped
  into the RF period C/h (the circumference without RF): exact for every
  map, it keeps the coordinate bounded;
* every ``stats_every`` turns the circulating particles' number, centroid
  and second moments (6x6) are kept, from which the emittances (numpy's
  pairwise sums, not BLAS: the same bits on every machine and in every run);
* snapshots of the whole distribution at chosen turns; tune probes tracked
  alongside, their coordinates kept for the first ``PROBE_RECORD_TURNS``
  turns, whose NAFF tunes give the tunes (a grid of amplitudes gives the
  footprint and the tune diffusion);
* a checkpoint every ``checkpoint_every`` turns: a run restarted from it
  ends bit for bit as the run that was not interrupted.

Coordinates are the ring's (:mod:`linac_gen.ring.coords`): x, y [m],
px, py = P/P0, zeta = s - beta0 c t [m], delta = (P - P0)/P0, relative to the
reference of the moment.  With space charge (:mod:`linac_gen.ring.sc`) the
turn is tracked from kick to kick, one turn per launch, the tune probes
feeling the field as test particles.  With a stripping foil — injection
settings (:mod:`linac_gen.ring.injection`) or a foil element in the ring —
the turn stops at the foil, one turn per launch: the circulating particles
that cross it are scattered and slowed down, an injection turn's particles
are added there (stripped, or waste), and the bump and painting magnets
follow their waveforms turn by turn.  Without a ramp, cavities that would
give the reference particle energy are refused (there is no closed orbit at
fixed energy then).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from linac_gen.ring import particles as P

ENGINE = "ring kernels, fixed energy or ramp, space charge, injection (M5)"
CHECKPOINT_VERSION = 3
#: checkpoint formats this HELIX continues (1: fixed energy, before the ramp;
#: 2: before injection)
CHECKPOINT_READABLE = (1, 2, 3)
#: tune probes: the amplitude of the single tune probe, in rms beam sizes
PROBE_SMALL_SIGMA = 0.01
#: ... with space charge: far enough above the jitter of the beam's centroid
#: (about 1/sqrt(n) rms sizes), which drives a probe at the bare tune; a
#: Gaussian beam's tune shift there is about 1.2 % below its centre's
PROBE_SMALL_SIGMA_SC = 0.2
#: fewest periods of a motion in the turn-by-turn record for its NAFF tune
MIN_PERIODS = 3.0
#: the probes' amplitudes are sized with at least these emittances (x and y
#: [m rad], zeta-delta [m]): a beam of zero emittance still gets probes far
#: above rounding
PROBE_MIN_EMIT = (1e-9, 1e-9, 1e-7)
#: the probes' turn-by-turn coordinates are kept for this many turns (their
#: NAFF tunes come from them; the probes are tracked to the end of the run)
PROBE_RECORD_TURNS = 4096
#: a run note when the line density keeps less of the bunch's peak than this
LINE_PEAK_NOTE = 0.97
#: the fewest particles a statistics row's bunch length is judged from
LINE_MIN_PARTICLES = 100
#: a run note when the beam at the start is further than this from the ring's
#: matched beam in a normal mode (BMAG; at 1.01 the beam's beta swings about
#: 14 % either side of the matched one)
MISMATCH_NOTE = 1.01
#: matched_beam: closed-orbit coordinates below this (m, P/P0, delta) are zero
CO_ZERO = 1e-8


@dataclass
class RingResults:
    """A ring run (ring coordinates, SI units).

    ``turns[k]`` is the number of turns completed when statistics row k was
    taken (0: the start); ``n_alive``, ``mean`` (6) and ``cov`` (6x6 central
    second moments) describe the circulating particles then.  ``snapshots``
    maps a turn to its particles (``coords`` (6, m), ``pid``); ``losses``
    holds per lost particle its ``pid``, ``turn`` (the turns it completed: 0
    for a loss in the first turn, so it is missing from the statistics row
    ``turn + 1`` on), ``node``, ``element`` (index), ``s`` [m] and ``cause``;
    ``probes`` the tune probes' turn-by-turn coordinates (``tbt`` (n,
    min(turns, PROBE_RECORD_TURNS) + 1, 6)), amplitudes and NAFF tunes;
    ``tunes`` the tunes of the smallest probe; ``optics`` the compiled ring's
    periodic optics; ``ledger`` the particle bookkeeping; ``injection`` the
    foil's and the injection's record (empty without a foil).
    """
    engine: str
    n_turns_requested: int
    turns: np.ndarray
    n_alive: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    snapshots: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    probes: dict = field(default_factory=dict)
    tunes: dict = field(default_factory=dict)
    optics: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)
    truncated: bool = False
    wall_s: float = 0.0
    meta: dict = field(default_factory=dict)
    # the reference when each statistics row was taken (it changes in a ramp)
    row_beta0gamma0: np.ndarray | None = None
    row_p0c_mev: np.ndarray | None = None
    # a ramp's turn-by-turn record (empty at fixed energy): per turn, at its
    # start, t_s, kinetic_mev, gain_mev, voltage_mv, phi_deg, frequency_hz,
    # above_transition, capped; and transition_turn, gamma_t, n_capped, the end
    ramp: dict = field(default_factory=dict)
    # the foil's and the injection's turn-by-turn record (empty without a foil):
    # per turn injected, stripped, h0, hminus, missed, injected_charge,
    # foil_hits, foil_hits_injected, deposit_mev; closed_orbit_foil (turns + 1,
    # 4), waveforms; the foil's settings and physics, its maps, the hits
    # per particle (see linac_gen.ring.injection.injector)
    injection: dict = field(default_factory=dict)
    # the run's periodic optics at every element's exit (at the start of a
    # ramp): name, type (lists of str); s [m], betx, alfx, bety, alfy, mux,
    # muy (tunes), dx, dpx, dy, dpy (per delta), x, px, y, py (closed orbit)
    optics_table: dict = field(default_factory=dict)

    @property
    def turns_done(self) -> int:
        return int(self.turns[-1]) if len(self.turns) else 0

    @property
    def emit(self) -> np.ndarray:
        """(k, 3) projected rms emittances: x, y [m rad] (px, py are P/P0),
        z [m] (zeta-delta)."""
        return np.stack([_emit2(self.cov, 0), _emit2(self.cov, 2), _emit2(self.cov, 4)], axis=1)

    @property
    def emit_normalized(self) -> np.ndarray:
        """(k, 2) normalized rms transverse emittances [m rad] (beta0 gamma0 x,
        with the reference of each row)."""
        bg = (self.row_beta0gamma0 if self.row_beta0gamma0 is not None
              else float(self.meta.get("beta0gamma0", float("nan"))))
        return self.emit[:, :2] * np.reshape(bg, (-1, 1))

    @property
    def emit_long_evs(self) -> np.ndarray:
        """(k,) rms longitudinal emittance in eV s (zeta-delta area x p0c/c,
        with the reference of each row)."""
        p = (self.row_p0c_mev if self.row_p0c_mev is not None
             else float(self.meta.get("p0c_mev", float("nan"))))
        return self.emit[:, 2] * p * 1e6 / C_LIGHT

    @property
    def survival(self) -> np.ndarray:
        n0 = float(self.n_alive[0]) if len(self.n_alive) else float("nan")
        return self.n_alive / n0 if n0 else np.full(len(self.n_alive), np.nan)

    @property
    def sigma(self) -> np.ndarray:
        """(k, 6) rms sizes."""
        return np.sqrt(np.maximum(np.diagonal(self.cov, axis1=1, axis2=2), 0.0))


C_LIGHT = 299792458.0


def _emit2(cov: np.ndarray, i: int) -> np.ndarray:
    c = np.asarray(cov)
    if c.size == 0:
        return np.zeros(0)
    det = c[:, i, i] * c[:, i + 1, i + 1] - c[:, i, i + 1] ** 2
    return np.sqrt(np.maximum(det, 0.0))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def wrap_length(rl) -> float | None:
    """The period of the maps in zeta: C/h for the common harmonic h of the
    cavities with a voltage (h = the greatest common divisor of their
    harmonics), C without RF; None when a cavity is not at a harmonic of the
    revolution frequency (then zeta is not wrapped)."""
    T0 = rl.revolution_time
    hs = []
    for c in rl.cavities:
        if not rl.cavity_voltage(c):
            continue
        h = c["frequency_hz"] * T0
        if abs(h - round(h)) > 1e-9 * max(1.0, abs(h)) or round(h) < 1:
            return None
        hs.append(int(round(h)))
    if not hs:
        return float(rl.length)
    g = hs[0]
    for h in hs[1:]:
        g = math.gcd(g, h)
    return float(rl.length) / g


def lattice_digest(rl, wrap) -> str:
    """A fingerprint of the compiled ring a checkpoint belongs to."""
    h = hashlib.sha256()
    for a in (rl.kinds, rl.offsets, rl.params, rl.ref_beta0, rl.ref_p0c, rl.ref_bg2):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((float(rl.length), wrap, float(rl.reference.p0c), float(rl.reference.mass)))
             .encode())
    return h.hexdigest()


def _apply(M, Z) -> np.ndarray:
    """M @ Z for a small matrix M, the rows of Z combined one by one in a
    fixed order: the same bits on every machine and in every run, which a
    BLAS library's matrix product does not promise (Intel MKL's may change
    with the memory alignment or its number of threads)."""
    M = np.asarray(M, dtype=float)
    Z = np.asarray(Z, dtype=float)
    out = np.empty((M.shape[0], Z.shape[1]))
    for i in range(M.shape[0]):
        acc = M[i, 0] * Z[0]
        for k in range(1, M.shape[1]):
            acc += M[i, k] * Z[k]
        out[i] = acc
    return out


def _moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and central second moments of X (d, n), C-ordered, from numpy's
    pairwise sums (no BLAS, see _apply)."""
    n = X.shape[1]
    mu = X.mean(axis=1)
    d = X - mu[:, None]
    m = X.shape[0]
    C = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            C[i, j] = C[j, i] = np.sum(d[i] * d[j]) / n
    return mu, C


def _cholesky(C: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a small positive-definite matrix (ValueError
    when it is not)."""
    m = C.shape[0]
    L = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            v = float(C[i, j]) - sum(float(L[i, k] * L[j, k]) for k in range(j))
            if i == j:
                if not v > 0.0:
                    raise ValueError("not positive definite")
                L[i, i] = math.sqrt(v)
            else:
                L[i, j] = v / L[j, j]
    return L


def _whitened(Z: np.ndarray) -> np.ndarray:
    """Z (m, n) with its sample mean taken out and its sample second moments
    made exactly the identity (rows of L^-1 (Z - mean), L the Cholesky factor
    of their matrix): the cut tails and the sampling noise then leave the
    rms sizes exactly as asked.  Unchanged with too few particles (n <= m)."""
    m, n = Z.shape
    if n <= m:
        return Z
    mu, C = _moments(Z)
    try:
        L = _cholesky(C)
    except ValueError:
        return Z
    d = Z - mu[:, None]
    out = np.empty_like(d)
    for i in range(m):
        acc = d[i].copy()
        for k in range(i):
            acc -= L[i, k] * out[k]
        out[i] = acc / L[i, i]
    return out


def _normal_emittances(X: np.ndarray, Winv: np.ndarray, co: np.ndarray,
                       disp: np.ndarray | None = None) -> np.ndarray:
    """Second-moment emittances of each normal mode of W (rows of
    W^-1 (X - co)); with a 4-D W, X is (6, n) and the dispersion ``disp``
    times delta is removed first (betatron coordinates)."""
    if Winv.shape[0] == 4:
        d = X[5] - co[5]
        X = X[:4] - (disp[:, None] * d[None, :] if disp is not None else 0.0)
        co = co[:4]
    Z = _apply(Winv, X - co[:, None])
    n = Z.shape[0] // 2
    out = np.empty(n)
    for k in range(n):
        a, b = Z[2 * k], Z[2 * k + 1]
        a = a - a.mean()
        b = b - b.mean()
        out[k] = math.sqrt(max(float(np.mean(a * a) * np.mean(b * b) - np.mean(a * b) ** 2), 0.0))
    return out


def _normal_mismatch(X: np.ndarray, Winv: np.ndarray, co: np.ndarray,
                     disp: np.ndarray | None = None) -> np.ndarray:
    """Per normal mode of W, how far the beam X is from the matched one:
    BMAG = tr(S) / (2 sqrt(det S)), S the 2x2 second moments of the mode in
    the matched normalized coordinates (rows of W^-1 (X - co), as
    _normal_emittances) — 1 for a matched beam, (beta gamma_m - 2 alpha
    alpha_m + gamma beta_m)/2 for an uncoupled plane; NaN for a mode of no
    emittance."""
    if Winv.shape[0] == 4:
        d = X[5] - co[5]
        X = X[:4] - (disp[:, None] * d[None, :] if disp is not None else 0.0)
        co = co[:4]
    Z = _apply(Winv, X - co[:, None])
    n = Z.shape[0] // 2
    out = np.full(n, np.nan)
    for k in range(n):
        a, b = Z[2 * k] - Z[2 * k].mean(), Z[2 * k + 1] - Z[2 * k + 1].mean()
        saa, sbb, sab = float(np.mean(a * a)), float(np.mean(b * b)), float(np.mean(a * b))
        det = saa * sbb - sab * sab
        if det > 0.0:
            out[k] = (saa + sbb) / (2.0 * math.sqrt(det))
    return out


def _truncated_gaussian(rng, n: int, dims: int, cutoff: float | None) -> np.ndarray:
    """(2*dims, n) standard normal pairs, each pair cut at the action
    (u^2 + v^2)/2 < cutoff^2/2 (a 2-D Gaussian truncated at ``cutoff`` sigma;
    its rms is then below 1 — 0.949 at 3 sigma — until _whitened)."""
    out = np.empty((2 * dims, n))
    for k in range(dims):
        got = 0
        while got < n:
            u = rng.standard_normal((2, max(2 * (n - got), 64)))
            if cutoff:
                u = u[:, (u[0] ** 2 + u[1] ** 2) < cutoff * cutoff]
            m = min(u.shape[1], n - got)
            out[2 * k:2 * k + 2, got:got + m] = u[:, :m]
            got += m
    return out


# ---------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------
class RingSimulation:
    """Multi-turn tracking of a ring.

    ``lattice`` — a HELIX Lattice describing one turn; ``beam_config`` — the
    BeamConfig the particles come from (the ``"beam"`` source generates its
    distribution at the ring's species and energy; its emittances define a
    ``"matched"`` one, its frequency the unit of its phases and of
    ``emit_z``); ``ring_config`` — a :class:`~linac_gen.ring.config.RingConfig`.
    ``beam`` — a HELIX Beam to start from instead of generating one (at the
    ring's species and energy; used whatever the source); ``coords`` — the
    particles' ring coordinates (6, n) to start from (the API's way to place
    chosen particles); ``ring_lattice`` — the compiled ring, when the caller
    already has it.
    """

    def __init__(self, lattice, beam_config, ring_config, *, beam=None, coords=None,
                 ring_lattice=None):
        self.lattice = lattice
        self.beam_config = beam_config
        self.config = ring_config
        self._beam = beam
        self._coords = None if coords is None else np.array(coords, dtype=float, copy=True)
        if self._coords is not None and (self._coords.ndim != 2 or self._coords.shape[0] != 6):
            raise ValueError("coords must have shape (6, n)")
        self._rl = ring_lattice
        self._ramp_cfg = None           # the ramp's programs, resolved at compile
        self._compile_notes: list[str] = []
        self._notes: list[str] = []      # (a run's: the compile's and its own)
        self._should_stop = None         # (matched_beam's stop request, polled by the optics)

    # -- set-up ---------------------------------------------------------------
    def compile(self):
        if self._rl is None:
            if self.lattice is None or not len(self.lattice.elements):
                raise ValueError("the lattice has no elements")
            from linac_gen.ring.compile import compile_from_config
            cfg = self.config
            rl = compile_from_config(self.lattice, cfg, beam=self.beam_config)
            if getattr(cfg, "ramp", None) is not None:
                # a ramp starts at its program's energy: the ring's reference is
                # compiled there (the magnets keep the lattice's normalized
                # strengths whatever it is)
                import dataclasses
                ref = rl.reference
                rc = self._resolved_ramp(ref)
                e0 = rc.energy.total_energy(rc.t_start_s, ref.mass, ref.p0c)
                w0 = e0 - ref.mass
                if not math.isclose(w0, ref.kinetic0, rel_tol=1e-12, abs_tol=0.0):
                    rl = compile_from_config(self.lattice,
                                             dataclasses.replace(cfg, design_energy_mev=w0),
                                             beam=self.beam_config)
                    note = (f"the ramp starts at {w0:.9g} MeV: the ring's reference is "
                            "compiled there")
                    self._compile_notes.append(note)
                    self._notes.append(note)
            self._rl = rl
        return self._rl

    def _optics(self, rl):
        from linac_gen.ring.optics import _check_stationary, twiss
        from linac_gen.ring.optics.normal_form import UnstableError
        rf_on = rl.rf_on
        if rf_on:
            try:
                _check_stationary(rl)
            except ValueError as exc:
                raise ValueError(f"{exc}.  (A run without a ramp is at fixed energy: give "
                                 "it a ramp to accelerate.)") from None
        try:
            opt = twiss(rl, six_d=rf_on, should_stop=self._should_stop)
        except UnstableError as exc:
            hint = (" — with the RF on, check the cavities' phase: below transition the "
                    "bucket is at a sine lag of 0 deg (HELIX phase -90), above it at 180 deg "
                    "(phase +90)") if rf_on else ""
            raise ValueError(f"{exc}{hint}") from None
        return opt, rf_on

    def ramp(self):
        """The run's ramp on the compiled ring (:class:`~linac_gen.ring.
        programs.Ramp`; gamma_t from the ring's 4-D optics unless the ramp
        gives it) — what :meth:`run` follows, for a look at the reference
        alone (``ramp().orbit(n)``, ``ramp().turns_to_end()``); None without a
        ramp."""
        if getattr(self.config, "ramp", None) is None:
            return None
        return self._make_ramp(self.compile())

    def _resolved_ramp(self, ref):
        """The run's RampConfig with a resonant cycle's blank start energy made
        the ring reference's (``ref``: the reference before the ring is
        compiled at the ramp's start), once: the cycle must not move with the
        reference it starts from."""
        if self._ramp_cfg is None:
            import dataclasses

            from linac_gen.ring.programs import RampConfig
            rc = RampConfig.from_dict(self.config.ramp)
            if rc.energy.kind == "sine" and rc.energy.start_kinetic_mev is None:
                rc = dataclasses.replace(rc, energy=dataclasses.replace(
                    rc.energy, start_kinetic_mev=float(ref.kinetic0)))
            self._ramp_cfg = rc
        return self._ramp_cfg

    def _make_ramp(self, rl):
        from linac_gen.ring.optics import twiss
        from linac_gen.ring.optics.normal_form import UnstableError
        from linac_gen.ring.programs import Ramp
        rc = self._resolved_ramp(rl.reference)
        gamma_t = rc.gamma_t
        if gamma_t is None:
            try:
                gamma_t = twiss(rl.with_rf(False), six_d=False, chromaticity=False,
                                should_stop=self._should_stop).gamma_t
            except UnstableError as exc:
                raise ValueError(f"the ring's 4-D optics: {exc}") from None
        return Ramp(rc, rl, gamma_t=gamma_t,
                    corrector_mode=getattr(self.config, "corrector_mode", "follow"))

    def _ramp_optics(self, rl):
        """(Ramp, optics, rf_on) at the start of a ramp: gamma_t from the
        ring's 4-D optics unless the ramp gives it; the optics — the matched
        beam's — of the stationary equivalent of the first turn (each cavity
        at lag 0 with V_k cos(lag_k): the same linear RF focusing as the
        accelerating bucket's)."""
        from linac_gen.ring.optics import twiss
        from linac_gen.ring.optics.normal_form import UnstableError
        ramp = self._make_ramp(rl)
        t0, e0 = ramp.start()
        plan0 = ramp.plan(0, t0, e0)
        ramp.n_capped = 0
        # (the RF's focusing at the start, V_k cos(lag_k): none — no voltage,
        # or the phase at +-90 deg — and the matched beam is a coasting one)
        focus = float(np.sum(np.abs(plan0.volts_mv * np.cos(np.radians(plan0.lags_deg)))))
        rf_on = focus > 1e-9 * max(float(np.sum(np.abs(plan0.volts_mv))), 1e-300)
        if plan0.voltage_mv > 0.0 and not rf_on:
            self._notes.append("the RF gives no focusing at the start of the ramp (phase at "
                               "+-90 deg): the matched beam is a coasting beam")
        eff = ramp.stationary_equivalent(plan0)
        try:
            opt = twiss(eff, six_d=rf_on, should_stop=self._should_stop)
        except UnstableError as exc:
            raise ValueError(f"the ring at the start of the ramp: {exc}") from None
        return ramp, opt, rf_on, eff

    def _start(self):
        """The ring at the start of a run: (rl, rl_start, ramp, opt, rf_on,
        inj_cfg, programs) — the compiled ring (the run's own copy of its node
        table when the injection magnets or a ramp write it), the ring of the
        first turn's optics (a ramp's stationary equivalent of turn 0), the
        Ramp or None, the optics and whether the RF focuses then, the
        injection settings or None and the magnets' waveforms or None."""
        import dataclasses
        cfg = self.config
        rl = self.compile()
        ramp = None
        inj_cfg = None
        programs = None
        if getattr(cfg, "injection", None) is not None:
            from linac_gen.ring.injection.config import InjectionConfig
            inj_cfg = InjectionConfig.from_dict(cfg.injection)
            if inj_cfg.magnets:
                # (the magnets' waveforms write the node table every turn: this
                # run's own copy; at the start, the optics and the matched beam
                # see the magnets of turn 0)
                from linac_gen.ring.injection.programs import MagnetPrograms
                rl = dataclasses.replace(rl, params=rl.params.copy())
                programs = MagnetPrograms(inj_cfg.magnets, rl)
                programs.apply(0.0, rl.params)
        rl_start = rl
        if getattr(cfg, "ramp", None) is not None:
            # (the ramp writes the node table every turn: this run's own copy,
            # so a ring handed in is left as it was)
            rl = dataclasses.replace(rl, params=rl.params.copy(),
                                     ref_beta0=rl.ref_beta0.copy(), ref_p0c=rl.ref_p0c.copy(),
                                     ref_bg2=rl.ref_bg2.copy())
            ramp, opt, rf_on, rl_start = self._ramp_optics(rl)
        else:
            opt, rf_on = self._optics(rl)
        return rl, rl_start, ramp, opt, rf_on, inj_cfg, programs

    def matched_beam(self, frequency_mhz: float | None = None, *, should_stop=None) -> dict:
        """The beam matched to the ring at the start of the run, in the Beam
        tab's terms — the beam the ``"matched"`` source generates, so that the
        Beam tab's distribution (the ``"beam"`` source) can be that beam:

        * ``alpha_x, beta_x, alpha_y, beta_y`` [m] — the periodic Twiss
          functions at the start of the ring;
        * ``disp_x, disp_y`` [mm/MeV], ``disp_xp, disp_yp`` [mrad/MeV] — the
          periodic dispersion per MeV of kinetic energy, D/(beta0 p0c);
        * ``alpha_z, beta_z`` [deg/MeV] — with the RF focusing at the start,
          the longitudinal Twiss of the 6-D normal mode (its zeta-delta block
          B, normalized: beta = (B B^T)_00 / sqrt(det B B^T)), the phase in
          degrees at ``frequency_mhz`` (default the Beam tab's frequency, else
          the RF's), alpha of opposite sign to zeta's (the phase grows as zeta
          falls); None without RF (the beam coasts);
        * ``centroid_x`` [mm], ``centroid_xp`` [mrad], ``centroid_y``,
          ``centroid_yp`` and, with the RF, ``centroid_dphi`` [deg],
          ``centroid_dw`` [MeV] — the closed orbit at the start (zero on a
          ring without errors or bump), which the matched beam is centred on.

        The optics are those of the run's first turn: the injection magnets
        of turn 0, a ramp's stationary equivalent of turn 0 (the first turn's
        RF focusing).  Also ``species``, ``kinetic_mev``, ``beta0``,
        ``p0c_mev``, ``frequency_mhz`` (the unit of the phase), ``rf_on``,
        ``rf_frequency_hz`` (the first turn's frequency of the first cavity
        with a voltage; None without RF),
        ``ramp`` (True with a ramp), the tunes ``qx, qy, qs`` and ``notes``.
        ``should_stop()``, when given, is asked before every pass of particles
        around the ring of the optics: when it says yes the computation stops
        with :class:`~linac_gen.core.cancelled.OperationCancelled`."""
        from linac_gen.ring import coords as C
        notes, self._notes = self._notes, list(self._compile_notes)
        self._should_stop = should_stop
        try:
            rl, rl_start, ramp, opt, rf_on, _inj, _prog = self._start()
            start_notes = list(self._notes)
        finally:
            self._notes = notes
            self._should_stop = None
        ref = rl.reference
        # (the first turn's RF frequency, as tracked: a ramp's stationary
        # equivalent holds the turn's in the node table)
        f_rf = next((float(rl_start.params[rl_start.offsets[c["node"]] + 1])
                     for c in rl_start.cavities if rl_start.cavity_voltage(c)),
                    None) if rf_on else None
        if frequency_mhz is None:
            cfg_f = getattr(self.beam_config, "frequency", None)
            frequency_mhz = (float(cfg_f) if cfg_f else
                             f_rf * 1e-6 if f_rf else ref.beta0 * C_LIGHT / rl.length * 1e-6)
        f = float(frequency_mhz)
        if not f > 0.0:
            raise ValueError(f"the phase's frequency must be positive (got {frequency_mhz})")
        st = opt.start
        per_mev = 1e3 / (ref.beta0 * ref.p0c)       # per delta → per MeV, m → mm
        co = np.array(opt.closed_orbit, dtype=float)
        # (what the closed-orbit search leaves on a ring without errors — 1e-9
        # in delta and what dispersion makes of it — is zero)
        co[np.abs(co) < CO_ZERO] = 0.0
        h = C.to_helix(co[:, None], ring_ref=ref, w_ref=ref.kinetic0, frequency_mhz=f)[0]
        out = {"species": "H-" if ref.charge < 0 else "proton",
               "kinetic_mev": float(ref.kinetic0), "beta0": float(ref.beta0),
               "p0c_mev": float(ref.p0c), "frequency_mhz": f, "rf_on": bool(rf_on),
               "rf_frequency_hz": f_rf, "ramp": ramp is not None,
               "alpha_x": float(st["alfx"]), "beta_x": float(st["betx"]),
               "alpha_y": float(st["alfy"]), "beta_y": float(st["bety"]),
               "disp_x": float(st["dx"]) * per_mev, "disp_xp": float(st["dpx"]) * per_mev,
               "disp_y": float(st["dy"]) * per_mev, "disp_yp": float(st["dpy"]) * per_mev,
               "alpha_z": None, "beta_z": None,
               # (+ 0.0: a zero, not -0.0, from the sign conventions)
               "centroid_x": float(h[0]) + 0.0, "centroid_xp": float(h[1]) + 0.0,
               "centroid_y": float(h[2]) + 0.0, "centroid_yp": float(h[3]) + 0.0,
               "centroid_dphi": None, "centroid_dw": None,
               "qx": float(opt.qx), "qy": float(opt.qy),
               "qs": None if opt.qs is None else float(opt.qs), "notes": start_notes}
        if rf_on:
            B = np.asarray(opt.W, dtype=float)[4:6, 4:6]
            S = B @ B.T
            e = math.sqrt(max(float(np.linalg.det(S)), 0.0))
            if not e > 0.0:
                raise ValueError("the ring's longitudinal normal mode is degenerate at the start")
            # zeta-delta → phase-energy: dphi/dzeta = -360 f/(beta0 c) [deg/m],
            # dW/ddelta = beta0^2 E0 [MeV]
            k = 360.0 * f * 1e6 / (ref.beta0 * C_LIGHT)
            m = ref.beta0 ** 2 * ref.energy0
            out["beta_z"] = float(S[0, 0] / e * k / m)
            out["alpha_z"] = float(S[0, 1] / e)        # -alpha_zeta = +(B B^T)_01/e
            out["centroid_dphi"] = float(h[4]) + 0.0
            out["centroid_dw"] = float(h[5]) + 0.0
        return out

    def _beam_from_config(self, rl):
        """The Beam tab's distribution, converted to ring coordinates.  It is
        generated at the ring's species and energy (those of the beam the
        lattice declares, or of the energy asked for) with every other
        setting the Beam tab's — its frequency the unit of its phases; a beam
        given ready-made (``beam=``) must be at the ring's reference."""
        import dataclasses

        from linac_gen.distributions.factory import create_beam
        from linac_gen.ring import coords as C
        cfg = self.beam_config
        ring = rl.reference
        ring_sp = "H-" if ring.charge < 0 else "proton"
        if self._beam is not None:
            beam = self._beam
        else:
            if cfg is None:
                raise ValueError("no beam: set the Beam tab, or use the matched source")
            gen = cfg
            if (str(cfg.species).lower() != ring_sp.lower()
                    or not math.isclose(float(cfg.energy), ring.kinetic0, rel_tol=1e-9)):
                gen = dataclasses.replace(cfg, species=ring_sp, energy=float(ring.kinetic0))
                self._notes.append(
                    f"the Beam tab's distribution was generated at the ring's reference, "
                    f"{ring_sp} {ring.kinetic0:.6g} MeV (the Beam tab: {cfg.species} "
                    f"{float(cfg.energy):.6g} MeV)")
            beam = create_beam(gen, seed=int(self.config.seed))
        ref = beam.ref
        sp = ref.species.name.lower()
        if sp != ring_sp.lower() or not math.isclose(ref.w_kin, ring.kinetic0, rel_tol=1e-9):
            raise ValueError(
                f"the beam's reference ({ref.species.name} {ref.w_kin:.6g} MeV) is not the "
                f"ring's ({ring_sp} {ring.kinetic0:.6g} MeV) — give the Beam tab the ring's "
                "beam, or use the matched source")
        alive = ~np.asarray(beam.lost, dtype=bool)
        X = C.from_helix(np.asarray(beam.particles, dtype=float)[alive], w_ref=ref.w_kin,
                         mass=ref.species.mass, frequency_mhz=float(beam.ref.frequency),
                         ring_ref=ring)
        return X

    def _matched(self, rl, opt, rf_on, n, rng):
        """A Gaussian matched to the ring's periodic optics with the Beam tab's
        emittances: each normal mode cut at the cutoff (in amplitude), then
        given exactly its emittance — mean and second moments set (_whitened),
        as HELIX's linac generator sets its beams'."""
        cfg = self.beam_config
        if cfg is None:
            raise ValueError("the matched source needs the Beam tab's emittances")
        ref = rl.reference
        bg = ref.beta0 * ref.gamma0
        ex = float(cfg.emit_nx) * 1e-6 / bg                 # pi mm mrad, normalized rms → m rad
        ey = float(cfg.emit_ny) * 1e-6 / bg
        # emit_z [deg MeV at f] → zeta-delta [m]: dzeta/dphi = beta0 c/(360 f), ddelta/dW = 1/(beta0^2 E0)
        f = float(cfg.frequency) * 1e6
        ez = float(cfg.emit_z) * (ref.beta0 * C_LIGHT / (360.0 * f)) / (ref.beta0 ** 2 * ref.energy0)
        cut = float(getattr(cfg, "cutoff", 0.0) or 0.0) or None
        W = np.asarray(opt.W)
        co = np.asarray(opt.closed_orbit, dtype=float)
        if rf_on:
            Z = _whitened(_truncated_gaussian(rng, n, 3, cut))
            for k, e in enumerate((ex, ey, ez)):
                Z[2 * k:2 * k + 2] *= math.sqrt(e)
            return co[:, None] + _apply(W, Z)
        Z = _truncated_gaussian(rng, n, 2, cut)
        # a coasting beam: delta Gaussian with the Beam tab's rms energy spread,
        # sqrt(emit_z gamma_z), gamma_z = (1 + alpha_z^2)/beta_z; zeta uniform
        # over the period
        beta_z = float(getattr(cfg, "beta_z", 0.0) or 0.0)   # deg/MeV
        alpha_z = float(getattr(cfg, "alpha_z", 0.0) or 0.0)
        if beta_z > 0.0:
            sig_w = math.sqrt(float(cfg.emit_z) * (1.0 + alpha_z ** 2) / beta_z)    # MeV
            sig_d = sig_w / (ref.beta0 ** 2 * ref.energy0)
        else:
            sig_d = 0.0
        u = rng.standard_normal(n)
        if cut:
            bad = np.abs(u) > cut
            while bad.any():
                u[bad] = rng.standard_normal(int(bad.sum()))
                bad = np.abs(u) > cut
        if sig_d > 0.0:
            U = _whitened(np.vstack([Z, u]))
            Z, d = U[:4], U[4] * sig_d
        else:
            Z, d = _whitened(Z), np.zeros(n)
        for k, e in enumerate((ex, ey)):
            Z[2 * k:2 * k + 2] *= math.sqrt(e)
        X = np.empty((6, n))
        X[:4] = co[:4, None] + _apply(W[:4, :4], Z)
        X[5] = d
        D = np.array([opt.start["dx"], opt.start["dpx"], opt.start["dy"], opt.start["dpy"]])
        X[:4] += D[:, None] * d[None, :]
        L = wrap_length(rl) or rl.length
        X[4] = (rng.random(n) - 0.5) * L
        return X

    def _probes(self, opt, rf_on, eps):
        """Tune probes at the closed orbit: one small amplitude (the tunes) or
        an n x n grid of amplitudes up to ``probe_max_sigma`` (the footprint),
        in units of the beam's rms size in each normal mode."""
        n = int(getattr(self.config, "tune_probes", 0) or 0)
        if n <= 0:
            return None, None
        W = np.asarray(opt.W)
        dims = W.shape[0] // 2
        if n == 1:
            a = (PROBE_SMALL_SIGMA_SC if getattr(self.config, "space_charge", None) is not None
                 else PROBE_SMALL_SIGMA)
            amps = np.array([[a, a]])
        else:
            amax = float(getattr(self.config, "probe_max_sigma", 3.0) or 3.0)
            g = np.linspace(0.1, amax, n)
            amps = np.array([[a, b] for a in g for b in g])
        e = [max(float(eps[k]), PROBE_MIN_EMIT[k]) if math.isfinite(float(eps[k]))
             else PROBE_MIN_EMIT[k] for k in range(3)]
        Z = np.zeros((2 * dims, len(amps)))
        Z[0] = amps[:, 0] * math.sqrt(e[0])
        Z[2] = amps[:, 1] * math.sqrt(e[1])
        if dims == 3:
            a_z = (PROBE_SMALL_SIGMA_SC if getattr(self.config, "space_charge", None) is not None
                   else PROBE_SMALL_SIGMA)
            Z[4] = a_z * math.sqrt(e[2])
        X = np.zeros((6, len(amps)))
        X[:2 * dims] = opt.closed_orbit[:2 * dims, None] + _apply(W, Z)
        if dims == 2:
            X[4:] = opt.closed_orbit[4:, None]
        return X, amps

    # -- the run --------------------------------------------------------------
    def run(self, *, progress_callback=None, should_abort=None, live_callback=None,
            checkpoint_path=None, restart_from=None) -> RingResults:
        """Track the turns.  ``progress_callback(turn, n_turns)`` after every
        launch of turns; ``should_abort()`` is asked before each (a stop takes
        effect at the end of the turns being tracked — a turn of a large
        bunch can take seconds); ``live_callback(stats, n_turns, reference)``
        after every statistics row, with the lists of rows so far (each row's
        reference in ``stats["ref_bg"]``, ``stats["ref_p0c"]``: it changes in
        a ramp) and the reference at the start (``beta0gamma0``, ``p0c_mev``);
        ``checkpoint_path`` is written every ``checkpoint_every`` turns;
        ``restart_from`` continues a checkpoint's run — its tune probes,
        their largest amplitude and its statistics interval must be this
        run's too."""
        from linac_gen.ring.config import snapshot_turn_set
        from linac_gen.ring.optics import _energy
        from linac_gen.ring.programs import RampError
        t_start = time.perf_counter()
        cfg = self.config
        n_turns = int(cfg.n_turns)
        if n_turns < 1:
            raise ValueError(f"n_turns must be at least 1 (got {n_turns})")
        # (this run's notes: the compile's, then its own — a second run on the
        # same simulation does not repeat them)
        self._notes = list(self._compile_notes)
        rl, rl_start, ramp, opt, rf_on, inj_cfg, programs = self._start()
        has_foil = inj_cfg is not None or "Foil" in rl.element_types
        ref = rl.reference
        if getattr(cfg, "zeta_wrap", "auto") == "off":
            zwrap = None
        elif ramp is not None:
            # (f = h beta c / C every turn: the RF period in zeta is C/h)
            zwrap = float(rl.length) / int(rl.harmonic)
        else:
            zwrap = wrap_length(rl)
        digest = lattice_digest(rl, zwrap)
        sc = self._space_charge(rl, zwrap, rf_on)
        # (the bunch core's width at the statistics rows: what the line density must resolve)
        # (with bunched space charge: each statistics row records the core of
        # the bunch, on the circle of the RF period, and the line filter's width)
        zcore = sc.period if sc is not None and sc.bunched else None
        inj = None
        if has_foil:
            from linac_gen.ring.injection.injector import Injection, bucket_length, summary_text
            inj = Injection(inj_cfg, rl, lattice=self.lattice, beam_config=self.beam_config,
                            seed=int(cfg.seed), rf_on=rf_on,
                            period_m=bucket_length(rl, rf_on), n_turns=n_turns,
                            programs=programs)
            self._notes.append(summary_text(inj))
        T0 = float(rl.revolution_time)
        W = np.asarray(opt.W)
        Winv = np.linalg.inv(W)
        co = np.asarray(opt.closed_orbit, dtype=float)
        dims = W.shape[0] // 2
        every = max(1, int(getattr(cfg, "stats_every", 1) or 1))
        snap_set = snapshot_turn_set(cfg.snapshot_turns, n_turns)
        live_ref = {"beta0gamma0": float(ref.beta0 * ref.gamma0), "p0c_mev": float(ref.p0c)}
        n_record = min(n_turns, PROBE_RECORD_TURNS)

        rng = np.random.default_rng(int(cfg.seed))
        stats = {"turns": [], "n_alive": [], "mean": [], "cov": [], "ref_bg": [], "ref_p0c": [],
                 "zeta_core": []}
        snapshots: dict = {}
        self._run_info = {"stats_every": every, "probe_record_turns": n_record}
        # the reference (its time and total energy) and a ramp's record
        t_ref, e_ref = (ramp.start() if ramp is not None else (0.0, float(ref.energy0)))
        rec = _ramp_record(n_turns) if ramp is not None else None
        if restart_from is not None:
            ck = load_checkpoint(restart_from)
            if (ramp is None) != (ck.get("ramp") is None):
                raise ValueError("the checkpoint's run " + ("had a" if ramp is None else "had no")
                                 + " ramp — continue it with the same settings")
            saved_cfg = ck["meta"].get("config") or {}
            if ramp is not None and (
                    saved_cfg.get("ramp") != cfg.ramp
                    or saved_cfg.get("corrector_mode", "follow") != cfg.corrector_mode):
                raise ValueError("the checkpoint's run followed another ramp (its programs, "
                                 "gamma_t or correctors differ) — continue it with the same "
                                 "ramp")
            if (saved_cfg.get("space_charge") or None) != (cfg.space_charge or None):
                raise ValueError("the checkpoint's run had other space-charge settings — "
                                 "continue it with the same ones")
            if (saved_cfg.get("injection") or None) != (cfg.injection or None):
                raise ValueError("the checkpoint's run had other injection settings — "
                                 "continue it with the same ones")
            if ck["digest"] != digest:
                raise ValueError("the checkpoint belongs to another ring (its compiled lattice "
                                 "or element model differs) — it cannot be continued here")
            _check_restart(ck["meta"].get("config") or {}, cfg)
            parts = ck["particles"]
            turn0 = int(ck["turn"])
            if turn0 > n_turns:
                raise ValueError(f"the checkpoint is at turn {turn0}, past the {n_turns} "
                                 "turns asked for")
            stats = ck["stats"]
            if not stats.get("ref_bg"):
                # (a checkpoint of fixed energy written before the ramp)
                stats["ref_bg"] = [float(ref.beta0gamma0)] * len(stats["turns"])
                stats["ref_p0c"] = [float(ref.p0c)] * len(stats["turns"])
            snapshots = ck["snapshots"]
            probe0, probe_tbt, amps = ck["probe"], ck["probe_tbt"], ck["probe_amps"]
            eps0 = ck["eps0"]
            if ramp is not None:
                t_ref, e_ref = ck["ramp"]["t_ref"], ck["ramp"]["e_ref"]
                for k, v in ck["ramp"]["record"].items():
                    rec[k][:min(turn0, len(v))] = v[:turn0]
                ramp.n_capped = int(ck["ramp"]["n_capped"])
            if sc is not None:
                sc.restore_frozen(ck.get("sc_frozen"))
                sc.restore_bins(ck["meta"].get("sc_bins"))
                sc.restore_diagnostics(ck["meta"].get("sc_diagnostics"))
            if inj is not None:
                inj.restore(ck.get("injection") or {}, ck["meta"].get("injection") or {}, turn0)
            self._run_info.update({"restarted_from": str(restart_from), "restart_turn": turn0})
        else:
            turn0 = 0
            src = getattr(cfg, "source", "matched")
            if self._coords is not None:
                X = self._coords
            elif src == "empty":
                X = np.zeros((6, 0))
            elif src == "matched" and self._beam is None:
                X = self._matched(rl_start, opt, rf_on, int(cfg.n_particles), rng)
            else:
                X = self._beam_from_config(rl)
            if X.shape[1] == 0 and inj_cfg is None:
                raise ValueError("the beam has no particles")
            n_inj = 0 if inj_cfg is None else inj_cfg.particles_total
            cap = int(cfg.capacity or X.shape[1] + n_inj)
            parts = P.RingParticles.allocate(max(cap, X.shape[1] + n_inj, 1))
            weight = self._weight(X.shape[1])
            # (with injection the space-charge intensity is refused: a beam the
            # ring starts with takes the Beam tab's current, as without)
            if sc is not None and X.shape[1]:
                if sc.cfg.intensity is not None:
                    weight = float(sc.cfg.intensity) / X.shape[1]
                elif not sc.bunched:
                    # (a coasting beam fills the ring: its particles are the
                    # current times the revolution time)
                    weight = self._weight(X.shape[1]) * float(self.beam_config.frequency) \
                        * 1e6 * float(rl.revolution_time)
                if not weight > 0.0:
                    if inj_cfg is not None:
                        raise ValueError(
                            "space charge with a beam at the start and injection: the starting "
                            "beam's intensity comes from the Beam tab's current — give one, or "
                            "start with an empty ring")
                    raise ValueError(
                        "space charge needs the bunch's intensity: give it (real particles "
                        "in the bunch) with the space-charge settings, or a beam current in "
                        "the Beam tab")
            parts.add(X, weight, ref=ref, turn=0)
            sl = slice(0, parts.n_used)
            pt, rp, rv = _energy(parts.coords[5, sl], ref.beta0)
            parts.ptau[sl], parts.rpp[sl], parts.rvv[sl] = pt, rp, rv
            disp = np.array([opt.start["dx"], opt.start["dpx"], opt.start["dy"],
                             opt.start["dpy"]])
            # (a ring that starts empty: the emittances of the beam to inject,
            # carried from the foil to the start of the ring — the probes'
            # amplitudes are sized with them)
            Xe = (parts.coords[:, sl] if parts.n_used or inj is None or inj.pool is None
                  else _pool_at_start(rl_start, inj, ref))
            eps0 = _normal_emittances(Xe, Winv, co, disp) if Xe.shape[1] else \
                np.full(dims if dims == 3 else 2, np.nan)
            if dims == 2:
                # (no RF: the zeta-delta second-moment emittance of the beam)
                c = (_moments(np.ascontiguousarray(Xe[4:6]))[1]
                     if Xe.shape[1] > 1 else np.zeros((2, 2)))
                eps0 = np.append(eps0, math.sqrt(max(c[0, 0] * c[1, 1] - c[0, 1] ** 2, 0.0)))
            if parts.n_used > 2:
                # (the beam the ring starts with against the ring's matched one,
                # mode by mode: x, y and, with the RF, z)
                bm = _normal_mismatch(parts.coords[:, sl], Winv, co, disp)
                self._run_info["start_mismatch"] = [None if not math.isfinite(v) else float(v)
                                                    for v in bm]
                if np.nanmax(np.append(bm, 1.0)) > MISMATCH_NOTE:
                    self._notes.append(
                        "the beam at the start is mismatched to the ring's optics: BMAG "
                        + ", ".join(f"{p} {v:.4f}" for p, v in zip("xyz", bm, strict=False)
                                    if math.isfinite(v))
                        + " (1 when matched)")
            probeX, amps = self._probes(opt, rf_on, eps0)
            probe_tbt = None
            probe0 = None
            if probeX is not None:
                probe_tbt = np.full((probeX.shape[1], n_record + 1, 6), np.nan)
                probe_tbt[:, 0, :] = probeX.T
                pt, rp, rv = _energy(probeX[5], ref.beta0)
                probe0 = {"X": np.ascontiguousarray(probeX, dtype=float),
                          "ptau": np.ascontiguousarray(pt), "rpp": np.ascontiguousarray(rp),
                          "rvv": np.ascontiguousarray(rv),
                          "state": np.ones(probeX.shape[1], dtype=np.int8)}
            if zwrap:
                _wrap(parts.coords[4, :parts.n_used], zwrap)
            self._record(stats, parts, 0, *_ref_of(e_ref, ref), core=zcore, sc=sc)
            if sc is not None and sc.auto_z:
                sc.start_bins(stats["zeta_core"][-1], stats["n_alive"][-1])
                # (row 0 with the filter its turn uses: set just now)
                stats["line_filter_m"][-1] = float(sc.filter_width())
            if 0 in snap_set:
                snapshots[0] = _snapshot(parts)
                if ramp is not None:
                    snapshots[0]["reference"] = _snapshot_reference(ramp, 0, t_ref, e_ref,
                                                                    float(opt.alpha_c))

        if probe_tbt is not None and probe_tbt.shape[1] < n_record + 1:
            grown = np.full((probe_tbt.shape[0], n_record + 1, 6), np.nan)
            grown[:, :probe_tbt.shape[1]] = probe_tbt
            probe_tbt = grown

        # (with a foil: every slot tracked — the injection fills them — and
        # the beam written by the host at the foil)
        tracker = _make_tracker(cfg, rl, parts, zwrap or 0.0, full=inj is not None)
        probes = None if probe0 is None else _ProbeTracker(rl, probe0, zwrap or 0.0)
        snaps_at = {t for t in snap_set if 0 < t <= n_turns}
        ck_every = int(getattr(cfg, "checkpoint_every", 0) or 0)
        turn = turn0
        refused = None
        chunk = 1
        t_ref0 = ramp.start()[0] if ramp is not None else 0.0
        try:
            while turn < n_turns:
                if (stats["n_alive"][-1] == 0 and stats["turns"][-1] == turn
                        and not (inj is not None and inj.pending(turn))):
                    break                      # (every particle lost: nothing left to track)
                if should_abort is not None and should_abort():
                    break
                # turns launched together up to the next event (statistics,
                # snapshot, checkpoint, end); zeta is wrapped in the kernel
                # after every turn, so the grouping changes no bit
                nxt = min([n_turns, (turn // every + 1) * every]
                          + [t for t in snaps_at if t > turn]
                          + ([(turn // ck_every + 1) * ck_every] if ck_every else []))
                k = max(1, min(chunk, nxt - turn))
                if sc is not None or inj is not None:
                    k = 1
                plan = None
                if ramp is not None:
                    # a ramp: one turn at a time, its programs written first
                    k = 1
                    try:
                        plan = ramp.plan(turn, t_ref, e_ref)
                    except RampError as exc:
                        # (a gain the RF cannot give, refused: the run ends here,
                        # with the turns done)
                        refused = str(exc)
                        self._notes.append(f"stopped: {refused}")
                        break
                    ramp.apply(plan, rl.params, rl.ref_beta0, rl.ref_p0c, rl.ref_bg2)
                    tracker.update_table(rl)
                    _record_turn(rec, plan)
                foil = None
                if inj is not None:
                    # the turn's time from the start of the run, and its length
                    t_run = (t_ref - t_ref0) if ramp is not None else turn * T0
                    t_len = plan.turn_time_s if plan is not None else T0
                    if inj.has_programs:
                        inj.apply_programs(turn, t_run, rl.params)
                        tracker.update_table(rl)
                        if ramp is None:
                            inj.record_closed_orbit(turn, rl)
                    foil = (inj.node, _FoilStep(inj, tracker, parts, rl, ref, turn, t_run, t_len))
                t0 = time.perf_counter()
                if sc is not None:
                    # (a frozen model is measured again every turn that injects,
                    # and on the turn after the last: the beam is growing)
                    _sc_turn(sc, tracker, probes, turn, rl.n_nodes,
                             probe_tbt if turn < n_record else None, foil=foil,
                             refresh_frozen=inj is not None and (inj.injecting(turn)
                                                                 or inj.injecting(turn - 1)))
                elif foil is not None:
                    _foil_turn(tracker, probes, turn, rl.n_nodes, foil,
                               probe_tbt if turn < n_record else None)
                else:
                    tracker.track(turn, k)
                    if probes is not None:
                        hi = min(turn + k, n_record)
                        if hi > turn:
                            got = probes.track(turn, hi - turn)
                            probe_tbt[:, turn + 1:hi + 1, :] = got
                        if turn + k > hi:
                            probes.advance(max(turn, hi), turn + k - max(turn, hi))
                dt = time.perf_counter() - t0
                # (about a quarter of a second of work per launch: a stop
                # request is seen soon, few launches for a long run)
                chunk = int(min(4096, max(1, chunk * 2 if dt < 0.12 else
                                          (chunk // 2 if dt > 0.5 else chunk))))
                turn += k
                if plan is not None:
                    t_ref += plan.turn_time_s
                    e_ref = plan.energy_end_mev
                if turn % every == 0 or turn == n_turns:
                    tracker.sync_host()
                    self._record(stats, parts, turn, *_ref_of(e_ref, ref), core=zcore,
                                 sc=sc)
                    if live_callback is not None:
                        live_callback(stats, n_turns, live_ref)
                if turn in snaps_at:
                    tracker.sync_host()
                    snapshots[turn] = _snapshot(parts)
                    if ramp is not None:
                        snapshots[turn]["reference"] = _snapshot_reference(
                            ramp, turn, t_ref, e_ref, float(opt.alpha_c))
                if ck_every and checkpoint_path and turn % ck_every == 0 and turn < n_turns:
                    tracker.sync_host()
                    save_checkpoint(checkpoint_path, parts=parts, turn=turn, stats=stats,
                                    snapshots=snapshots,
                                    probe=None if probes is None else probes.arrays(),
                                    probe_tbt=probe_tbt, probe_amps=amps, eps0=eps0,
                                    digest=digest, config=cfg.to_dict(),
                                    ramp=None if ramp is None else {
                                        "t_ref": t_ref, "e_ref": e_ref,
                                        "n_capped": ramp.n_capped,
                                        "record": {k2: v2[:turn] for k2, v2 in rec.items()}},
                                    sc_frozen=None if sc is None else sc.frozen_arrays(),
                                    sc_bins=None if sc is None else sc.bins_state(),
                                    sc_diag=None if sc is None else sc.diagnostics_state(),
                                    injection=None if inj is None else
                                    (inj.state_arrays(), inj.state_meta()))
                if progress_callback is not None:
                    progress_callback(turn, n_turns)
        finally:
            tracker.sync_host()
            tracker.close()
        if stats["turns"][-1] != turn:
            self._record(stats, parts, turn, *_ref_of(e_ref, ref), core=zcore, sc=sc)
        led = parts.ledger()
        # (a ring that never held a circulating particle — its injection all
        # waste, or not reached — has lost nothing)
        circulated = led["circulating"] + led["lost"] > 0
        all_lost = stats["n_alive"][-1] == 0 and circulated
        empty = stats["n_alive"][-1] == 0 and not circulated
        if inj is not None and inj.has_programs:
            # (the magnets after the last turn, and the closed orbit then)
            import dataclasses
            t_end = (t_ref - t_ref0) if ramp is not None else turn * T0
            p_end = rl.params.copy()
            inj.apply_programs(turn, t_end, p_end)
            if ramp is None:
                inj.record_closed_orbit(turn, dataclasses.replace(rl, params=p_end))

        res = self._results(rl_start, opt, rf_on, parts, stats, snapshots, probe_tbt, amps,
                            tracker, probes, Winv, co, eps0, turn, n_turns, zwrap, digest)
        res.row_beta0gamma0 = np.asarray(stats["ref_bg"], dtype=float)
        res.row_p0c_mev = np.asarray(stats["ref_p0c"], dtype=float)
        if inj is not None:
            res.injection = inj.results(turn, parts)
            if empty and inj.cfg is not None and turn > inj.cfg.start_turn:
                res.injection["notes"].append("nothing circulated: every injected particle "
                                              "was waste (it missed the foil, or left it "
                                              "unstripped)")
            res.meta["notes"] = list(self._notes) + list(res.injection.get("notes", []))
        if ramp is not None:
            # (the program as it ran: a cycle's blank start energy filled in)
            res.ramp = _ramp_results(ramp, rec, turn, t_ref, e_ref, ramp.cfg.to_dict())
            res.meta["final_kinetic_mev"] = float(e_ref - ref.mass)
            # the bucket at the start (the stationary equivalent's is not it)
            res.optics.pop("bucket", None)
            if rf_on:
                from linac_gen.ring.optics import accelerating_bucket
                t0, e0 = ramp.start()
                sr = _snapshot_reference(ramp, 0, t0, e0, float(opt.alpha_c))
                b = accelerating_bucket(sr["voltage_eff_mv"], sr["phi_eff_deg"], sr["harmonic"],
                                        sr["eta"], sr["beta0"], sr["energy_mev"], sr["length_m"])
                res.optics["bucket"] = {k: (float(v) if not isinstance(v, bool) else v)
                                        for k, v in b.items() if k != "separatrix"}
        if sc is not None:
            # (the intensity used: the settings' or, left blank, from the beam's
            # current — over its frequency for a bunched beam, times the revolution
            # time for a coasting one: the real particles the bunch started with)
            extra = {}
            if inj is not None and inj.cfg is not None:
                # (with injection: the beam circulating at the end — its
                # intensity from the injected current, its tune shift then)
                n_b = float(led["circulating_charge"])
                Xe = np.ascontiguousarray(parts.coords[:, :parts.n_used][:, parts.alive])
                d0 = np.array([opt.start["dx"], opt.start["dpx"], opt.start["dy"],
                               opt.start["dpy"]])
                las = (_laslett_estimate(sc, opt, stats, _normal_emittances(Xe, Winv, co, d0),
                                         parts, rl_start, row=-1, intensity=n_b)
                       if Xe.shape[1] > 1 else None)
                extra["intensity_from"] = "injection"
            else:
                n_b = float(parts.ledger()["injected_charge"])
                las = _laslett_estimate(sc, opt, stats, eps0, parts, rl_start)
            lr = _line_resolution(sc, stats)
            if lr is not None:
                extra["line_density_resolution"] = lr
                if lr["peak_ratio"] < LINE_PEAK_NOTE:
                    from linac_gen.ring.sc.solver import AUTO_NZ_MAX
                    more = {
                        None: "more z_bins (with enough particles in each) resolve it",
                        "particles": (f"the line density was smoothed over "
                                      f"{lr['line_filter_m'] * 1e3:.3g} mm (rms) to hold down "
                                      f"the noise of its {lr.get('particles', 0)} particles — "
                                      "more particles resolve it"),
                        "largest": f"the bins were at their largest number, {AUTO_NZ_MAX}",
                        "follow": ("the bins follow the core a turn late and in factors of "
                                   "two" + (", on the turns the frozen model measures the "
                                            "beam" if sc.frozen else "")
                                   + ", and it shortened faster")}[lr.get("limited_by")]
                    res.meta.setdefault("notes", []).append(
                        f"space charge: the bunch's core was as short as "
                        f"{lr['zeta_core_rms_m'] * 100:.3g} cm rms (turn {lr['turn']}), "
                        f"{lr['bins_per_sigma']:.3g} line-density bins of "
                        f"{sc.period / lr['z_bins'] * 1e3:.3g} mm — the bins and the filter keep "
                        f"about {lr['peak_ratio'] * 100:.0f} % of the peak of a Gaussian that "
                        f"short, and of the space-charge kick at its centre; {more}")
            res.meta["space_charge"] = dict(sc.cfg.to_dict(), **sc.summary(),
                                            # (counted by the deposit: not in the frozen model)
                                            outside_grid_max=(None if sc.frozen
                                                              else int(sc.n_outside_max)),
                                            intensity_used=n_b, laslett_gaussian=las, **extra)
        res.truncated = turn < n_turns and not (all_lost or empty)
        if all_lost:
            res.meta["all_lost_at_turn"] = turn
        if refused is not None:
            res.meta["ramp_refused"] = refused
        res.wall_s = time.perf_counter() - t_start
        return res

    def _space_charge(self, rl, zwrap, rf_on):
        """The run's space-charge kicks (None without space charge)."""
        spec = getattr(self.config, "space_charge", None)
        if spec is None:
            return None
        from linac_gen.ring.sc.solver import SpaceCharge
        return SpaceCharge(spec, rl, zwrap=zwrap, rf_on=rf_on)

    def _weight(self, n: int) -> float:
        """Real particles per macroparticle: the Beam tab's bunch charge
        (current / bunch frequency) over the macroparticles."""
        cfg = self.beam_config
        try:
            q = float(cfg.current) * 1e-3 / (float(cfg.frequency) * 1e6)
        except (AttributeError, TypeError, ZeroDivisionError, ValueError):
            return 0.0
        return q / 1.602176634e-19 / max(n, 1)

    @staticmethod
    def _record(stats, parts, turn: int, ref_bg: float, ref_p0c: float, *,
                core: float | None = None, sc=None) -> None:
        """A statistics row; with ``core`` (the RF period of a bunched beam with
        space charge) also the rms width of the bunch's core in zeta, 1.4826
        times the median absolute deviation on the circle of the period (what
        the line density must resolve; a halo does not hide it, nor does a
        bunch lying across the period's ends), and the width of the line
        filter in use (``sc``), NaN otherwise."""
        stats["ref_bg"].append(float(ref_bg))
        stats["ref_p0c"].append(float(ref_p0c))
        m = parts.alive
        # (a C-ordered copy: boolean indexing may hand back a column-ordered
        # array, whose sums numpy takes in another order — other last bits)
        X = np.ascontiguousarray(parts.coords[:, :parts.n_used][:, m])
        n = X.shape[1]
        stats["turns"].append(int(turn))
        stats["n_alive"].append(int(n))
        if n:
            mu, C = _moments(X)
            stats["mean"].append(mu)
            stats["cov"].append(C)
        else:
            stats["mean"].append(np.full(6, np.nan))
            stats["cov"].append(np.full((6, 6), np.nan))
        stats.setdefault("zeta_core", []).append(
            _periodic_core(X[4], core) if core and n else math.nan)
        stats.setdefault("line_filter_m", []).append(
            float(sc.filter_width()) if core and sc is not None else math.nan)

    def _results(self, rl, opt, rf_on, parts, stats, snapshots, probe_tbt, amps, tracker,
                 probes, Winv, co, eps0, turn, n_turns, zwrap, digest) -> RingResults:
        nu = parts.n_used
        lost = parts.state[:nu] == P.LOST
        idx = np.flatnonzero(lost)
        nodes = parts.lost_node[idx]
        elem = np.where(nodes >= 0, rl.node_element[np.clip(nodes, 0, None)], -1)
        losses = {"pid": parts.pid[idx].copy(), "turn": parts.lost_turn[idx].copy(),
                  "node": nodes.copy(), "element": elem.astype(np.int64),
                  "s": np.where(nodes >= 0, rl.node_s[np.clip(nodes, 0, None)], np.nan),
                  "cause": parts.lost_cause[idx].copy(),
                  "coords": parts.coords[:, idx].copy()}
        tunes = {}
        if probe_tbt is None:
            probes = {}
        else:
            probes, tunes = _probe_tunes(probe_tbt[:, :min(turn, PROBE_RECORD_TURNS) + 1],
                                         amps, Winv, co, probes.state(), rf_on)
        ref = rl.reference
        res = RingResults(
            engine=ENGINE, n_turns_requested=n_turns,
            turns=np.asarray(stats["turns"], dtype=np.int64),
            n_alive=np.asarray(stats["n_alive"], dtype=np.int64),
            mean=np.asarray(stats["mean"], dtype=float).reshape(-1, 6),
            cov=np.asarray(stats["cov"], dtype=float).reshape(-1, 6, 6),
            snapshots=snapshots, losses=losses, probes=probes, tunes=tunes,
            optics=opt.summary(), ledger=parts.ledger(), optics_table=_optics_table(opt))
        if rf_on:
            # (what tracking measures with the RF on: the 6-D modes, whose
            # transverse tunes differ from the 4-D ones by the synchro-betatron
            # coupling of dispersion at the cavities)
            from linac_gen.ring.optics.normal_form import normal_form
            _w, mu = normal_form(np.asarray(opt.R))
            q6 = (mu / (2.0 * math.pi)) % 1.0
            res.optics["eigen_tunes_6d"] = [float(q6[0]), float(q6[1]),
                                            float(min(q6[2], 1.0 - q6[2]))]
            from linac_gen.ring.optics import rf_bucket
            try:
                b = rf_bucket(rl, opt)
                res.optics["bucket"] = {k: float(v) for k, v in b.items()
                                        if k != "separatrix" and v is not None}
            except ValueError:
                pass
        res.meta = {
            "n_particles": int(nu), "length_m": float(rl.length), "n_nodes": int(rl.n_nodes),
            "species": "H-" if ref.charge < 0 else "proton", "kinetic_mev": float(ref.kinetic0),
            "p0c_mev": float(ref.p0c), "beta0": float(ref.beta0), "gamma0": float(ref.gamma0),
            "beta0gamma0": float(ref.beta0 * ref.gamma0), "mass_mev": float(ref.mass),
            "harmonic": rl.harmonic, "rf_on": bool(rf_on), "zeta_wrap_m": zwrap,
            "revolution_time_s": float(rl.revolution_time),
            "element_names": list(rl.element_names), "element_s": np.asarray(rl.element_s),
            "element_length": np.asarray(rl.element_length),
            "options": dict(rl.options), "warnings": list(rl.warnings),
            "device": tracker.device, "threads": tracker.threads,
            "compiled": tracker.compiled, "lattice_digest": digest,
            "config": self.config.to_dict(), "eps0_normal_modes": [float(v) for v in eps0],
            "closed_orbit": [float(v) for v in co],
            "notes": list(self._notes),
        }
        res.meta.update(getattr(self, "_run_info", {}))
        return res


# ---------------------------------------------------------------------------
# small helpers of the turn loop
# ---------------------------------------------------------------------------
def _wrap(z: np.ndarray, L: float) -> None:
    """zeta into [-L/2, L/2) in place."""
    z -= L * np.floor(z / L + 0.5)


def _ref_of(energy: float, ref) -> tuple[float, float]:
    """(beta0 gamma0, p0c [MeV]) of the reference at total energy
    ``energy``: the compiled reference's own values at its energy."""
    if energy == float(ref.energy0):
        return float(ref.beta0gamma0), float(ref.p0c)
    g = energy / ref.mass
    bg = math.sqrt(g * g - 1.0)
    return bg, bg * ref.mass


#: a ramp's turn-by-turn record (each turn at its start)
_RAMP_FIELDS = ("t_s", "energy_mev", "gain_mev", "voltage_mv", "phi_deg", "frequency_hz",
                "above_transition", "capped")


def _ramp_record(n_turns: int) -> dict:
    rec = {k: np.full(n_turns, np.nan) for k in _RAMP_FIELDS[:6]}
    rec["above_transition"] = np.zeros(n_turns, dtype=bool)
    rec["capped"] = np.zeros(n_turns, dtype=bool)
    return rec


def _optics_table(opt) -> dict:
    """The optics' table at every element's exit as plain values: names and
    types as lists of str, the rest float arrays."""
    out = {}
    for k, v in (getattr(opt, "table", None) or {}).items():
        out[k] = ([str(x) for x in v] if k in ("name", "type")
                  else np.asarray(v, dtype=float).copy())
    return out


def _record_turn(rec: dict, plan) -> None:
    n = plan.turn
    rec["t_s"][n], rec["energy_mev"][n], rec["gain_mev"][n] = plan.t_s, plan.energy_mev, \
        plan.gain_mev
    rec["voltage_mv"][n], rec["phi_deg"][n] = plan.voltage_mv, plan.phi_deg
    rec["frequency_hz"][n] = plan.frequency_hz
    rec["above_transition"][n], rec["capped"][n] = plan.above_transition, plan.capped


def _snapshot_reference(ramp, turn: int, t: float, energy: float, alpha_c: float) -> dict:
    """The reference and the RF of the turn after a snapshot, and what its
    bucket needs (linac_gen.ring.optics.accelerating_bucket); the RF is NaN
    when that turn would be refused (the gain beyond the voltage)."""
    from linac_gen.ring.programs import RampError
    g = energy / ramp.mass
    out = {"t_s": float(t), "kinetic_mev": float(energy - ramp.mass),
           "energy_mev": float(energy), "beta0": math.sqrt(1.0 - 1.0 / (g * g)),
           "eta": float(alpha_c - 1.0 / (g * g)), "harmonic": int(ramp.h),
           "length_m": float(ramp.length)}
    try:
        pl = ramp.plan(turn, t, energy)
    except RampError:
        nan = float("nan")
        out.update(voltage_mv=nan, phi_deg=nan, voltage_eff_mv=nan, phi_eff_deg=nan,
                   frequency_hz=nan, above_transition=bool(
                       ramp.gamma_t is not None and g > ramp.gamma_t))
        return out
    ramp.n_capped -= int(pl.capped)          # (a look ahead, not a turn)
    out.update(voltage_mv=float(pl.voltage_mv), phi_deg=float(pl.phi_deg),
               voltage_eff_mv=float(pl.voltage_mv * ramp.r_unit),
               phi_eff_deg=float(pl.phi_deg + math.degrees(ramp.arg)),
               frequency_hz=float(pl.frequency_hz), above_transition=bool(pl.above_transition))
    return out


#: the most turns of the program's clock walked for a run's "turns_to_end"
#: (about a microsecond each; a longer program reports None)
TURNS_TO_END_CAP = 2_000_000


def _ramp_results(ramp, rec: dict, turns: int, t_end: float, e_end: float, cfg: dict) -> dict:
    from linac_gen.ring.programs import RampError, transition_crossing
    out = {k: v[:turns].copy() for k, v in rec.items()}
    out["kinetic_mev"] = out.pop("energy_mev") - ramp.mass
    above = out["above_transition"]
    try:
        n_end = ramp.turns_to_end(max_turns=TURNS_TO_END_CAP)
    except RampError:
        n_end = None
    out.update({"transition_turn": transition_crossing(above),
                "above_transition_at_start": bool(above[0]) if len(above) else None,
                "gamma_t": ramp.gamma_t, "n_capped": int(ramp.n_capped),
                "voltage_eff_factor": float(ramp.r_unit),
                "phase_offset_deg": float(math.degrees(ramp.arg)),
                "t_end_s": float(t_end), "kinetic_end_mev": float(e_end - ramp.mass),
                "gamma_end": float(e_end / ramp.mass), "program": cfg,
                "turns_to_end": n_end})
    return out


def _tune_step(q_a: float, q_b: float) -> float:
    """q_a - q_b for tunes, fractions of a turn: across an integer (0.999
    then 0.001) the tune moved by 0.002, not 0.998; a step under half a turn
    is the plain difference, bit for bit."""
    d = q_a - q_b
    return d - 1.0 if d > 0.5 else (d + 1.0 if d < -0.5 else d)


#: the settings a checkpoint's run and its continuation must share (the seed:
#: an injection draws every turn's bunch and foil scattering from it)
_RESTART_KEYS = ("tune_probes", "probe_max_sigma", "stats_every", "seed")


def _check_restart(saved: dict, cfg) -> None:
    """Refuse to continue a checkpoint with other probes, another
    statistics interval or another seed: the results would mix the two."""
    diff = [f"{k}={saved[k]!r}" for k in _RESTART_KEYS
            if k in saved and saved[k] != getattr(cfg, k)]
    if diff:
        raise ValueError("the checkpoint's run had " + ", ".join(diff) + " — continue it with "
                         "the same settings (this run: " + ", ".join(
                             f"{k}={getattr(cfg, k)!r}" for k in _RESTART_KEYS) + ")")


def _snapshot(parts) -> dict:
    m = parts.alive
    sl = slice(0, parts.n_used)
    return {"coords": np.ascontiguousarray(parts.coords[:, sl][:, m]),
            "pid": parts.pid[sl][m].copy()}


def _probe_tunes(tbt, amps, Winv, co, state, rf_on):
    """NAFF tunes of every probe that survived (normalized coordinates)."""
    from linac_gen.ring import naff
    n_probe, n_t, _ = tbt.shape
    dims = Winv.shape[0] // 2
    q = np.full((n_probe, 3), np.nan)
    diff = np.full(n_probe, np.nan)
    for i in range(n_probe):
        X = tbt[i].T
        if not np.all(np.isfinite(X)) or n_t < 16:
            continue
        Z = Winv @ (X[:2 * dims] - co[:2 * dims, None])
        for k in range(dims):
            qk = naff.tunes_from_normalized(Z[2 * k], Z[2 * k + 1])
            if k == 2:
                qk = min(qk, 1.0 - qk)            # (the rotation reverses above transition)
            # (NAFF needs a few periods of the motion in the record: fewer
            # than MIN_PERIODS and the tune is not reported)
            if min(qk, 1.0 - qk) * (n_t - 1) < MIN_PERIODS:
                qk = float("nan")
            q[i, k] = qk
        if n_t >= 64:
            h = n_t // 2
            q1 = [naff.tunes_from_normalized(Z[2 * k, :h], Z[2 * k + 1, :h]) for k in range(2)]
            q2 = [naff.tunes_from_normalized(Z[2 * k, h:], Z[2 * k + 1, h:]) for k in range(2)]
            d = math.hypot(*(_tune_step(a, b) for a, b in zip(q1, q2, strict=True)))
            diff[i] = d
    probes = {"tbt": tbt, "amplitudes_sigma": amps, "tunes": q, "diffusion": diff,
              "state": state}
    k0 = int(np.argmin(np.hypot(amps[:, 0], amps[:, 1])))
    tunes = {}
    for k, name in enumerate(("qx", "qy", "qs")):
        if k < dims and np.isfinite(q[k0, k]) and (k < 2 or rf_on):
            tunes[name] = float(q[k0, k])
    return probes, tunes


def _periodic_core(z, period: float) -> float:
    """1.4826 times the median absolute deviation of ``z`` on the circle of
    ``period``: z is first moved by its circular mean, so that a bunch lying
    across the period's ends is measured whole."""
    z = np.asarray(z, dtype=float)
    ang = (2.0 * math.pi / period) * z
    c = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))) \
        * period / (2.0 * math.pi)
    zz = np.mod(z - c + 0.5 * period, period) - 0.5 * period
    return 1.4826 * float(np.median(np.abs(zz - np.median(zz))))


def _line_resolution(sc, stats) -> dict | None:
    """The shortest bunch core over the statistics rows (of at least
    LINE_MIN_PARTICLES particles) and how much of the peak line density of a
    Gaussian that short the space charge's bins and filter keep (None for a
    coasting beam or without such rows)."""
    if not sc.bunched:
        return None
    best = None
    widths = stats.get("line_filter_m") or []
    for i, (turn, n, sz) in enumerate(zip(stats["turns"], stats["n_alive"],
                                          stats.get("zeta_core", []), strict=False)):
        if n < LINE_MIN_PARTICLES or not np.isfinite(sz) or sz <= 0.0:
            continue
        nz = sc.bins_at(max(int(turn) - 1, 0))
        lw = widths[i] if i < len(widths) and np.isfinite(widths[i]) else None
        r = sc.peak_ratio(sz, nz=nz, line_width=lw)
        if best is None or r < best["peak_ratio"]:
            best = {"zeta_core_rms_m": float(sz), "turn": int(turn), "z_bins": int(nz),
                    "bins_per_sigma": float(sz) * nz / sc.period, "peak_ratio": r,
                    "line_filter_m": (float(lw) if lw is not None else
                                      float(sc.cfg.z_smoothing_bins) * sc.period / nz)}
            if sc.auto_z:
                best["limited_by"] = sc.auto_limit(float(sz), float(n), nz, lw)
                best["particles"] = int(n)
    return best


def _laslett_estimate(sc, opt, stats, eps0, parts, rl, *, row: int = 0,
                      intensity: float | None = None) -> list | None:
    """The analytic tune shift (dQx, dQy) at the centre of a Gaussian beam with
    the emittances ``eps0``, the energy spread and bunch length of the
    statistics' row ``row`` (the start) and the intensity (the ledger's
    unless given) (:mod:`linac_gen.ring.sc.laslett`); None when the optics
    cannot give it."""
    from linac_gen.ring.sc.laslett import tune_shift
    try:
        c0 = np.asarray(stats["cov"][row], dtype=float)
        sig_d = math.sqrt(max(float(c0[5, 5]), 0.0))
        sig_z = math.sqrt(max(float(c0[4, 4]), 0.0))
        f = (1.0 / (math.sqrt(2.0 * math.pi) * sig_z) if sc.bunched and sig_z > 0.0
             else 1.0 / float(rl.length))
        n_b = float(parts.ledger()["injected_charge"]) if intensity is None else intensity
        dq = tune_shift(opt, emit_x=float(eps0[0]), emit_y=float(eps0[1]),
                        intensity=n_b, line_density=f,
                        sigma_delta=sig_d, mass_mev=float(rl.reference.mass))
    except (KeyError, IndexError, ValueError, ZeroDivisionError, FloatingPointError):
        return None
    return [float(dq[0]), float(dq[1])] if all(math.isfinite(v) for v in dq) else None


def _sc_turn(sc, tracker, probes, turn: int, n_nodes: int, probe_tbt, foil=None,
             refresh_frozen: bool = False) -> None:
    """One turn from kick to kick (space charge): the beam and the tune probes
    tracked to each kick, then kicked by the beam's field (the probes feel it
    without adding to it); the probes' coordinates after the turn put in
    ``probe_tbt[:, turn + 1]`` (None: not kept).  ``foil`` — (node, step):
    the foil's step (:class:`_FoilStep`) is one more event, before node
    ``node`` (before a kick at the same node: the particles it injects feel
    that kick).  ``refresh_frozen``: a frozen model measures the beam again
    this turn whatever its settings."""
    beam = tracker.sc_beam()
    parr = None if probes is None else probes.kick_arrays()
    refresh = sc.needs_refresh(turn) or (sc.frozen and refresh_frozen)
    sc.begin_turn(turn, refresh)
    n_b = None

    def frozen_intensity():
        from linac_gen.ring.sc.solver import combine_moments
        return combine_moments(beam.moments(0.0, 0.0), 0.0, 0.0)["w"]

    if sc.frozen and not refresh:
        n_b = frozen_intensity()
    # (zeta is wrapped once, at the end of the turn, as without space charge:
    # in a ramp the reference changes within the turn and so does the RF
    # period in zeta — a wrap by C/h between two kicks would not be exact;
    # the line density is periodic, so a particle a little beyond the period
    # between kicks is binned where it belongs)
    events = [(int(j), 1, k) for k, j in enumerate(sc.nodes)]
    if foil is not None:
        events.append((int(foil[0]), 0, -1))
        events.sort()
    j_prev = 0
    for j, what, k in events:
        if j > j_prev:
            tracker.track_nodes(turn, j_prev, j, wrap=False)
            if probes is not None:
                probes.track_nodes(turn, j_prev, j, wrap=False)
        if what == 0:
            added = foil[1]()
            if added and n_b is not None:
                n_b = frozen_intensity()
        else:
            sc.kick(k, beam, parr, n_b=n_b, refresh=refresh)
        j_prev = j
    if j_prev < n_nodes or foil is not None:
        tracker.track_nodes(turn, j_prev, n_nodes, wrap=True)
        if probes is not None:
            probes.track_nodes(turn, j_prev, n_nodes, wrap=True)
    if probes is not None and probe_tbt is not None:
        probe_tbt[:, turn + 1, :] = probes.coords_nan()
    sc.end_turn(turn)


def _foil_turn(tracker, probes, turn: int, n_nodes: int, foil, probe_tbt) -> None:
    """One turn with a foil (no space charge): tracked to the foil, its step
    (``foil`` = (node, step)), then to the end of the turn (zeta wrapped
    there); the probes alongside, their coordinates after the turn put in
    ``probe_tbt[:, turn + 1]`` (None: not kept)."""
    node, step = foil
    if node > 0:
        tracker.track_nodes(turn, 0, node, wrap=False)
        if probes is not None:
            probes.track_nodes(turn, 0, node, wrap=False)
    step()
    tracker.track_nodes(turn, node, n_nodes, wrap=True)
    if probes is not None:
        probes.track_nodes(turn, node, n_nodes, wrap=True)
        if probe_tbt is not None:
            probe_tbt[:, turn + 1, :] = probes.coords_nan()


def _ref_at(rl, j: int) -> tuple[float, float]:
    """(p0c, beta0) of the reference before node j (j = n_nodes: at the end of
    the turn, after a last reference change)."""
    n = rl.n_nodes
    if j < n:
        return float(rl.ref_p0c[j]), float(rl.ref_beta0[j])
    from linac_gen.ring.kernels import nodes as nd
    last = n - 1
    if int(rl.kinds[last]) == nd.N_REFSHIFT:
        o = int(rl.offsets[last])
        return float(rl.params[o + 1]), float(rl.params[o + 2])
    return float(rl.ref_p0c[last]), float(rl.ref_beta0[last])


class _FoilStep:
    """The foil's step of one turn, on the host: the circulating particles that
    cross the foil, then the turn's injection (on a CUDA GPU only the particles
    inside the foil and the slots the injection fills cross to the host and
    back).  Calling it returns whether particles were added."""

    def __init__(self, inj, tracker, parts, rl, ref, turn, t_run, t_len):
        self.inj, self.tracker, self.parts, self.rl = inj, tracker, parts, rl
        self.ref, self.turn, self.t_run, self.t_len = ref, int(turn), float(t_run), float(t_len)

    def __call__(self) -> bool:
        import dataclasses
        inj, parts, tr = self.inj, self.parts, self.tracker
        p0c, b0 = _ref_at(self.rl, inj.node)
        local = self.ref if p0c == float(self.ref.p0c) else dataclasses.replace(self.ref, p0c=p0c)
        kin = float(local.kinetic0)
        if tr.device_foil:
            # (on a GPU: the particles inside the foil found there, those
            # alone through the foil on the host, the injected slots sent)
            idx = tr.foil_slots(parts.n_used, inj.foil.bounds)
            X = tr.gather(idx)
            got = inj.foil_cross(X, parts.weight[idx], self.turn, p0c=p0c, beta0=b0,
                                 kinetic_mev=kin, crossings=parts.foil_hits[idx])
            if got is not None:
                tr.scatter(idx, X, *got)
                parts.foil_hits[idx] += 1
            sl = inj.inject(parts, self.turn, t_turn=self.t_run, turn_time_s=self.t_len,
                            ref=local)
            tr.push_slots(sl)
            return sl is not None
        tr.sync_host()
        inj.foil_pass((parts.coords, parts.ptau, parts.rpp, parts.rvv, parts.state,
                       parts.foil_hits), parts.weight, parts.n_used, self.turn, p0c=p0c,
                      beta0=b0, kinetic_mev=kin)
        sl = inj.inject(parts, self.turn, t_turn=self.t_run, turn_time_s=self.t_len, ref=local)
        tr.push_host()
        return sl is not None


def _pool_at_start(rl, inj, ref) -> np.ndarray:
    """The beam to inject (its pool, at most 20000 particles) at the foil,
    carried from there to the end of the turn — where the ring starts — with
    no apertures (the emittances a ring that starts empty sizes its probes
    with)."""
    from linac_gen.ring.injection import source as S
    from linac_gen.ring.kernels.cpu import backend
    from linac_gen.ring.optics import _energy
    c = inj.cfg
    n = min(inj.pool.n, 20000)
    X = np.ascontiguousarray(S.to_ring(inj.pool, np.arange(n), ring_ref=ref, centroid=(
        c.source.x_m, c.source.px, c.source.y_m, c.source.py)))
    pt, rp, rv = (np.ascontiguousarray(a, dtype=float) for a in _energy(X[5], ref.beta0))
    out = np.empty((0, 6, n))
    backend().track_probe(X, pt, rp, rv, *rl.node_table(), int(inj.node), rl.n_nodes, out, False)
    return np.ascontiguousarray(X[:, np.all(np.isfinite(X), axis=0)])


# ---------------------------------------------------------------------------
# trackers: the particles on the CPU or on a CUDA device
# ---------------------------------------------------------------------------
_PROBE_KEYS = ("X", "ptau", "rpp", "rvv", "state")


def _make_tracker(cfg, rl, parts, zwrap, *, full: bool = False):
    """The beam's tracker; ``full``: every slot of the particles' capacity
    (the injection fills them; the empty ones are not tracked)."""
    if getattr(cfg, "device", "cpu") == "cuda":
        from linac_gen.ring.kernels import cuda as kc
        if not kc.available():
            raise ValueError("device 'cuda': no CUDA GPU is available to numba on this machine "
                             "(the CPU runs the same kernels: device 'cpu')")
        return _CudaTracker(rl, parts, zwrap, full=full)
    return _CpuTracker(cfg, rl, parts, zwrap, full=full)


def _particle_views(parts, full: bool = False):
    n = parts.capacity if full else parts.n_used
    return (parts.coords[:, :n], parts.ptau[:n], parts.rpp[:n], parts.rvv[:n],
            parts.state[:n], parts.lost_turn[:n], parts.lost_node[:n], parts.lost_cause[:n])


class _CpuTracker:
    """The beam on the CPU: numba's multi-core driver (plain Python without numba)."""
    device = "cpu"
    device_foil = False

    def __init__(self, cfg, rl, parts, zwrap, *, full: bool = False):
        from linac_gen.ring.kernels.cpu import PARALLEL_LOCK, backend
        self.be = backend()
        self.lock = PARALLEL_LOCK
        # (with a foil, every slot of the capacity: while the ring fills, the
        # slots used so far are the ones tracked — the rest are empty — so
        # that every thread gets its share of the beam)
        self.full = bool(full)
        self._parts = parts
        self._n_active = int(parts.n_used)
        self.compiled = self.be.compiled
        self.table = rl.node_table()
        self.n_nodes = rl.n_nodes
        self.zwrap = float(zwrap)
        self.threads = 1
        if self.compiled:
            from linac_gen.parallel import threads as _th
            from linac_gen.ring.kernels.cpu import set_threads
            want = getattr(cfg, "threads", None)
            self.threads = set_threads(int(want) if want else _th.get_threads())
        self._arrays = _particle_views(parts, full)
        # (numba wants contiguous arrays: contiguous working copies where a
        # view is not, copied back by sync_host)
        self._work = tuple(np.ascontiguousarray(a) for a in self._arrays)
        self._copied = any(w is not a for w, a in zip(self._work, self._arrays, strict=True))
        n = parts.capacity if full else parts.n_used
        self._weight = np.ascontiguousarray(parts.weight[:n])
        self._weight_src = parts.weight[:n]
        self._sc_beam = None

    def track(self, turn: int, k: int) -> None:
        self._launch(0, self.n_nodes, int(k), int(turn), self.zwrap)

    def track_nodes(self, turn: int, j0: int, j1: int, *, wrap: bool) -> None:
        """Nodes j0..j1-1 of turn ``turn`` (space charge: from kick to kick);
        zeta wrapped after them only with ``wrap`` (the end of the turn)."""
        self._launch(int(j0), int(j1), 1, int(turn), self.zwrap if wrap else 0.0)

    def _launch(self, j0, j1, k, turn, zwrap) -> None:
        fn = self.be.track_turns_parallel if self.compiled else self.be.track_turns
        w = self._work
        if self.full and self._n_active < w[4].shape[0]:
            n = self._n_active
            w = (w[0][:, :n], *(a[:n] for a in w[1:]))
        with self.lock:
            fn(*w, *self.table, j0, j1, k, turn, zwrap)

    def sc_beam(self):
        """The beam as the space-charge kicks see it (its working arrays)."""
        if self._sc_beam is None:
            from linac_gen.ring.sc.beams import CpuBeam
            w = self._work
            self._sc_beam = CpuBeam(w[0], w[1], w[2], w[3], w[4], self._weight)
        return self._sc_beam

    def update_table(self, rl) -> None:
        """(the node table is the ring's own arrays, written in place by the ramp)"""

    def sync_host(self) -> None:
        if self._copied:
            for w, a in zip(self._work, self._arrays, strict=True):
                a[...] = w

    def push_host(self) -> None:
        """The particles as the host left them (the foil, an injection) into
        the working arrays (the same arrays unless they are copies)."""
        self._n_active = int(self._parts.n_used)
        if self._copied:
            for w, a in zip(self._work, self._arrays, strict=True):
                w[...] = a
        if self._weight is not self._weight_src:
            self._weight[...] = self._weight_src

    def close(self) -> None:
        pass


class _CudaTracker:
    """The beam on a CUDA GPU: it stays on the device between turns; the
    host gets it back for statistics, snapshots and checkpoints.  The turns
    are launched in short, synchronized launches (kernels.cuda.Launcher), so
    the time a launch of turns takes is the GPU's.  With a foil, its step
    finds the particles inside the foil on the device: only those, and the
    slots an injection fills, cross to the host and back."""
    device = "cuda"
    device_foil = True

    def __init__(self, rl, parts, zwrap, *, full: bool = False):
        from linac_gen.ring.kernels import cuda as kc
        kc.kernel()
        self.kc = kc
        self.cuda = kc.cuda
        self.compiled = True
        self.threads = kc.THREADS_PER_BLOCK
        self._parts = parts
        self._arrays = _particle_views(parts, full)
        n = parts.capacity if full else parts.n_used
        self._dev = [self.cuda.to_device(np.ascontiguousarray(a)) for a in self._arrays]
        self._table = [self.cuda.to_device(np.ascontiguousarray(a)) for a in rl.node_table()]
        # (with a foil, every slot of the capacity is on the device; the ones
        # used so far are launched — the rest are empty)
        self.full = bool(full)
        # (the foil step on the device: a tracker of every slot only)
        self.device_foil = self.full
        self.n_active = int(parts.n_used) if full else int(n)
        self.launcher = kc.Launcher(self._dev, self._table, self.n_active, 0, rl.n_nodes,
                                    float(zwrap))
        self.n = int(n)
        self.zwrap = float(zwrap)
        self._weight_src = parts.weight[:n]
        self._weight = np.ascontiguousarray(self._weight_src)
        self._segments = {}
        self._sc_beam = None
        self._mask = None

    def track(self, turn: int, k: int) -> None:
        self.launcher.track(int(turn), int(k))

    def track_nodes(self, turn: int, j0: int, j1: int, *, wrap: bool) -> None:
        """Nodes j0..j1-1 of turn ``turn`` (space charge: from kick to kick),
        each stretch with its own launcher (its own launch sizes); zeta
        wrapped after them only with ``wrap`` (the end of the turn)."""
        from linac_gen.ring.kernels import cuda as kc
        key = (int(j0), int(j1), bool(wrap))
        la = self._segments.get(key)
        if la is None:
            la = self._segments[key] = kc.Launcher(self._dev, self._table, self.n_active, j0,
                                                   j1, self.zwrap if wrap else 0.0)
        la.track(int(turn), 1)

    def _set_active(self, n: int) -> None:
        """The slots launched from now on (with a foil: the ones used)."""
        if self.full:
            self.n_active = int(n)
            self.launcher.n = self.n_active
            for la in self._segments.values():
                la.n = self.n_active

    def foil_slots(self, n_used: int, bounds) -> np.ndarray:
        """The circulating particles inside the foil's rectangle ``bounds``,
        found on the device: their slots, ascending."""
        if self._mask is None:
            self._mask = self.cuda.device_array(max(1, self.n), dtype=np.uint8)
        return self.kc.foil_slots(self._dev, int(n_used), bounds, self._mask)

    def gather(self, idx: np.ndarray) -> np.ndarray:
        """The coordinates of slots ``idx`` from the device (6, k)."""
        return self.kc.gather(self._dev, idx)

    def scatter(self, idx: np.ndarray, X, pt, rp, rv) -> None:
        """Slots ``idx`` back to the device: their coordinates and energy terms."""
        self.kc.scatter(self._dev, idx, X, pt, rp, rv)

    def push_slots(self, sl) -> None:
        """The slots an injection filled (``sl``; None: none) to the device,
        and the weights to the space-charge kicks."""
        if sl is not None and sl.stop > sl.start:
            self.kc.put_slots(self._dev, self._arrays, int(sl.start), int(sl.stop))
            if self._weight is not self._weight_src:
                self._weight[sl] = self._weight_src[sl]
            if self._sc_beam is not None:
                self._sc_beam.weight[sl.start:sl.stop].copy_to_device(
                    np.ascontiguousarray(self._weight[sl]))
        self._set_active(self._parts.n_used)

    def sc_beam(self):
        if self._sc_beam is None:
            from linac_gen.ring.sc.beams import CudaBeam
            self._sc_beam = CudaBeam(self._dev[:5], self._weight)
        return self._sc_beam

    def update_table(self, rl) -> None:
        """The ramp's turn: the parameters and the reference copied to the
        device (in place: the launcher keeps the same arrays)."""
        table = rl.node_table()
        for i in (2, 3, 4, 5):
            self._table[i].copy_to_device(np.ascontiguousarray(table[i]))

    def sync_host(self) -> None:
        """The particles back to the host — with a foil, the slots used so far
        (the device never changes an empty slot: every kernel skips them)."""
        n = int(self._parts.n_used) if self.full else self.n
        for d, a in zip(self._dev, self._arrays, strict=True):
            if n == a.shape[-1]:
                a[...] = d.copy_to_host()
            elif n == 0:
                continue
            elif a.ndim == 2:
                for r in range(a.shape[0]):
                    a[r, :n] = d[r, :n].copy_to_host()
            else:
                a[:n] = d[:n].copy_to_host()

    def push_host(self) -> None:
        """The particles as the host left them (the foil, an injection) back to
        the device (in place: the launchers keep the same arrays), and the
        weights to the space-charge kicks."""
        for d, a in zip(self._dev, self._arrays, strict=True):
            d.copy_to_device(np.ascontiguousarray(a))
        self._weight = np.ascontiguousarray(self._weight_src)
        if self._sc_beam is not None:
            self._sc_beam.weight.copy_to_device(self._weight)
        self._set_active(self._parts.n_used)

    def close(self) -> None:
        self._dev = []
        self._table = []
        self._segments = {}
        self._sc_beam = None
        self._mask = None


class _ProbeTracker:
    """The tune probes: a few particles, turn by turn on the CPU (serial),
    whatever the beam's device."""

    def __init__(self, rl, probe, zwrap):
        from linac_gen.ring.kernels.cpu import backend
        self.be = backend()
        self.table = rl.node_table()
        self.n_nodes = rl.n_nodes
        self.zwrap = float(zwrap)
        k = probe["X"].shape[1]
        self._a = [np.ascontiguousarray(probe[key]).copy() for key in _PROBE_KEYS] + \
            [np.full(k, -1, np.int32), np.full(k, -1, np.int32), np.zeros(k, np.int8)]

    def track(self, turn: int, k: int) -> np.ndarray:
        """k turns → (n_probe, k, 6) coordinates after each (NaN once lost)."""
        n = self._a[0].shape[1]
        tbt = np.full((int(k), 6, n), np.nan)
        self.be.track_turns_tbt(*self._a, *self.table, int(k), int(turn), self.zwrap, tbt)
        return np.transpose(tbt, (2, 0, 1))

    def advance(self, turn: int, k: int) -> None:
        """k turns, nothing kept (past the record)."""
        self.be.track_turns(*self._a, *self.table, 0, self.n_nodes, int(k), int(turn),
                            self.zwrap)

    def track_nodes(self, turn: int, j0: int, j1: int, *, wrap: bool) -> None:
        """Nodes j0..j1-1 of turn ``turn`` (space charge: from kick to kick);
        zeta wrapped after them only with ``wrap`` (the end of the turn)."""
        self.be.track_turns(*self._a, *self.table, int(j0), int(j1), 1, int(turn),
                            self.zwrap if wrap else 0.0)

    def kick_arrays(self) -> tuple:
        """(coords, ptau, rpp, rvv, state) — the probes' own arrays, for the
        space-charge kicks."""
        return tuple(self._a[:5])

    def coords_nan(self) -> np.ndarray:
        """(n_probe, 6): the probes' coordinates, NaN once lost."""
        X = self._a[0].T.copy()
        X[self._a[4] != 1] = np.nan
        return X

    def arrays(self) -> dict:
        return {key: a.copy() for key, a in zip(_PROBE_KEYS, self._a[:5], strict=True)}

    def state(self) -> np.ndarray:
        return self._a[4].copy()


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
_PARTICLE_FIELDS = ("coords", "ptau", "rpp", "rvv", "weight", "state", "pid", "inj_turn",
                    "foil_hits", "lost_turn", "lost_node", "lost_s", "lost_cause")


def save_checkpoint(path, *, parts, turn, stats, snapshots, probe, probe_tbt, probe_amps, eps0,
                    digest, config, ramp=None, sc_frozen=None, injection=None,
                    sc_bins=None, sc_diag=None) -> None:
    """Everything a run needs to continue bit for bit (written to a temporary
    file, then renamed: a checkpoint is never half written)."""
    path = Path(path)
    arrays = {f"p_{k}": getattr(parts, k) for k in _PARTICLE_FIELDS}
    arrays.update({
        "s_turns": np.asarray(stats["turns"], dtype=np.int64),
        "s_n_alive": np.asarray(stats["n_alive"], dtype=np.int64),
        "s_mean": np.asarray(stats["mean"], dtype=float).reshape(-1, 6),
        "s_cov": np.asarray(stats["cov"], dtype=float).reshape(-1, 6, 6),
        "s_ref_bg": np.asarray(stats.get("ref_bg", []), dtype=float),
        "s_ref_p0c": np.asarray(stats.get("ref_p0c", []), dtype=float),
        "s_zeta_core": np.asarray(stats.get("zeta_core", []), dtype=float),
        "s_line_filter_m": np.asarray(stats.get("line_filter_m", []), dtype=float),
        "eps0": np.asarray(eps0, dtype=float)})
    snap_refs = {}
    for t, snap in snapshots.items():
        arrays[f"snap_{int(t)}_coords"] = snap["coords"]
        arrays[f"snap_{int(t)}_pid"] = snap["pid"]
        if "reference" in snap:
            snap_refs[str(int(t))] = snap["reference"]
    if ramp is not None:
        for k, v in ramp["record"].items():
            arrays[f"ramp_{k}"] = np.asarray(v)
    if sc_frozen is not None:
        arrays["scf_gauss"] = np.asarray(sc_frozen["gauss"], dtype=float)
        arrays["scf_line"] = np.asarray(sc_frozen["line"], dtype=float)
    if injection is not None:
        arrays.update({k: np.asarray(v) for k, v in injection[0].items()})
    if probe is not None:
        arrays.update({f"probe_{k}": probe[k] for k in _PROBE_KEYS})
        arrays.update({"probe_tbt": probe_tbt, "probe_amps": probe_amps})
    meta = {"version": CHECKPOINT_VERSION, "turn": int(turn), "digest": digest,
            "n_used": int(parts.n_used), "next_pid": int(parts.next_pid), "config": config,
            "engine": ENGINE, "snapshot_references": snap_refs,
            "ramp": None if ramp is None else {"t_ref": float(ramp["t_ref"]).hex(),
                                               "e_ref": float(ramp["e_ref"]).hex(),
                                               "n_capped": int(ramp["n_capped"])},
            "injection": None if injection is None else injection[1],
            "sc_bins": sc_bins, "sc_diagnostics": sc_diag}
    arrays["meta_json"] = np.array(json.dumps(meta))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def checkpoint_config(path) -> dict:
    """The ring settings a checkpoint's run had (read without its particles)."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
    return dict(meta.get("config") or {})


def load_checkpoint(path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        if meta.get("version") not in CHECKPOINT_READABLE:
            raise ValueError(f"checkpoint {path}: format {meta.get('version')} is not one this "
                             f"HELIX reads ({list(CHECKPOINT_READABLE)})")
        cap = d["p_coords"].shape[1]
        parts = P.RingParticles.allocate(cap)
        for k in _PARTICLE_FIELDS:
            getattr(parts, k)[...] = d[f"p_{k}"]
        parts.n_used = int(meta["n_used"])
        parts.next_pid = int(meta["next_pid"])
        stats = {"turns": [int(t) for t in d["s_turns"]],
                 "n_alive": [int(v) for v in d["s_n_alive"]],
                 "mean": list(d["s_mean"]), "cov": list(d["s_cov"]),
                 "ref_bg": [float(v) for v in d["s_ref_bg"]] if "s_ref_bg" in d.files else [],
                 "ref_p0c": [float(v) for v in d["s_ref_p0c"]] if "s_ref_p0c" in d.files else []}
        # (a checkpoint written before the core widths were recorded: none for its rows)
        stats["zeta_core"] = ([float(v) for v in d["s_zeta_core"]] if "s_zeta_core" in d.files
                              else [math.nan] * len(stats["turns"]))
        stats["line_filter_m"] = ([float(v) for v in d["s_line_filter_m"]]
                                  if "s_line_filter_m" in d.files
                                  else [math.nan] * len(stats["turns"]))
        snaps = {}
        snap_refs = meta.get("snapshot_references") or {}
        for k in d.files:
            if k.startswith("snap_") and k.endswith("_coords"):
                t = int(k.split("_")[1])
                snaps[t] = {"coords": d[k].copy(), "pid": d[f"snap_{t}_pid"].copy()}
                if str(t) in snap_refs:
                    snaps[t]["reference"] = dict(snap_refs[str(t)])
        ramp = None
        if meta.get("ramp"):
            r = meta["ramp"]
            ramp = {"t_ref": float.fromhex(r["t_ref"]), "e_ref": float.fromhex(r["e_ref"]),
                    "n_capped": int(r["n_capped"]),
                    "record": {k[5:]: d[k].copy() for k in d.files if k.startswith("ramp_")}}
        has_probe = "probe_X" in d.files
        sc_frozen = ({"gauss": d["scf_gauss"].copy(), "line": d["scf_line"].copy()}
                     if "scf_gauss" in d.files else None)
        injection = {k: d[k].copy() for k in d.files if k.startswith("inj_")}
        return {"meta": meta, "turn": int(meta["turn"]), "digest": meta["digest"],
                "sc_frozen": sc_frozen, "injection": injection,
                "particles": parts, "stats": stats, "snapshots": snaps,
                "probe": ({k: d[f"probe_{k}"].copy() for k in _PROBE_KEYS} if has_probe
                          else None),
                "probe_tbt": d["probe_tbt"].copy() if has_probe else None,
                "probe_amps": d["probe_amps"].copy() if has_probe else None,
                "eps0": d["eps0"].copy(), "ramp": ramp}
