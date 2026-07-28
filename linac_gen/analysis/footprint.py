"""Frozen-SC incoherent tune footprint.

Per-particle tunes of a (quasi-)periodic cell, measured by tracking a
test distribution repeatedly through the cell with the REAL nonlinear
element transport while the space-charge field is FROZEN — recorded on
the first pass and replayed identically on every later turn.

Model honesty
-------------
The frozen field is the 2-D Gaussian-equivalent transverse SC field
(round-beam analytic / Bassetti-Erskine — the same formula as
``pic_solver.kick_continuous_2d_gauss``), with σ_x, σ_y and the line
density λ taped at every kick of a reference pass of the matched
Gaussian FIELD beam.  λ is the DC density I/(βc) for a continuous
beam, and the bunch-CENTER slice density Q/(√2π·σ_z(s)) (Q = I/f_bunch)
for a bunched beam — the test ladder is launched at φ = ΔW = 0, so the
center-slice field is the field it actually rides in; the DC average
would be too weak by the bunching factor.  An upstream
``SPACE_CHARGE_COMP`` factor scales the kick, mirroring the matched
solve.  The Gaussian field is nonlinear (core particles see the full
gradient, tail particles a weaker one), which is what produces a REAL
amplitude-dependent tune spread — a linear frozen map would give every
particle identical tunes and no footprint at all.  It is NOT a
self-consistent PIC footprint (future work), and the longitudinal
DYNAMICS of the test particles are not modelled (only the field-beam
σ_φ(s) enters, through λ).  Accelerating cells are handled in the
frozen-energy approximation: the reference is reset to the cell
entrance every turn (valid for ΔW/W ≪ 1 per cell).

Tune extraction: windowed FFT of the Courant-Snyder-normalized complex
coordinate with parabolic peak interpolation (NAFF-lite, adequate to
~1e-4 at 256 turns).
"""
from __future__ import annotations

import math

import numpy as np

from linac_gen.analysis.period_detect import PeriodicStructure

__all__ = ["tune_footprint", "FrozenGaussianKicker"]


class FrozenGaussianKicker:
    """``pic_solver``-compatible kicker: 2-D Gaussian-equivalent SC.

    ``mode="record"``: σ_x, σ_y and the line density λ are measured from
    the live beam at every kick and taped (the recording beam is the
    matched Gaussian FIELD beam); ``mode="replay"``: the taped values are
    reused in order — the field a test particle sees is then independent
    of the test distribution (frozen).

    Line-density convention (must stay strength-consistent with the
    matched Σ the ladder is launched from):

    * DC / continuous beam — λ = I/(βc), the coasting-beam density (the
      matched envelope used the 2-D DC kernel with the same λ);
    * bunched beam — λ = Q/(√2π·σ_z(s)) with Q = I/f_bunch: the
      bunch-CENTER slice density of the Gaussian bunch, taped per kick
      from the field beam's live σ_φ.  The test ladder is launched at
      φ = ΔW = 0 (it rides the bunch center), so the center-slice field
      is exactly the field those particles see.  The DC average I/(βc)
      would be too weak by the bunching factor (~5-20×) and the ladder
      would not be matched to its own frozen field.
    * an upstream ``SPACE_CHARGE_COMP`` factor scales the kick, mirroring
      the envelope walk that produced Σ.
    """

    # The Tracker calls .kick(beam, ds_mm) on the bunched-beam path; DC
    # routing is keyed off ``beam.continuous`` (Beam default False), NOT
    # off this attribute — ``config`` is only probed for grid parameters
    # by the real PIC solver.  Beams built here must keep
    # ``continuous=False`` so the Tracker never bypasses the tape and
    # calls a live DC kernel on the sparse ladder.
    config = None

    def __init__(self, *, continuous: bool = True,
                 f_bunch_MHz: "float | None" = None,
                 sc_factor: float = 1.0):
        self.tape: list = []
        self.mode = "record"
        self.continuous = bool(continuous)
        self.f_bunch_MHz = f_bunch_MHz
        self.sc_factor = float(sc_factor)
        self._i = 0

    def reset(self, mode: str) -> None:
        self.mode = mode
        self._i = 0

    def _line_density(self, beam, alive, beta: float, v_m_s: float) -> float:
        """λ [C/m] seen by the bunch-center test particles (see class
        docstring).  Falls back to the DC density when the longitudinal
        information is missing (σ_φ = 0, e.g. an emit_z = 0 input)."""
        from linac_gen.core.constants import C_LIGHT

        lam_dc = abs(beam.current) * 1e-3 / v_m_s
        if self.continuous or not self.f_bunch_MHz:
            return lam_dc
        sig_phi = float(np.std(beam.particles[alive, 4]))    # deg
        f_local = float(beam.ref.frequency)                  # σ_φ is in
        if sig_phi <= 0 or f_local <= 0:                     # LOCAL deg
            return lam_dc
        wavelength_m = C_LIGHT / (f_local * 1e6)
        sigma_z_m = sig_phi * beta * wavelength_m / 360.0
        Q = abs(beam.current) * 1e-3 / (self.f_bunch_MHz * 1e6)
        return Q / (math.sqrt(2.0 * math.pi) * sigma_z_m)

    def kick(self, beam, ds_mm: float) -> None:
        from linac_gen.core.constants import C_LIGHT
        from linac_gen.pic.pic_solver import _gauss_field_2d, E_CHARGE

        if beam.current == 0 or ds_mm <= 0:
            return
        # Retire runaway particles before they overflow: a large-amplitude
        # test particle on a resonance of the FROZEN nonlinear field can
        # grow exponentially (frozen ≠ self-consistent), and the aperture
        # may not catch it between element boundaries.  Non-finite or
        # > 1 m coordinates are unphysical here → mark lost so they neither
        # get further kicks nor poison the diagnostics covariance.
        alive = beam.alive_mask
        runaway = alive & (
            ~np.isfinite(beam.particles[:, 0])
            | ~np.isfinite(beam.particles[:, 2])
            | (np.abs(beam.particles[:, 0]) > 1.0e3)
            | (np.abs(beam.particles[:, 2]) > 1.0e3)
        )
        if runaway.any():
            beam.lost[runaway] = True
            alive = beam.alive_mask
        alive_idx = np.where(alive)[0]
        if alive_idx.size == 0:
            return
        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]

        ref = beam.ref
        beta = float(ref.beta)
        gamma = float(ref.gamma)
        v_m_s = beta * C_LIGHT
        if v_m_s <= 0:
            return

        if self.mode == "record":
            # The "live beam" here is the matched Gaussian FIELD beam
            # (drawn from Σ) — never the sparse probe ladder, whose std
            # depends on the amplitude ladder and would give an arbitrary
            # field strength.
            if alive_idx.size < 2:
                self.tape.append((0.0, 0.0, 0.0))
                return
            sx_m = float(np.std(xs)) * 1e-3
            sy_m = float(np.std(ys)) * 1e-3
            lam = self._line_density(beam, alive, beta, v_m_s)
            self.tape.append((sx_m, sy_m, lam))
        else:
            if self._i >= len(self.tape):
                return
            sx_m, sy_m, lam = self.tape[self._i]
            self._i += 1
        if sx_m <= 0 or sy_m <= 0 or lam <= 0:
            return

        mass_MeV = float(ref.species.mass)
        q_abs = abs(ref.species.charge) * E_CHARGE
        Ex, Ey = _gauss_field_2d(xs * 1e-3, ys * 1e-3, sx_m, sy_m, lam)
        mc2_J = mass_MeV * 1e6 * E_CHARGE
        pre = (self.sc_factor * q_abs * (ds_mm * 1e-3)
               / (beta * beta * gamma * mc2_J))
        beam.particles[alive_idx, 1] += (pre * Ex) * 1e3
        beam.particles[alive_idx, 3] += (pre * Ey) * 1e3


class _NullRecorder:
    """Minimal recorder satisfying the Tracker interface with no work.

    The footprint reads ``beam.particles`` directly after each turn, so no
    diagnostics are needed.  Skipping the real recorder both removes the
    per-element normal-mode-emittance computation (which raises on a
    non-finite covariance — a NaN test particle whose ``NaN > aperture``
    check never retired it) and makes the tracking substantially faster.
    """

    def __init__(self):
        self.s: list = []
        self.element_exit_idx: list = []

    def record(self, beam, s, element_name=None):  # noqa: D401
        pass

    def save_snapshot(self, beam, s):
        pass


def _sample_from_sigma(Sigma, n: int, rng, ndim: int = 4) -> np.ndarray:
    """Draw ``n`` samples from the leading ``ndim``×``ndim`` block of
    ``Sigma`` and rescale so the sample covariance equals it exactly.

    ``ndim=4`` samples the transverse block only — the longitudinal
    block is degenerate for a DC/continuous matched beam and would fail
    the Cholesky.  ``ndim=6`` includes (φ, ΔW), needed by the bunched
    field beam so the kicker can measure σ_φ(s) for the center-slice
    line density.  The exact rescaling removes sampling noise so the
    taped σ(s) is deterministic given ``rng``."""
    S = np.asarray(Sigma, dtype=float)[:ndim, :ndim]
    S = 0.5 * (S + S.T)
    jit = 1e-12 * max(np.trace(S) / ndim, 1e-30)
    try:
        L = np.linalg.cholesky(S + jit * np.eye(ndim))
    except np.linalg.LinAlgError:
        # Fully degenerate Σ block — fall back to independent diagonal
        # sampling (zero-variance planes stay zero).
        L = np.diag(np.sqrt(np.clip(np.diag(S), 0.0, None)))
    samp = rng.standard_normal((n, ndim)) @ L.T
    # Rescale to match S exactly (as the Gaussian generator does).
    Ss = (samp.T @ samp) / n
    try:
        A = L @ np.linalg.inv(np.linalg.cholesky(Ss + jit * np.eye(ndim)))
        samp = samp @ A.T
    except np.linalg.LinAlgError:
        pass
    return samp


def _fft_tune(u: np.ndarray) -> float:
    """Fractional tune of a complex quasi-periodic signal u_n ~ e^{-2πiQn}.

    Hann-windowed FFT + parabolic interpolation of the log-magnitude
    peak (NAFF-lite).  Returns Q in [0, 1)."""
    n = u.size
    if n < 8 or not np.all(np.isfinite(u)):
        return float("nan")
    w = np.hanning(n)
    U = np.fft.fft((u - u.mean()) * w)
    mag = np.abs(U)
    k = int(np.argmax(mag))
    if mag[k] <= 0:
        return float("nan")
    # Parabolic interpolation on log-magnitude (guard the edges).
    km = (k - 1) % n
    kp = (k + 1) % n
    a, b, c = (math.log(max(mag[km], 1e-300)),
               math.log(max(mag[k], 1e-300)),
               math.log(max(mag[kp], 1e-300)))
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-300 else 0.0
    delta = max(-0.5, min(0.5, delta))
    q = ((k + delta) / n) % 1.0
    # The rotation sense of the normalized coordinate depends on sign
    # conventions; the physical (folded, < 180°/cell) tune is the
    # principal alias.
    return float(min(q, 1.0 - q))


def tune_footprint(lattice, ref, period: PeriodicStructure,
                   base_initial: dict, current: float, *,
                   n_turns: int = 256, n_particles: int = 200,
                   amp_max_sigma: float = 3.0, seed: int = 0,
                   should_stop=None, progress=None) -> dict:
    """Frozen-SC per-particle tune footprint of the period's first cell.

    Particles are launched on a radial ladder in normalized amplitude
    (0..``amp_max_sigma`` σ, alternating x/y/diagonal rays) from the
    SC-matched Σ of :func:`matching.periodic.find_matched_period_sigma`,
    then tracked ``n_turns`` times through the cell with frozen
    Gaussian-equivalent SC (see module docstring).

    ``progress`` — optional callable ``progress(turn_done, n_turns)``,
    invoked once per completed turn (turn 0 = matched solve + record
    pass finished).  Cost scales as n_turns × n_particles × (element
    transport); field-map cells (RK4) are ~two orders costlier per
    turn than hard-edge cells — budget accordingly.

    Returns
    -------
    dict
        ``qx``, ``qy`` — per-particle tunes (fraction of a cell, i.e.
        μ/360; NaN for lost particles and for identically-zero signals
        such as the x-tune of a pure y-ray); ``ax_sigma``, ``ay_sigma``
        — launch amplitudes in units of σ; ``qx_core``, ``qy_core`` —
        small-amplitude (core) tunes (median below 0.3 σ);
        ``mu_x_core_deg``, ``mu_y_core_deg`` — the same in degrees;
        ``mu_{x,y}_spread_pp_deg`` / ``mu_{x,y}_spread_rms_deg`` — the
        footprint width (peak-to-peak / rms over finite tunes);
        ``n_turns``, ``n_particles``, ``model`` (label string),
        ``matched_state`` (fixed-point metadata).
    """
    from linac_gen.core.beam import Beam
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.lattice_commands import LatticeCommand
    from linac_gen.matching.periodic import find_matched_period_sigma
    from linac_gen.tracking.tracker import Tracker

    ms = find_matched_period_sigma(lattice, ref, period, current,
                                   base_initial)
    Sigma = np.asarray(ms["sigma_entry"], dtype=float)
    ref_entry = ms["ref_entry"]

    a0, b0 = period.spans()[0]
    cell = Lattice()
    for e in lattice.elements[a0:b0]:
        cell.add(e)
    step_cfg = getattr(lattice, "step_config", None)
    if step_cfg is not None:
        cell.step_config = step_cfg
    prefix_cmds = [e for e in lattice.elements[:a0]
                   if isinstance(e, LatticeCommand)]

    # Matched projected Twiss for launching and normalization.
    def _twiss(i, j):
        eps = math.sqrt(max(Sigma[i, i] * Sigma[j, j]
                            - Sigma[i, j] ** 2, 1e-30))
        return (-Sigma[i, j] / eps, Sigma[i, i] / eps, eps)

    ax_t, bx_t, ex_t = _twiss(0, 1)
    ay_t, by_t, ey_t = _twiss(2, 3)
    sx = math.sqrt(Sigma[0, 0])
    sy = math.sqrt(Sigma[2, 2])

    # Radial amplitude ladder: rays along x, y, and the diagonal.
    n_rays = 3
    per_ray = max(2, n_particles // n_rays)
    amps = np.linspace(0.05, amp_max_sigma, per_ray)
    launch = []
    for amp in amps:
        launch.append((amp * sx, 0.0))          # x-ray
        launch.append((0.0, amp * sy))          # y-ray
        launch.append((amp * sx / math.sqrt(2),
                       amp * sy / math.sqrt(2)))  # diagonal
    launch = launch[:max(n_particles, 3)]
    n_p = len(launch)

    # Strength consistency with the matched Σ: DC/bunched state, bunch
    # repetition frequency, and the upstream SPACE_CHARGE_COMP factor all
    # come from the SAME prefix walk that produced Σ (they are returned
    # by find_matched_period_sigma precisely for replay paths like this).
    continuous = bool(ms.get("continuous", False))
    kicker = FrozenGaussianKicker(
        continuous=continuous,
        f_bunch_MHz=ms.get("bunch_frequency", None),
        sc_factor=float(ms.get("sc_factor", 1.0)),
    )
    entry_state = ref_entry.copy()

    # --- Record pass: tape the matched-beam SC field σ(s) from a Gaussian
    #     FIELD beam drawn from the matched Σ.  This is what makes the
    #     frozen field physical: the σ(s) profile the test particles see
    #     is the matched beam's own breathing field, independent of the
    #     sparse probe ladder (whose std would depend on amp_max_sigma). ---
    if current != 0:
        rng = np.random.default_rng(seed)
        n_field = 1000
        # Bunched beams need the (φ, ΔW) block too — the kicker measures
        # σ_φ(s) for the bunch-center slice density.  Degenerate z blocks
        # (emit_z = 0) sample as zeros → the kicker falls back to the DC
        # density.
        ndim = 6 if (not continuous and Sigma[4, 4] > 0) else 4
        field = _sample_from_sigma(Sigma, n_field, rng, ndim=ndim)
        fbeam = Beam(entry_state.copy(), n_particles=n_field, current=current)
        fbeam.particles[:, :ndim] = field
        kicker.reset("record")
        ftr = Tracker(cell, fbeam, pic_solver=kicker, recorder=_NullRecorder())
        for cmd in prefix_cmds:
            try:
                cmd.apply_command(ftr.track_state)
            except Exception:                                # noqa: BLE001
                pass
        ftr.run()

    beam = Beam(ref_entry.copy(), n_particles=n_p, current=current)
    for i, (x0, y0) in enumerate(launch):
        beam.particles[i, 0] = x0
        beam.particles[i, 2] = y0

    xs = np.empty((n_turns, n_p))
    xps = np.empty((n_turns, n_p))
    ys = np.empty((n_turns, n_p))
    yps = np.empty((n_turns, n_p))

    if progress is not None:
        progress(0, n_turns)          # matched solve + record pass done

    for turn in range(n_turns):
        if should_stop is not None and should_stop():
            from linac_gen.core.cancelled import OperationCancelled
            raise OperationCancelled("tune footprint cancelled")
        # Always replay the taped matched-beam field (frozen).  At I = 0
        # the tape is empty and every kick is a no-op.
        kicker.reset("replay")
        # Frozen-energy approximation: reset the reference (the beam's
        # coordinates persist turn to turn).
        beam.ref = entry_state.copy()
        tracker = Tracker(cell, beam, pic_solver=kicker,
                          recorder=_NullRecorder())
        for cmd in prefix_cmds:
            try:
                cmd.apply_command(tracker.track_state)
            except Exception:                            # noqa: BLE001
                pass
        tracker.run()
        xs[turn] = beam.particles[:, 0]
        xps[turn] = beam.particles[:, 1]
        ys[turn] = beam.particles[:, 2]
        yps[turn] = beam.particles[:, 3]
        if progress is not None:
            progress(turn + 1, n_turns)

    # CS-normalized complex coordinates → per-particle tunes.
    def _tunes(pos, ang, alpha, beta_t):
        u = (pos / math.sqrt(beta_t)
             - 1j * (alpha * pos + beta_t * ang) / math.sqrt(beta_t))
        return np.array([_fft_tune(u[:, i]) for i in range(u.shape[1])])

    qx = _tunes(xs, xps, ax_t, bx_t)
    qy = _tunes(ys, yps, ay_t, by_t)

    # Particles that became lost (aperture or runaway) never completed a
    # clean quasi-periodic signal — their tune is meaningless.
    lost = np.asarray(beam.lost, dtype=bool)
    qx[lost] = float("nan")
    qy[lost] = float("nan")

    amp_x_sigma = np.array([x0 / sx if sx > 0 else 0.0
                            for (x0, _y0) in launch])
    amp_y_sigma = np.array([y0 / sy if sy > 0 else 0.0
                            for (_x0, y0) in launch])

    # Core tune = the small-amplitude tune, taken as the median of the
    # ray particles below a FIXED amplitude threshold (0.3 σ) so it does
    # not depend on amp_max_sigma.  This is the most strongly depressed
    # end of the footprint (the Gaussian core sees ~2× the rms-equivalent
    # gradient, so the core tune sits well BELOW the rms channel tune —
    # that ordering, not a core≈centroid identity, is the physics here).
    # Near strong depression the absolute value approaches the FFT
    # resolution floor (~360/n_turns per cell) and is only approximate.
    _CORE_AMP = 0.3

    def _core(q, amp_plane, ray_mask):
        near = ray_mask & np.isfinite(q) & (amp_plane > 0) & (amp_plane <= _CORE_AMP)
        if near.any():
            return float(np.median(q[near]))
        fin = ray_mask & np.isfinite(q) & (amp_plane > 0)
        if not fin.any():
            return float("nan")
        return float(q[fin][np.argmin(amp_plane[fin])])

    x_ray = (amp_x_sigma > 0) & (amp_y_sigma == 0)
    y_ray = (amp_y_sigma > 0) & (amp_x_sigma == 0)
    qx_core = _core(qx, amp_x_sigma, x_ray)
    qy_core = _core(qy, amp_y_sigma, y_ray)

    # Tune SPREAD (the primary footprint quantity): peak-to-peak and rms
    # over all finite-tune particles, per plane, in degrees/cell.
    def _spread(q):
        fin = np.isfinite(q)
        if int(fin.sum()) < 2:
            return 0.0, 0.0
        qd = q[fin] * 360.0
        return float(qd.max() - qd.min()), float(np.std(qd))

    dx_pp, dx_rms = _spread(qx)
    dy_pp, dy_rms = _spread(qy)

    return {
        "qx": qx, "qy": qy,
        "ax_sigma": amp_x_sigma, "ay_sigma": amp_y_sigma,
        "qx_core": qx_core, "qy_core": qy_core,
        "mu_x_core_deg": qx_core * 360.0,
        "mu_y_core_deg": qy_core * 360.0,
        "mu_x_spread_pp_deg": dx_pp, "mu_y_spread_pp_deg": dy_pp,
        "mu_x_spread_rms_deg": dx_rms, "mu_y_spread_rms_deg": dy_rms,
        "n_turns": n_turns, "n_particles": n_p,
        "model": ("frozen 2-D Gaussian-equivalent SC (transverse, "
                  + ("DC λ=I/βc" if continuous
                     else "bunch-center slice λ=Q/√2π·σ_z")
                  + "); frozen-energy per turn"),
        "matched_state": {
            "converged": ms["converged"], "residual": ms["residual"],
        },
    }
