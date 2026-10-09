# linac_gen/errors/ensemble.py
"""Envelope ensembles for transfer-line error studies.

:class:`~linac_gen.errors.error_model.ErrorStudy.run` tracks particles and
loops seeds serially — right for losses, far too slow for the thousands of
seeds a tolerance / sorting question needs on a 300 m line.  This module
runs the same seeds (same draws: ``ErrorStudy._apply_errors`` /
``_apply_beam_errors``) through the envelope solver in a process pool:

* :func:`design_reference` — the error-free run every seed is compared
  with (betatron β at every record, end Twiss, lattice dispersion);
* :func:`seed_metrics` — per-seed numbers at the observation point (the
  last record, e.g. the stripping foil) and along the line;
* :func:`run_ensemble` — draw → optional :class:`OrmCorrector` correction
  (per-seed BPM offsets and noise) → envelope run → metrics;
* :func:`run_jitter` — pulse-to-pulse jitter on ONE frozen, corrected
  seed, every pulse drawing only the jitter definitions from an
  independent stream, not re-corrected; orbits are reported RELATIVE to
  the frozen seed's orbit (the jitter alone).

Definitions.  β and α are BETATRON values: the Σ correlations with ΔW are
removed first (Schur complement, as in the phase-advance BMAG), so neither
the arcs' dispersive beam size nor space-charge x–ΔW correlation reads as
β-beat.  Dispersion is the LATTICE dispersion function of each errored
lattice (:func:`linac_gen.analysis.dispersion.dispersion_along_s`, no
input dispersion), current-independent; the statistical ⟨xδ⟩/⟨δ²⟩ of the
Σ is not used because at the BTL's 4.8 mA space charge adds an x–ΔW
correlation of its own.

Losses need particles: verify chosen seeds with the multi-particle tracker
(``ErrorStudy.draw_overrides(seed).lattice`` is the UNCORRECTED copy;
re-apply the seed's kicks, kept by ``run_ensemble(keep_kicks=True)``).
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")


# --------------------------------------------------------------- helpers
def _twiss_betatron(S, i):
    """(α, β[m], ε[mm·mrad]) of rows (i, i+1) after removing the ΔW part."""
    s55 = S[5, 5]
    if s55 > 0.0:
        a = S[i, i] - S[i, 5] ** 2 / s55
        b = S[i, i + 1] - S[i, 5] * S[i + 1, 5] / s55
        c = S[i + 1, i + 1] - S[i + 1, 5] ** 2 / s55
    else:
        a, b, c = S[i, i], S[i, i + 1], S[i + 1, i + 1]
    e2 = a * c - b * b
    if not e2 > 0.0:
        return np.nan, np.nan, 0.0
    e = np.sqrt(e2)
    return -b / e, a / e, e


def _lattice_dispersion_end(lattice, beam_config):
    """(Dx, Dx', Dy, Dy') [m, rad, δ-normalised] of the lattice function at
    the end (no input dispersion)."""
    from linac_gen.analysis.dispersion import dispersion_along_s
    from linac_gen.cli.common import build_ref
    ref = build_ref(beam_config)
    r = dispersion_along_s(lattice, ref)
    # η is per MeV: D = η·β²γm (mm→m, mrad→rad: ×1e-3); take β²γm at the
    # end from the walk's own ratio where η ≠ 0, else at the entrance
    # (a transfer line does not accelerate)
    k = ref.beta ** 2 * ref.gamma * ref.species.mass * 1e-3
    for d, e in (("disp_x_m", "eta_x"), ("disp_y_m", "eta_y")):
        if abs(float(r[e][-1])) > 1e-12:
            k = float(r[d][-1]) / float(r[e][-1])
            break
    return (float(r["disp_x_m"][-1]), float(r["eta_xp"][-1]) * k,
            float(r["disp_y_m"][-1]), float(r["eta_yp"][-1]) * k)


def design_reference(lattice, beam_config) -> dict:
    """Error-free envelope run: betatron β per record, end Twiss, dispersion."""
    from linac_gen.cli.common import run_envelope_sim
    res = run_envelope_sim(lattice, beam_config)
    S = np.asarray(res.sigma_matrix)
    return {"s_m": np.asarray(res.s) / 1e3,
            "beta_x": np.array([_twiss_betatron(s, 0)[1] for s in S]),
            "beta_y": np.array([_twiss_betatron(s, 2)[1] for s in S]),
            "twiss_end": (_twiss_betatron(S[-1], 0)[:2], _twiss_betatron(S[-1], 2)[:2]),
            "disp_end": _lattice_dispersion_end(lattice, beam_config),
            "sigma_end": (float(res.sigma_x[-1]), float(res.sigma_y[-1])),
            "n_rows": len(S)}


def seed_metrics(res, design, *, lattice=None, beam_config=None,
                 bpm_idx=(), bpm_planes=None) -> dict:
    """Per-seed figures of merit against :func:`design_reference`.

    Pass ``lattice`` and ``beam_config`` (the seed's) for the dispersion
    change; without them the dispersion keys are omitted.
    """
    from linac_gen.analysis.phase_advance import mismatch_factor
    S = np.asarray(res.sigma_matrix)
    C = np.asarray(res.centroid)
    c = C[-1]
    out = dict(x_end_mm=float(c[0]), xp_end_mrad=float(c[1]),
               y_end_mm=float(c[2]), yp_end_mrad=float(c[3]),
               x_max_mm=float(np.max(np.abs(C[:, 0]))),
               y_max_mm=float(np.max(np.abs(C[:, 2]))))
    if len(bpm_idx):
        ex = res.element_exit_idx
        xs, ys = [], []
        for i in bpm_idx:
            p = (bpm_planes or {}).get(i, "xy")
            row = C[ex[i]]
            if "x" in p:
                xs.append(row[0])
            if "y" in p:
                ys.append(row[2])
        out["x_bpm_rms_mm"] = float(np.sqrt(np.mean(np.square(xs)))) if xs else np.nan
        out["y_bpm_rms_mm"] = float(np.sqrt(np.mean(np.square(ys)))) if ys else np.nan
    if len(S) == design["n_rows"]:
        dbx = np.array([_twiss_betatron(s, 0)[1] for s in S]) / design["beta_x"] - 1.0
        dby = np.array([_twiss_betatron(s, 2)[1] for s in S]) / design["beta_y"] - 1.0
        out.update(beat_x_max=float(np.nanmax(np.abs(dbx))),
                   beat_y_max=float(np.nanmax(np.abs(dby))),
                   beat_x_rms=float(np.sqrt(np.nanmean(dbx ** 2))),
                   beat_y_rms=float(np.sqrt(np.nanmean(dby ** 2))))
    (axd, bxd), (ayd, byd) = design["twiss_end"]
    ax, bx, _ = _twiss_betatron(S[-1], 0)
    ay, by, _ = _twiss_betatron(S[-1], 2)
    out.update(bmag_x_end=float(mismatch_factor(axd, bxd, ax, bx)),
               bmag_y_end=float(mismatch_factor(ayd, byd, ay, by)),
               dbeta_x_end=float(bx / bxd - 1.0), dalpha_x_end=float(ax - axd),
               dbeta_y_end=float(by / byd - 1.0), dalpha_y_end=float(ay - ayd),
               sigx_end_mm=float(res.sigma_x[-1]), sigy_end_mm=float(res.sigma_y[-1]))
    if lattice is not None and beam_config is not None:
        d = _lattice_dispersion_end(lattice, beam_config)
        dd = design["disp_end"]
        out.update(dDx_end_m=d[0] - dd[0], dDpx_end=d[1] - dd[1],
                   dDy_end_m=d[2] - dd[2], dDpy_end=d[3] - dd[3])
    return out


# ------------------------------------------------------------------ seeds
def _job(args) -> dict:
    (study, seed, corrector, corr_kw, design, bpm_idx, bpm_planes,
     offset_sigma_mm, noise_mm, keep_kicks) = args
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.errors.error_model import stream_entropy
    lat = study._apply_errors(seed)
    cfg = study._apply_beam_errors(seed)
    rec = {"seed": int(seed)}
    if corrector is not None:
        # BPM offsets / noise: a tagged stream of its own, keyed by the
        # study's stream so independent ensembles get independent BPMs
        rng = np.random.default_rng(stream_entropy(seed, 5, *study._rng_key))
        offs = (rng.normal(0.0, 1.0, size=len(corrector.rows)) * _row_sigmas(
            corrector, offset_sigma_mm) if offset_sigma_mm is not None else None)
        info = corrector.correct(lat, offsets=offs, noise_mm=_row_sigmas(corrector, noise_mm),
                                 rng=rng, beam_config=cfg, **(corr_kw or {}))
        # active planes only: (bx_l, by_l) per steerer, x kick = by_l
        names = [lat.elements[i].name for i in corrector.steerer_idx]
        k = np.array([info["kicks_Tm"][names[j]][1 if kp == "x" else 0]
                      for j, kp in corrector.cols])
        rec.update(rms_bpm_before_mm=info["rms_before_mm"],
                   rms_bpm_after_mm=info["rms_after_mm"],
                   kick_max_Tm=float(np.max(np.abs(k))) if k.size else 0.0,
                   kick_rms_Tm=float(np.sqrt(np.mean(k ** 2))) if k.size else 0.0,
                   n_at_limit=int(info["n_clipped"]))
        if keep_kicks:
            rec["kicks_Tm"] = info["kicks_Tm"]
    res = run_envelope_sim(lat, cfg)
    rec.update(seed_metrics(res, design, lattice=lat, beam_config=cfg,
                            bpm_idx=bpm_idx, bpm_planes=bpm_planes))
    return rec


def _row_sigmas(corrector, sigma):
    """Per-reading σ [mm]: a scalar, or ``{bpm element index: σ}``."""
    if isinstance(sigma, dict):
        return np.array([float(sigma.get(corrector.bpm_idx[n], 0.0))
                         for n, _ in corrector.rows])
    return np.full(len(corrector.rows), float(sigma or 0.0))


def _init_worker():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                          # noqa: BLE001
        pass


def run_ensemble(study, seeds, *, design=None, corrector=None, corr_kw=None,
                 bpm_idx=(), bpm_planes=None, offset_sigma_mm=None,
                 noise_mm=0.0, workers=1, keep_kicks=False) -> list[dict]:
    """Envelope ensemble over ``seeds``; one metrics dict per seed (in order).

    ``corrector`` (an :class:`~linac_gen.errors.orm_correction.OrmCorrector`
    built on the design lattice) corrects each errored copy before the
    measured run, with per-seed BPM offsets ``N(0, offset_sigma_mm)``
    and reading noise ``noise_mm`` (each a scalar or ``{bpm index: σ}``),
    from a tagged per-seed stream.  With a corrector the BPM metrics default to
    its BPMs in their measured planes (the orbit everywhere is in
    ``x_max_mm`` / ``y_max_mm``).  ``workers > 1`` runs seeds in a spawn
    process pool with BLAS threads limited to 1 per worker.
    """
    if design is None:
        design = design_reference(study.lattice, study.beam_config)
    if corrector is not None:
        if not len(bpm_idx):
            bpm_idx = corrector.bpm_idx
        if bpm_planes is None:
            bpm_planes = dict(corrector.bpm_planes)
    args = [(study, int(s), corrector, corr_kw, design, tuple(bpm_idx), bpm_planes,
             offset_sigma_mm, noise_mm, bool(keep_kicks)) for s in seeds]
    if int(workers) <= 1:
        return [_job(a) for a in args]
    # The spawn child imports numpy before any initializer runs, so the
    # thread variables must already be in the environment it inherits.
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: "1" for v in _THREAD_VARS})
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx,
                                 initializer=_init_worker) as ex:
            return list(ex.map(_job, args,
                               chunksize=max(1, len(args) // (4 * int(workers)))))
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


# ------------------------------------------------------------------ jitter
_ORBIT_KEYS = ("x_end_mm", "xp_end_mrad", "y_end_mm", "yp_end_mrad")


def run_jitter(lattice, beam_config, add_defs, pulses, *, static_seed=0,
               design=None) -> list[dict]:
    """Pulse-to-pulse jitter on a frozen (already corrected) lattice.

    ``add_defs(study)`` registers the jitter definitions on a fresh
    :class:`ErrorStudy` (``add_error`` with ``circuit_mode="common"`` for
    power supplies, ``add_beam_error`` for launch jitter, …).  Pulse ``p``
    draws from the tagged stream ``(p, ·, 3, static_seed)`` — independent of
    every static draw and of other static seeds' pulses.

    Every metric is relative to the frozen lattice: the orbit keys are the
    pulse orbit MINUS the frozen seed's orbit (``x_end_mm`` … and the
    along-the-line maxima ``x_max_mm`` / ``y_max_mm`` = the jitter alone),
    β-beat / BMAG / dispersion use the frozen lattice as design.
    """
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.errors.error_model import ErrorStudy
    js = ErrorStudy(lattice, beam_config, n_seeds=0, rng_stream=(3, int(static_seed)))
    js._errors, js._beam_errors = [], []          # never re-draw deck ERROR_* cards
    add_defs(js)
    if design is None:
        design = design_reference(lattice, beam_config)
    fres = run_envelope_sim(lattice, beam_config)
    frozen = seed_metrics(fres, design)
    c0 = np.asarray(fres.centroid)
    out = []
    for p in range(int(pulses)):
        lat = js._apply_errors(p)
        cfg = js._apply_beam_errors(p)
        res = run_envelope_sim(lat, cfg)
        rec = {"pulse": p}
        rec.update(seed_metrics(res, design, lattice=lat, beam_config=cfg))
        for k in _ORBIT_KEYS:
            rec[k] -= frozen[k]
        dc = np.asarray(res.centroid) - c0       # same layout: row by row
        rec["x_max_mm"] = float(np.max(np.abs(dc[:, 0])))
        rec["y_max_mm"] = float(np.max(np.abs(dc[:, 2])))
        out.append(rec)
    return out
