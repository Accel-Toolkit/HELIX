"""Model-ORM SVD correction (linac_gen.errors.orm_correction) and the
envelope ensemble runner (linac_gen.errors.ensemble).

* the probe-built response equals kick-and-read with the envelope solver;
* labels set the planes (HT/VT steerers, HP/VP BPMs);
* correction flattens an orbit driven by dipole field errors and rolls;
  the readings are centred, so the true orbit lands on the BPM offsets;
* vmax clips; an SV cut keeps only the requested modes;
* the ensemble gives the same numbers serially and in a process pool, an
  error-free seed has zero orbit / beat and BMAG 1, and jitter pulses use
  their own RNG stream (never replaying the static seed's draws).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.orm_correction import OrmCorrector, bpm_plane, steerer_plane


def _cell_lattice(n_cells=6):
    """FODO with a bend per half cell, an H and a V steerer each followed
    (after a drift) by an H / V BPM — a square, invertible response
    (labels as in the BTL corrector deck)."""
    lat = Lattice()
    for k in range(n_cells):
        for e in (
            Quadrupole(f"QF{k}", length=200.0, gradient=-6.0),
            Drift(f"a{k}", length=300.0),
            Steerer(f"STH{k}"),
            Drift(f"b{k}", length=300.0),
            Marker(f"BH{k}"),
            Dipole(f"BA{k}a", angle=3.0, rho=20000.0),
            Drift(f"c{k}", length=300.0),
            Quadrupole(f"QD{k}", length=200.0, gradient=6.0),
            Drift(f"d{k}", length=300.0),
            Steerer(f"STV{k}"),
            Drift(f"e{k}", length=300.0),
            Marker(f"BV{k}"),
            Dipole(f"BA{k}b", angle=3.0, rho=20000.0),
            Drift(f"f{k}", length=300.0),
        ):
            lat.add(e)
    for e in lat.elements:
        if isinstance(e, Steerer):
            e.label = ("HT" if e.name.startswith("STH") else "VT") + e.name[3:]
        if isinstance(e, Marker):
            e.is_bpm = True
            e.label = ("HP" if e.name.startswith("BH") else "VP") + e.name[2:]
    return lat


def _cfg():
    return BeamConfig(species="H-", energy=800.0, frequency=162.5, current=0.0,
                      alpha_x=0.0, beta_x=8.0, alpha_y=0.0, beta_y=8.0)


def _idx(lat, cls):
    return [i for i, e in enumerate(lat.elements) if isinstance(e, cls)
            and (cls is not Marker or getattr(e, "is_bpm", False))]


def _errored(lat, seed=1):
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(lat)
    for e in out.elements:
        if isinstance(e, Dipole):
            e.field_rel = float(rng.normal(0, 1e-3))
            e.tilt_deg = float(rng.normal(0, 0.1))
        if isinstance(e, Quadrupole):
            e.dx = float(rng.normal(0, 0.1))
            e.dy = float(rng.normal(0, 0.1))
    return out


def test_planes_from_labels():
    lat = _cell_lattice(1)
    st = [lat.elements[i] for i in _idx(lat, Steerer)]
    bp = [lat.elements[i] for i in _idx(lat, Marker)]
    assert [steerer_plane(e) for e in st] == ["x", "y"]
    assert [bpm_plane(e) for e in bp] == ["x", "y"]


def test_model_response_equals_kick_and_read():
    from linac_gen.cli.common import run_envelope_sim
    lat = _cell_lattice()
    cfg = _cfg()
    oc = OrmCorrector(lat, cfg, _idx(lat, Steerer), _idx(lat, Marker))
    base, _, _ = oc.read(lat)
    dbl = 1e-4
    for c, (j, kp) in enumerate(oc.cols):
        l2 = copy.deepcopy(lat)
        st = l2.elements[oc.steerer_idx[j]]
        setattr(st, "by_l" if kp == "x" else "bx_l", dbl)
        meas, _, _ = oc.read(l2)
        np.testing.assert_allclose((meas - base) / dbl, oc.R[:, c], rtol=1e-7, atol=1e-9)


def test_correction_flattens_the_orbit_and_honours_offsets():
    lat = _cell_lattice()
    cfg = _cfg()
    oc = OrmCorrector(lat, cfg, _idx(lat, Steerer), _idx(lat, Marker))
    err = _errored(lat)
    info = oc.correct(err, n_iter=2)
    assert info["rms_before_mm"] > 0.3
    # square system; the design model vs the errored machine differ by
    # ~1e-4, so the iteration converges geometrically
    assert info["rms_after_mm"] < 1e-5 * info["rms_before_mm"]
    # BPM offsets: the correction centres the READINGS, so the true orbit
    # sits at the offsets (reading = true − offset = 0)
    err2 = _errored(lat)
    offs = np.full(len(oc.rows), 0.2)
    oc.correct(err2, n_iter=4, offsets=offs)
    _, true, _ = oc.read(err2)
    np.testing.assert_allclose(true, offs, atol=1e-6)


def test_errored_model_does_not_blow_up():
    """A perturbed model with a (numerically) zero mode: the default
    relative SV cut keeps the correction finite."""
    lat = _cell_lattice()
    lat.add(Steerer("STX")); lat.elements[-1].label = "HTX"   # no BPM after it
    cfg = _cfg()
    err = _errored(lat)
    oc = OrmCorrector(err, cfg, _idx(lat, Steerer), _idx(lat, Marker))
    assert oc.n_kept == len(oc.cols) - 1
    info = oc.correct(_errored(lat), n_iter=2)
    assert info["rms_after_mm"] < 1e-6


def test_vmax_clips_and_sv_cut():
    lat = _cell_lattice()
    cfg = _cfg()
    oc = OrmCorrector(lat, cfg, _idx(lat, Steerer), _idx(lat, Marker),
                      vmax_Tm=1e-5, n_sv=4)
    assert oc.n_kept == 4
    err = _errored(lat)
    info = oc.correct(err, n_iter=1)
    assert info["n_clipped"] > 0
    assert max(abs(v) for kk in info["kicks_Tm"].values() for v in kk) <= 1e-5 + 1e-18


def test_ensemble_serial_equals_pool_and_error_free_seed_is_clean():
    from linac_gen.errors.error_model import ErrorStudy
    from linac_gen.errors.ensemble import design_reference, run_ensemble, seed_metrics
    from linac_gen.cli.common import run_envelope_sim
    lat = _cell_lattice()
    cfg = _cfg()
    design = design_reference(lat, cfg)
    m0 = seed_metrics(run_envelope_sim(lat, cfg), design)
    assert m0["x_end_mm"] == 0.0 and m0["beat_x_max"] == 0.0
    assert m0["bmag_x_end"] == pytest.approx(1.0, abs=1e-12)
    st = ErrorStudy(lat, cfg, n_seeds=0)
    st.add_error("BA*", "field_rel", sigma=1e-3, cutoff=2.0)
    st.add_error("Q*", "gradient_rel", sigma=4e-3, cutoff=2.0)
    # this layout is ill-conditioned (each BPM 0.3 m behind its steerer,
    # far BPMs ~30x more sensitive): with noise and offsets an SV cut is
    # needed, as in any real correction
    oc = OrmCorrector(lat, cfg, _idx(lat, Steerer), _idx(lat, Marker), rcond=1e-2)
    kw = dict(design=design, corrector=oc,
              offset_sigma_mm=0.1, noise_mm=0.01)
    a = run_ensemble(st, range(4), workers=1, **kw)
    b = run_ensemble(st, range(4), workers=2, **kw)
    assert a == b
    # readings are centred, so the TRUE orbit settles at the BPM-offset
    # level (σ 0.1 mm), whatever the uncorrected orbit was
    assert all(r["rms_bpm_after_mm"] < 0.3 for r in a)
    assert max(r["rms_bpm_before_mm"] for r in a) > 0.3
    assert all(r["beat_x_max"] > 0 for r in a)


def test_jitter_pulses_use_their_own_stream():
    from linac_gen.errors.error_model import ErrorStudy
    from linac_gen.errors.ensemble import run_jitter
    lat = _cell_lattice()
    cfg = _cfg()
    circuits = {"ARC": [e.name for e in lat.elements if isinstance(e, Dipole)]}

    def defs(js):
        js.add_error("BA*", "field_rel", sigma=1e-4, circuits=circuits,
                     circuit_mode="common")
    p = run_jitter(lat, cfg, defs, 3, static_seed=0)
    q = run_jitter(lat, cfg, defs, 3, static_seed=1)
    assert len({r["x_end_mm"] for r in p}) == 3            # pulses differ
    assert p[0]["x_end_mm"] != q[0]["x_end_mm"]            # per static seed
    # a static study with the same definition draws a different value
    st = ErrorStudy(lat, cfg, n_seeds=0)
    defs(st)
    static = st._apply_errors(0)
    js = ErrorStudy(lat, cfg, n_seeds=0, rng_stream=(3, 0))
    defs(js)
    pulse0 = js._apply_errors(0)
    b = next(e for e in static.elements if isinstance(e, Dipole))
    c = next(e for e in pulse0.elements if isinstance(e, Dipole))
    assert b.field_rel != c.field_rel


def test_jitter_orbit_is_relative_to_the_frozen_seed():
    """Zero jitter on a corrected, errored lattice: every pulse reports
    zero orbit change (the static orbit is NOT counted as jitter)."""
    from linac_gen.errors.ensemble import run_jitter
    lat = _cell_lattice()
    cfg = _cfg()
    err = _errored(lat)                              # frozen seed with an orbit
    out = run_jitter(err, cfg, lambda js: js.add_error("BA*", "field_rel", sigma=0.0), 2)
    for r in out:
        assert r["x_end_mm"] == 0.0 and r["y_end_mm"] == 0.0
        assert r["bmag_x_end"] == pytest.approx(1.0, abs=1e-12)


def test_jitter_line_maxima_are_relative_and_kick_rms_counts_active_planes():
    from linac_gen.errors.ensemble import run_jitter, run_ensemble, design_reference
    from linac_gen.errors.error_model import ErrorStudy
    lat = _cell_lattice()
    cfg = _cfg()
    err = _errored(lat)
    out = run_jitter(err, cfg, lambda js: js.add_error("BA*", "field_rel", sigma=0.0), 1)
    assert out[0]["x_max_mm"] == 0.0 and out[0]["y_max_mm"] == 0.0
    # kick rms over the ACTIVE plane of each single-plane steerer
    st = ErrorStudy(lat, cfg, n_seeds=0)
    st.add_error("BA*", "field_rel", sigma=1e-3, cutoff=2.0)
    oc = OrmCorrector(lat, cfg, _idx(lat, Steerer), _idx(lat, Marker), rcond=1e-2)
    r = run_ensemble(st, [0], design=design_reference(lat, cfg), corrector=oc,
                     keep_kicks=True)[0]
    k = [r["kicks_Tm"][lat.elements[oc.steerer_idx[j]].name][1 if p == "x" else 0]
         for j, p in oc.cols]
    assert r["kick_rms_Tm"] == pytest.approx(float(np.sqrt(np.mean(np.square(k)))), rel=1e-12)
