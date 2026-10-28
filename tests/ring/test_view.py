"""A ring run seen through a linac run's result names (linac_gen.ring.view):
one entry per statistics row, the turn in ``s``; the conversion to the linac
units against the exact per-particle conversion (coords.to_helix), the
recorder's own definitions, the reference of each row, injection, no RF."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.ring import coords as C
from linac_gen.ring.reference import RingReference
from linac_gen.ring.view import RingResultsView, _jacobian, received_by_turn

ROOT = Path(__file__).resolve().parents[2]
BOOSTER = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
SERIES = ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w", "emit_x", "emit_y", "emit_z",
          "emit_z_mmmrad", "emit_nx", "emit_ny", "emit_nz", "alpha_x", "beta_x", "alpha_y",
          "beta_y", "alpha_z", "beta_z", "emit_4d", "emit_n1", "emit_n2", "emit_e1", "emit_e2",
          "emit_e3", "sigma_matrix", "centroid", "transmission", "ref_w_kin", "ref_beta",
          "ref_gamma", "ref_bg", "ref_frequency", "continuous_at")


def _beam():
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0,
                      emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0, cutoff=4.0)


def _run(**cfg):
    from linac_gen.cli.common import load_lattice
    from linac_gen.ring import RingConfig, RingSimulation
    lat = load_lattice(str(BOOSTER))
    base = dict(n_turns=4, n_particles=2000, source="matched", tune_probes=0,
                snapshot_turns=[0, "last"])
    base.update(cfg)
    return RingSimulation(lat, _beam(), RingConfig(**base)).run()


@pytest.fixture(scope="module")
def run():
    return _run()


def _emp_cov(P):
    d = P - P.mean(axis=0)
    return d.T @ d / len(P)


def test_one_entry_per_statistics_row_with_the_turn_in_s(run):
    v = RingResultsView(run)
    n = len(run.turns)
    assert v.x_domain == "turn" and v.is_ring_view and v.ring is run
    assert v.s == [float(t) for t in run.turns]
    assert all(len(getattr(v, k)) == n for k in SERIES), [k for k in SERIES
                                                          if len(getattr(v, k)) != n]
    # (lists, 2-D series lists of arrays — as a linac results file reads back)
    assert isinstance(v.sigma_x, list) and isinstance(v.sigma_matrix, list)
    assert isinstance(v.sigma_matrix[0], np.ndarray) and v.sigma_matrix[0].shape == (6, 6)
    assert v.element_names == [] and v.element_exit_idx == []
    # what a ring run does not record stays absent (those tiles are disabled)
    for k in ("x_max", "y_max", "halo_x", "loss_table", "n_macro", "beam", "_snapshots"):
        assert not hasattr(v, k), k


def test_the_matrix_against_the_exact_conversion_of_the_particles(run):
    v = RingResultsView(run)
    t = max(run.snapshots)
    k = list(run.turns).index(t)
    ref = RingReference.from_kinetic("proton", float(run.meta["kinetic_mev"]))
    P = C.to_helix(run.snapshots[t]["coords"], ring_ref=ref, w_ref=ref.kinetic0,
                   frequency_mhz=v.f0_hz * 1e-6)
    S_emp, S = _emp_cov(P), v.sigma_matrix[k]
    scale = np.sqrt(np.outer(np.diag(S_emp), np.diag(S_emp)))
    lin = np.ix_([0, 2, 4], [0, 2, 4])                       # x, y, dphi: a linear map
    assert np.max(np.abs(S - S_emp)[lin] / scale[lin]) < 1e-13
    # x', y', dW enter to first order about the centroid (sampling noise ~ 1e-5 here)
    assert np.max(np.abs(S - S_emp) / scale) < 1e-4
    np.testing.assert_allclose(v.centroid[k][:5], P.mean(axis=0)[:5], rtol=0, atol=1e-5)


def test_a_bumped_off_momentum_centroid_needs_the_full_jacobian():
    # a beam on a bumped orbit (px 0.02) and off momentum (delta 1e-3), with a
    # correlated dispersive spread: J C J^T against the exact conversion; the
    # diagonal first-order J would be off by percents on the x' entries
    rng = np.random.default_rng(3)
    n = 200_000
    p0c, mass, f0 = 1463.296, 938.272088, 44.7e6
    d = 1e-3 + 1e-3 * rng.standard_normal(n)
    X = np.vstack([0.01 + 2e-3 * rng.standard_normal(n) + 3.0 * (d - 1e-3),
                   0.02 + 1e-4 * rng.standard_normal(n) + 0.1 * (d - 1e-3),
                   1e-3 * rng.standard_normal(n), 2e-4 * rng.standard_normal(n),
                   0.1 * rng.standard_normal(n), d])
    ref = RingReference(mass=mass, charge=1, p0c=p0c)
    P = C.to_helix(X, ring_ref=ref, w_ref=ref.kinetic0, frequency_mhz=f0 * 1e-6)
    c, J = _jacobian(X.mean(axis=1), p0c, ref.beta0, mass, f0)
    S = J @ np.cov(X, bias=True) @ J.T
    S_emp = _emp_cov(P)
    scale = np.sqrt(np.outer(np.diag(S_emp), np.diag(S_emp)))
    assert np.max(np.abs(S - S_emp) / scale) < 5e-3
    J1 = np.diag(np.diag(J))
    J1[1, 1] = J1[3, 3] = 1e3                                 # x' = px, first order
    S1 = J1 @ np.cov(X, bias=True) @ J1.T
    assert np.max(np.abs(S1 - S_emp) / scale) > 1e-2
    np.testing.assert_allclose(c, P.mean(axis=0), rtol=0, atol=2e-3)


def test_longitudinal_quantities_and_dispersion_as_the_tiles_compute_them(run):
    v = RingResultsView(run)
    for k, c in enumerate(run.cov):
        # the recorder's (z, z') emittance in mm mrad: the ring's zeta-delta one
        assert v.emit_z_mmmrad[k] == pytest.approx(math.sqrt(np.linalg.det(c[4:6, 4:6])) * 1e6,
                                                   rel=1e-6)
        assert v.emit_nz[k] == pytest.approx(v.emit_z_mmmrad[k] * v.ref_bg[k], rel=1e-14)
        # the dispersion tile: D = S05 / S55 * beta^2 gamma m / 1000 [m] = <x delta>/<delta^2>
        S = v.sigma_matrix[k]
        bgm = v.ref_beta[k] ** 2 * v.ref_gamma[k] * v.mass_mev * 1e-3
        assert S[0, 5] / S[5, 5] * bgm == pytest.approx(c[0, 5] / c[5, 5], rel=1e-5)
        # sigma_z from the phase as the KPI computes it: the bunch length
        lam_mm = 299792458.0 / (v.ref_frequency[k] * 1e6) * 1e3
        assert v.sigma_phi[k] * v.ref_beta[k] * lam_mm / 360.0 == pytest.approx(
            math.sqrt(c[4, 4]) * 1e3, rel=1e-12)
    assert len(set(v.ref_frequency)) == 1                    # one f0: no "frequency jump"


def test_the_kpi_and_the_turn_by_turn_ring_series(run):
    v = RingResultsView(run)
    c = run.cov[-1]
    assert v.kpi["sigma_x_mm"] == pytest.approx(math.sqrt(c[0, 0]) * 1e3, rel=1e-14)
    assert v.kpi["sigma_y_mm"] == pytest.approx(math.sqrt(c[2, 2]) * 1e3, rel=1e-14)
    assert v.kpi["sigma_z_mm"] == pytest.approx(math.sqrt(c[4, 4]) * 1e3, rel=1e-14)
    assert v.kpi["ex_growth"] == pytest.approx(v.emit_nx_betatron[-1] / v.emit_nx_betatron[0],
                                               rel=1e-14)
    assert v.kpi["transmission"] == 100.0 and v.kpi["loss"] == 0.0
    # the RING RUN series are the run's own, bit for bit
    assert v.emit_nx_projected == [float(x) for x in np.asarray(run.emit_normalized)[:, 0] * 1e6]
    assert v.emit_long_evs == [float(x) for x in run.emit_long_evs]
    # the matched beam's betatron emittance is the Beam tab's 1.5 pi mm mrad (the
    # x-px block conditioned on delta is the 6-D normal mode up to the RF's
    # synchro-betatron coupling at dispersion: 2e-7 here)
    assert v.emit_nx_betatron[0] == pytest.approx(1.5, rel=1e-6)


def test_received_and_transmission_with_injection():
    from linac_gen.ring.engine import RingResults
    res = RingResults(engine="cpu", n_turns_requested=3, turns=np.array([0, 1, 2, 3]),
                      n_alive=np.array([0, 9, 17, 16]), mean=np.zeros((4, 6)),
                      cov=np.tile(np.eye(6) * 1e-6, (4, 1, 1)),
                      injection={"settings": {"turns": 2}, "stripped": [10, 8, 0]},
                      meta={"mass_mev": 938.272088, "beta0gamma0": 1.5, "p0c_mev": 1407.4,
                            "revolution_time_s": 1.88e-6, "harmonic": 84, "rf_on": True})
    assert list(received_by_turn(res, res.turns)) == [0.0, 10.0, 18.0, 18.0]
    v = RingResultsView(res)
    assert math.isnan(v.transmission[0])                     # nothing received yet
    assert v.transmission[1:] == pytest.approx([90.0, 100.0 * 17 / 18, 100.0 * 16 / 18])
    assert v.kpi["transmission"] == pytest.approx(100.0 * 16 / 18)


def test_a_row_without_particles_is_nan_not_an_error():
    from linac_gen.ring.engine import RingResults
    mean = np.zeros((2, 6))
    cov = np.tile(np.eye(6) * 1e-6, (2, 1, 1))
    mean[1] = np.nan
    cov[1] = np.nan
    res = RingResults(engine="cpu", n_turns_requested=1, turns=np.array([0, 1]),
                      n_alive=np.array([10, 0]), mean=mean, cov=cov,
                      meta={"mass_mev": 938.272088, "beta0gamma0": 1.5, "p0c_mev": 1407.4,
                            "revolution_time_s": 1.88e-6, "harmonic": 84, "rf_on": True})
    v = RingResultsView(res)
    assert math.isnan(v.sigma_x[1]) and np.isnan(v.sigma_matrix[1]).all()
    assert v.transmission == [100.0, 0.0]


def test_a_ramp_rows_carry_their_own_reference():
    ramp = {"energy": {"kind": "table", "times_s": [0.0, 1e-4], "values": [800.0, 830.0]},
            "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
            "gamma_t": 5.449167323}
    res = _run(n_turns=6, n_particles=300, ramp=ramp, snapshot_turns=[])
    v = RingResultsView(res)
    p0c = np.asarray(res.row_p0c_mev)
    m = v.mass_mev
    w = p0c * p0c / (np.sqrt(p0c * p0c + m * m) + m)
    assert v.ref_w_kin == [float(x) for x in w] and v.ref_w_kin[-1] > v.ref_w_kin[0]
    assert v.f0_hz == float(np.asarray(res.ramp["frequency_hz"])[0])
    assert v.ref_bg == [float(x) for x in np.asarray(res.row_beta0gamma0)]


def test_without_rf_the_beam_is_continuous():
    from linac_gen.ring.engine import RingResults
    res = RingResults(engine="cpu", n_turns_requested=1, turns=np.array([0, 1]),
                      n_alive=np.array([10, 10]), mean=np.zeros((2, 6)),
                      cov=np.tile(np.eye(6) * 1e-6, (2, 1, 1)),
                      meta={"mass_mev": 938.272088, "beta0gamma0": 1.5, "p0c_mev": 1407.4,
                            "revolution_time_s": 2e-6, "harmonic": 84, "rf_on": False})
    v = RingResultsView(res)
    # (this meta records no fold of zeta: not periodic — a real run without RF
    # folds it at C, test_without_rf_the_phase_is_at_the_revolution_frequency)
    assert v.continuous_at == [True, True] and v.periodic_phase is False
    assert v.f0_hz == pytest.approx(5e5)                     # the revolution frequency


def test_the_run_keeps_its_optics_table_and_its_file_too(run, tmp_path):
    import h5py

    from linac_gen.ring.results import load_ring_hdf5, save_ring_hdf5
    t = run.optics_table
    n_el = len(run.meta["element_names"])
    assert t["name"] == list(run.meta["element_names"]) and len(t["s"]) == n_el
    assert np.all(np.diff(t["s"]) >= 0.0) and t["s"][-1] == pytest.approx(run.meta["length_m"])
    # (the table at the last element's exit is the periodic start)
    assert t["betx"][-1] == pytest.approx(run.optics["betx_start"], rel=1e-9)
    p = save_ring_hdf5(run, tmp_path / "r.h5")
    back = load_ring_hdf5(p)
    assert back.optics_table["name"] == t["name"] and back.optics_table["type"] == t["type"]
    assert all(np.array_equal(back.optics_table[k], t[k]) for k in t if k not in ("name", "type"))
    # a file written before the table existed: loads with none
    with h5py.File(p, "a") as f:
        del f["optics_table"]
    assert load_ring_hdf5(p).optics_table == {}


def test_the_derived_emittances_are_the_recorders_row_by_row(run):
    # the view computes every row at once; each value is the linac recorder's
    # helper on that row's matrix, bit for bit — and the rows those helpers
    # treat apart (no particle spread: det <= 0) come out as they give them
    from types import SimpleNamespace

    from linac_gen.diagnostics.eigenemittance import eigenemittances
    from linac_gen.diagnostics.recorder import (
        _convert_emit_z_to_mmmrad,
        _normal_mode_emittances,
    )
    from linac_gen.ring.engine import RingResults
    from linac_gen.ring.view import C_LIGHT
    small = _run(n_turns=3, n_particles=500)
    k = len(run.turns)
    flat = dict(run.__dict__)
    # (two more rows: one particle left — a covariance of zeros — and none;
    # each row with its own reference, as in a ramp)
    flat["turns"] = np.concatenate([run.turns, [k, k + 1]])
    flat["n_alive"] = np.concatenate([run.n_alive, [1, 0]])
    flat["mean"] = np.concatenate([run.mean, run.mean[-1:], np.full((1, 6), np.nan)])
    flat["cov"] = np.concatenate([run.cov, np.zeros((1, 6, 6)), np.full((1, 6, 6), np.nan)])
    bg = float(run.meta["beta0gamma0"]) * (1.0 + 1e-3 * np.arange(k + 2))
    flat["row_beta0gamma0"] = bg
    flat["row_p0c_mev"] = bg * float(run.meta["mass_mev"])
    res = RingResults(**{f: flat[f] for f in RingResults.__dataclass_fields__})
    assert len(set(RingResultsView(res).ref_beta)) == k + 2
    for r in (res, small):
        v = RingResultsView(r)
        for i in range(len(v.s)):
            S = np.asarray(v.sigma_matrix[i])
            if not np.all(np.isfinite(S)):
                assert np.isnan(v.emit_e1[i]) and np.isnan(v.emit_n1[i])
                continue
            ref = SimpleNamespace(beta=v.ref_beta[i], gamma=v.ref_gamma[i],
                                  wavelength=C_LIGHT / v.f0_hz * 1e3,
                                  species=SimpleNamespace(mass=v.mass_mev))
            assert v.emit_z_mmmrad[i] == _convert_emit_z_to_mmmrad(v.emit_z[i], ref)
            assert (v.emit_n1[i], v.emit_n2[i]) == _normal_mode_emittances(S[:4, :4])
            assert (v.emit_e1[i], v.emit_e2[i], v.emit_e3[i]) == eigenemittances(S)
            assert v.emit_4d[i] == math.sqrt(max(float(np.linalg.det(S[:4, :4])), 0.0))
    # (the one-particle row: zero spread, every emittance 0; the empty row NaN)
    v = RingResultsView(res)
    assert v.emit_x[k] == 0.0 and v.emit_e1[k] == 0.0 and v.beta_x[k] == 0.0
    assert np.isnan(v.sigma_x[k + 1]) and np.isnan(v.centroid[k + 1]).all()


def test_a_long_run_is_seen_quickly(run):
    # (statistics on every turn up to 100 000 turns: the view of a long run is
    # built at once, not row by row — well under a second for 100 000 rows)
    import dataclasses
    import time
    k = len(run.turns)
    n = 100_000
    reps = -(-n // k)
    upd = {f.name: np.concatenate([getattr(run, f.name)] * reps)[:n]
           for f in dataclasses.fields(run)
           if isinstance(getattr(run, f.name), np.ndarray) and getattr(run, f.name).ndim
           and getattr(run, f.name).shape[0] == k}
    upd["turns"] = np.arange(n)
    big = dataclasses.replace(run, **upd)
    t0 = time.perf_counter()
    v = RingResultsView(big)
    dt = time.perf_counter() - t0
    assert len(v.s) == n and v.emit_e1[-1] == RingResultsView(run).emit_e1[(n - 1) % k]
    assert dt < 5.0, dt


def test_the_phase_frequency_without_a_declared_harmonic(run):
    # (a cavity given by its frequency, or a ring from another format: no
    # harmonic in the run's meta — the frequency still follows the RF, from
    # the period the run folds zeta into, C/h)
    import copy

    from linac_gen.ring.view import C_LIGHT, rf_frequency_hz
    meta = run.meta
    f_h = float(meta["harmonic"]) / float(meta["revolution_time_s"])
    assert rf_frequency_hz(run) == pytest.approx(f_h, rel=1e-12)
    bare = copy.copy(run)
    bare.meta = dict(meta, harmonic=None)
    assert rf_frequency_hz(bare) == pytest.approx(f_h, rel=1e-12)
    assert rf_frequency_hz(bare) == float(meta["beta0"]) * C_LIGHT / float(meta["zeta_wrap_m"])
    assert RingResultsView(bare).sigma_phi == pytest.approx(RingResultsView(run).sigma_phi,
                                                            rel=1e-12)
    # (no fold — zeta wrap off — and no harmonic: the revolution frequency, said
    # by the run's meta only)
    unfolded = copy.copy(run)
    unfolded.meta = dict(meta, harmonic=None, zeta_wrap_m=None)
    assert rf_frequency_hz(unfolded) == pytest.approx(1.0 / float(meta["revolution_time_s"]))
    assert RingResultsView(unfolded).periodic_phase is False
    assert RingResultsView(run).periodic_phase is True


def test_without_rf_the_phase_is_at_the_revolution_frequency():
    from linac_gen.cli.common import load_lattice
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.view import rf_frequency_hz
    lat = load_lattice(str(BOOSTER))
    rl = compile_from_config(lat, RingConfig(), beam=_beam()).with_rf(False)
    res = RingSimulation(load_lattice(str(BOOSTER)), _beam(),
                         RingConfig(n_turns=2, n_particles=200, source="matched",
                                    tune_probes=0), ring_lattice=rl).run()
    assert not res.meta["rf_on"]
    assert rf_frequency_hz(res) == pytest.approx(1.0 / res.meta["revolution_time_s"],
                                                 rel=1e-12)
    v = RingResultsView(res)
    assert all(v.continuous_at) and v.periodic_phase is True
