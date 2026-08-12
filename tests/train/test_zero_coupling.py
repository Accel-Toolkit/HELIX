"""THE zero-coupling contract: with all physics off, an N-bunch train is
BIT-IDENTICAL to N independent single-bunch runs (mp AND envelope — the
dual-regime rule)."""
from __future__ import annotations

import numpy as np

from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.train import PulsePattern, TrainConfig, TrainRunner

_CMP_KEYS = ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
             "emit_x", "emit_y", "emit_z", "ref_w_kin", "ref_phi_s")


def _lattice():
    from linac_gen.elements.ncells import NCells
    lat = Lattice()
    for _ in range(3):
        lat.add(Quadrupole("QF", 40.0, gradient=18.0, aperture=20.0))
        lat.add(Drift("D1", 30.0, aperture=20.0))
        lat.add(RFGap(name="G", voltage=0.05, phase=-30.0, frequency=162.5))
        lat.add(Drift("D2", 30.0, aperture=20.0))
        lat.add(Quadrupole("QD", 40.0, gradient=-18.0, aperture=20.0))
        lat.add(Drift("D3", 30.0, aperture=20.0))
    # SET_SYNC_PHASE cavity so the design pass + psi-pin lifecycle is
    # exercised end-to-end (adversarial F7)
    lat.add(NCells("NC", mode=1, n_cells=4, beta_g=0.081, eot_v_per_m=8e5,
                   theta_s_deg=-40.0, aperture_mm=20.0, sync_phase=True,
                   frequency_mhz=162.5))
    return lat


def _cfg(**kw):
    base = dict(species="proton", energy=3.0, frequency=162.5,
                current=5.0, n_particles=800, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                emit_z=0.15, alpha_z=0.0, beta_z=1.2)
    base.update(kw)
    return BeamConfig(**base)


def _assert_identical(res_a, res_b):
    for k in _CMP_KEYS:
        a, b = getattr(res_a, k, None), getattr(res_b, k, None)
        if a is None and b is None:
            continue
        assert np.array_equal(np.asarray(a, float), np.asarray(b, float)), k


def test_mp_train_bit_identical_to_singles():
    cfg = _cfg()
    sc = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0)
    tc = TrainConfig(bunch_frequency_MHz=162.5,
                     pattern=PulsePattern.from_rle("1*2 0*3 1*1"))
    train = TrainRunner(_lattice(), cfg, tc, sc_config=sc).run()
    assert train.slots == [0, 1, 5]
    for res in train.bunch_results:
        single = Simulation(_lattice(), create_beam(cfg, seed=tc.seed),
                            space_charge=sc).run()
        _assert_identical(res, single)


def test_envelope_train_bit_identical():
    cfg = _cfg()
    tc = TrainConfig(bunch_frequency_MHz=162.5, mode="envelope",
                     pattern=PulsePattern.uniform(3))
    train = TrainRunner(_lattice(), cfg, tc).run()
    assert len(train.bunch_results) == 3
    _assert_identical(train.bunch_results[0], train.bunch_results[2])
    from linac_gen.cli.common import run_envelope_sim
    import dataclasses
    cfg_f = dataclasses.replace(cfg, bunch_frequency_MHz=162.5)
    single = run_envelope_sim(_lattice(), cfg_f)
    _assert_identical(train.bunch_results[0], single)


def test_post_train_lattice_is_clean():
    """K2 regression: a normal single-bunch run on the SAME lattice
    object AFTER a train must be bit-identical to a fresh-lattice run —
    no pins or any other state may survive the train (zero-coupling
    contract, plan 3b)."""
    cfg = _cfg(n_particles=300)
    lat = _lattice()
    tc = TrainConfig(bunch_frequency_MHz=162.5,
                     pattern=PulsePattern.uniform(2))
    TrainRunner(lat, cfg, tc).run()
    for el in lat.elements:
        pin = getattr(el, "sync_phase_pin", None)
        assert pin is None, f"pin leaked on {getattr(el, 'name', el)}"
    after = Simulation(lat, create_beam(cfg, seed=7)).run()
    fresh = Simulation(_lattice(), create_beam(cfg, seed=7)).run()
    _assert_identical(after, fresh)


def test_periodic_phase_refused():
    import pytest
    cfg = _cfg(periodic_phase=True)
    tc = TrainConfig(bunch_frequency_MHz=162.5,
                     pattern=PulsePattern.uniform(2))
    with pytest.raises(ValueError, match="periodic_phase"):
        TrainRunner(_lattice(), cfg, tc)


def test_hdf5_roundtrip(tmp_path):
    cfg = _cfg(n_particles=200)
    tc = TrainConfig(bunch_frequency_MHz=162.5,
                     pattern=PulsePattern.from_rle("1*2 0*2"))
    train = TrainRunner(_lattice(), cfg, tc).run()
    out = tmp_path / "train.h5"
    train.save_hdf5(str(out))
    import h5py
    with h5py.File(out) as f:
        assert f["train"].attrs["n_bunches_tracked"] == 2
        assert f["train"].attrs["pattern_rle"] == "1*2 0*2"
        assert "bunches/b_0000" in f and "bunches/b_0001" in f
        s = f["train/summary/sigma_x"][:]
        assert s.shape == (2,) and np.isfinite(s).all()
