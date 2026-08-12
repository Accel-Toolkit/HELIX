"""M8 surface guarantees at the core level.

1. Config honesty guards: knobs no v1 runner consumes (charge_scale,
   jitter, explicit select_bunches outside hybrid) are REFUSED at
   construction — a study must never accept an input and ignore it.
2. Tracked-mode abort marker: a mid-train should_abort leaves a partial
   TrainResults with ``truncated=True`` that saves and loads intact
   (the tracked mirror of FastPulseSummary.truncated).
"""
from __future__ import annotations

import json

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.train import (PulsePattern, TrainConfig, TrainJitter,
                             TrainPhysics, TrainRunner)
from linac_gen.train.results import load_train_results

F = 162.5


def _lattice():
    lat = Lattice()
    lat.add(Drift("D0", 50.0, aperture=30.0))
    lat.add(RFGap(name="CAV1", voltage=0.8, phase=0.0, frequency=F))
    lat.add(Drift("D1", 50.0, aperture=30.0))
    return lat


def _cfg(n=200):
    return BeamConfig(species="proton", energy=3.0, frequency=F,
                      current=5.0, n_particles=n,
                      distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                      emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                      emit_z=0.15, alpha_z=0.0, beta_z=1.2)


def _sidecar(tmp_path):
    p = tmp_path / "cav.json"
    p.write_text(json.dumps(
        {"CAV*": {"r_over_q": 200.0, "q_loaded": 5.0e6}}))
    return str(p)


# ---------------------------------------------------------------------------
# 1. config honesty guards
# ---------------------------------------------------------------------------
def test_charge_scale_refused_as_inert():
    with pytest.raises(ValueError, match="charge_scale"):
        TrainConfig(bunch_frequency_MHz=F,
                    pattern=PulsePattern.uniform(4),
                    charge_scale=lambda k: 1.0)


def test_jitter_refused_as_inert():
    with pytest.raises(ValueError, match="jitter"):
        TrainConfig(bunch_frequency_MHz=F,
                    pattern=PulsePattern.uniform(4),
                    jitter=TrainJitter(phase_deg_rms=0.5))


def test_select_bunches_refused_outside_hybrid():
    for mode in ("mp", "envelope", "fast"):
        with pytest.raises(ValueError, match="hybrid"):
            TrainConfig(bunch_frequency_MHz=F,
                        pattern=PulsePattern.uniform(4),
                        mode=mode, select_bunches=[0, 1])
    # the default passes everywhere
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.uniform(4), mode="fast")
    assert tc.select_bunches == "auto"


# ---------------------------------------------------------------------------
# 2. tracked abort → truncated partial, save/load intact
# ---------------------------------------------------------------------------
def test_tracked_abort_truncated_roundtrip(tmp_path):
    side = _sidecar(tmp_path)
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*8"), mode="mp",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=side)
    done = []

    def abort_after_three():
        return len(done) >= 3

    runner = TrainRunner(_lattice(), _cfg(), tc,
                         progress_callback=lambda i, n: done.append(i),
                         should_abort=abort_after_three)
    with pytest.warns(UserWarning, match="aborted after 3/8"):
        res = runner.run()
    assert res.truncated
    assert len(res.slots) == 3

    out = tmp_path / "partial.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))
    assert ld.truncated
    assert ld.n_bunches_tracked == 3
    assert len(ld.summary["ref_w_kin"]) == 3
    # The loading ledger covers exactly the processed prefix.
    assert sorted({s for (s, _i, _n) in ld.applied_loading}) == [0, 1, 2]


def test_completed_run_not_truncated(tmp_path):
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.uniform(2), mode="mp")
    res = TrainRunner(_lattice(), _cfg(), tc).run()
    assert not res.truncated
    out = tmp_path / "full.h5"
    res.save_hdf5(str(out))
    assert not load_train_results(str(out)).truncated
