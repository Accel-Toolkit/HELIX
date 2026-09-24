"""WP4 — ScanPoint.loss_metrics / snapshot_elements: inert by default, loss
power at the project duty cycle, the 17.6 kW anchor, HDF5 cross-check."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.beam import LOSS_DTYPE
from linac_gen.parallel.scan_pool import (LOSS_METRIC_KEYS, ScanPoint,
                                          _loss_metrics, _run_one_point_worker)
from linac_gen.study.engine import METRIC_KEYS

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo" / "reliability_demo.lgproj"
FOIL_S_MM = 3 * 750.0 + 3 * 150.0     # three 750 mm cells + the tail up to FOIL_001


def _point(mode="envelope", cli=None, element_overrides=(), **beam):
    from linac_gen.cli import common
    return common.build_scan_point(str(DEMO), beam_overrides=beam or None,
                                   mode=mode, cli=cli,
                                   element_overrides=element_overrides)


def _fake_results(n_lost, energy_mev, s_mm, n_macro, s_end_mm):
    lt = np.zeros(n_lost, dtype=LOSS_DTYPE)
    lt["particle_id"] = np.arange(n_lost)
    lt["s"] = s_mm
    lt["energy"] = energy_mev
    lt["element_name"] = "APER_X"
    return SimpleNamespace(loss_table=lt, n_macro=n_macro, s=[0.0, s_end_mm])


def test_defaults_are_inert():
    p = _point()
    assert p.snapshot_elements == () and p.loss_metrics is False
    row = _run_one_point_worker(p)
    assert row["error"] is None
    assert set(row) == set(METRIC_KEYS) | {"error"}
    assert not any(k in row for k in LOSS_METRIC_KEYS)


def test_loss_metrics_anchor_17_6_kw_at_project_duty():
    """2 mA x 1.1 % duty, all 5000 macroparticles lost at 800 MeV in one
    1 m bin -> 17.6 kW total and 17.6 kW/m at s = 0.5 m; the same
    record at 100 % duty is the CW 1.6 MW (both regimes)."""
    res = _fake_results(5000, 800.0, 400.0, 5000, 2000.0)
    m = _loss_metrics(res, SimpleNamespace(current=2.0, duty_cycle=1.1))
    assert m["loss_w_total"] == pytest.approx(17600.0, rel=1e-12)
    assert m["loss_w_per_m_peak"] == pytest.approx(17600.0, rel=1e-12)
    assert m["loss_w_per_m_peak_s_m"] == pytest.approx(0.5)
    assert m["n_lost"] == 5000
    cw = _loss_metrics(res, SimpleNamespace(current=2.0, duty_cycle=100.0))
    assert cw["loss_w_total"] == pytest.approx(1.6e6, rel=1e-12)


def test_loss_metrics_none_without_a_loss_record():
    m = _loss_metrics(SimpleNamespace(), SimpleNamespace(current=2.0, duty_cycle=1.1))
    assert m == {k: None for k in LOSS_METRIC_KEYS}
    row = _run_one_point_worker(dataclasses.replace(_point(), loss_metrics=True))
    assert all(row[k] is None for k in LOSS_METRIC_KEYS)
    assert set(row) == set(METRIC_KEYS) | {"error"} | set(LOSS_METRIC_KEYS)


def test_mp_point_loss_metrics_snapshot_and_hdf5_cross_check(tmp_path):
    """A fat beam scrapes the 8 mm collimators; the row's loss watts equal
    the loss-power summary recomputed from the HDF5 losses/ group, the foil
    snapshot lands in particles/, and neither flag perturbs the shared
    metrics of the same seed."""
    import h5py
    from linac_gen.analysis.loss_power import loss_power_summary
    from linac_gen.io.hdf5_output import load_results_hdf5
    # The foil's scattering kicks come from an UNSEEDED generator (until
    # WP6 adds the seed), so the end-of-line widths differ run to run;
    # a zero-thickness foil makes the two rows comparable.
    base = _point("mp", cli={"nx": 16}, n_particles=300, beta_x=40.0,
                  beta_y=40.0, current=2.0, duty_cycle=1.1,
                  element_overrides=(("FOIL_001.thickness_ug_cm2", 0.0),))
    out = tmp_path / "results.h5"
    p = dataclasses.replace(base, loss_metrics=True,
                            snapshot_elements=("FOIL_001",), out_path=str(out))
    row = _run_one_point_worker(p)
    assert row["error"] is None
    assert row["n_lost"] > 0 and row["loss_w_total"] > 0.0
    saved = load_results_hdf5(str(out))
    assert len(saved["loss_table"]) == row["n_lost"]
    summ = loss_power_summary(saved["loss_table"], current_mA=2.0, duty_pct=1.1,
                              n_macro=saved["n_macro"])
    assert saved["n_macro"] == 300
    assert row["loss_w_total"] == pytest.approx(summ["lost_w"], rel=1e-12)
    with h5py.File(out, "r") as f:
        snaps = list(f["particles"].keys())
        s_snap = float(f["particles"][snaps[0]].attrs["s"])
    assert len(snaps) == 1 and s_snap == pytest.approx(FOIL_S_MM)
    plain = _run_one_point_worker(base)
    for k in METRIC_KEYS:
        if k == "elapsed":
            continue
        assert plain[k] == row[k], k
