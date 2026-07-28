"""End-to-end test of the FailureStudy sweep runner (serial, envelope)."""
from __future__ import annotations

import numpy as np

from linac_gen.failures.failure_mode import FailureKind
from linac_gen.failures.scenario import enumerate_scenarios
from linac_gen.failures.study import FailureStudy
from linac_gen.io.tracewin_parser import parse_tracewin

LAT = "examples/ml_bayesopt/bo_demo.dat"      # 2 solenoids + 2 RF cavities
BEAM = {"energy": 2.5, "frequency": 162.5, "current": 5.0,
        "emit_nx": 0.30, "emit_ny": 0.30, "emit_z": 0.40,
        "alpha_x": -1.2, "beta_x": 0.32,
        "alpha_y": 2.0, "beta_y": 0.05, "beta_z": 10.0}


def test_study_single_serial_envelope():
    lat, _ = parse_tracewin(LAT)
    scn, n2c, names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                          combination="single")
    res = FailureStudy(LAT, beam_overrides=BEAM, mode="envelope").run(
        scn, names, n2c, combination="single", serial=True)

    assert res.baseline.get("ref_w_kin") is not None
    assert len(res.impacts) == len(scn) == len(names)
    # ranking is criticality-descending
    crit = [im.criticality for im in res.impacts]
    assert [crit[i] for i in res.ranking] == sorted(crit, reverse=True)
    # a cavity OFF must drop exit energy (lost acceleration)
    gap = next(im for im in res.impacts
               if im.scenario.element_names == ("GAP_001",))
    assert gap.d_energy_mev is not None and gap.d_energy_mev > 0.0


def test_inmemory_mp_metrics_not_none():
    """Regression: the in-memory MP forward pass must return REAL transmission
    and energy. run_mp_sim returns (recorder, beam); reading the tuple instead
    of the recorder gave None for everything (flat criticality + all beam_lost).
    """
    from linac_gen.cli.common import apply_beam_overrides
    from linac_gen.core.config import BeamConfig
    beam = BeamConfig()
    apply_beam_overrides(beam, {"energy": 2.5, "frequency": 162.5, "current": 0.0,
                                "n_particles": 200, "beta_x": 0.6, "beta_y": 0.6})
    lat, _ = parse_tracewin(LAT)
    scn, n2c, names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                          combination="single")
    res = FailureStudy(lattice=lat, beam_config=beam, mode="mp").run(
        scn, names, n2c, combination="single")
    assert res.baseline.get("transmission") is not None
    assert res.baseline.get("ref_w_kin") is not None
    # normalized RMS emittances now exposed for the table
    for k in ("emit_nx", "emit_ny", "emit_nz"):
        assert res.baseline.get(k) is not None, k
    assert all(im.metrics.get("transmission") is not None for im in res.impacts)
    assert all(im.metrics.get("emit_nz") is not None for im in res.impacts)
    # criticality must vary (not all collapsed to 0)
    assert len({round(im.criticality, 6) for im in res.impacts}) > 1


def test_recorder_metrics_schema_matches_scan_metrics():
    """Contract: in-memory recorder_metrics and path-mode _scan_metrics expose
    the SAME key schema, so any consumer sees one schema regardless of executor.
    (scan_pool._scan_metrics carries a "keep the two in sync" comment.)"""
    from linac_gen.core.config import BeamConfig
    from linac_gen.cli.common import apply_beam_overrides, run_envelope_sim
    from linac_gen.failures.study import recorder_metrics
    from linac_gen.parallel.scan_pool import _scan_metrics

    beam = BeamConfig()
    apply_beam_overrides(beam, {"energy": 2.5, "frequency": 162.5, "current": 0.0,
                                "n_particles": 200, "beta_x": 0.6, "beta_y": 0.6})
    lat, _ = parse_tracewin(LAT)
    rec = run_envelope_sim(lat, beam, "matrix")     # the API study.py uses
    rk = set(recorder_metrics(rec))
    sk = set(_scan_metrics(rec, 0.0))
    assert rk == sk, f"schema drift: only-in-recorder={rk - sk}, only-in-scan={sk - rk}"


def test_envelope_emit_nz_fallback():
    """Envelope records only geometric emittances; recorder_metrics must still
    expose εnz = βγ·εz_mmmrad (transmission stays None — envelope has no loss)."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.cli.common import apply_beam_overrides, run_envelope_sim
    from linac_gen.failures.study import recorder_metrics
    beam = BeamConfig()
    apply_beam_overrides(beam, {"energy": 2.5, "frequency": 162.5, "current": 0.0,
                                "n_particles": 200, "beta_x": 0.6, "beta_y": 0.6})
    lat, _ = parse_tracewin(LAT)
    rec = run_envelope_sim(lat, beam, "matrix")
    m = recorder_metrics(rec)
    assert m["emit_nx"] is not None and m["emit_ny"] is not None
    assert m["emit_nz"] is not None                       # the new fallback
    bg = rec.ref_beta[-1] * rec.ref_gamma[-1]
    assert abs(m["emit_nz"] - rec.emit_z_mmmrad[-1] * bg) < 1e-9
    assert m["transmission"] is None                      # envelope: no loss


def test_study_pairs_matrix():
    lat, _ = parse_tracewin(LAT)
    scn, n2c, names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                          combination="pairs")
    res = FailureStudy(LAT, beam_overrides=BEAM, mode="envelope").run(
        scn, names, n2c, combination="pairs", serial=True)
    n = len(names)
    assert res.pair_matrix.shape == (n, n)
    assert not np.isnan(res.pair_matrix).any()          # diag + off-diag filled
    assert np.allclose(res.pair_matrix, res.pair_matrix.T)  # symmetric
