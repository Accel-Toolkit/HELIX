"""StepConfig.rfq_stats_accelerated: particles that leave a time-domain RFQ
unaccelerated stay in the beam and the transmission, but the recorded
statistics leave them out from the record plane where they fall behind the
bunch (a handful ~2 MeV below the bunch otherwise dominate the energy
spread)."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.rfq_td.chain import RfqTdDriver


def _states(P, n):
    return np.zeros((P, n, 6), np.float32), np.ones((P, n), bool)


def test_the_plane_from_which_a_particle_is_left_out():
    P, w_in = 6, 0.03
    w_ref = np.linspace(w_in, 2.0, P)
    gain = w_ref - w_in
    s, crossed = _states(P, 8)
    s[3:, 1, 5] = -gain[3:]          # 1: stays at the entrance energy from 3
    s[2, 2, 5] = -gain[2]            # 2: behind at 2, back at 3, behind from 4
    s[4:, 2, 5] = -gain[4:]
    s[2:, 3, 5] = -gain[2:]          # 3: falls behind at 2, lost after 3
    crossed[4:, 3] = False
    s[:, 4, 5] = -0.4 * gain         # 4: slow, but more than half the gain
    s[2:4, 5, 5] = -0.2 * gain[2:4]  # 5: slips out of the bucket slowly:
    s[4:, 5, 5] = -0.9 * gain[4:]    #    left out where it first lags 10 %
    s[0, 6, 5] = -0.3                # 6: lost at once (the reference had
    crossed[1:, 6] = False           #    not gained a tenth of its gain yet)
    crossed[3:, 7] = False           # 7: lost in the bunch, not behind it
    out = RfqTdDriver._behind_planes(s, crossed, w_ref, w_in)
    assert list(out) == [-1, 3, 4, 2, -1, 2, -1, -1]


def test_an_rfq_that_hardly_accelerates_leaves_nobody_out():
    P, w_in = 6, 0.03
    s, crossed = _states(P, 2)
    s[:, 1, 5] = -1e-3
    w_ref = np.full(P, w_in * 1.05)
    assert (RfqTdDriver._behind_planes(s, crossed, w_ref, w_in) == -1).all()


def _beam(n=400, seed=3):
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    cfg = BeamConfig(species="proton", energy=2.5, frequency=352.21,
                     current=0.0, n_particles=n)
    return create_beam(cfg, seed=seed)


def test_the_recorder_leaves_marked_particles_out_of_the_statistics():
    from linac_gen.diagnostics.moments import compute_moments
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    beam = _beam()
    beam.particles[:3, 5] -= 2.0                 # three far behind in energy
    beam.lost[10] = True
    beam.stats_excluded = np.zeros(beam.lost.shape, bool)
    beam.stats_excluded[:3] = True
    rec = DiagnosticRecorder()
    rec.record(beam, 0.0)
    keep = beam.alive_mask & ~beam.stats_excluded
    m = compute_moments(beam.particles[keep])
    assert rec.sigma_w[-1] == m["sigma_w"]
    assert rec.sigma_x[-1] == m["sigma_x"]
    assert rec.transmission[-1] == pytest.approx(100.0 * 399 / 400)
    # unmarked: every alive particle counts, as before
    beam.stats_excluded = None
    rec.record(beam, 1.0)
    full = compute_moments(beam.alive_particles)
    assert rec.sigma_w[-1] == full["sigma_w"] > m["sigma_w"]


def test_the_setting_from_config_project_and_command_line():
    import argparse
    import importlib
    from linac_gen.cli.common import rfq_settings
    from linac_gen.core.step_config import StepConfig
    assert StepConfig().rfq_stats_accelerated is True
    assert StepConfig(rfq_stats_accelerated=0).rfq_stats_accelerated is False
    assert rfq_settings({}, {})["rfq_stats_accelerated"] is True
    assert rfq_settings({"rfq_stats_accelerated": False},
                        {})["rfq_stats_accelerated"] is False
    assert rfq_settings({"rfq_stats_accelerated": "off"},
                        {})["rfq_stats_accelerated"] is False
    # the command line wins over the project
    assert rfq_settings({"rfq_stats_accelerated": False},
                        {"rfq_stats": "accelerated"}
                        )["rfq_stats_accelerated"] is True
    assert rfq_settings({}, {"rfq_stats": "all"})["rfq_stats_accelerated"] \
        is False
    with pytest.raises(ValueError, match="rfq_stats"):
        rfq_settings({}, {"rfq_stats": "some"})
    for module in ("run", "scan"):
        p = argparse.ArgumentParser()
        importlib.import_module(f"linac_gen.cli.{module}").add_arguments(p)
        base = ["deck.dat"] + (["--vary", "q.gradient=1:2:1"]
                               if module == "scan" else [])
        assert p.parse_args(base).rfq_stats is None
        assert p.parse_args(base + ["--rfq-stats", "all"]).rfq_stats == "all"
        with pytest.raises(SystemExit):
            p.parse_args(base + ["--rfq-stats", "some"])
