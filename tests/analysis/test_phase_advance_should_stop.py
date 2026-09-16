"""The coupled per-cell walks can be cancelled — and a cancel RAISES.

Added 2026-09-08 with the results-tab worker that runs these walks off the
GUI thread.  Both functions wrap per-cell / per-element work in
``except Exception`` ladders, and ``OperationCancelled`` is an Exception:
polled in the wrong place a cancel would be swallowed into ``continue`` and
a NaN-filled dict returned — and cached by the popup as valid.  These pin
that a set stop hook raises out of both, that ``should_stop=None`` is the
historical call (bit-identical output), and that a never-firing hook changes
nothing.
"""
import warnings

import numpy as np
import pytest

from linac_gen.cli.common import build_ref, load_lattice
from linac_gen.core.cancelled import OperationCancelled
from linac_gen.core.config import BeamConfig
from linac_gen.analysis.period_detect import detect_periods
from linac_gen.analysis.phase_advance import (
    coupled_beam_phase_advance_per_cell_via_M, coupled_phase_advance_per_cell,
    run_phase_probe,
)
from linac_gen.cli.common import _envelope_initial


@pytest.fixture(scope="module")
def coupled_line():
    lat = load_lattice("examples/solenoid_channel.dat")
    cfg = BeamConfig(species="H-", energy=2.1226695, frequency=162.5, current=5.0)
    ref = build_ref(cfg)
    period = detect_periods(lat)[0]
    probe = run_phase_probe(lat, ref, _envelope_initial(cfg, ref), current=cfg.current)
    legacy = run_phase_probe(lat, ref, _envelope_initial(cfg, ref), current=cfg.current,
                             phase_probe=False)
    assert not getattr(legacy, "element_maps_dep", None)
    return lat, ref, period, probe, legacy


def _eq(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_cpc_stop_raises_and_never_returns(coupled_line):
    lat, ref, period, *_ = coupled_line
    with pytest.raises(OperationCancelled):
        coupled_phase_advance_per_cell(lat, ref.copy(), period, should_stop=lambda: True)


def test_cpc_none_and_false_hooks_are_the_historical_call(coupled_line):
    lat, ref, period, *_ = coupled_line
    a = coupled_phase_advance_per_cell(lat, ref.copy(), period)
    b = coupled_phase_advance_per_cell(lat, ref.copy(), period, should_stop=None)
    c = coupled_phase_advance_per_cell(lat, ref.copy(), period, should_stop=lambda: False)
    _eq(a, b); _eq(a, c)
    assert np.isfinite(a["mu_I_deg"]).all()


def test_via_m_stop_raises_on_both_branches(coupled_line):
    lat, ref, period, probe, legacy = coupled_line
    with pytest.raises(OperationCancelled):
        coupled_beam_phase_advance_per_cell_via_M(lat, ref.copy(), period, probe,
                                                  should_stop=lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # the legacy-branch UserWarning
        with pytest.raises(OperationCancelled):
            coupled_beam_phase_advance_per_cell_via_M(lat, ref.copy(), period, legacy,
                                                      should_stop=lambda: True)


def test_via_m_stop_fires_mid_walk_on_the_legacy_branch(coupled_line):
    """A hook that trips after a few elements must still raise, not return."""
    lat, ref, period, _probe, legacy = coupled_line
    calls = {"n": 0}
    def later():
        calls["n"] += 1
        return calls["n"] > 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(OperationCancelled):
            coupled_beam_phase_advance_per_cell_via_M(lat, ref.copy(), period, legacy,
                                                      should_stop=later)
    assert calls["n"] > 3


def test_via_m_none_hook_is_the_historical_call(coupled_line):
    lat, ref, period, probe, legacy = coupled_line
    for res in (probe, legacy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = coupled_beam_phase_advance_per_cell_via_M(lat, ref.copy(), period, res)
            b = coupled_beam_phase_advance_per_cell_via_M(lat, ref.copy(), period, res,
                                                          should_stop=lambda: False)
        _eq(a, b)
