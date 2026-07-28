"""End-to-end test for ``examples/twiss_matching_demo.dat``.

Inverse Twiss match: variables = input α_x/β_x/α_y/β_y, constraints =
exit α_x/β_x/α_y/β_y locked to a target.  Asserts:

* the .dat parses without warnings,
* one ADJUST_BEAM_TWISS card → exactly four optimiser variables,
* one SET_TWISS card → constraint with kα_x=kβ_x=kα_y=kβ_y=1,
* the matcher converges (success, low cost),
* a forward envelope run with the matched input lands the exit Twiss
  on (α_x, β_x, α_y, β_y) = (0, 1.5, 0, 1.5) within a tight tolerance,
* the matcher mutates the BeamConfig in place — the four input Twiss
  attributes change.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.lattice_commands import AdjustBeamTwiss, SetTwiss
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import collect_variables, collect_constraints, match
from linac_gen.tracking.envelope import EnvelopeSolver


REPO = Path(__file__).resolve().parent.parent.parent
DEMO = REPO / "examples" / "twiss_matching_demo.dat"


def _baseline_cfg() -> BeamConfig:
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
        emit_z=0.3,    alpha_z=0.0, beta_z=10.0,
    )


def _run_envelope(lattice, cfg):
    beam = create_beam(cfg, seed=42)
    bg = max(beam.ref.bg, 1e-9)
    initial = dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
    )
    return EnvelopeSolver(lattice, beam.ref, initial, current=cfg.current).run()


@pytest.fixture
def demo_lattice():
    assert DEMO.exists(), f"missing fixture: {DEMO}"
    lat, meta = parse_tracewin(str(DEMO))
    assert meta["warnings"] == [], meta["warnings"]
    return lat


# ---------------------------------------------------------------------------
def test_demo_has_one_adjust_beam_twiss_and_one_set_twiss(demo_lattice):
    abts = [e for e in demo_lattice.elements
            if isinstance(e, AdjustBeamTwiss)]
    sts  = [e for e in demo_lattice.elements if isinstance(e, SetTwiss)]
    assert len(abts) == 1
    assert len(sts) == 1
    abt = abts[0]; st = sts[0]
    # Flag tail expanded into 4 enabled axes (α_x, β_x, α_y, β_y).
    assert abt.flags == [1, 1, 1, 1, 0, 0]
    # SET_TWISS at end with k-flags only on x and y Twiss.
    assert (st.kax, st.kbx, st.kay, st.kby) == (1, 1, 1, 1)
    assert (st.kaz, st.kbz) == (0, 0)
    assert (st.alpha_x, st.beta_x) == pytest.approx((0.0, 1.5))
    assert (st.alpha_y, st.beta_y) == pytest.approx((0.0, 1.5))


def test_collect_variables_yields_four_beam_twiss_dofs(demo_lattice):
    cfg = _baseline_cfg()
    vars_ = collect_variables(demo_lattice, cfg)
    attrs = sorted(v.attr for v in vars_)
    assert attrs == ["alpha_x", "alpha_y", "beta_x", "beta_y"]
    # All four target the same BeamConfig instance.
    assert all(v.target is cfg for v in vars_)


def test_collect_constraints_emits_one_set_twiss(demo_lattice):
    constraints = collect_constraints(demo_lattice)
    labels = [c.label for c in constraints]
    assert any("SET_TWISS" in lbl for lbl in labels)
    assert len(constraints) == 1


def test_matcher_drives_exit_twiss_to_target(demo_lattice):
    cfg = _baseline_cfg()
    lat = copy.deepcopy(demo_lattice)
    result = match(lat, cfg, max_iter=400)

    assert result.success, result.message
    assert result.cost < 1e-10, f"cost {result.cost:.3e} too high"

    # The matcher should have written four variables to BeamConfig.
    # Verify the BeamConfig in place and that a forward run hits the target.
    assert cfg.alpha_x != 0.0 or cfg.beta_x != 1.0
    assert cfg.alpha_y != 0.0 or cfg.beta_y != 1.0

    res = _run_envelope(lat, cfg)
    assert res.alpha_x[-1] == pytest.approx(0.0, abs=1e-3)
    assert res.beta_x[-1]  == pytest.approx(1.5, abs=1e-3)
    assert res.alpha_y[-1] == pytest.approx(0.0, abs=1e-3)
    assert res.beta_y[-1]  == pytest.approx(1.5, abs=1e-3)
