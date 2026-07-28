"""Smoke test for ``examples/matching_demo.dat`` + ``matching_demo.py``.

Asserts the example lattice still solves cleanly:

* ``parse_tracewin`` ingests it without warnings,
* ``match`` converges (success flag, low cost),
* the matched gradient brings ``σ_x`` at the exit to the SET_SIZE
  target within numerical tolerance,
* the linked-group ADJUST cards leave both quads at the same
  *magnitude* of gradient (signs may differ — they're a focus/defocus
  pair).

Keeps the matcher entry point honest end-to-end.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import match
from linac_gen.tracking.envelope import EnvelopeSolver


REPO = Path(__file__).resolve().parent.parent.parent
DEMO_DAT = REPO / "examples" / "matching_demo.dat"


def _beam_cfg() -> BeamConfig:
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
        emit_z=0.3,    alpha_z=0.0, beta_z=10.0,
    )


def _run_envelope(lattice, cfg: BeamConfig):
    beam = create_beam(cfg, seed=42)
    bg = max(beam.ref.bg, 1e-9)
    initial = dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
    )
    solver = EnvelopeSolver(lattice, beam.ref, initial, current=cfg.current)
    return solver.run()


@pytest.fixture
def demo_lattice():
    assert DEMO_DAT.exists(), f"missing fixture: {DEMO_DAT}"
    lat, meta = parse_tracewin(str(DEMO_DAT))
    assert meta["warnings"] == [], meta["warnings"]
    return lat


# ---------------------------------------------------------------------------
def test_demo_parses_with_two_adjust_and_one_constraint(demo_lattice):
    """Baseline: the example file must parse two ADJUST cards on QUAD
    gradients and exactly one SET_SIZE constraint."""
    from linac_gen.elements.lattice_commands import Adjust, SetSize
    adjusts = [e for e in demo_lattice.elements if isinstance(e, Adjust)]
    sizes  = [e for e in demo_lattice.elements if isinstance(e, SetSize)]
    quads = [e for e in demo_lattice.elements if isinstance(e, Quadrupole)]
    assert len(adjusts) == 2
    assert len(sizes) == 1 and sizes[0].x_mm == 4.0
    assert len(quads) == 2
    # Both ADJUSTs share link group 1 → matcher should yield 1 column.
    assert all(a.link_group == 1 for a in adjusts)


def test_demo_matcher_converges(demo_lattice):
    """End-to-end: match the lattice and assert convergence + final σ_x."""
    cfg = _beam_cfg()
    lat = copy.deepcopy(demo_lattice)
    result = match(lat, cfg, max_iter=200)

    assert result.success, result.message
    # Linked group → exactly one optimiser column.
    assert result.x_final.shape == (1,)
    # Cost should be at noise level after convergence.
    assert result.cost < 1e-6, f"cost {result.cost:.3e} too high"

    # Forward simulation with the matched gradients should hit σ_x = 4 mm.
    res = _run_envelope(lat, cfg)
    sigma_x_end = float(res.sigma_x[-1])
    assert sigma_x_end == pytest.approx(4.0, abs=1e-3), sigma_x_end


def test_demo_link_group_keeps_quads_equal(demo_lattice):
    """Both quads share link group 1 → matcher must drive them to the
    same gradient (same column)."""
    cfg = _beam_cfg()
    lat = copy.deepcopy(demo_lattice)
    match(lat, cfg, max_iter=200)
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]
    assert len(quads) == 2
    # Both quads end up holding the linked column's value verbatim.
    assert quads[0].gradient == pytest.approx(quads[1].gradient)
