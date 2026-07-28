"""Tests for :mod:`linac_gen.matching.variables`."""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    Adjust, AdjustBeamCurrent, AdjustBeamEmit, AdjustBeamTwiss,
    AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.matching.variables import (
    MatchingConfigError, collect_variables,
)


def _bcfg() -> BeamConfig:
    return BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      n_particles=10, distribution="waterbag")


# ---------------------------------------------------------------------------
def test_resolves_quad_family_target():
    lat = Lattice()
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                    link_group=0, vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=50.0, gradient=12.0, aperture=5.0))
    vars_ = collect_variables(lat, _bcfg())
    assert len(vars_) == 1
    v = vars_[0]
    assert v.attr == "gradient"
    assert v.x0 == 12.0
    assert v.vmin == -30 and v.vmax == 30
    # Re-assignment lands on the element
    v.assign(7.5)
    assert lat.elements[1].gradient == 7.5


def test_resolves_quad_diag_index():
    lat = Lattice()
    lat.add(Drift("D_001", length=100.0, aperture=5.0))
    lat.add(Quadrupole("QUAD_001", length=50.0, gradient=10.0, aperture=5.0))
    # ADJUST 2 2 — second non-command element, parameter 2 (gradient)
    lat.add(Adjust("CMD1", target="2", param_idx=2,
                    link_group=0, vmin=-30, vmax=30))
    vars_ = collect_variables(lat, _bcfg())
    assert len(vars_) == 1
    assert vars_[0].target.name == "QUAD_001"
    assert vars_[0].attr == "gradient"


def test_unknown_target_warns_and_skips():
    """Unresolvable ADJUST target → warning + skipped variable, not an
    exception.  Lattices in active development often have ADJUST cards
    pointing at elements that haven't been added yet; the matcher
    should be tolerant of that."""
    lat = Lattice()
    lat.add(Adjust("CMD1", target="DOES_NOT_EXIST", param_idx=1,
                    link_group=0, vmin=0, vmax=1))
    with pytest.warns(UserWarning, match="DOES_NOT_EXIST"):
        out = collect_variables(lat, _bcfg())
    assert out == []


# ---------------------------------------------------------------------------
def test_link_groups_share_column():
    lat = Lattice()
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                    link_group=7, vmin=-30, vmax=30))
    lat.add(Quadrupole("QUAD_001", length=50.0, gradient=12.0, aperture=5.0))
    lat.add(Adjust("CMD2", target="QUAD", param_idx=2,
                    link_group=7, vmin=-30, vmax=30))
    lat.add(Quadrupole("QUAD_002", length=50.0, gradient=-12.0, aperture=5.0))
    vars_ = collect_variables(lat, _bcfg())
    assert len(vars_) == 2
    assert vars_[0].link_group == 7 and vars_[1].link_group == 7


# ---------------------------------------------------------------------------
def test_adjust_steerer_resolves():
    lat = Lattice()
    lat.add(AdjustSteerer("CMD1", diag_n=1, vmax=0.05, first_step=1e-3))
    lat.add(Steerer("STEER_001", bx_l=0.0, by_l=0.0))
    vars_ = collect_variables(lat, _bcfg())
    assert len(vars_) == 2  # bx_l and by_l
    attrs = sorted(v.attr for v in vars_)
    assert attrs == ["bx_l", "by_l"]


def test_adjust_steerer_bx_only():
    lat = Lattice()
    lat.add(AdjustSteererBx("CMD1", diag_n=1, vmax=0.05))
    lat.add(Steerer("STEER_001", bx_l=0.0, by_l=0.0))
    vars_ = collect_variables(lat, _bcfg())
    assert len(vars_) == 1
    assert vars_[0].attr == "bx_l"


# ---------------------------------------------------------------------------
def test_adjust_beam_twiss_flag_expansion():
    cfg = _bcfg()
    lat = Lattice()
    # Adjust α_x and β_x (1 1), couple α_y to β_x (2), skip rest (0 0 0)
    lat.add(AdjustBeamTwiss("CMD1", 1, 1, 1, 2, 0, 0, 0))
    vars_ = collect_variables(lat, cfg)
    attrs = [v.attr for v in vars_]
    assert "alpha_x" in attrs and "beta_x" in attrs and "alpha_y" in attrs
    assert "beta_y" not in attrs


def test_adjust_beam_emit_flags():
    cfg = _bcfg()
    lat = Lattice()
    lat.add(AdjustBeamEmit("CMD1", 1, 1, 0, 1))
    vars_ = collect_variables(lat, cfg)
    attrs = [v.attr for v in vars_]
    assert "emit_nx" in attrs and "emit_z" in attrs
    assert "emit_ny" not in attrs


def test_adjust_beam_current_emits_one():
    cfg = _bcfg()
    lat = Lattice()
    lat.add(AdjustBeamCurrent("CMD1", 1, 1))
    vars_ = collect_variables(lat, cfg)
    assert len(vars_) == 1 and vars_[0].attr == "current"
