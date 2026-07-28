"""Dipole-field lattice tiles (bend_b, bend_bl) in the Results tab.

A BEND card stores geometry (θ, ρ); the field only exists once a beam
rigidity is folded in, B = Bρ/|ρ|.  These tests pin both rigidity
branches (entrance beam-config preview and per-element from a run), the
BTL invariant B·L = Bρ·θ, and the popup smoke path.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BTL = REPO / "examples" / "pipii" / "btl" / "btl.dat"

pytestmark = pytest.mark.skipif(
    not BTL.exists(), reason="BTL example lattice not present")

# BTL main-dipole geometry straight off the BEND cards (mm / deg).
RHO_MAIN_MM = 21384.55
THETA_MAIN_DEG = 6.5637426
N_MAINS = 32
N_BENDS_TOTAL = 36          # 32 mains + 4 short bends


class _Cfg:
    species = "H-"
    energy = 800.0
    frequency = 162.5
    current = 4.84
    alpha_x = 0.0
    beta_x = 5.0
    alpha_y = 0.0
    beta_y = 5.0


class _State:
    def __init__(self, lattice, results=None):
        self.lattice = lattice
        self.results = results
        self.beam_config = _Cfg()


@pytest.fixture(scope="module")
def btl_lattice():
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(BTL))
    return lat


def _brho(w_kin_mev: float) -> float:
    """Independent anchor: Bρ [T·m] for H⁻ (HELIX mass convention)."""
    from linac_gen.core.particle import H_MINUS
    gamma = 1.0 + w_kin_mev / H_MINUS.mass
    bg = math.sqrt(gamma * gamma - 1.0)
    return bg * H_MINUS.mass / 299.792458


def test_bend_b_series_entrance_rigidity(btl_lattice):
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns
    fns = _build_series_fns(_State(btl_lattice))
    out = fns["bend_b"](None)
    assert out is not None
    xs, ys = out
    assert ys.size == N_BENDS_TOTAL
    assert xs.size == ys.size
    expected = _brho(800.0) / (RHO_MAIN_MM * 1e-3)
    mains = ys[np.isclose(ys, expected, rtol=1e-3)]
    assert mains.size == N_MAINS
    # per-magnet ρ rounding in the .dat gives ~2e-5 relative spread
    assert np.allclose(mains, expected, rtol=1e-4)


def test_bend_bl_equals_brho_theta(btl_lattice):
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns
    fns = _build_series_fns(_State(btl_lattice))
    xs, ys = fns["bend_bl"](None)
    assert ys.size == N_BENDS_TOTAL
    expected = _brho(800.0) * math.radians(THETA_MAIN_DEG)   # B·L = Bρ·θ
    mains = ys[np.isclose(ys, expected, rtol=1e-3)]
    assert mains.size == N_MAINS
    # fixed angle ⇒ B·L identical across the family (up to .dat rounding)
    assert np.ptp(mains) / np.mean(mains) < 1e-4


def test_no_beam_config_gives_placeholder(btl_lattice):
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns
    state = _State(btl_lattice)
    state.beam_config = None
    fns = _build_series_fns(state)
    assert fns["bend_b"](None) is None
    assert fns["bend_bl"](None) is None


def test_results_rigidity_branch_wins(btl_lattice):
    """Per-element γ from a run must override the entrance energy."""
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns
    from linac_gen.core.particle import H_MINUS

    class _Res:
        pass

    res = _Res()
    n = len(btl_lattice.elements)
    gamma = 1.0 + 1000.0 / H_MINUS.mass          # pretend a 1 GeV run
    res.ref_gamma = np.full(8, gamma)
    res.element_exit_idx = [7] * n               # len must match elements
    fns = _build_series_fns(_State(btl_lattice, results=res))
    xs, ys = fns["bend_b"](None)
    expected = _brho(1000.0) / (RHO_MAIN_MM * 1e-3)
    mains = ys[np.isclose(ys, expected, rtol=1e-3)]
    assert mains.size == N_MAINS


def test_results_len_mismatch_falls_back(btl_lattice):
    """Stale results (exit-idx length ≠ element count) → entrance βγ."""
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns

    class _Res:
        ref_gamma = np.full(8, 3.0)
        element_exit_idx = [7, 7, 7]             # wrong length on purpose

    fns = _build_series_fns(_State(btl_lattice, results=_Res()))
    xs, ys = fns["bend_b"](None)
    expected = _brho(800.0) / (RHO_MAIN_MM * 1e-3)
    assert np.isclose(ys, expected, rtol=1e-3).sum() == N_MAINS


def test_popup_smoke(qapp, btl_lattice):
    """Popup construction + refresh through the real value_fn/transform."""
    from linac_gen_gui.interphase.tabs.results_tab import (
        _LatticeParamPopup, _bend_field_T)
    from linac_gen.elements.dipole import Dipole

    state = _State(btl_lattice)
    pop = _LatticeParamPopup(
        None, state,
        title="Dipole field  —  |B| [T] per BEND",
        element_types=(Dipole,), attr=None,
        value_fn=_bend_field_T(state),
        ylabel="|B|", yunits="T",
        type_name="dipole (BEND)",
    )
    pop.refresh(None)            # must not raise; data comes from lattice
    pop.close()

    pop_bl = _LatticeParamPopup(
        None, state,
        title="Dipole ∫B·dl  —  B·L [T·m] per BEND",
        element_types=(Dipole,), attr=None,
        value_fn=_bend_field_T(state),
        transform=lambda el, v: v * (float(el.length) * 1e-3),
        ylabel="∫B·dl", yunits="T·m",
        type_name="dipole (BEND)",
    )
    pop_bl.refresh(None)
    pop_bl.close()
