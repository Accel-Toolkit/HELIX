"""Floor-plan survey tile/popup: analytic geometry anchors + BTL integration.

The survey walks the reference orbit in 3-D; anchors below are hand-derived
(not survey-derived) so a symmetric sign/convention bug cannot cancel out.
"""
from __future__ import annotations

from tests.dataguard import needs, require  # noqa: E402

import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
BTL = REPO / "examples" / "pipii" / "btl" / "btl.dat"


def _mklat(*elements):
    class _Lat:
        pass
    lat = _Lat()
    lat.elements = list(elements)
    return lat


def _bend(angle_deg, rho_mm, hv=0):
    from linac_gen.elements.dipole import Dipole
    return Dipole("B", angle=angle_deg, rho=rho_mm, hv=hv)


def _drift(L_mm):
    from linac_gen.elements.drift import Drift
    return Drift("D", length=L_mm, aperture=0.0)


def test_single_90deg_horizontal_bend():
    """Analytic: drift a, then 90° bend of radius rho, then drift b.

    Endpoint: z = a + rho, x = rho + b (exit tangent along +x).
    """
    from linac_gen_gui.interphase.tabs.results_tab import _survey_lattice
    a, rho, b = 1000.0, 2000.0, 500.0            # mm
    sv = _survey_lattice(_mklat(_drift(a), _bend(90.0, rho), _drift(b)),
                         ds_mm=1.0)
    assert abs(sv["z"][-1] - (a + rho) * 1e-3) < 1e-4
    assert abs(sv["x"][-1] - (rho + b) * 1e-3) < 1e-4
    assert abs(sv["y"]).max() < 1e-12            # purely horizontal
    # path length = a + arc + b
    assert abs(sv["s"][-1] - (a + rho * math.pi / 2 + b) * 1e-3) < 1e-6


def test_single_vertical_bend_sign():
    """Positive vertical angle curves toward +y; x untouched."""
    from linac_gen_gui.interphase.tabs.results_tab import _survey_lattice
    rho = 3000.0
    sv = _survey_lattice(_mklat(_bend(30.0, rho, hv=1), _drift(1000.0)),
                         ds_mm=1.0)
    th = math.radians(30.0)
    y_exp = (rho * (1 - math.cos(th)) + 1000.0 * math.sin(th)) * 1e-3
    z_exp = (rho * math.sin(th) + 1000.0 * math.cos(th)) * 1e-3
    assert abs(sv["y"][-1] - y_exp) < 1e-4
    assert abs(sv["z"][-1] - z_exp) < 1e-4
    assert abs(sv["x"]).max() < 1e-12
    assert len(sv["dip_v"]) == 1 and len(sv["dip_h"]) == 0


def test_negative_horizontal_bend_sign():
    from linac_gen_gui.interphase.tabs.results_tab import _survey_lattice
    sv = _survey_lattice(_mklat(_bend(-45.0, 2000.0)), ds_mm=1.0)
    assert sv["x"][-1] < 0.0                     # curves toward -x


@pytest.mark.skipif(not BTL.exists(), reason="BTL example not present")
def test_btl_integration():
    from linac_gen_gui.interphase.tabs.results_tab import _survey_lattice
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(BTL))
    sv = _survey_lattice(lat)
    # Σ|θ| from the raw cards (grep census): the old btl.dat mixes 8 mains
    # at 6.5637426° with 24 rounded to 6.564°, plus 2×1.409° + 2×1.494°.
    expected = 8 * 6.5637426 + 24 * 6.564 + 2 * 1.409 + 2 * 1.494
    assert abs(sv["bend_deg"] - expected) < 1e-6
    assert len(sv["dip_h"]) == 32 and len(sv["dip_v"]) == 4
    assert abs(sv["s"][-1] - 304.9063) < 1e-2    # path length = lattice length
    # 210° of horizontal bending folds the line back: z-extent << s
    assert sv["z"].max() - sv["z"].min() < 0.75 * sv["s"][-1]


@needs("examples/pipii/btl/btl.dat")
def test_tile_series_and_popup(qapp):
    """Real entry path: series fn from _build_series_fns + popup refresh."""
    from linac_gen_gui.interphase.tabs.results_tab import (
        _build_series_fns, _FloorPlanPopup)
    from linac_gen.io.tracewin_parser import parse_tracewin

    class _State:
        results = None
        beam_config = None
    st = _State()
    st.lattice, _ = parse_tracewin(str(BTL))
    xs, ys = _build_series_fns(st)["floorplan"](None)
    assert xs.size == ys.size and xs.size > 100
    pop = _FloorPlanPopup(None, st)
    pop.refresh(None)
    assert "304.9" in pop._info.text() and "32 horizontal" in pop._info.text()
    pop.close()
    # no lattice → clean placeholder, no exception
    st2 = _State(); st2.lattice = None
    pop2 = _FloorPlanPopup(None, st2)
    pop2.refresh(None)
    assert "No lattice" in pop2._info.text()
    pop2.close()


def test_negative_drift_steps_backward():
    """MAD-style negative drifts must translate backward, not be skipped
    (skipping inflates path length — bit us on the v0703 conversion)."""
    from linac_gen_gui.interphase.tabs.results_tab import _survey_lattice
    sv = _survey_lattice(_mklat(_drift(1000.0), _drift(-204.288),
                                _drift(500.0)), ds_mm=1.0)
    assert abs(sv["s"][-1] - (1000.0 - 204.288 + 500.0) * 1e-3) < 1e-9
    assert abs(sv["z"][-1] - (1000.0 - 204.288 + 500.0) * 1e-3) < 1e-9
