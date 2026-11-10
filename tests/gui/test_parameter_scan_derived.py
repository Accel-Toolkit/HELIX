"""The Parameter Scan offers the parameters an element is built from, not those that follow from them."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from linac_gen_gui.interphase.dialogs.parameter_scan import _scannable_params  # noqa: E402

from linac_gen.elements.dipole import Dipole  # noqa: E402
from linac_gen.elements.ncells import NCells  # noqa: E402


def test_the_scan_offers_an_ncells_field_but_not_its_length():
    c = NCells("NC", mode=1, n_cells=4, beta_g=0.2, eot_v_per_m=1.0e6, theta_s_deg=-30.0, aperture_mm=15.0,
               p_flag=2, frequency_mhz=325.0)
    names = _scannable_params(c)
    assert {"eot_v_per_m", "beta_g", "theta_s_deg", "frequency_mhz"} <= set(names)
    assert "length" not in names and "beta_s" not in names


def test_the_scan_offers_a_pipes_length_but_not_a_bends():
    """A bend's length follows from its angle and radius; a bend with no angle is a pipe and takes a length."""
    bend, pipe = Dipole("B", angle=10.0, rho=800.0), Dipole("P", angle=0.0, rho=1.0e9)
    pipe.length = 300.0
    assert {"angle", "rho"} <= set(_scannable_params(bend)) and "length" not in _scannable_params(bend)
    assert "length" in _scannable_params(pipe)
