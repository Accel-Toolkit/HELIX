"""The GUI state traces curved field-map clusters (SUPERPOSE_MAP_OUT) for
the Beam tab's beam and tells the lattice views when a length changed."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt6")


def _bend_lattice():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from tests.elements.test_curved_field_map import RHO, _sector
    cl = _sector(30.0)
    lat = Lattice()
    lat.elements = [Drift("D0", 100.0), cl, Drift("D1", 100.0)]
    return lat, cl, RHO * math.radians(30.0)


def test_beam_change_retraces_and_redraws(qapp):
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.config import BeamConfig
    lat, cl, arc = _bend_lattice()
    st = AppState()
    seen = []
    st.lattice_changed.connect(seen.append)
    st.set_lattice(lat, None)                    # no beam yet: provisional
    assert cl.length_provisional
    n0 = len(seen)
    st.set_beam_config(BeamConfig(species="proton", energy=2.5,
                                  frequency=352.21))
    assert cl.length == pytest.approx(arc, rel=1e-12)
    assert len(seen) == n0 + 1                   # the views redraw once
    st.set_beam_config(BeamConfig(species="proton", energy=2.5,
                                  frequency=352.21))
    assert len(seen) == n0 + 1                   # nothing moved: no redraw
    st.set_beam_config(BeamConfig(species="proton", energy=2.6,
                                  frequency=352.21))
    assert len(seen) == n0 + 2
    assert cl.length != pytest.approx(arc, rel=1e-6)
