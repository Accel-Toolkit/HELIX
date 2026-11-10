"""An element can refuse a value (a derived or whole-number attribute).  An edit of several knobs through the GUI is then
undone as a whole, and the assistant hears the element's reason, not that the GUI was busy."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _state(qapp):
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.ncells import NCells
    lat = Lattice()
    lat.add(NCells("NC", mode=1, n_cells=4, beta_g=0.2, eot_v_per_m=1.0e6, theta_s_deg=-30.0, aperture_mm=15.0,
                   p_flag=2, frequency_mhz=325.0))
    lat.add(Drift("D1", 200.0))
    st = AppState()
    st.set_lattice(lat, "<test>")
    st.set_beam_config(BeamConfig(species="proton", energy=19.0, frequency=325.0, current=0.0, duty_cycle=100.0,
                                  n_particles=100, distribution="gaussian", cutoff=3.0, emit_nx=0.25, alpha_x=0.0,
                                  beta_x=1.0, emit_ny=0.25, alpha_y=0.0, beta_y=1.0, emit_z=0.30, alpha_z=0.0,
                                  beta_z=5.0))
    return st


def test_a_macro_command_is_undone_as_a_whole_when_an_element_refuses(qapp):
    from linac_gen_gui.interphase.commands import MacroCommand, ParamChangeCommand
    st = _state(qapp)
    nc, d1 = st.lattice.elements
    length = nc.length
    cmd = MacroCommand([ParamChangeCommand(d1, "length", 200.0, 300.0),
                        ParamChangeCommand(nc, "eot_v_per_m", 1.0e6, 1.2e6),
                        ParamChangeCommand(nc, "length", length, 400.0)])
    with pytest.raises(AttributeError):
        cmd.do(st.lattice)
    assert d1.length == 200.0 and nc.eot_v_per_m == 1.0e6 and nc.length == length


def test_the_assistant_hears_why_an_edit_was_refused(qapp, tmp_path):
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    st = _state(qapp)
    nc, d1 = st.lattice.elements
    ctx = _make_context(st, str(tmp_path))
    with pytest.raises(RuntimeError, match="follows from"):
        ctx.apply_param_changes([(d1, "length", 300.0), (nc, "length", 400.0)], label="Assistant edit")
    assert d1.length == 200.0 and len(st.bus._undo) == 0
    with pytest.raises(RuntimeError, match="whole number"):
        ctx.apply_param_changes([(nc, "n_cells", 5.5)], label="Assistant edit")
    assert nc.n_cells == 4 and len(st.bus._undo) == 0
