"""LEBT SCC popup (results_tab._SccPopup).

Real _open_popup seam, compute in both modes, pressure scan, DC gating,
export, and the confirm-gated Apply (QMessageBox mocked — offscreen
modals hang forever, per the house gotcha).
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import _SccPopup

_noop = lambda *a, **k: None                                 # noqa: E731


class _Cfg:
    species = "H-"
    current = 15.0   # mA — the popup passes this through to the driver


def _dc_lebt():
    lat = Lattice()
    lat.add(Drift(name="D1", length=300.0, aperture=60.0))
    lat.add(Solenoid(name="S1", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D2", length=900.0, aperture=60.0))
    lat.add(Solenoid(name="S2", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D3", length=300.0, aperture=60.0))
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5)
    init = dict(alpha_x=0.0, beta_x=0.6, emit_x=3.0,
                alpha_y=0.0, beta_y=0.6, emit_y=3.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EnvelopeSolver(lat, ref.copy(), init, current=15.0).run()
    return lat, res


def _state(lat, res):
    st = AppState()
    st.set_lattice(lat, path=None)
    st.set_beam_config(_Cfg())
    st.set_results(res)
    return st


def _wait(qapp, pop, timeout_s=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        qapp.processEvents()
        w = pop._worker
        if w is not None and not w.isRunning():
            qapp.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("SCC worker did not finish in time")


def test_open_and_compute_via_real_seam(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    lat, res = _dc_lebt()
    tab = ResultsTab(_state(lat, res), _noop, _noop, _noop)
    tab._open_popup("scc")
    pop = tab._popups.get("scc")
    assert isinstance(pop, _SccPopup)
    assert "Compute" in pop._info.text()          # DC guidance shown

    pop._gas.setCurrentText("N2")
    pop._btn.click()
    assert pop._worker is not None
    _wait(qapp, pop)
    assert pop._payload is not None and pop._payload["reason"] is None
    x, y = pop._c_fc.getData()
    assert x is not None and len(x) > 0
    assert np.all((np.asarray(y) >= 0) & (np.asarray(y) <= 1.0))
    assert "computed balance" in pop._info.text()
    assert pop._btn_export.isEnabled() and pop._btn_apply.isEnabled()
    pop.close()


def test_assumed_mode_and_scan(qapp):
    lat, res = _dc_lebt()
    pop = _SccPopup(None, _state(lat, res))
    pop.refresh(res)
    pop._mode.setCurrentIndex(1)                  # Assumed
    assert pop._eta.isEnabled()
    pop._eta.setValue(0.5)
    pop._btn.click()
    _wait(qapp, pop)
    assert "assumed eta = 0.5" in pop._info.text()

    pop._btn_scan.click()
    _wait(qapp, pop)
    assert "pressure scan" in pop._info.text()
    x, _ = pop._c_tau.getData()
    assert x is not None and len(x) == 13
    assert "pressure" in pop._plot_tau.getAxis("bottom").labelText

    # Compute after a scan must restore the z axes (adversarial F3).
    pop._btn.click()
    _wait(qapp, pop)
    assert "assumed eta" in pop._info.text()
    assert pop._plot_tau.getAxis("bottom").labelText == "z"
    pop.close()


def test_bunched_results_are_refused(qapp):
    class Bunched:
        continuous = False
    lat, _ = _dc_lebt()
    st = AppState()
    st.set_lattice(lat, path=None)
    st.set_beam_config(_Cfg())
    pop = _SccPopup(None, st)
    pop.refresh(Bunched())
    assert "Bunched-beam" in pop._info.text()
    pop.close()


def test_apply_inserts_cards_confirm_gated(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    import linac_gen_gui.interphase.tabs.results_tab as rt

    lat, res = _dc_lebt()
    st = _state(lat, res)
    pop = _SccPopup(None, st)
    pop.refresh(res)
    pop._btn.click()
    _wait(qapp, pop)
    n_cards = len(pop._payload["scc_cards"])
    assert n_cards > 0

    # Cancel first: nothing changes.
    monkeypatch.setattr(rt.QMessageBox, "question",
                        staticmethod(lambda *a, **k:
                                     QMessageBox.StandardButton.Cancel))
    pop._apply_cards()
    assert not any(isinstance(e, SpaceChargeComp) for e in lat.elements)

    # Confirm: cards inserted, results cleared (new-lattice semantics).
    monkeypatch.setattr(rt.QMessageBox, "question",
                        staticmethod(lambda *a, **k:
                                     QMessageBox.StandardButton.Ok))
    cards = list(pop._payload["scc_cards"])
    base_elems = [e for e in lat.elements
                  if not isinstance(e, SpaceChargeComp)]
    bounds = np.concatenate(([0.0], np.cumsum(
        [float(getattr(e, "length", 0.0) or 0.0)
         for e in base_elems]) * 1e-3))
    expect_slots = {int(np.argmin(np.abs(bounds - c["z_m"])))
                    for c in cards}

    pop._apply_cards()
    sccs = [e for e in lat.elements if isinstance(e, SpaceChargeComp)]
    # Colliding cards merge: one element per DISTINCT nearest boundary
    # (the adversarial round showed searchsorted placement quantised up
    # to a full element upstream and let colliding cards shadow each
    # other — this pins the corrected placement semantics).
    assert len(sccs) == len(expect_slots) <= n_cards
    assert all(0.0 <= e.factor <= 1.0 for e in sccs)
    # each inserted card sits exactly at one of the expected boundaries
    got_slots = set()
    pos = 0.0
    idx_non_scc = 0
    for e in lat.elements:
        if isinstance(e, SpaceChargeComp):
            got_slots.add(int(np.argmin(np.abs(bounds - pos))))
        else:
            pos += float(getattr(e, "length", 0.0) or 0.0) * 1e-3
            idx_non_scc += 1
    assert got_slots == expect_slots
    assert st.results is None                     # set_lattice drops results
    assert "applied" in pop._info.text()
    pop.close()


def _real_cfg(continuous=True):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="H-", energy=0.030, frequency=162.5,
                      current=15.0, duty_cycle=100.0, n_particles=200,
                      distribution="gaussian", cutoff=3.0,
                      emit_nx=0.2, alpha_x=0.0, beta_x=0.6,
                      emit_ny=0.2, alpha_y=0.0, beta_y=0.6,
                      emit_z=0.0, alpha_z=0.0, beta_z=1.0,
                      continuous=continuous)


def test_iterate_button_end_to_end(qapp):
    """Iterate runs its own envelopes off the GUI thread and lands a
    normal analysis payload plus a convergence line — the loaded lattice
    stays untouched (only confirm-gated Apply writes cards)."""
    lat, res = _dc_lebt()
    st = _state(lat, res)
    st.set_beam_config(_real_cfg())
    pop = _SccPopup(None, st)
    try:
        pop._start_iterate()
        assert pop._worker is not None
        _wait(qapp, pop)
        qapp.processEvents()
        a = pop._payload
        assert a is not None and a.get("reason") is None
        assert a["iterate"]["converged"]
        assert "self-consistent" in pop._info.text()
        assert pop._btn_iter.isEnabled()
        assert pop._btn_iter.text() == "Iterate"
        from linac_gen.elements.space_charge_comp import SpaceChargeComp
        assert not any(isinstance(e, SpaceChargeComp) for e in lat.elements)
        assert pop._btn_apply.isEnabled()          # damped set is applicable
    finally:
        pop.close()
        pop.deleteLater()
        qapp.processEvents()


def test_iterate_gates_on_dc_config_not_results(qapp):
    """Bunched Beam tab -> friendly refusal, no worker.  A DC config with
    NO session results must still start (Apply nulls results; the button
    has to survive that)."""
    lat, res = _dc_lebt()
    st = _state(lat, res)
    st.set_beam_config(_real_cfg(continuous=False))
    pop = _SccPopup(None, st)
    try:
        pop._start_iterate()
        assert pop._worker is None
        assert "DC beam" in pop._info.text()
    finally:
        pop.close()
        pop.deleteLater()
        qapp.processEvents()

    st2 = _state(lat, res)
    st2.set_beam_config(_real_cfg())
    st2.set_results(None)                          # the post-Apply state
    pop2 = _SccPopup(None, st2)
    try:
        pop2._start_iterate()
        assert pop2._worker is not None            # started regardless
        _wait(qapp, pop2)
        qapp.processEvents()
        assert pop2._payload is not None
        assert pop2._payload.get("reason") is None
    finally:
        pop2.close()
        pop2.deleteLater()
        qapp.processEvents()


def test_apply_after_iterate_inserts_the_damped_set(qapp, monkeypatch):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen.elements.space_charge_comp import SpaceChargeComp

    lat, res = _dc_lebt()
    st = _state(lat, res)
    st.set_beam_config(_real_cfg())
    pop = _SccPopup(None, st)
    try:
        pop._start_iterate()
        _wait(qapp, pop)
        qapp.processEvents()
        payload_factors = [c["factor"] for c in pop._payload["scc_cards"]]
        monkeypatch.setattr(
            rt.QMessageBox, "question",
            staticmethod(lambda *a, **k: rt.QMessageBox.StandardButton.Ok))
        pop._apply_cards()
        sccs = [e for e in lat.elements if isinstance(e, SpaceChargeComp)]
        assert sccs, "Apply after iterate inserted nothing"
        lo, hi = min(payload_factors), max(payload_factors)
        assert all(lo - 1e-9 <= e.factor <= hi + 1e-9 for e in sccs)
        assert st.results is None                  # set_lattice contract
    finally:
        pop.close()
        pop.deleteLater()
        qapp.processEvents()


def test_cleared_region_controls_flow_to_analysis(qapp):
    lat, res = _dc_lebt()
    st = _state(lat, res)
    pop = _SccPopup(None, st)
    try:
        assert not pop._cl_from.isEnabled()          # off until ticked
        pop._chk_cleared.setChecked(True)
        assert pop._cl_from.isEnabled()
        assert pop._cl_from.maximum() == len(lat.elements) - 1
        pop._cl_from.setValue(2)
        pop._cl_to.setValue(2)
        kw = pop._beam_kwargs()
        assert kw["cleared_regions"] == [[2, 2]]
        assert kw["cleared_residual"] == 0.0
        pop._compute()
        _wait(qapp, pop)
        qapp.processEvents()
        a = pop._payload
        assert a is not None and a.get("reason") is None
        assert a["cleared_regions"] == [[2, 2]]
        assert "cleared el 2–2" in pop._info.text()
    finally:
        pop.close()
        pop.deleteLater()
        qapp.processEvents()
