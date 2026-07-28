"""Transfer-matrix model overlay in the Results-tab dispersion popup.

The popup's statistical curve comes from the results Σ-matrix; the new
"Transfer-matrix model" checkbox runs `analysis.dispersion` in a
background worker and overlays the machine-optics dispersion.  Covered
here: the real toggle path (checkbox → worker → curves), agreement of
the drawn curve with a direct `dispersion_along_s` call, state-less
degradation, and lattice-change invalidation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from linac_gen.analysis.dispersion import dispersion_along_s
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import _DispersionPopup


@dataclass
class _MockBeamConfig:
    species: str = "proton"
    energy: float = 3.0
    frequency: float = 352.21
    current: float = 0.0
    disp_x: float = 0.0
    disp_xp: float = 0.0
    disp_y: float = 0.0
    disp_yp: float = 0.0


class _MockResults:
    pass


def _bend_lattice():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, aperture=20.0, n_steps=5))
    lat.add(Drift("D1", 200.0, aperture=50.0))
    lat.add(Dipole("B1", angle=10.0, rho=2000.0, aperture=50.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, aperture=20.0, n_steps=5))
    lat.add(Drift("D2", 200.0, aperture=50.0))
    return lat


def _synthetic_results(n: int = 30):
    res = _MockResults()
    res.s = np.linspace(0.0, 600.0, n)
    res.ref_beta = np.full(n, 0.08)
    res.ref_gamma = np.full(n, 1.0032)
    res.ref_w_kin = np.full(n, 3.0)
    res.mass_mev = 938.272
    sm = np.zeros((n, 6, 6))
    sm[:, 0, 5] = 0.5      # ⟨x·ΔW⟩ mm·MeV
    sm[:, 2, 5] = -0.2
    sm[:, 5, 5] = 1e-4     # ⟨ΔW²⟩ MeV²
    res.sigma_matrix = list(sm)     # the recorder stores a LIST of (6,6)
    return res


def _wait_for_worker(qapp, pop, timeout_s: float = 15.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        qapp.processEvents()
        w = pop._worker
        if w is not None and not w.isRunning() and pop._model_out is not None:
            return
        time.sleep(0.01)
    raise AssertionError("model-dispersion worker did not finish in time")


def test_model_overlay_via_real_toggle_path(qapp):
    state = AppState()
    lat = _bend_lattice()
    state.set_lattice(lat)
    state.set_beam_config(_MockBeamConfig())

    pop = _DispersionPopup(parent=None, state=state)
    pop.refresh(_synthetic_results())

    # statistical curves populated
    x, y = pop._cx.getData()
    assert x is not None and len(x) > 0

    # model curves empty until the checkbox is ticked
    xm, _ = pop._cx_m.getData()
    assert xm is None or len(xm) == 0

    pop._chk_model.setChecked(True)          # the real user path
    _wait_for_worker(qapp, pop)
    qapp.processEvents()                     # deliver finished_signal

    xm, ym = pop._cx_m.getData()
    assert xm is not None and len(xm) == len(lat.elements) + 1

    # drawn curve == direct core call
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    expect = dispersion_along_s(lat, ref)
    np.testing.assert_allclose(ym, expect["disp_x_m"], rtol=1e-12)
    _, ym_y = pop._cy_m.getData()
    np.testing.assert_allclose(ym_y, expect["disp_y_m"], rtol=1e-12)
    assert "drawn" in pop._lbl_status.text()  # complete walk → confirmation

    # untick clears the overlay but keeps the statistical curves
    pop._chk_model.setChecked(False)
    xm, _ = pop._cx_m.getData()
    assert xm is None or len(xm) == 0
    x, _ = pop._cx.getData()
    assert x is not None and len(x) > 0
    pop.close()


def test_stateless_popup_disables_checkbox(qapp):
    pop = _DispersionPopup(parent=None, state=None)
    assert not pop._chk_model.isEnabled()
    pop.refresh(_synthetic_results())        # statistical path unaffected
    x, _ = pop._cx.getData()
    assert x is not None and len(x) > 0
    pop.close()


def test_missing_lattice_reports_not_crashes(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    pop = _DispersionPopup(parent=None, state=state)
    pop._chk_model.setChecked(True)
    qapp.processEvents()
    assert "lattice" in pop._lbl_status.text().lower()
    xm, _ = pop._cx_m.getData()
    assert xm is None or len(xm) == 0
    pop.close()


def test_real_open_popup_path_wires_state(qapp):
    """End-to-end through the REAL entry seam: ResultsTab._open_popup
    must hand the popup the app state so the model checkbox is live."""
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    _noop = lambda *a, **k: None            # noqa: E731
    state = AppState()
    state.set_lattice(_bend_lattice())
    state.set_beam_config(_MockBeamConfig())
    state.set_results(_synthetic_results())
    tab = ResultsTab(state, _noop, _noop, _noop)
    tab._open_popup("dispersion")
    pop = tab._popups.get("dispersion")
    assert isinstance(pop, _DispersionPopup)
    assert pop._chk_model.isEnabled()
    x, _ = pop._cx.getData()                # statistical curves refreshed
    assert x is not None and len(x) > 0
    pop._chk_model.setChecked(True)
    _wait_for_worker(qapp, pop)
    qapp.processEvents()
    xm, _ = pop._cx_m.getData()
    assert xm is not None and len(xm) > 0
    pop.close()


def test_replacing_running_worker_parks_thread(qapp, monkeypatch):
    """Lattice change while the walk is mid-flight: the running QThread
    must be parked (a GC'd live QThread aborts the process), the stale
    result ignored, and the new lattice's curve drawn."""
    import linac_gen.analysis.dispersion as disp_mod
    from linac_gen_gui.interphase.tabs.results_tab import _ZOMBIE_WORKERS

    real = disp_mod.dispersion_along_s

    def slow(lattice, ref, **kw):
        time.sleep(0.4)
        return real(lattice, ref, **kw)

    monkeypatch.setattr(disp_mod, "dispersion_along_s", slow)

    state = AppState()
    state.set_lattice(_bend_lattice())
    state.set_beam_config(_MockBeamConfig())
    pop = _DispersionPopup(parent=None, state=state)
    pop._chk_model.setChecked(True)
    time.sleep(0.05); qapp.processEvents()
    first = pop._worker
    assert first is not None and first.isRunning()

    lat2 = _bend_lattice()
    lat2.add(Drift("D3", 300.0, aperture=50.0))
    state.set_lattice(lat2)                  # replaces mid-flight
    assert (first in _ZOMBIE_WORKERS) or (not first.isRunning())

    _wait_for_worker(qapp, pop)
    qapp.processEvents()
    xm, _ = pop._cx_m.getData()
    assert xm is not None and len(xm) == len(lat2.elements) + 1

    t0 = time.time()                          # zombie self-prunes
    while first in _ZOMBIE_WORKERS and time.time() - t0 < 5.0:
        qapp.processEvents(); time.sleep(0.01)
    assert first not in _ZOMBIE_WORKERS
    pop.close()


def test_lattice_change_invalidates_and_recomputes(qapp):
    state = AppState()
    state.set_lattice(_bend_lattice())
    state.set_beam_config(_MockBeamConfig())
    pop = _DispersionPopup(parent=None, state=state)
    pop._chk_model.setChecked(True)
    _wait_for_worker(qapp, pop)
    qapp.processEvents()
    first_key = pop._model_key
    assert first_key is not None

    # New lattice (one more drift) → old result invalidated, recomputed.
    lat2 = _bend_lattice()
    lat2.add(Drift("D3", 300.0, aperture=50.0))
    state.set_lattice(lat2)
    _wait_for_worker(qapp, pop)
    qapp.processEvents()
    assert pop._model_key != first_key
    xm, _ = pop._cx_m.getData()
    assert xm is not None and len(xm) == len(lat2.elements) + 1
    pop.close()
