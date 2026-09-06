"""Tests for ConvergenceTab.current_sc_config — the single source of
truth for the space-charge configuration used by BOTH the toolbar MP
run and the Errors-tab Monte Carlo studies."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture()
def tab(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.convergence_tab import ConvergenceTab
    t = ConvergenceTab(AppState())
    yield t
    t.deleteLater()


def test_zero_current_means_no_sc(tab):
    assert tab.current_sc_config(0.0) is None
    assert tab.current_sc_config(-1.0) is None


def test_builder_mirrors_widgets(tab):
    tab._fixed_nx.setValue(64)
    tab._fixed_ext.setValue(5.0)
    tab._fixed_green.setCurrentText("point")
    tab._fixed_kernel.setCurrentText("tsc")
    tab._fixed_dc_kernel.setCurrentText("gaussian")
    tab._fixed_csr.setChecked(True)

    sc = tab.current_sc_config(5.0)
    assert sc is not None
    assert (sc.nx, sc.ny, sc.nz) == (64, 64, 64)
    assert sc.grid_extent == 5.0
    assert sc.green_kind == "point"
    assert sc.kernel == "tsc"
    assert sc.dc_kernel == "gaussian"
    assert sc.csr_enabled is True
    assert sc.sc_backend == "numpy"


def test_torch_backend_forces_adaptive_grid(tab):
    tab._fixed_sc_backend.setCurrentText("torch")
    sc = tab.current_sc_config(5.0)
    assert sc.sc_backend == "torch"
    assert sc.grid_mode == "adaptive"


def test_torch_backend_falls_back_to_numpy_for_dc_beam(tab):
    tab._fixed_sc_backend.setCurrentText("torch")
    sc = tab.current_sc_config(5.0, continuous=True)
    assert sc.sc_backend == "numpy"


def test_current_step_config_mirrors_widgets(tab):
    """Single source for every run path: spinboxes + the single-push checkbox."""
    tab._fixed_step1.setValue(30.0)
    tab._fixed_step2.setValue(15.0)
    tab._drift_single_push.setChecked(False)
    cfg = tab.current_step_config()
    assert (cfg.integration_steps_per_metre, cfg.sc_steps_per_metre, cfg.drift_single_push) == (30.0, 15.0, False)
    tab._drift_single_push.setChecked(True)
    assert tab.current_step_config().drift_single_push is True


def test_sync_step_from_lattice_sets_the_checkbox(tab):
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.step_config import StepConfig
    lat = Lattice()
    lat.step_config = StepConfig(integration_steps_per_metre=100.0, sc_steps_per_metre=50.0,
                                 drift_single_push=False)
    tab._sync_step_from_lattice(lat)
    assert tab._drift_single_push.isChecked() is False
    lat.step_config = StepConfig()
    tab._sync_step_from_lattice(lat)
    assert tab._drift_single_push.isChecked() is True
