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


# ---------------------------------------------------------------------------
# named space-charge method selector
# ---------------------------------------------------------------------------
def _choose(tab, name):
    tab._sc_method.setCurrentIndex(tab._sc_method.findData(name))


def test_default_method_matches_the_default_widgets(tab):
    assert tab.sc_method_name() == "cic_fixed"
    assert tab._sc_method_summary.text()
    assert not tab._fixed_kernel.isEnabled()          # locked by the method
    assert tab._fixed_nx.isEnabled()                  # a param stays free


def test_choosing_a_method_sets_its_settings(tab):
    from linac_gen.pic import methods
    _choose(tab, "tsc_adaptive")
    sc = tab.current_sc_config(5.0)
    want = methods.base_fields("tsc_adaptive")
    for key in ("nx", "grid_extent", "kernel", "grid_mode", "green_kind",
                "sc_backend", "grid_card", "green_build", "reuse_every"):
        assert getattr(sc, key) == want[key], key
    assert sc.method == "tsc_adaptive"
    assert not tab._fixed_grid_mode.isEnabled()


def test_reuse_and_torch_methods(tab):
    _choose(tab, "cic_adaptive_reuse4")
    sc = tab.current_sc_config(5.0)
    assert sc.reuse_every == 4 and sc.method == "cic_adaptive_reuse4"
    _choose(tab, "differentiable_pic")
    sc = tab.current_sc_config(5.0)
    assert (sc.sc_backend, sc.grid_mode, sc.reuse_every) == ("torch",
                                                             "adaptive", 1)
    assert sc.method == "differentiable_pic"


def test_custom_unlocks_and_keeps_its_choice(tab):
    _choose(tab, "custom")
    assert tab._fixed_kernel.isEnabled() and tab._fixed_grid_mode.isEnabled()
    tab._fixed_kernel.setCurrentText("tsc")
    assert tab.sc_method_name() == "custom"
    assert not tab._fixed_reuse.isEnabled()           # needs an adaptive grid
    tab._fixed_grid_mode.setCurrentText("adaptive")
    assert tab._fixed_reuse.isEnabled()
    tab._fixed_reuse.setValue(2)
    assert tab.current_sc_config(5.0).method == "tsc_adaptive_reuse2"
    tab._fixed_grid_mode.setCurrentText("fixed")
    assert tab._fixed_reuse.value() == 1


def test_a_locked_setting_changed_by_code_shows_the_matching_method(tab):
    tab._fixed_kernel.setCurrentText("tsc")        # a script, a project load
    assert tab.sc_method_name() == "tsc_fixed"
    tab._fixed_sc_backend.setCurrentText("torch")  # the engine rule
    assert tab.sc_method_name() == "differentiable_pic"
    tab._fixed_sc_backend.setCurrentText("numpy")
    assert tab.sc_method_name() == "tsc_fixed"
    assert not tab._fixed_grid_mode.isEnabled()    # locked again


def test_apply_sc_config_shows_every_setting(tab):
    from linac_gen.core.config import SpaceChargeConfig
    _choose(tab, "tsc_adaptive_reuse2")
    cfg = SpaceChargeConfig(nx=40, grid_extent=8.0)        # classic base
    tab.apply_sc_config(cfg)
    sc = tab.current_sc_config(5.0)
    assert (sc.nx, sc.grid_extent, sc.kernel, sc.grid_mode, sc.reuse_every,
            sc.green_build, sc.grid_card) == (40, 8.0, "cic", "fixed", 1,
                                              "exact", "apply")
    assert tab.sc_method_name() == "cic_fixed"
    tab.apply_sc_config(cfg, method="custom")
    assert tab.sc_method_name() == "custom"



def test_choosing_an_adaptive_method_after_torch_keeps_the_adaptive_grid(tab):
    """Regression (review): the engine change used to restore a stale
    grid mode after the method had set adaptive."""
    _choose(tab, "custom")
    tab._fixed_grid_mode.setCurrentText("fixed")
    tab._fixed_sc_backend.setCurrentText("torch")
    _choose(tab, "cic_adaptive")
    sc = tab.current_sc_config(5.0)
    assert (sc.sc_backend, sc.grid_mode, sc.method) == ("numpy", "adaptive",
                                                        "cic_adaptive")
    _choose(tab, "differentiable_pic")
    _choose(tab, "tsc_adaptive_reuse4")
    sc = tab.current_sc_config(5.0)
    assert (sc.grid_mode, sc.reuse_every, sc.method) == (
        "adaptive", 4, "tsc_adaptive_reuse4")


def test_extent_keeps_three_decimals(tab):
    tab._fixed_ext.setValue(6.125)
    assert tab.current_sc_config(5.0).grid_extent == 6.125
