"""GUI wiring of the field-map Sampling switch: apply-on-change (all run
types inherit), dirty-marking, and construction-time sync."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")

from linac_gen.elements import field_map_3d as fm3

pytestmark = pytest.mark.skipif(not fm3.kernel_available(),
                                reason="compiled kernel not built")


@pytest.fixture(autouse=True)
def _restore_switch():
    before = fm3.fused_kernel_enabled()
    env = os.environ.get("LINAC_GEN_FIELDMAP_KERNEL")
    yield
    fm3.use_fused_kernel(before)
    if env is None:
        os.environ.pop("LINAC_GEN_FIELDMAP_KERNEL", None)
    else:
        os.environ["LINAC_GEN_FIELDMAP_KERNEL"] = env


@pytest.fixture()
def tab(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.convergence_tab import ConvergenceTab
    t = ConvergenceTab(AppState())
    yield t
    t.deleteLater()


def test_combo_change_applies_immediately(tab):
    """Selecting scipy/kernel takes effect at once — envelope runs,
    scans and error studies then inherit it without any run-start hook."""
    assert tab._fieldmap_sampling.currentText() == "kernel"
    assert fm3.fused_kernel_enabled() is True
    tab._fieldmap_sampling.setCurrentText("scipy")
    assert fm3.fused_kernel_enabled() is False
    assert os.environ["LINAC_GEN_FIELDMAP_KERNEL"] == "0"
    tab._fieldmap_sampling.setCurrentText("kernel")
    assert fm3.fused_kernel_enabled() is True
    assert os.environ["LINAC_GEN_FIELDMAP_KERNEL"] == "1"


def test_construction_syncs_global_to_combo(tab, qapp):
    """A process started with the env kill-switch still ends up with the
    global matching the combo after the tab constructs."""
    fm3.use_fused_kernel(False)          # simulate env=0 startup state
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.convergence_tab import ConvergenceTab
    t2 = ConvergenceTab(AppState())
    try:
        assert t2._fieldmap_sampling.currentText() == "kernel"
        assert fm3.fused_kernel_enabled() is True
    finally:
        t2.deleteLater()


def test_combo_marks_project_dirty(tab):
    calls = []
    tab.state.mark_project_dirty = lambda: calls.append(1)
    tab._fieldmap_sampling.setCurrentText("scipy")
    assert calls, "Sampling change must mark the project dirty"
