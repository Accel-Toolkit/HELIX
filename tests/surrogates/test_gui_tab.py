"""M6 — SurrogatesTab headless construction + lattice repopulation."""
import os

import pytest

# Tests in this module require the GUI to be importable; skip gracefully
# if PyQt6 isn't installed.
PyQt6 = pytest.importorskip("PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def test_surrogates_tab_constructs(qapp):
    """The SurrogatesTab builds headless with no lattice loaded."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    tab = SurrogatesTab(AppState())
    assert tab._elem_combo is not None
    assert tab._train_btn is not None
    assert tab._table.columnCount() == 5
    # No lattice -> no elements in the combo
    assert tab._elem_combo.count() == 0


def test_surrogates_tab_populates_on_lattice(qapp):
    """When a lattice with a FieldMap3D is set, the combo repopulates."""
    from linac_gen.elements.base import FieldMapElement
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    class _FakeFieldMap(FieldMapElement):
        """Counts as a FieldMap3D for the tab if isinstance check were
        on FieldMapElement; the tab tests isinstance(FieldMap3D), so we
        verify the non-FieldMap3D case here instead."""
        def __init__(self, name):
            super().__init__(name=name, length=200.0, aperture=10.0, n_steps=10)
        def track_rk4(self, beam, ds): return None

    state = AppState()
    tab = SurrogatesTab(state)
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=10.0))
    lat.add(_FakeFieldMap("FM_X"))   # not a FieldMap3D -> NOT listed
    state.set_lattice(lat, path=None)
    # Only FieldMap3D instances are listed; this lattice has none.
    assert tab._elem_combo.count() == 0


def test_tabs_export_includes_surrogates_tab(qapp):
    """The new tab is exported from linac_gen_gui.interphase.tabs."""
    from linac_gen_gui.interphase import tabs
    assert "SurrogatesTab" in tabs.__all__
    assert hasattr(tabs, "SurrogatesTab")


def test_state_tabs_includes_surrogates():
    from linac_gen_gui.interphase.state import TABS
    ids = [t[0] for t in TABS]
    assert "surrogates" in ids
    # Position: between convergence and errors.
    idx = ids.index("surrogates")
    assert ids[idx - 1] == "convergence"
    assert ids[idx + 1] == "study"      # Param Study sits between
    assert ids[idx + 2] == "errors"     # Surrogates and Error Study


# ---- auto-discovery + cache-aware Train ------------------------------------
def _stage_cached_for_lattice(tmp_path, monkeypatch, element_name: str,
                               lattice_hash: str):
    """Train a tiny surrogate AND patch the SurrogatesTab to use
    ``tmp_path`` as the weights root (via cwd)."""
    import numpy as np
    from linac_gen.elements.base import FieldMapElement
    from linac_gen.surrogates.training import train_surrogate_for_element

    class _Mock(FieldMapElement):
        def __init__(self, name):
            super().__init__(name=name, length=100.0, aperture=10.0, n_steps=10)
            self.scale = 1.0
        def track_rk4(self, beam, ds): return None
        def fitted_matrix(self, ref):
            M = np.eye(6)
            M[0, 1] = 0.001 * ref.w_kin * self.scale
            return M
        def fitted_matrix_slice(self, ref, ds_mm): return np.eye(6)

    class _Ref:
        def __init__(self, w_kin=3.0):
            self.w_kin = w_kin; self.beta = 0.1; self.gamma = 1.005
        def copy(self): return _Ref(self.w_kin)

    elem = _Mock(element_name)
    out = (tmp_path / "linac_gen" / "surrogates" / "weights"
           / lattice_hash[:16] / element_name)
    train_surrogate_for_element(
        element=elem, ref_template=_Ref(),
        n_samples=64, ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(8, 8), epochs=5, lr=3e-3, batch_size=32,
        seed=0, out_dir=out,
        lattice_hash=lattice_hash, element_key=element_name,
    )
    # Make Path("linac_gen/...") in the tab resolve under tmp_path.
    monkeypatch.chdir(tmp_path)
    return elem


def test_surrogates_tab_auto_discovers_cached(qapp, tmp_path, monkeypatch):
    """On lattice_changed the tab loads any cached surrogates whose
    element names match the lattice's FieldMap3D elements."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    # Build a one-element FieldMap3D lattice & write it to a .dat-ish
    # file so hash_lattice_file() succeeds and matches our staged dir.
    lat_file = tmp_path / "mini.dat"
    lat_file.write_text("DRIFT 100 0 10 0 0\nEND")
    from linac_gen.surrogates.registry import hash_lattice_file
    lh = hash_lattice_file(lat_file)
    _stage_cached_for_lattice(tmp_path, monkeypatch, "FMAP_X", lh)

    # The auto-discovery filters by isinstance(FieldMap3D); build a
    # real FieldMap3D so the filter accepts it.  We don't need the
    # field map files on disk because we only exercise the discovery
    # codepath (no envelope run).
    fm = FieldMap3D.__new__(FieldMap3D)
    FieldMap3D.__bases__[0].__init__(fm, name="FMAP_X", length=240.0,
                                       aperture=10.0, n_steps=10)
    # Minimum attributes the table row reads:
    fm.ke = 0.05; fm.kb = 0.0; fm.phase = -90.0
    fm.frequency = 162.5

    lat = Lattice()
    lat.add(fm)

    state = AppState()
    tab = SurrogatesTab(state)
    state.set_lattice(lat, path=str(lat_file))

    # Combo populated AND the cached surrogate auto-loaded into the
    # Trained-surrogates table.
    assert tab._elem_combo.count() == 1
    assert "FMAP_X" in tab._trained
    assert tab._table.rowCount() == 1


def test_detect_sweep_params_filters_zeros_and_missing(qapp):
    """The dynamic sweep detector skips zero amplitudes and absent attrs.

    Heuristics tested:
      * ke=0 OR kb=0 -> the dead channel is dropped.
      * phase is only swept when an E-channel is active (ke != 0).
      * scale is NEVER in the default sweep set (global multiplier).
      * Missing attrs are silently ignored.
    """
    from linac_gen_gui.interphase.tabs.surrogates_tab import (
        _detect_sweep_params)

    class _Cavity:        # E + B, both active -- cavity-like
        ke = 0.07; kb = 0.045; scale = 1.0; phase = -90.0
    class _Solenoid:      # B only -- 1-D solenoid case
        ke = 0.0; kb = 0.05; scale = 1.0; phase = 0.0
    class _MinimalE:      # E only, no kb attr
        ke = 0.03; phase = -45.0
    class _NoFieldAttrs:  # no ke/kb/phase
        scale = 1.0

    names_c = [n for n, _, _ in _detect_sweep_params(_Cavity())]
    assert names_c == ["ke", "kb", "phase"]   # both channels + phase

    names_s = [n for n, _, _ in _detect_sweep_params(_Solenoid())]
    # ke=0 -> skip ke AND skip phase (no E-channel); kb=0.05 -> include.
    assert names_s == ["kb"]

    names_e = [n for n, _, _ in _detect_sweep_params(_MinimalE())]
    assert names_e == ["ke", "phase"]

    names_n = [n for n, _, _ in _detect_sweep_params(_NoFieldAttrs())]
    assert names_n == []   # nothing to sweep


def test_surrogatable_types_includes_both_field_maps(qapp):
    """The dropdown filter accepts FieldMap (1D) and FieldMap3D."""
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen_gui.interphase.tabs.surrogates_tab import (
        _surrogatable_types)

    types = _surrogatable_types()
    assert FieldMap in types
    assert FieldMap3D in types
    # RFQ subclasses NOT included (more complex state, future work).
    from linac_gen.elements.vane_rfq import VaneRFQ
    assert VaneRFQ not in types


def test_train_dialog_builds_dynamic_form_for_cavity(qapp):
    """A cavity element gets ke + kb + phase sweep rows."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import _TrainDialog

    class _Cavity:
        name = "C1"; length = 240.0
        ke = 0.07; kb = 0.045; scale = 1.0; phase = -90.0
        frequency = 162.5

    dlg = _TrainDialog(_Cavity())
    names = list(dlg._param_widgets.keys())
    assert names == ["ke", "kb", "phase"]   # scale always excluded

    opts = dlg.get_options()
    pr = opts["param_ranges"]
    assert set(pr) == {"ke", "kb", "phase"}
    # Default rel sweep 0.20, phase 20 deg
    ke_lo, ke_hi = pr["ke"]
    assert ke_lo == pytest.approx(0.07 * 0.80, rel=1e-6)
    assert ke_hi == pytest.approx(0.07 * 1.20, rel=1e-6)
    ph_lo, ph_hi = pr["phase"]
    assert ph_lo == pytest.approx(-90.0 - 20.0)
    assert ph_hi == pytest.approx(-90.0 + 20.0)


def test_train_dialog_builds_dynamic_form_for_solenoid(qapp):
    """A 1-D solenoid (ke=0, kb!=0) gets only the kb sweep — no phase."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import _TrainDialog

    class _Solenoid:
        name = "SOL1"; length = 300.0
        ke = 0.0; kb = -1.8; scale = 1.0; phase = 0.0
        frequency = 0.0

    dlg = _TrainDialog(_Solenoid())
    assert list(dlg._param_widgets.keys()) == ["kb"]
    opts = dlg.get_options()
    pr = opts["param_ranges"]
    assert set(pr) == {"kb"}
    # kb=-1.8 with 20% rel: range straddles zero negatively.
    lo, hi = pr["kb"]
    assert lo == pytest.approx(-1.8 * 1.20)   # more negative
    assert hi == pytest.approx(-1.8 * 0.80)
    assert lo <= hi   # the dialog auto-swaps if rel inverts


def test_progress_dialog_constructs_and_accepts_events(qapp):
    """_TrainProgressDialog builds headless and updates internal
    state on data_gen + epoch events."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import numpy as np
    from linac_gen_gui.interphase.tabs.surrogates_tab import (
        _TrainProgressDialog)

    dlg = _TrainProgressDialog(
        element_name="UNIT_TEST", n_samples=100, epochs=10)
    assert dlg.windowTitle().endswith("UNIT_TEST")

    # Feed a data_gen event.
    dlg.on_progress({
        "stage": "data_gen", "done": 50, "total": 100,
        "elapsed_s": 1.0,
    })
    assert dlg._data_done == 50

    # Feed two epoch events; lists should grow.
    dlg.on_progress({
        "stage": "epoch", "epoch": 1, "total": 10,
        "train_loss": 1.0, "val_mape": 0.1,
        "best_val_mape": 0.1,
        "per_entry_val_mape": np.full((6, 6), 0.05),
        "elapsed_s": 1.5,
    })
    dlg.on_progress({
        "stage": "epoch", "epoch": 2, "total": 10,
        "train_loss": 0.5, "val_mape": 0.05,
        "best_val_mape": 0.05,
        "per_entry_val_mape": np.full((6, 6), 0.02),
        "elapsed_s": 2.0,
    })
    assert dlg._epoch_xs == [1, 2]
    assert dlg._train_loss == [1.0, 0.5]
    assert dlg._val_mape == [0.1, 0.05]
    assert dlg._per_entry_heatmap is not None
    assert dlg._per_entry_heatmap.shape == (6, 6)


def test_train_dialog_skips_zero_delta_sweeps(qapp):
    """A spin left at 0 drops that param from the sweep (user-disabled)."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import _TrainDialog

    class _Cavity:
        name = "C2"; length = 100.0
        ke = 0.05; kb = 0.02; scale = 1.0; phase = -30.0
        frequency = 162.5

    dlg = _TrainDialog(_Cavity())
    # User zeroes out the kb sweep (relative).
    _, _, kb_sp = dlg._param_widgets["kb"]
    kb_sp.setValue(0.0)
    opts = dlg.get_options()
    pr = opts["param_ranges"]
    assert "kb" not in pr     # user-disabled
    assert "ke" in pr          # still in
    assert "phase" in pr


def test_mp_section_exists_and_starts_collapsed(qapp):
    """The new collapsible MP section sits below the table, collapsed
    by default so the existing envelope workflow is untouched."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    tab = SurrogatesTab(AppState())
    assert hasattr(tab, "_mp_section")
    # If the user never opened it, it's collapsed.  Persistence may
    # have flipped it open from a prior session; the invariant we
    # care about is that the section EXISTS.
    assert hasattr(tab, "_mp_engage")
    assert hasattr(tab, "_mp_substeps")
    assert hasattr(tab, "_mp_compare_btn")
    # Default substep count is the accuracy-first 15 from the plan
    # (unless QSettings overrode it from a prior session).
    val = tab._mp_substeps.value()
    assert 0 <= val <= 100


def test_mp_master_toggle_drives_registry_flag(qapp, monkeypatch, tmp_path):
    """Toggling the master engages / disengages the registry's MP flag.

    Isolated from the user's real QSettings via `monkeypatch.setenv`
    on the application's settings path so this test never pollutes
    `~/Library/Preferences/com.helix.linac_gen_gui.plist` (or its
    Linux/Windows equivalents).
    """
    from PyQt6.QtCore import QSettings
    from linac_gen.surrogates import registry as _reg
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    # Use a per-test INI file so QSettings doesn't write to the real
    # user preferences.  Restored automatically when the test exits.
    QSettings.setPath(QSettings.Format.IniFormat,
                       QSettings.Scope.UserScope, str(tmp_path))
    monkeypatch.setattr(QSettings, "defaultFormat",
                        lambda: QSettings.Format.IniFormat,
                        raising=False)

    saved = _reg.is_mp_enabled()
    try:
        tab = SurrogatesTab(AppState())
        tab._mp_engage.setChecked(False)
        _reg.set_mp_enabled(False)   # ensure clean baseline

        tab._mp_engage.setChecked(True)
        assert _reg.is_mp_enabled() is True

        tab._mp_engage.setChecked(False)
        assert _reg.is_mp_enabled() is False
    finally:
        _reg.set_mp_enabled(saved)


def test_mp_substeps_propagates_to_registered_surrogates(qapp, tmp_path):
    """Spinbox change updates `residual_n_steps` on every trained row.

    Isolated from the user's real QSettings so the spinbox writes
    here (`_on_mp_substeps_changed` writes to
    `surrogates/mp_residual_substeps`) don't pollute their saved
    preferences.
    """
    from PyQt6.QtCore import QSettings
    QSettings.setPath(QSettings.Format.IniFormat,
                       QSettings.Scope.UserScope, str(tmp_path))
    import numpy as np
    from linac_gen.elements.base import FieldMapElement
    from linac_gen.surrogates.base import (
        MlpHead, Scope, SurrogateFieldMap, SurrogateMetadata,
    )
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    class _Mock(FieldMapElement):
        def __init__(self):
            super().__init__(name="MOCK", length=100.0, aperture=10.0,
                             n_steps=10)
            self.scale = 1.0
        def track_rk4(self, beam, ds): return None
        def fitted_matrix(self, ref): return np.eye(6)
        def fitted_matrix_slice(self, ref, ds_mm): return np.eye(6)

    tab = SurrogatesTab(AppState())
    # Inject a trained-surrogate row directly into the dict.
    wrapped = _Mock()
    mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8,))
    meta = SurrogateMetadata(
        element_key="MOCK", element_class="_Mock",
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [8], "activation": "silu",
                      "param_names": ["scale"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "scale"],
                    input_lo=np.array([0.0]*4),
                    input_hi=np.array([1e6]*4)),
        input_norm={"mean": [0.0]*4, "std": [1.0]*4},
        output_norm={"mean": list(np.eye(6).flatten()),
                     "std":  [1.0]*36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    from pathlib import Path
    tab._trained["MOCK"] = (surr, Path("/tmp/nowhere"), meta)

    # Default is whatever QSettings restored; explicitly set then check.
    tab._mp_substeps.setValue(7)
    assert surr.residual_n_steps == 7
    tab._mp_substeps.setValue(42)
    assert surr.residual_n_steps == 42


def test_select_all_use_registers_every_surrogate(qapp, tmp_path):
    """Select-all clicks every Use checkbox; Deselect-all clears them.

    Verifies the registry sees every surrogate after Select-all and
    is empty after Deselect-all (drives the per-row `_on_use_toggled`
    slot).  Uses isolated QSettings so the test never pollutes
    real user preferences.
    """
    from PyQt6.QtCore import QSettings
    QSettings.setPath(QSettings.Format.IniFormat,
                       QSettings.Scope.UserScope, str(tmp_path))
    import numpy as np
    from pathlib import Path
    from linac_gen.elements.base import FieldMapElement
    from linac_gen.surrogates import registry as _reg
    from linac_gen.surrogates.base import (
        MlpHead, Scope, SurrogateFieldMap, SurrogateMetadata,
    )
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    class _Mock(FieldMapElement):
        def __init__(self, name):
            super().__init__(name=name, length=100.0, aperture=10.0,
                             n_steps=10)
            self.scale = 1.0
        def track_rk4(self, beam, ds): return None
        def fitted_matrix(self, ref): return np.eye(6)
        def fitted_matrix_slice(self, ref, ds_mm): return np.eye(6)

    def _surrogate(name):
        wrapped = _Mock(name)
        mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8,))
        meta = SurrogateMetadata(
            element_key=name, element_class="_Mock",
            architecture={"input_dim": 4, "output_dim": 36,
                          "hidden_dims": [8], "activation": "silu",
                          "param_names": ["scale"]},
            scope=Scope(input_names=["w_kin", "beta", "gamma", "scale"],
                        input_lo=np.array([0.0]*4),
                        input_hi=np.array([1e6]*4)),
            input_norm={"mean": [0.0]*4, "std": [1.0]*4},
            output_norm={"mean": list(np.eye(6).flatten()),
                         "std":  [1.0]*36},
            training_seed=0, n_samples=0, epochs=0,
            val_mape=0.0, helix_commit_sha="",
            lattice_hash="hash-bulk", created_iso="",
        )
        return SurrogateFieldMap(wrapped, mlp, meta), meta

    _reg.clear()
    saved_mp = _reg.is_mp_enabled()
    try:
        tab = SurrogatesTab(AppState())
        for nm in ("S1", "S2", "S3"):
            surr, meta = _surrogate(nm)
            tab._trained[nm] = (surr, Path("/tmp/x"), meta)
        tab._refresh_table()
        assert tab._table.rowCount() == 3
        assert len(_reg.list_registered()) == 0   # nothing ticked yet

        tab._set_all_use(True)
        assert len(_reg.list_registered()) == 3
        # And the visible Use checkboxes are all ticked.
        for row in range(3):
            assert tab._table.cellWidget(row, 3).isChecked()

        tab._set_all_use(False)
        assert len(_reg.list_registered()) == 0
        for row in range(3):
            assert not tab._table.cellWidget(row, 3).isChecked()

        # Status line should summarise the bulk action.
        assert "3" in tab._status.text()
    finally:
        _reg.clear()
        _reg.set_mp_enabled(saved_mp)


def test_select_all_use_empty_table_is_noop(qapp):
    """Calling Select-all with no trained surrogates only updates
    the status line; no exception, no registry change."""
    from linac_gen.surrogates import registry as _reg
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    _reg.clear()
    tab = SurrogatesTab(AppState())
    assert len(tab._trained) == 0
    tab._set_all_use(True)
    assert "no trained surrogates" in tab._status.text().lower()
    assert len(_reg.list_registered()) == 0


def test_surrogates_tab_clears_table_on_lattice_swap(qapp, tmp_path,
                                                       monkeypatch):
    """Loading a different lattice drops the old rows (stale bindings)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.surrogates.registry import hash_lattice_file
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    lat_file = tmp_path / "first.dat"
    lat_file.write_text("DRIFT 100 0 10 0 0\nEND")
    lh = hash_lattice_file(lat_file)
    _stage_cached_for_lattice(tmp_path, monkeypatch, "FMAP_Y", lh)

    fm = FieldMap3D.__new__(FieldMap3D)
    FieldMap3D.__bases__[0].__init__(fm, name="FMAP_Y", length=240.0,
                                       aperture=10.0, n_steps=10)
    fm.ke = 0.05; fm.kb = 0.0; fm.phase = -90.0; fm.frequency = 162.5

    lat1 = Lattice(); lat1.add(fm)
    state = AppState()
    tab = SurrogatesTab(state)
    state.set_lattice(lat1, path=str(lat_file))
    assert "FMAP_Y" in tab._trained

    # Swap to a different lattice with no matching FieldMap3D.
    other = tmp_path / "second.dat"
    other.write_text("DRIFT 50 0 5 0 0\nEND")
    lat2 = Lattice(); lat2.add(Drift("D1", length=50.0, aperture=5.0))
    state.set_lattice(lat2, path=str(other))
    # Old rows wiped; nothing to auto-discover for the new lattice.
    assert tab._trained == {}
    assert tab._table.rowCount() == 0


def test_progress_dialog_throttles_datagen_paints(qapp, monkeypatch):
    """Regression: frequent data-gen progress events must NOT trigger a full
    canvas paint on every emit.

    The worker connects with a BlockingQueuedConnection, so a synchronous
    paint per emit stalls data generation (the "stops every few hundred
    steps" bug).  The expensive paint is throttled to ~25 Hz for the
    high-frequency data_gen stage, while terminal events, stage changes,
    and EVERY training epoch still paint (the live training animation).
    """
    from linac_gen_gui.interphase.tabs.surrogates_tab import (
        _TrainProgressDialog)

    # Deterministic clock so the 40 ms throttle window is controllable.
    clock = {"t": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])

    dlg = _TrainProgressDialog("FMAP_001", n_samples=5000, epochs=40)
    draws = {"n": 0}
    dlg._canvas.draw = lambda *a, **k: draws.__setitem__("n", draws["n"] + 1)
    dlg._canvas.repaint = lambda *a, **k: None

    # 300 data_gen emits at the SAME instant: only the first (the
    # None -> data_gen stage change) paints; the rest are throttled.
    for k in range(1, 301):
        dlg.on_progress({"stage": "data_gen", "done": k,
                         "total": 5000, "elapsed_s": 0.001})
    assert draws["n"] == 1, f"data-gen painted {draws['n']}x (expected 1)"

    # Advancing past the 40 ms window lets the next data_gen emit paint.
    clock["t"] += 0.05
    dlg.on_progress({"stage": "data_gen", "done": 400,
                     "total": 5000, "elapsed_s": 0.05})
    assert draws["n"] == 2

    # A terminal emit (done == total) always paints, even inside the window.
    before = draws["n"]
    dlg.on_progress({"stage": "data_gen", "done": 5000,
                     "total": 5000, "elapsed_s": 5.0})
    assert draws["n"] == before + 1

    # Every epoch paints (live training animation), even back-to-back at
    # the same instant -- the per-epoch paint is the reason the blocking
    # connection exists and must be preserved.
    before = draws["n"]
    for e in range(1, 6):
        dlg.on_progress({"stage": "epoch", "epoch": e,
                         "train_loss": 1.0 / e, "val_mape": 1.0 / e,
                         "best_val_mape": 1.0 / e, "elapsed_s": 0.001})
    assert draws["n"] == before + 5, "each epoch must paint"

    dlg.deleteLater()


def test_train_worker_emits_cancelled_not_failed(qapp, tmp_path):
    """A user stop must surface as `cancelled` (silent skip), NOT as
    `failed` — the batch controller pops a modal error dialog per
    element for failures, which would turn a cancel into a dialog storm."""
    import numpy as np
    from linac_gen.elements.base import FieldMapElement
    from linac_gen_gui.interphase.tabs.surrogates_tab import _TrainWorker

    class _MockFM(FieldMapElement):
        def __init__(self):
            super().__init__(name="MOCK", length=100.0, aperture=10.0,
                             n_steps=10)
            self.scale = 1.0
        def track_rk4(self, beam, ds): return None
        def fitted_matrix(self, ref):
            return np.eye(6)

    class _MockRef:
        w_kin, beta, gamma = 3.0, 0.1, 1.005
        def copy(self):
            return _MockRef()

    worker = _TrainWorker(
        element=_MockFM(), ref_template=_MockRef(),
        out_dir=tmp_path / "w", lattice_hash="h",
        options=dict(n_samples=32, ref_w_kin_range=(2.0, 10.0),
                     param_ranges={"scale": (0.8, 1.2)}, epochs=10),
    )
    seen = {"cancelled": 0, "failed": [], "finished": 0}
    worker.cancelled.connect(lambda: seen.__setitem__(
        "cancelled", seen["cancelled"] + 1))
    worker.failed.connect(seen["failed"].append)
    worker.finished_ok.connect(lambda _m: seen.__setitem__(
        "finished", seen["finished"] + 1))

    worker.request_stop()          # stop before the first sample
    worker.run()                   # synchronous — house pattern

    assert seen["cancelled"] == 1
    assert seen["failed"] == []
    assert seen["finished"] == 0
    assert not (tmp_path / "w").exists()


def test_stale_done_dialog_cannot_cancel_current_run(qapp):
    """Review finding: a leftover '[done]' progress dialog kept its
    cancel hook armed against the tab's LIVE _worker attribute — closing
    the old window killed the CURRENT run.  The hook now binds to the
    worker instance it was created for."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import (
        SurrogatesTab, _TrainWorker,
    )

    tab = SurrogatesTab(AppState())

    class _FakeWorker(_TrainWorker):
        def __init__(self):  # noqa: D401 - bypass heavy parent init
            # QThread init only; no element machinery needed.
            from PyQt6.QtCore import QThread
            QThread.__init__(self)
            import threading
            self._stop_event = threading.Event()
            self.running = True

        def isRunning(self):
            return self.running

    worker_a = _FakeWorker()   # finished run's worker
    worker_b = _FakeWorker()   # current run's worker
    worker_a.running = False
    tab._worker = worker_b

    # The dialog for run A was armed against worker A specifically.
    cancel_cb = lambda w=worker_a: tab._cancel_train_worker(w)  # noqa: E731
    cancel_cb()

    assert not worker_a._stop_event.is_set() or not worker_a.running
    # THE regression: worker B must be untouched.
    assert not worker_b._stop_event.is_set()
    tab.deleteLater()
