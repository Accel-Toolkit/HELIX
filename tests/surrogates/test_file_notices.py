"""What is wrong with a weight file is said at every load, not only at the first one of a process.

Python shows a warning once per line it is attributed to, and every load of a file is
attributed to the same line, so the GUI's status bar heard a stale or species-less file
once per session: load it, clear the list, load it again, and it was not there.
"""
import dataclasses
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates.training import (
    discover_cached_surrogates,
    load_surrogate,
    save_surrogate,
    train_surrogate_for_element,
)


class _Map(FieldMapElement):
    """A map whose longitudinal determinant is ``dz``: 2 is what a stale file's targets hold."""

    def __init__(self, name="MAP", dz=1.0):
        super().__init__(name=name, length=100.0, aperture=10.0, n_steps=10)
        self.scale = 1.0
        self.dz = dz

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        M = np.eye(6)
        M[0, 1] = 0.001 * ref.w_kin * self.scale
        M[4, 4] = self.dz
        return M

    def fitted_matrix_slice(self, ref, ds_mm):
        return np.eye(6)


class _Ref:
    """No clock, so the file records none: a legacy file, which is the one checked for staleness."""

    def __init__(self, w_kin=3.0):
        self.w_kin, self.beta, self.gamma = w_kin, 0.1, 1.005

    def copy(self):
        return _Ref(self.w_kin)


def _saved(root, name="MAP", *, dz=1.0, species="", mass=0.0):
    mlp, meta = train_surrogate_for_element(
        element=_Map(name, dz), ref_template=_Ref(), n_samples=32,
        ref_w_kin_range=(2.0, 10.0), param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(4,), epochs=2, lr=3e-3, batch_size=16, seed=0,
        out_dir=root / "trained" / name, lattice_hash="h", element_key=name)
    out = root / name
    save_surrogate(mlp, dataclasses.replace(meta, species_name=species, species_mass_mev=mass), out)
    return out


#: one file per notice the load gives, with the words that name it
NOTICES = {
    "stale": (dict(dz=2.0, species="H-", mass=H_MINUS.mass), "is stale"),
    "no species": (dict(), "no species-mass"),
    "old mass": (dict(species="H-", mass=938.27208816), "STALE kinematics"),
    "unknown species": (dict(species="muon", mass=1.0), "UNVERIFIABLE"),
}


@pytest.mark.parametrize("case", NOTICES)
def test_every_load_says_what_is_wrong(tmp_path, case):
    kw, words = NOTICES[case]
    d = _saved(tmp_path, **kw)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")        # Python's own: once per line
        for _ in range(3):
            load_surrogate(d)
    said = [str(w.message) for w in rec if words in str(w.message)]
    assert len(said) == 3, [str(w.message) for w in rec]


def test_the_notice_names_the_line_that_loaded(tmp_path):
    """As ``warnings.warn(stacklevel=2)`` did: the GUI keeps only HELIX's own warnings, by file."""
    d = _saved(tmp_path, dz=2.0, species="H-", mass=H_MINUS.mass)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        load_surrogate(d)
    assert [os.path.basename(w.filename) for w in rec] == [os.path.basename(__file__)]
    assert all(issubclass(w.category, UserWarning) for w in rec)


@pytest.mark.parametrize("action, n", [("ignore", 0), ("once", 1), ("module", 1), ("always", 3)])
def test_the_filters_still_decide(tmp_path, action, n):
    d = _saved(tmp_path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter(action)
        for _ in range(3):
            load_surrogate(d)
    assert len([w for w in rec if "no species-mass" in str(w.message)]) == n


def test_an_error_filter_still_raises(tmp_path):
    d = _saved(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="no species-mass"):
            load_surrogate(d)


def test_a_quiet_load_says_nothing(tmp_path):
    """For a caller that has said it already: the tab, reloading after an edit."""
    d = _saved(tmp_path, "FMAP_1", dz=2.0, species="H-", mass=H_MINUS.mass)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _mlp, meta = load_surrogate(d, notices=False)
        assert "FMAP_1" in discover_cached_surrogates(tmp_path, notices=False)
    assert rec == [] and meta.stale_factor is not None


@pytest.mark.parametrize("how", ["-c", "-m"])
def test_a_script_run_with_python_c_or_m_loads_on(tmp_path, how):
    """Handing the caller's globals to the warning made Python ask ``__main__``'s loader for
    the source line, which raised ImportError under ``python -c`` and ``python -m``."""
    import subprocess
    import sys
    d = _saved(tmp_path, "MAP", dz=2.0, species="H-", mass=H_MINUS.mass)
    line = tmp_path / "line.dat"
    line.write_text("DRIFT 100 10\nEND\n")
    repo = Path(__file__).resolve().parents[2]
    cmd = ([sys.executable, "-c", f"from linac_gen.surrogates.training import load_surrogate; "
                                  f"load_surrogate({str(d)!r}); print('loaded')"]
           if how == "-c" else
           [sys.executable, "-m", "linac_gen.surrogates.cli", "register-multi", "--source", str(d),
            "--lattice", str(line), "--names", "MAP"])
    env = {**os.environ, "PYTHONPATH": f"{repo}{os.pathsep}{repo / 'gui'}", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert "ImportError" not in r.stderr and "Traceback" not in r.stderr, r.stderr[-2000:]
    assert "is stale" in r.stderr
    if how == "-c":
        assert r.returncode == 0 and "loaded" in r.stdout
    else:
        assert "SKIP MAP: not in lattice" in r.stderr      # it went on past the load


def test_discovery_says_it_each_time(tmp_path):
    """The Surrogates tab finds its files through ``discover_cached_surrogates``."""
    _saved(tmp_path, "FMAP_1", dz=2.0, species="H-", mass=H_MINUS.mass)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("default")
        for _ in range(2):
            assert "FMAP_1" in discover_cached_surrogates(tmp_path)
    assert len([w for w in rec if "is stale" in str(w.message)]) == 2


# ---------------------------------------------------------------------------
# the status bar
# ---------------------------------------------------------------------------
def test_the_status_bar_hears_it_once_per_load(tmp_path, monkeypatch):
    """Once when the project loads (set_lattice signals twice), not again for an edit of the
    lattice, which reloads the files, and again after Clear when the project is loaded again."""
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])      # noqa: F841 - kept alive for the widgets
    from linac_gen_gui.interphase.chrome.statusbar import StatusBar
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    from linac_gen_gui.interphase.warning_log import install

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.surrogates.registry import hash_lattice_file

    lat_file = tmp_path / "mini.dat"
    lat_file.write_text("DRIFT 100 0 10 0 0\nEND")
    root = tmp_path / "linac_gen" / "surrogates" / "weights" / hash_lattice_file(lat_file)[:16]
    _saved(root, "FMAP_X", dz=2.0, species="H-", mass=H_MINUS.mass)
    monkeypatch.chdir(tmp_path)                 # the tab's weights folder is relative
    def project():
        fm = FieldMap3D.__new__(FieldMap3D)
        FieldMapElement.__init__(fm, name="FMAP_X", length=100.0, aperture=10.0, n_steps=10)
        fm.ke, fm.kb, fm.phase, fm.frequency = 0.05, 0.0, -90.0, 162.5
        lat = Lattice()
        lat.add(fm)
        return lat, fm

    def times():
        stale = [n for text, n in bar._warnings.items() if "is stale" in text]
        return stale[0] if stale else 0

    state = AppState()
    bar = StatusBar(state)
    install(state)
    tab = SurrogatesTab(state)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            lat, fm = project()
            state.set_lattice(lat, path=str(lat_file))
            assert times() == 1
            for ke in (0.051, 0.052, 0.053):
                state.bus.do(ParamChangeCommand(fm, "ke", fm.ke, ke))
            assert "FMAP_X" in tab._trained and times() == 1
            bar.clear_warnings()
            assert bar.warning_text() == ""
            state.set_lattice(project()[0], path=str(lat_file))      # loaded again
            assert times() == 1
            state.set_lattice(None)
            assert tab._discovered_from is None             # the closed lattice is let go
    finally:
        bar.clear_warnings()
        tab.deleteLater()
        bar.deleteLater()
