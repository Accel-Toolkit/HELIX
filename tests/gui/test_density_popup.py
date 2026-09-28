"""Density-vs-s heatmap: columns sit at their TRUE s.

Recorded steps are unevenly spaced (dense short elements, long drifts).
The heatmap used to give every step an equal-width column, so a step's
column was drawn at (index / n_steps) of the line instead of at its s —
e.g. 30 steps in the first 0.9 m of a 4.9 m line were spread over
0-4.6 m, against the ±σ band drawn at the true s.  The image is now built
on an even s grid (plot_style.resample_on_even_s), shared with the halo
action scan window.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.beam import Beam                          # noqa: E402
from linac_gen.core.lattice import Lattice                    # noqa: E402
from linac_gen.core.particle import PROTON                    # noqa: E402
from linac_gen.core.reference import ReferenceParticle        # noqa: E402
from linac_gen.core.simulation import Simulation              # noqa: E402
from linac_gen.elements.drift import Drift                    # noqa: E402
from linac_gen.elements.quadrupole import Quadrupole          # noqa: E402
from linac_gen_gui.interphase.plots.plot_style import (       # noqa: E402
    resample_on_even_s)

_noop = lambda *a, **k: None                                  # noqa: E731


# --------------------------------------------------------------------------
class TestResampleOnEvenS:
    def test_each_column_shows_the_step_at_or_before_it(self):
        s = np.array([0.0, 1.0, 2.0, 10.0])
        vals = np.arange(4.0)[:, None] * np.ones((1, 2))
        img, s0, s1 = resample_on_even_s(vals, s, min_cols=10, max_cols=10)
        assert (s0, s1) == (0.0, 10.0) and img.shape == (10, 2)
        # 1 s-unit per column: steps at 0, 1, 2 in columns 0-2, the step
        # at 10 in the last column; columns 3-8 hold the step at s = 2
        np.testing.assert_array_equal(
            img[:, 0], [0, 1, 2, 2, 2, 2, 2, 2, 2, 3])

    def test_pooling_of_steps_sharing_a_column(self):
        s = np.array([0.0, 0.1, 0.2, 10.0])
        vals = np.array([[1.0], [5.0], [3.0], [7.0]])
        mx, _, _ = resample_on_even_s(vals, s, pool="max",
                                      min_cols=10, max_cols=10)
        mean, _, _ = resample_on_even_s(vals, s, pool="mean",
                                        min_cols=10, max_cols=10)
        assert mx[0, 0] == 5.0 and mean[0, 0] == pytest.approx(3.0)
        # the empty columns hold the LAST step of column 0 (3), not the
        # pooled value
        np.testing.assert_array_equal(mx[1:9, 0], 3.0)
        np.testing.assert_array_equal(mean[1:9, 0], 3.0)
        assert mx[9, 0] == 7.0

    def test_nan_ignored_backward_steps_and_degenerate_s(self):
        s = np.array([0.0, 5.0, 4.8, 10.0])           # a negative drift
        vals = np.array([[1.0], [np.nan], [2.0], [4.0]])
        img, s0, s1 = resample_on_even_s(vals, s, pool="mean",
                                         min_cols=10, max_cols=10)
        assert img[5, 0] == 2.0                      # NaN ignored in the mean
        assert (s0, s1) == (0.0, 10.0)
        flat, a, b = resample_on_even_s(np.ones((3, 1)), [2.0, 2.0, 2.0])
        assert (a, b) == (2.0, 3.0) and np.isfinite(flat).all()
        with pytest.raises(ValueError):
            resample_on_even_s(vals, s, pool="median")


# --------------------------------------------------------------------------
def _uneven_run():
    """30 steps in the first 0.9 m (quads), then ONE step after a 4 m
    drift — the spacing that exposed the defect."""
    lat = Lattice()
    for i in range(15):
        lat.add(Quadrupole(f"Q{i}", length=30.0,
                           gradient=(25.0 if i % 2 else -25.0)))
        lat.add(Drift(f"D{i}", length=30.0))
    lat.add(Drift("LONG", length=4000.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=5000, current=0.0)
    rng = np.random.default_rng(1)
    beam.particles[:] = rng.standard_normal((5000, 6)) * np.array(
        [1.0, 0.6, 1.0, 0.6, 5.0, 0.002])
    rec = Simulation(lat, beam, space_charge="off",
                     density_axes=("x",)).run()
    return lat, rec


def test_heatmap_columns_sit_at_the_true_s(qapp, monkeypatch):
    import scipy.ndimage

    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import (
        ResultsTab, _DensityPopup)
    # compare exact histograms: no display smoothing for this test
    monkeypatch.setattr(scipy.ndimage, "gaussian_filter",
                        lambda H, **k: H)
    lat, rec = _uneven_run()
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(rec)
    tab = ResultsTab(state, _noop, _noop, _noop)
    tab._open_popup("density_s")
    pop = tab._popups["density_s"]
    assert isinstance(pop, _DensityPopup)
    pop._chk_log.setChecked(False)
    img = pop._image.image                      # (bins, columns)
    rect = pop._image.mapRectToParent(pop._image.boundingRect())
    s = np.asarray(rec.s, float)
    dens = rec.density_array("x").astype(float)
    assert rect.left() == pytest.approx(s[0])
    assert rect.right() == pytest.approx(s[-1])
    n_cols = img.shape[1]

    def column_at(s_mm):
        return img[:, min(int((s_mm - s[0]) / (s[-1] - s[0]) * n_cols),
                          n_cols - 1)]
    # inside the long drift (s = 3 m): the last step before it (s = 0.9 m)
    i_before = int(np.searchsorted(s, 3000.0)) - 1
    assert s[i_before] == pytest.approx(900.0)
    np.testing.assert_array_equal(column_at(3000.0), dens[i_before])
    # in the dense section every step sits in its own column at its s
    for i in (5, 12, 20):
        np.testing.assert_array_equal(column_at(s[i] + 1.0), dens[i])
    # and the end of the drift shows the last (large) beam
    np.testing.assert_array_equal(column_at(s[-1]), dens[-1])
    pop.close()
