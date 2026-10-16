"""DensityPanel colour bar: wide enough for its tick values, titled with what
the image shows (log₁₀(1+N) of the smoothed counts, or the scatter colour
quantity), the title on the axis pyqtgraph draws it on (the left one of a
vertical bar; the right one carries the tick values)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _panel(**kw):
    from linac_gen_gui.interphase.plots.plot_style import DensityPanel
    return DensityPanel(title="φ – W", xlabel="Δφ", xunits="deg",
                        ylabel="ΔW", yunits="MeV", **kw)


def _title(panel):
    left = panel._cbar.getAxis("left")
    return left.labelText if left.label.isVisible() else None


def _data(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=n)


@pytest.mark.parametrize("profiles", [False, True])
def test_bar_is_as_wide_as_its_parts(qapp, profiles):
    """Title axis + gradient + tick axis (45 px in pyqtgraph): at the old
    fixed 48 px the tick values were cut off ("2.0" drawn as "2.C")."""
    p = _panel(profiles=profiles)
    try:
        p.resize(560, 420)
        p.set_data(*_data())
        p.show()
        qapp.processEvents()
        bar = p._cbar
        right = bar.getAxis("right")
        need = (bar.getAxis("left").minimumWidth()
                + bar.layout.columnMinimumWidth(1) + right.minimumWidth())
        assert bar.maximumWidth() >= need
        assert bar.geometry().width() == pytest.approx(bar.maximumWidth())
        assert right.geometry().right() <= bar.geometry().width() + 0.5
    finally:
        p.deleteLater()


def test_image_is_log10_of_the_counts_and_the_bar_says_so(qapp):
    from scipy.ndimage import gaussian_filter
    x, y = _data()
    p = _panel()
    try:
        p.set_data(x, y)
        rng = ((x.mean() - 4.5 * x.std(), x.mean() + 4.5 * x.std()),
               (y.mean() - 4.5 * y.std(), y.mean() + 4.5 * y.std()))
        H, _, _ = np.histogram2d(x, y, bins=p._bins, range=rng)
        want = np.log10(1.0 + gaussian_filter(H, sigma=p._sigma,
                                              mode="nearest")).T
        assert np.allclose(p._image.image, want)
        assert p._cbar.levels() == pytest.approx((want.min(), want.max()))
        assert "log₁₀(1+N)" in _title(p)
    finally:
        p.deleteLater()


def test_scatter_mode_titles_the_bar_and_density_mode_restores_it(qapp):
    x, y = _data(n=500)
    p = _panel()
    try:
        right = p._cbar.getAxis("right")
        p.set_scatter(x, y, y, label="ΔW (MeV)")
        assert "ΔW (MeV)" in _title(p)
        assert not right.label.isVisible()        # the tick axis stays bare
        p.set_scatter(x, y, None)                 # one colour: no quantity
        assert "ΔW" not in _title(p) and "log" not in _title(p)
        p.set_data(x, y)
        assert "log₁₀(1+N)" in _title(p)
        assert not right.label.isVisible()
    finally:
        p.deleteLater()
