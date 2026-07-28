"""Centroid popup: DIAG_POSITION goal-orbit overlay + rms-to-goal banner.

The achieved orbit (curves) and the requested orbit (goal points from the
deck's DIAG_POSITION targets or a loaded BPM-targets file) share one
plot; the banner reports the rms gap per plane.  Envelope results carry
a real centroid and get the same rms banner; only genuinely
centroid-less results (loaded archives, foreign objects) fall back to
a "results carry no centroid" note instead of a bogus comparison.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker


def _lattice_with_targets():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("B1", is_bpm=True, diag_family=1,
                   x_target_mm=0.5, y_target_mm=-0.2))
    lat.add(Drift("D2", 100.0))
    lat.add(Marker("B2", is_bpm=True, diag_family=1,
                   x_target_mm=None, y_target_mm=0.1))   # x plane free
    return lat


def _mp_results(n_elem, cx=0.4, cy=-0.1):
    s = np.arange(n_elem + 1, dtype=float) * 100.0
    cent = [np.array([cx, 0, cy, 0, 0, 0], dtype=float)] * (n_elem + 1)
    return SimpleNamespace(s=s, centroid=cent,
                           element_exit_idx=list(range(1, n_elem + 1)))


def _popup(qapp, lat):
    from linac_gen_gui.interphase.tabs.results_tab import _CentroidPopup
    state = SimpleNamespace(lattice=lat)
    return _CentroidPopup(None, state)


def test_goal_points_and_rms_banner_mp(qapp):
    lat = _lattice_with_targets()
    dlg = _popup(qapp, lat)
    dlg.refresh(_mp_results(len(lat.elements)))
    x_pts = dlg._tx.getData()
    y_pts = dlg._ty.getData()
    assert len(x_pts[0]) == 1 and x_pts[1][0] == pytest.approx(0.5)
    assert len(y_pts[0]) == 2                     # both BPMs constrain y
    assert dlg._goal_lbl.isVisibleTo(dlg)
    txt = dlg._goal_lbl.text()
    # rms Δx = |0.4 − 0.5| = 0.1; Δy over (−0.1+0.2, −0.1−0.1)
    assert "0.1000 mm" in txt
    assert "1x + 2y" in txt
    dlg.close()


def test_centroidless_results_show_goals_with_note(qapp):
    lat = _lattice_with_targets()
    dlg = _popup(qapp, lat)
    n = len(lat.elements)
    dlg.refresh(SimpleNamespace(s=np.arange(n + 1, dtype=float) * 100.0,
                                centroid=None))
    assert len(dlg._tx.getData()[0]) == 1         # goals still drawn
    assert "no centroid" in dlg._goal_lbl.text()
    dlg.close()


def test_envelope_results_with_centroid_get_rms_banner(qapp):
    """EnvelopeResults now carry a centroid — the popup must show the
    achieved-vs-goal rms banner for envelope runs too, not the note."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat = _lattice_with_targets()
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    res = EnvelopeSolver(lat, ref,
                         dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                              alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                              alpha_z=0.0, beta_z=10.0, emit_z=0.3,
                              centroid=[0.4, 0, -0.1, 0, 0, 0]),
                         current=0.0).run()
    dlg = _popup(qapp, lat)
    dlg.refresh(res)
    txt = dlg._goal_lbl.text()
    assert "achieved-vs-goal" in txt and "no centroid" not in txt
    dlg.close()


def test_no_targets_hides_banner(qapp):
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("B1", is_bpm=True))            # bare BPM, no targets
    dlg = _popup(qapp, lat)
    dlg.refresh(_mp_results(len(lat.elements)))
    assert len(dlg._tx.getData()[0] or []) == 0
    assert not dlg._goal_lbl.isVisibleTo(dlg)
    dlg.close()


def test_file_override_moves_goal_points(qapp):
    lat = _lattice_with_targets()
    b1 = next(e for e in lat.elements if e.name == "B1")
    b1.diag_target_override = (0.9, None, None)
    dlg = _popup(qapp, lat)
    dlg.refresh(_mp_results(len(lat.elements)))
    assert dlg._tx.getData()[1][0] == pytest.approx(0.9)
    assert len(dlg._ty.getData()[0]) == 1         # B1's y freed by override
    dlg.close()


def test_none_results_clears_everything(qapp):
    dlg = _popup(qapp, _lattice_with_targets())
    dlg.refresh(_mp_results(4))
    dlg.refresh(None)
    assert not dlg._goal_lbl.isVisibleTo(dlg)
    dlg.close()
