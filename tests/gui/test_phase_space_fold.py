"""Phase Space popup: the 'fold φ' checkbox and everything that follows it.

An RFQ makes a bunch train (one bunch per RF period).  HELIX seeds one
period, so particles that slip into a neighbouring bucket are stored
360° away and the raw φ–ΔW panel shows a row of stripes.  The checkbox
folds them back for the single-bunch view TraceWin/Toutatis draw.

The plot, the beam-parameters table and the Ctrl+S export must ALL obey
the same checkbox — an adversarial review (2026-07-30) found the table
and the export ignoring it, so a folded picture sat next to a raw
σ_φ = 183°, and a CSV mixed the two conventions.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _train_recorder():
    """Recorder whose exit beam is a three-bucket train (±360°)."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rec = DiagnosticRecorder()
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    rec.s = [0.0, 400.0]
    rec.element_names = ["INPUT", "RFQ_203"]
    rng = np.random.default_rng(3)
    beam = Beam(ref=ref, n_particles=900, current=5.0)
    p = np.zeros((900, 6))
    p[:, 0] = rng.normal(0, 0.5, 900)
    p[:, 2] = rng.normal(0, 0.5, 900)
    p[:, 4] = rng.normal(0, 4.0, 900)
    p[:, 5] = rng.normal(0, 0.02, 900)
    p[300:600, 4] += 360.0
    p[600:, 4] -= 360.0
    beam.particles[:] = p
    rec.beam = beam
    return rec


@pytest.fixture()
def state_train(qapp):
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    st.set_results(_train_recorder())
    return st


def _cell(dlg, name):
    tbl = dlg._table
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 0)
        if it is not None and it.text().strip() == name:
            return float(tbl.item(r, 1).text())
    raise AssertionError(f"row '{name}' not found")


def test_checkbox_exists_and_defaults_to_folded(state_train):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_train)
    try:
        dlg.refresh(state_train.results)
        assert dlg._wrap_phi.isChecked() is True
    finally:
        dlg.close()


def test_table_follows_the_checkbox(state_train):
    """σ_φ in the parameters table must match the plotted convention."""
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_train)
    try:
        dlg.refresh(state_train.results)
        dlg._params_btn.setChecked(True)          # show the table
        folded_sigma = _cell(dlg, "σ_φ")
        dlg._wrap_phi.setChecked(False)           # -> raw train
        dlg._fill_table()
        raw_sigma = _cell(dlg, "σ_φ")
        assert folded_sigma == pytest.approx(4.0, rel=0.3)
        assert raw_sigma > 200.0                  # the train-wide value
        assert raw_sigma / folded_sigma > 20.0
    finally:
        dlg.close()


def test_export_follows_the_checkbox(state_train):
    """Ctrl+S must not mix a folded picture with a raw φ column."""
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_train)
    try:
        dlg.refresh(state_train.results)

        def phi_std():
            for panel in dlg._extra_panels():
                for c in panel.curves:
                    if c.name == "phi_deg":
                        return float(np.std(c.y))
            raise AssertionError("phi_deg curve not exported")

        assert phi_std() == pytest.approx(4.0, rel=0.4)   # folded
        dlg._wrap_phi.setChecked(False)
        assert phi_std() > 200.0                          # raw
    finally:
        dlg.close()


def test_fold_precedes_the_tracewin_basis_conversion(state_train):
    """Folding must happen in DEGREES, before Δφ → z, or z gets folded."""
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_train)
    try:
        dlg.refresh(state_train.results)
        for i in range(dlg._basis.count()):
            if dlg._basis.itemData(i) == "tracewin":
                dlg._basis.setCurrentIndex(i)
                break
        dlg._redraw()                       # must not raise in either state
        dlg._wrap_phi.setChecked(False)
        dlg._redraw()
    finally:
        dlg.close()


def test_display_fold_is_disabled_for_a_periodic_phase_run(qapp):
    """ADVERSARIAL FIND (2026-07-30): the display fold is median-anchored
    with a HARD 360° period, while a `periodic_phase` run has already
    folded about the synchronous particle using the true bunch spacing —
    720° downstream of a 162.5 → 325 MHz jump.  Leaving the checkbox
    live (and ticked by default) meant the GUI silently sliced a
    legitimately ±360°-wide bunch in half and reported σ_φ 35 % low.

    The run now stamps its provenance on the recorder, and the popup
    takes the control away when there is nothing left to fold.
    """
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup

    rec = _train_recorder()
    rec.periodic_phase = True                 # this run folded already
    st = AppState()
    st.set_results(rec)
    dlg = _PhaseSpacePopup(None, st)
    try:
        dlg.refresh(st.results)
        assert dlg._wrap_phi.isEnabled() is False
        assert dlg._wrap_phi.isChecked() is False
        assert "already folded" in dlg._wrap_phi.toolTip()
    finally:
        dlg.close()


def test_display_fold_stays_live_for_an_unflagged_run(state_train):
    """Control for the test above — an ordinary run keeps the control."""
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    assert getattr(state_train.results, "periodic_phase", False) is False
    dlg = _PhaseSpacePopup(None, state_train)
    try:
        dlg.refresh(state_train.results)
        assert dlg._wrap_phi.isEnabled() is True
        assert dlg._wrap_phi.isChecked() is True
    finally:
        dlg.close()
