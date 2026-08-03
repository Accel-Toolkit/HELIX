"""EnvelopeSolver's TraceWin-compatible DC mode (``rfq_dc_envelope``).

TraceWin's envelope mode never bunches: its genuine ENV+SC export of
the 2-solenoid PXIE LEBT+RFQ line carries NO longitudinal envelope
(σφ ≡ σ_dp/p ≡ σ_W ≡ 0 through the RFQ) — the RFQ beam stays DC with
2-D transverse space charge.  HELIX's default envelope instead seeds a
uniform 103.9° bunch at the first RF element and evolves it.

Both are idealisations; they are NOT comparable to each other with SC
on.  Measured on the PXIE line at 5 mA against the TW export:
bunched 27–64 % σ deviation inside the RFQ, DC-through-RFQ 4–6 %
full-line.  These tests pin both facts so neither silently drifts.

Skips wherever the PXIE project data is absent (any public checkout).
"""
from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl

DECK = (pathlib.Path(__file__).resolve().parents[2]
        / "examples" / "lebt_plus_rfq" / "lebt2sol_plus_rfq.dat")
MC2 = 939.294308


@pytest.fixture(scope="module")
def tw_export():
    p = rl.reference_path("2sol_lebt_plus_rfq_env+sc.txt")
    if p is None:
        pytest.skip("2-sol ENV+SC export not present on this machine")
    a = np.loadtxt(p, skiprows=2)
    return dict(z=a[:, 0], sx=a[:, 12] * 1e3, sy=a[:, 14] * 1e3,
                sphi=a[:, 19], W=a[:, 1] * MC2 * 1e3)


def _run(dc):
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.tracking.envelope import EnvelopeSolver
    cwd = os.getcwd()
    os.chdir(DECK.parent)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lat, _ = parse_tracewin(DECK.name)
            ref = ReferenceParticle(species=H_MINUS, w_kin=0.030,
                                    frequency=162.5)
            bg = ref.beta * ref.gamma
            init = dict(alpha_x=-6.1145021, beta_x=1.3870467,
                        emit_x=0.1370111 / bg,
                        alpha_y=-6.07846, beta_y=1.3799354,
                        emit_y=0.1370329 / bg,
                        alpha_z=0.0, beta_z=1.0, emit_z=0.0,
                        continuous=True)
            res = EnvelopeSolver(lat, ref, init, current=5.0,
                                 record_substeps=True,
                                 rfq_dc_envelope=dc).run()
    finally:
        os.chdir(cwd)
    return (np.asarray(res.s) * 1e-3, np.asarray(res.sigma_x),
            np.asarray(res.sigma_y), np.asarray(res.ref_w_kin) * 1e3)


def _dev(tw, s, v, col, z0=0.02, z1=5.80):
    m = (tw["z"] > z0) & (tw["z"] < z1)
    hi = np.interp(tw["z"][m], s, v)
    ref = tw[col][m]
    return (100 * np.abs(hi - ref) / np.maximum(ref, 1e-9)).mean()


@pytest.mark.slow
def test_dc_mode_matches_the_tracewin_env_sc_export(tw_export):
    if not DECK.is_file():
        pytest.skip("2-solenoid deck not present")
    s, sx, sy, W = _run(dc=True)
    assert _dev(tw_export, s, sx, "sx") < 8.0
    assert _dev(tw_export, s, sy, "sy") < 8.0
    # energy: the card model, exactly
    assert W[-1] == pytest.approx(tw_export["W"][-1], abs=1.0)


@pytest.mark.slow
def test_default_bunched_mode_is_not_tw_comparable(tw_export):
    """Documents the model split: the DEFAULT (bunched) envelope must
    NOT be validated against a TraceWin ENV+SC export.  If this ever
    starts agreeing, the default's physics changed — investigate."""
    if not DECK.is_file():
        pytest.skip("2-solenoid deck not present")
    s, sx, _sy, W = _run(dc=False)
    m = (tw_export["z"] > 1.3887) & (tw_export["z"] < 5.80)
    hi = np.interp(tw_export["z"][m], s, sx)
    dev = (100 * np.abs(hi - tw_export["sx"][m])
           / np.maximum(tw_export["sx"][m], 1e-9)).mean()
    assert dev > 15.0, dev
    # the ENERGY ramp is SC-independent — identical in both modes
    assert W[-1] == pytest.approx(tw_export["W"][-1], abs=1.0)


def test_flag_defaults_off_and_tw_export_is_dc():
    """The export itself proves TW-env never bunches: every longitudinal
    rms column is identically zero."""
    p = rl.reference_path("2sol_lebt_plus_rfq_env+sc.txt")
    if p is None:
        pytest.skip("2-sol ENV+SC export not present on this machine")
    a = np.loadtxt(p, skiprows=2)
    assert np.all(a[:, 19] == 0.0)      # rms phase
    assert np.all(a[:, 21] == 0.0)      # rms energy
    import inspect
    from linac_gen.tracking.envelope import EnvelopeSolver
    sig = inspect.signature(EnvelopeSolver.__init__)
    assert sig.parameters["rfq_dc_envelope"].default is False
