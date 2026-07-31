"""The phase-fold VIEW helper — and the pin that it stays out of stats.

An RFQ makes a bunch train (one bunch per RF period); HELIX seeds one
period, so particles that slip into a neighbouring bucket are stored
360° away and a raw φ–ΔW plot shows stripes.  `wrap_phase_column` folds
that for DISPLAY.

It is deliberately NOT wired into σ_φ / ε_z / z-Twiss.  Two adversarial
reviews (2026-07-30) measured folding on the real PXIE deck and found it
a biased estimator (ε_z +123 %, σ_W +69 %, α_z sign-flipped, because the
satellite buckets carry a space-charge chirp and are not periodic images
of the core) with no shot-noise-robust way to decide when to apply it
(271 % swings from a 0.9 % input change; a drift's exactly-constant ε_z
became a staircase; a mismatched beam's genuine debunching was
under-reported 10×).  The tests below pin BOTH halves: the helper works,
and the moments do not use it.

The real fix landed instead at the TRACKING level —
``BeamConfig.periodic_phase`` / ``Tracker._fold_phase``, pinned by
`tests/tracking/test_periodic_phase.py` — so a flagged run never forms
the train and its moments are single-bunch values with no statistics
treatment at all.  This helper remains the display fold for the
UNFLAGGED runs, and these tests must keep holding for them.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.diagnostics.moments import (compute_emittance, compute_halo,
                                           compute_moments,
                                           compute_twiss_from_particles,
                                           wrap_phase_column)


def _train(n_side=200, n_main=400, sigma_phi=5.0, seed=1,
           dW_chirp=0.0, dphi_chirp=0.0):
    """Three-bucket train.  With the chirp arguments the satellites carry
    the offsets MEASURED on the PXIE deck at the RFQ exit — the leading
    bucket ~+35 keV and +3.3° inside its own bucket, the trailing one
    mirrored — i.e. they are NOT periodic images of the core."""
    rng = np.random.default_rng(seed)
    n = n_main + 2 * n_side
    p = np.zeros((n, 6))
    p[:, 4] = rng.normal(0.0, sigma_phi, n)
    p[:, 5] = rng.normal(0.0, 0.02, n)
    lead = slice(n_main, n_main + n_side)
    trail = slice(n_main + n_side, n)
    p[lead, 4] += -360.0 + dphi_chirp
    p[lead, 5] += dW_chirp
    p[trail, 4] += 360.0 - dphi_chirp
    p[trail, 5] -= dW_chirp
    return p


# --------------------------------------------------------------- helper
def test_noop_for_a_normal_bunch_is_the_same_object():
    rng = np.random.default_rng(0)
    p = rng.normal(0.0, 1.0, (500, 6))
    p[:, 4] *= 30.0                                  # ±4σ ≈ 120°, one bucket
    out, n_folded = wrap_phase_column(p)
    assert out is p and n_folded == 0


def test_fold_recovers_the_single_bunch_and_leaves_the_input_alone():
    p = _train()
    raw_sigma = p[:, 4].std()
    out, n_folded = wrap_phase_column(p)
    assert n_folded == 400 and out is not p
    assert raw_sigma > 200.0
    assert out[:, 4].std() == pytest.approx(5.0, rel=0.15)
    assert p[:, 4].std() == raw_sigma                # caller's array intact


def test_fold_is_idempotent():
    """The GUI may fold an already-folded array; it must be a no-op."""
    once, n1 = wrap_phase_column(_train())
    twice, n2 = wrap_phase_column(once)
    assert n1 == 400 and n2 == 0 and twice is once


def test_median_anchor_survives_an_asymmetric_train():
    """Mean-anchoring would fold the beam in half; median must not."""
    out, _ = wrap_phase_column(_train(n_side=300, n_main=100, seed=3))
    assert out[:, 4].std() == pytest.approx(5.0, rel=0.25)


# ------------------------------------------------- the deliberate no-wire
def test_moments_do_NOT_fold():
    """Pin the decision, with the reason in the name.  If someone wires
    the fold into the moments, these numbers move by 30-40× and this
    test fails — read the module docstring before changing it."""
    p = _train()
    m = compute_moments(p)
    assert m["sigma_phi"] > 200.0                    # train-wide, raw
    assert "n_phase_wrapped" not in m                # no schema surprise
    assert compute_emittance(p, "z") > 1.0
    assert compute_twiss_from_particles(p, "z")["beta"] > 1e3
    assert compute_halo(p, "z") > 0.0


def test_folding_would_bias_the_estimator_which_is_why_it_is_view_only():
    """Reproduce the reviews' headline in miniature: with the real
    −35 keV/bucket chirp, folded moments are NOT the single-bunch
    moments — ε_z inflates and the (φ,W) correlation flips sign."""
    p = _train(dW_chirp=0.035, dphi_chirp=3.3)       # the PXIE offsets
    main = p[:400]                                   # the core bucket
    folded, _ = wrap_phase_column(p)
    ez_folded = compute_emittance(folded, "z")
    ez_main = compute_emittance(main, "z")
    assert ez_folded > 1.5 * ez_main                 # 1.8× here, 2.2× on PXIE
    a_folded = compute_twiss_from_particles(folded, "z")["alpha"]
    a_main = compute_twiss_from_particles(main, "z")["alpha"]
    assert np.sign(a_folded) != np.sign(a_main)      # the sign flip


def test_debunched_beam_folds_to_a_meaningless_compact_bunch():
    """Why no automatic gate: a genuinely debunched single bunch folds
    into a plausible-looking blob, and statistics cannot tell it from a
    train.  Only the caller (a user ticking a checkbox) knows."""
    rng = np.random.default_rng(5)
    p = np.zeros((2000, 6))
    p[:, 4] = rng.uniform(-900.0, 900.0, 2000)       # ~5 periods, real spread
    p[:, 5] = rng.normal(0.0, 0.02, 2000)
    folded, n_folded = wrap_phase_column(p)
    assert n_folded > 0
    assert p[:, 4].std() > 400.0                     # the truth
    assert folded[:, 4].std() < 110.0                # the flattering picture
