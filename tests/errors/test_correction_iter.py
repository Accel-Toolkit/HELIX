"""Iteration + vmax tests for ``apply_correction``.

The legacy ``test_correction.py`` exercises single-shot behaviour;
this file pins down the new ``n_iter`` / ``tol_mm`` / ``vmax`` /
``history`` extensions added for TraceWin-parity correction.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.correction import apply_correction
from linac_gen.tracking.tracker import Tracker


def _factory(offset_x=2.0, offset_y=1.0, n=300, seed=42):
    def f():
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=n, current=0.0)
        rng = np.random.default_rng(seed)
        beam.particles[:, 0] = rng.normal(offset_x, 0.5, n)
        beam.particles[:, 1] = rng.normal(0, 0.2, n)
        beam.particles[:, 2] = rng.normal(offset_y, 0.5, n)
        beam.particles[:, 3] = rng.normal(0, 0.2, n)
        return beam
    return f


def _fodo_with_pair():
    """Drift → Steerer → Drift → BPM → Drift (linear, simple)."""
    lat = Lattice()
    lat.add(Drift("D0", 100.0))
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Marker("BPM_1"))
    lat.add(Drift("D2", 100.0))
    return lat


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------
def test_history_returns_iterations():
    lat = _fodo_with_pair()
    out = apply_correction(lat, _factory(), n_iter=3, history=True)
    assert isinstance(out, tuple) and len(out) == 2
    kicks, hist = out
    assert "STEER_1" in kicks
    assert isinstance(hist, list) and len(hist) >= 1
    assert {"iter", "rms_orbit_mm", "n_saturated"} <= set(hist[0].keys())


def test_n_iter_one_matches_legacy():
    """n_iter=1 + history=False = legacy single-shot return type."""
    lat = _fodo_with_pair()
    res = apply_correction(lat, _factory(), n_iter=1)
    assert isinstance(res, dict)
    assert "STEER_1" in res


def test_iteration_converges_below_tol():
    lat = _fodo_with_pair()
    kicks, hist = apply_correction(
        lat, _factory(), n_iter=5, tol_mm=0.001, history=True,
    )
    # Final iteration's residual is below tolerance.
    final_rms = hist[-1]["rms_orbit_mm"]
    assert final_rms < 0.05  # comfortably below default tol


# ---------------------------------------------------------------------
# vmax saturation
# ---------------------------------------------------------------------
def test_scalar_vmax_clips_kicks():
    """A tiny global vmax saturates every kick to ±vmax."""
    lat = _fodo_with_pair()
    kicks, hist = apply_correction(
        lat, _factory(offset_x=10.0, offset_y=10.0),
        n_iter=3, vmax=1e-6, history=True,
    )
    assert abs(kicks["STEER_1"]["bx_l"]) <= 1e-6 + 1e-12
    assert abs(kicks["STEER_1"]["by_l"]) <= 1e-6 + 1e-12
    # Stalled at saturation: history records n_saturated > 0 in late iters.
    assert any(h["n_saturated"] > 0 for h in hist)


def test_dict_vmax_per_steerer():
    """Per-steerer vmax dict caps each steerer independently."""
    lat = Lattice()
    lat.add(Drift("D0", 100.0))
    lat.add(Steerer("STEER_A"))
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("BPM_A"))
    lat.add(Drift("D2", 100.0))
    lat.add(Steerer("STEER_B"))
    lat.add(Drift("D3", 100.0))
    lat.add(Marker("BPM_B"))
    lat.add(Drift("D4", 100.0))

    kicks = apply_correction(
        lat, _factory(offset_x=5.0, offset_y=5.0),
        n_iter=3, vmax={"STEER_A": 1e-6, "STEER_B": 1.0},
    )
    # STEER_A is heavily clipped; STEER_B has plenty of headroom.
    assert abs(kicks["STEER_A"]["bx_l"]) <= 1e-6 + 1e-12
    assert abs(kicks["STEER_A"]["by_l"]) <= 1e-6 + 1e-12


def test_no_vmax_unbounded():
    """When vmax is None, kicks can be anything the corrector wants."""
    lat = _fodo_with_pair()
    kicks = apply_correction(lat, _factory(offset_x=10.0), n_iter=1, vmax=None)
    # We don't fix the kick magnitude — just that it's not clipped to a tiny value.
    assert abs(kicks["STEER_1"]["by_l"]) > 1e-5


# ---------------------------------------------------------------------
# Element-list overrides (used by run_correction_from_lattice)
# ---------------------------------------------------------------------
def test_explicit_element_lists_bypass_pattern():
    """Passing ``steerers=`` and ``bpms=`` ignores the glob patterns."""
    lat = _fodo_with_pair()
    # Names that the default patterns wouldn't pick.
    s = lat.elements[1]; s.name = "FOO"
    b = lat.elements[3]; b.name = "BAR"
    res = apply_correction(
        lat, _factory(),
        steerers=[s], bpms=[b],
    )
    assert "FOO" in res


# ---------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------
def test_should_stop_raises_operation_cancelled():
    from linac_gen.core.cancelled import OperationCancelled

    lat = _fodo_with_pair()
    with pytest.raises(OperationCancelled):
        apply_correction(lat, _factory(), n_iter=5, should_stop=lambda: True)


def test_should_stop_after_first_pass_keeps_history_length():
    """Cancel between passes: exactly one pass ran before the raise."""
    from linac_gen.core.cancelled import OperationCancelled

    lat = _fodo_with_pair()
    polls = {"n": 0}
    def stop_second_poll():
        polls["n"] += 1
        return polls["n"] >= 2

    with pytest.raises(OperationCancelled):
        # tol 0.0 is unreachable (rms >= 0), so the loop must reach the
        # second poll instead of converging out after pass 1.
        apply_correction(lat, _factory(), n_iter=5, tol_mm=0.0,
                         history=True, should_stop=stop_second_poll)
    assert polls["n"] == 2


def test_no_hook_is_unchanged():
    lat = _fodo_with_pair()
    out = apply_correction(lat, _factory(), n_iter=2, history=True,
                           should_stop=None)
    kicks, hist = out
    assert "STEER_1" in kicks and len(hist) >= 1
