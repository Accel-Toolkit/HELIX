"""Tests for the one-sided ``MIN_EMIT_GROWTH`` constraint."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.lattice_commands import (
    MinEmit4DGrowth, MinEmitGrowth,
)
from linac_gen.matching.constraints import (
    _make_min_emit_4d_growth_evaluator,
    _make_min_emit_growth_evaluator,
    collect_constraints,
)


def _results(*, emit_x_in, emit_x_out,
             emit_y_in=0.30, emit_y_out=0.30,
             emit_z_in=0.40, emit_z_out=0.40,
             beta_in=0.0731, gamma_in=1.00266,
             beta_out=0.0731, gamma_out=1.00266):
    """Fake ``EnvelopeResults`` with just the fields the evaluator reads."""
    return SimpleNamespace(
        emit_x=[emit_x_in, emit_x_out],
        emit_y=[emit_y_in, emit_y_out],
        emit_z=[emit_z_in, emit_z_out],
        ref_beta=[beta_in, beta_out],
        ref_gamma=[gamma_in, gamma_out],
    )


# ---------------------------------------------------------------------------
# X-plane: normalised emittance (geom * bg)
# ---------------------------------------------------------------------------
def test_x_plane_growth_residual_positive():
    """Geometric emit going up while bg constant → residual = Δε_n."""
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "X", 1.0))
    # bg = 0.0731  →  ε_n,in = 4.1 * 0.0731 = 0.30 ; ε_n,out = 5.0 * 0.0731 = 0.366
    res = ev(_results(emit_x_in=4.1, emit_x_out=5.0), None)
    assert res.shape == (1,)
    assert res[0] == pytest.approx(0.366 - 0.30, abs=1e-3)


def test_x_plane_drop_is_free():
    """Drop below ε_in produces zero residual (one-sided penalty)."""
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "X", 1.0))
    res = ev(_results(emit_x_in=4.1, emit_x_out=3.0), None)
    assert res[0] == 0.0


def test_x_plane_adiabatic_damping_does_not_trigger_residual():
    """Geometric ε drops due to acceleration (bg goes up) but ε_n stays flat.

    Without normalisation this would look like a "drop" and pass.
    With normalisation the evaluator must see ε_n,out ≈ ε_n,in and
    return zero — i.e. it correctly treats damping as a non-event.
    """
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "X", 1.0))
    # bg_in=0.0731, bg_out=0.146 (~doubled by acceleration)
    # ε_n,in = 4.1 * 0.0731 = 0.30
    # ε_n,out = 2.05 * 0.146 = 0.299
    res = ev(_results(emit_x_in=4.1, emit_x_out=2.05,
                      beta_in=0.0731, gamma_in=1.00266,
                      beta_out=0.146, gamma_out=1.0106),
             None)
    assert res[0] < 0.01


# ---------------------------------------------------------------------------
# Z-plane: native (deg·MeV) — NOT bg-scaled
# ---------------------------------------------------------------------------
def test_z_plane_growth_residual_positive():
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "Z", 1.0))
    res = ev(_results(emit_x_in=4.1, emit_x_out=4.1,
                      emit_z_in=0.40, emit_z_out=0.55),
             None)
    assert res[0] == pytest.approx(0.15, abs=1e-9)


def test_z_plane_drop_is_free():
    """Coupling-resonance drop in ε_z is explicitly allowed."""
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "Z", 1.0))
    res = ev(_results(emit_x_in=4.1, emit_x_out=4.1,
                      emit_z_in=0.40, emit_z_out=0.25),
             None)
    assert res[0] == 0.0


# ---------------------------------------------------------------------------
# Z-plane: frequency-jump invariance (deg·MeV is clock-referenced)
# ---------------------------------------------------------------------------
def test_z_plane_freq_jump_alone_is_not_growth():
    """Across a FREQ jump the native deg·MeV emit_z inflates by
    f_new/f_old with zero physical growth (Δφ = 360·f·Δt).  The
    evaluator must anchor both ends to one clock and return zero."""
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "Z", 1.0))
    res_obj = _results(emit_x_in=4.1, emit_x_out=4.1,
                       emit_z_in=0.40, emit_z_out=0.80)   # exact 2x rescale
    res_obj.ref_frequency = [162.5, 325.0]
    assert ev(res_obj, None)[0] == pytest.approx(0.0, abs=1e-12)


def test_z_plane_real_growth_across_freq_jump_still_detected():
    """Genuine growth on top of the rescale must survive the anchor:
    0.40 deg·MeV @162.5 == 0.80 @325; exit 0.90 => 0.10 real growth."""
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "Z", 1.0))
    res_obj = _results(emit_x_in=4.1, emit_x_out=4.1,
                       emit_z_in=0.40, emit_z_out=0.90)
    res_obj.ref_frequency = [162.5, 325.0]
    assert ev(res_obj, None)[0] == pytest.approx(0.10, abs=1e-12)


def test_z_plane_fixed_frequency_residual_unchanged():
    """With ref_frequency present but constant the residual is
    bit-identical to the no-frequency-info case (ratio = 1)."""
    ev = _make_min_emit_growth_evaluator(MinEmitGrowth("g", "Z", 1.0))
    res_obj = _results(emit_x_in=4.1, emit_x_out=4.1,
                       emit_z_in=0.40, emit_z_out=0.55)
    res_obj.ref_frequency = [162.5, 162.5]
    assert ev(res_obj, None)[0] == pytest.approx(0.15, abs=1e-12)


def test_4d_z_arm_freq_jump_alone_is_not_growth():
    """The MIN_EMIT_4D_GROWTH z-residual gets the same clock anchor."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    res_obj = _results_4d(emit_z_in=0.40, emit_z_out=0.80)
    res_obj.ref_frequency = [162.5, 325.0]
    assert ev(res_obj, None)[1] == pytest.approx(0.0, abs=1e-12)


def test_z_clock_ratio_helper_all_regimes():
    """The shared anchor (also used by the engine's seqscan reversal
    threshold): jump -> f_out/f_in; fixed, missing, empty, or zero
    frequency -> exactly 1.0 (degrade to a no-op, never divide)."""
    from linac_gen.matching.constraints import z_clock_ratio
    r = SimpleNamespace(ref_frequency=[162.5, 325.0])
    assert z_clock_ratio(r) == pytest.approx(2.0)
    assert z_clock_ratio(SimpleNamespace(ref_frequency=[162.5, 162.5])) == 1.0
    assert z_clock_ratio(SimpleNamespace(ref_frequency=[])) == 1.0
    assert z_clock_ratio(SimpleNamespace(ref_frequency=[0.0, 325.0])) == 1.0
    assert z_clock_ratio(SimpleNamespace()) == 1.0


# ---------------------------------------------------------------------------
# Plane validation + collection
# ---------------------------------------------------------------------------
def test_invalid_plane_raises():
    with pytest.raises(ValueError):
        MinEmitGrowth("g", plane="W", weight=1.0)


def test_collect_constraints_picks_up_min_emit_growth():
    lat = Lattice()
    lat.add(MinEmitGrowth("x", plane="X", weight=2.0))
    lat.add(MinEmitGrowth("y", plane="Y", weight=1.5))
    cs = collect_constraints(lat)
    labels = sorted(c.label for c in cs)
    assert labels == ["MIN_EMIT_GROWTH:X", "MIN_EMIT_GROWTH:Y"]
    weights = {c.label: c.weight for c in cs}
    assert weights["MIN_EMIT_GROWTH:X"] == 2.0
    assert weights["MIN_EMIT_GROWTH:Y"] == 1.5


def test_collect_constraints_skips_zero_weight():
    lat = Lattice()
    lat.add(MinEmitGrowth("x", plane="X", weight=0.0))
    cs = collect_constraints(lat)
    assert cs == []


def test_weight_applied_through_evaluate():
    """The Constraint.evaluate wrapper multiplies in the weight."""
    lat = Lattice()
    lat.add(MinEmitGrowth("g", plane="Z", weight=2.5))
    cs = collect_constraints(lat)
    assert len(cs) == 1
    res = cs[0].evaluate(
        _results(emit_x_in=4.1, emit_x_out=4.1,
                 emit_z_in=0.40, emit_z_out=0.50),
        lat,
    )
    # raw residual = 0.10 ; weighted = 0.25
    assert res[0] == pytest.approx(0.25, abs=1e-9)


# ===========================================================================
# MIN_EMIT_4D_GROWTH -- coupled 4-D transverse + normalised z, with tolerance
# ===========================================================================

def _results_4d(*, emit_4d_in=1.0, emit_4d_out=1.0,
                emit_z_in=0.40, emit_z_out=0.40,
                beta_in=0.0731, gamma_in=1.00266,
                beta_out=0.0731, gamma_out=1.00266):
    """Fake ``EnvelopeResults`` with the fields the 4D evaluator reads."""
    return SimpleNamespace(
        emit_4d=[emit_4d_in, emit_4d_out],
        emit_z=[emit_z_in, emit_z_out],
        ref_beta=[beta_in, beta_out],
        ref_gamma=[gamma_in, gamma_out],
    )


def test_4d_residual_zero_when_emit_unchanged():
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    res = ev(_results_4d(emit_4d_in=1.0, emit_4d_out=1.0,
                         emit_z_in=0.40, emit_z_out=0.40), None)
    assert res.shape == (2,)
    np.testing.assert_allclose(res, [0.0, 0.0])


def test_4d_residual_accepts_exchange():
    """ε_4D is invariant under x-y exchange: εx doubles while εy halves
    leaves det(Σ_4x4) unchanged.  We model this by keeping emit_4d
    constant — the evaluator only sees the determinant via emit_4d, so
    arbitrary x-y exchange that preserves the 4D volume yields a zero
    residual."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    # emit_4d unchanged: free exchange between transverse planes is OK.
    res = ev(_results_4d(emit_4d_in=1.0, emit_4d_out=1.0), None)
    assert res[0] == 0.0


def test_4d_residual_positive_on_real_growth():
    """Honest 4-D growth (det Σ_4x4 increased) shows up as r_4d > 0."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    # bg constant; ε_4D_norm = ε_4D · (bg)² = 1.0 · (0.0731·1.00266)²
    # → in = 0.00538; out = 1.3 · 0.00538 = 0.00699; r_4d = 0.00161
    res = ev(_results_4d(emit_4d_in=1.0, emit_4d_out=1.3), None)
    bg2 = (0.0731 * 1.00266) ** 2
    expected = (1.3 - 1.0) * bg2
    assert res[0] == pytest.approx(expected, abs=1e-6)


def test_4d_tolerance_band_eats_growth():
    """tol_4d = 1.3 should make a 30 % 4-D growth produce zero residual."""
    ev = _make_min_emit_4d_growth_evaluator(
        MinEmit4DGrowth("g", tol_4d=1.3))
    res = ev(_results_4d(emit_4d_in=1.0, emit_4d_out=1.3), None)
    assert res[0] == 0.0
    # Slightly over the tolerance band -- residual reappears.
    res2 = ev(_results_4d(emit_4d_in=1.0, emit_4d_out=1.31), None)
    assert res2[0] > 0.0


def test_4d_z_norm_uses_betagamma():
    """r_z is computed on the normalised z emittance (emit_z * βγ),
    not the native emit_z."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    # bg constant 0.0731 · 1.00266 ≈ 0.0733
    res = ev(_results_4d(emit_z_in=0.40, emit_z_out=0.50), None)
    bg = 0.0731 * 1.00266
    expected = (0.50 - 0.40) * bg
    assert res[1] == pytest.approx(expected, abs=1e-6)


def test_4d_z_normalisation_immunity_to_adiabatic_damping():
    """Under acceleration, emit_z_geom drops by βγ ratio so the
    normalised emit_z (= emit_z · βγ) stays constant.  Residual = 0."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    bg_in = 0.0731 * 1.00266        # ≈ 0.0733
    bg_out = 0.151 * 1.011          # ≈ 0.1527 (accelerated ~10x mass-energy)
    # Pick emit_z_out such that emit_z_out · bg_out = emit_z_in · bg_in
    emit_z_in_native = 0.40
    emit_z_out_native = emit_z_in_native * (bg_in / bg_out)
    res = ev(_results_4d(emit_z_in=emit_z_in_native,
                         emit_z_out=emit_z_out_native,
                         beta_in=0.0731, gamma_in=1.00266,
                         beta_out=0.151, gamma_out=1.011), None)
    assert res[1] == pytest.approx(0.0, abs=1e-9)


def test_4d_z_drop_is_free():
    """One-sided semantics on z plane: emit dropping below entry value
    contributes zero residual."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    res = ev(_results_4d(emit_z_in=0.40, emit_z_out=0.30), None)
    assert res[1] == 0.0


def test_4d_empty_results_returns_zero():
    """A constraint with no data (empty envelope results) does not crash
    — it returns the two-zero residual so the matcher continues."""
    ev = _make_min_emit_4d_growth_evaluator(MinEmit4DGrowth("g"))
    empty = SimpleNamespace(
        emit_4d=[], emit_z=[], ref_beta=[], ref_gamma=[])
    res = ev(empty, None)
    np.testing.assert_allclose(res, [0.0, 0.0])


def test_4d_collect_constraints_picks_up():
    lat = Lattice()
    lat.add(MinEmit4DGrowth("g4d", weight=2.0, tol_4d=1.1, tol_z=1.2))
    cs = collect_constraints(lat)
    assert len(cs) == 1
    assert cs[0].label == "MIN_EMIT_4D_GROWTH"
    assert cs[0].weight == 2.0


def test_4d_collect_constraints_skips_zero_weight():
    lat = Lattice()
    lat.add(MinEmit4DGrowth("g", weight=0.0))
    cs = collect_constraints(lat)
    assert cs == []


def test_4d_tolerance_below_one_rejected():
    """tol < 1.0 doesn't make physical sense (1.0 = strict, >1 = relaxed).
    Constructor rejects it to surface the bug at parse time, not silently
    mis-direct the optimiser."""
    with pytest.raises(ValueError, match="tolerances must be >= 1.0"):
        MinEmit4DGrowth("g", tol_4d=0.9)
    with pytest.raises(ValueError, match="tolerances must be >= 1.0"):
        MinEmit4DGrowth("g", tol_z=0.5)


def test_4d_round_trip_parser_writer(tmp_path):
    """Write a MIN_EMIT_4D_GROWTH command to a .dat file, read it back,
    confirm fields round-trip correctly."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    lat = Lattice()
    lat.add(MinEmit4DGrowth("g", weight=0.5, tol_4d=1.05, tol_z=1.20))
    out = tmp_path / "out.dat"
    write_tracewin(lat, str(out))
    lat2, _ = parse_tracewin(str(out))

    cmds = [e for e in lat2.elements if isinstance(e, MinEmit4DGrowth)]
    assert len(cmds) == 1
    assert cmds[0].weight == pytest.approx(0.5)
    assert cmds[0].tol_4d == pytest.approx(1.05)
    assert cmds[0].tol_z == pytest.approx(1.20)
