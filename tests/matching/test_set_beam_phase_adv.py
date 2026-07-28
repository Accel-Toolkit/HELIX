"""SET_BEAM_PHASE_ADV evaluator — TraceWin span/zero-target semantics.

The card applies from ITS OWN position through the N following
non-command elements (TW manual: "it applies to the entrance of the
element where it is placed").  A zero target disables that plane;
invalid β spans produce a fixed penalty (never a fabricated β = 1);
μ_z is measured via the effective (z, z′) β.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import SetBeamPhaseAdv
from linac_gen.matching.constraints import (
    _make_set_beam_phase_adv_evaluator, collect_constraints,
)

_MRAD_TO_DEG = (180.0 / math.pi) * 1e-3


def _fake_results(lattice: Lattice, beta: float = 2.0):
    """One-record-per-element results with constant β; s accumulates
    the ACTUAL element lengths (zero-length cards advance nothing)."""
    from linac_gen.tracking.envelope import EnvelopeResults
    s_vals = [0.0]
    for e in lattice.elements:
        s_vals.append(s_vals[-1] + float(getattr(e, "length", 0.0) or 0.0))
    n = len(s_vals)
    return EnvelopeResults(
        s=s_vals,
        beta_x=[beta] * n, beta_y=[beta] * n,
        alpha_x=[0.0] * n, alpha_y=[0.0] * n,
        sigma_x=[1.0] * n, sigma_y=[1.0] * n,
        sigma_phi=[5.0] * n, sigma_w=[0.01] * n,
        element_names=["INPUT"] + [getattr(e, "name", "?")
                                   for e in lattice.elements],
        element_exit_idx=list(range(1, n)),
    )


def _lattice(n_drifts: int, card_at: int, n_span: int) -> Lattice:
    lat = Lattice()
    for i in range(n_drifts):
        if i == card_at:
            lat.add(SetBeamPhaseAdv(name="SPA", k=1.0, n_elems=n_span,
                                    mu_x_deg=10.0, mu_y_deg=10.0))
        lat.add(Drift(name=f"D{i}", length=100.0, aperture=10.0))
    return lat


def test_span_is_n_elements_following_the_card():
    """Card mid-lattice: μ measured over exactly its N next elements."""
    lat = _lattice(n_drifts=6, card_at=2, n_span=3)
    cmd_index = 2  # the card sits at raw index 2 (before D2)
    assert isinstance(lat.elements[cmd_index], SetBeamPhaseAdv)
    cmd = lat.elements[cmd_index]
    res = _fake_results(lat)
    ev = _make_set_beam_phase_adv_evaluator(cmd, cmd_index=cmd_index)
    r = ev(res, lat)
    # 3 drifts × 100 mm at β = 2 mm/mrad.
    mu_expected = (300.0 / 2.0) * _MRAD_TO_DEG
    assert r[0] == pytest.approx(mu_expected - 10.0, rel=1e-9)
    assert r[1] == pytest.approx(mu_expected - 10.0, rel=1e-9)
    assert r[2] == 0.0                       # μ_z target 0 ⇒ disabled


def test_zero_target_disables_plane():
    lat = _lattice(n_drifts=4, card_at=0, n_span=2)
    cmd = lat.elements[0]
    cmd.mu_y_deg = 0.0                       # disable y
    res = _fake_results(lat)
    ev = _make_set_beam_phase_adv_evaluator(cmd, cmd_index=0)
    r = ev(res, lat)
    assert r[0] != 0.0
    assert r[1] == 0.0                       # disabled, NOT driven to zero
    assert r[2] == 0.0


def test_invalid_beta_penalizes_not_fabricates():
    lat = _lattice(n_drifts=4, card_at=0, n_span=2)
    cmd = lat.elements[0]
    res = _fake_results(lat)
    res.beta_x = [0.0] * len(res.beta_x)     # beam lost in x everywhere
    ev = _make_set_beam_phase_adv_evaluator(cmd, cmd_index=0)
    r = ev(res, lat)
    assert r.shape == (3,)                   # dimensionality stable
    assert r[0] == pytest.approx(1.0e3)      # penalty, not β=1 fabrication
    assert abs(r[1]) < 1.0e3


def test_mu_z_wired_via_effective_beta():
    lat = _lattice(n_drifts=4, card_at=0, n_span=2)
    cmd = lat.elements[0]
    cmd.mu_z_deg = 5.0
    res = _fake_results(lat)
    # Give the results a longitudinal σ-matrix + ref state so the
    # effective β̃_z exists.
    n = len(res.s)
    S = np.zeros((6, 6))
    S[4, 4], S[5, 5] = 16.0, 1e-4
    res.sigma_matrix = [S.copy() for _ in range(n)]
    res.ref_beta = [0.07] * n
    res.ref_gamma = [1.0027] * n
    res.ref_frequency = [162.5] * n
    res.ref_w_kin = [2.5] * n
    res.mass_mev = PROTON.mass
    ev = _make_set_beam_phase_adv_evaluator(cmd, cmd_index=0)
    r = ev(res, lat)
    assert np.isfinite(r[2]) and r[2] != 0.0 and abs(r[2]) < 1.0e3


def test_collect_constraints_passes_card_index():
    lat = _lattice(n_drifts=6, card_at=2, n_span=3)
    cons = [c for c in collect_constraints(lat)
            if c.label == "SET_BEAM_PHASE_ADV"]
    assert len(cons) == 1
    res = _fake_results(lat)
    r = cons[0].evaluator(res, lat)
    mu_expected = (300.0 / 2.0) * _MRAD_TO_DEG
    assert r[0] == pytest.approx(mu_expected - 10.0, rel=1e-9)


def test_trailing_card_disabled_not_penalized():
    """A card with no following elements must not poison the fit with a
    permanent, gradient-dead penalty — it is disabled with a warning."""
    lat = Lattice()
    for i in range(3):
        lat.add(Drift(name=f"D{i}", length=100.0, aperture=10.0))
    lat.add(SetBeamPhaseAdv(name="SPA", k=1.0, n_elems=2,
                            mu_x_deg=10.0, mu_y_deg=10.0))
    res = _fake_results(lat)
    ev = _make_set_beam_phase_adv_evaluator(lat.elements[3], cmd_index=3)
    with pytest.warns(UserWarning, match="no following elements"):
        r = ev(res, lat)
    assert np.all(r == 0.0)


def test_ne_zero_disables_card():
    """Ne = 0 follows TW's zero-means-off convention (as zero targets
    do), instead of inventing a 1-element span."""
    lat = _lattice(n_drifts=4, card_at=0, n_span=2)
    cmd = lat.elements[0]
    cmd.n_elems = 0
    res = _fake_results(lat)
    ev = _make_set_beam_phase_adv_evaluator(cmd, cmd_index=0)
    with pytest.warns(UserWarning, match="Ne <= 0"):
        r = ev(res, lat)
    assert np.all(r == 0.0)


def test_missing_plane_data_skipped_not_penalized():
    """μ_z target on results that never recorded a σ-matrix: the plane
    is unmeasurable for lack of DATA (not a lost beam) — skip with a
    warning instead of a constant penalty that distorts the whole fit."""
    lat = _lattice(n_drifts=4, card_at=0, n_span=2)
    cmd = lat.elements[0]
    cmd.mu_z_deg = 5.0
    res = _fake_results(lat)              # no sigma_matrix attached
    ev = _make_set_beam_phase_adv_evaluator(cmd, cmd_index=0)
    with pytest.warns(UserWarning, match="no z-plane data"):
        r = ev(res, lat)
    assert r[2] == 0.0                    # skipped
    assert abs(r[0]) < 1.0e3              # x/y still measured normally
