"""Tests for ``run_correction_from_lattice`` — the TraceWin-card driver."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
)
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.correction import run_correction_from_lattice


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


def _make_pair_lattice(n_pairs: int):
    """``ADJUST_STEERER N → STEERER → DRIFT → BPM`` repeated *n_pairs* times."""
    lat = Lattice()
    for n in range(1, n_pairs + 1):
        lat.add(Drift(f"D_pre{n}", 100.0))
        lat.add(AdjustSteerer(f"ADJ_{n}", diag_n=n, vmax=0.0,
                              first_step=1e-4))
        lat.add(Steerer(f"STEER_{n}", bx_l=0.0, by_l=0.0))
        lat.add(Drift(f"D_post{n}", 100.0))
        lat.add(Marker(f"BPM_{n}", is_bpm=True))
        lat.add(Drift(f"D_after{n}", 100.0))
    return lat


# ---------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------
def test_method_one_to_one_when_clean_pairing():
    """4 cards, 4 BPMs, 1:1 pairing → driver picks one_to_one."""
    lat = _make_pair_lattice(4)
    res = run_correction_from_lattice(lat, _factory(), history=True)
    assert res["method"] == "one_to_one"
    assert res["n_pairs"] == 4


def test_method_svd_when_overdetermined():
    """5 BPMs, 2 cards → underdetermined wrt BPMs → SVD."""
    lat = _make_pair_lattice(2)
    # Add 3 extra BPMs at the end so n_bpms (5) != n_cards (2).
    for n in range(3, 6):
        lat.add(Drift(f"D_x{n}", 50.0))
        lat.add(Marker(f"BPM_{n}", is_bpm=True))
    res = run_correction_from_lattice(lat, _factory(), history=True)
    assert res["method"] == "svd"


def test_override_method_force_svd():
    """User can force SVD even on a clean 1:1 lattice."""
    lat = _make_pair_lattice(3)
    res = run_correction_from_lattice(
        lat, _factory(), override_method="svd", history=True,
    )
    assert res["method"] == "svd"


# ---------------------------------------------------------------------
# diag_n resolution
# ---------------------------------------------------------------------
def test_diag_n_picks_nth_bpm_only():
    """ADJUST_STEERER 3 must point at the 3rd ``is_bpm`` marker (NOT
    the 3rd Marker overall)."""
    lat = Lattice()
    # Decoy snapshot markers (NOT BPMs).
    lat.add(Marker("MARK_a", snapshot=True))
    lat.add(Drift("D1", 50.0))
    lat.add(Marker("MARK_b", snapshot=False))
    lat.add(Drift("D2", 50.0))
    # Real BPMs.
    lat.add(Marker("BPM_1", is_bpm=True))
    lat.add(Drift("D3", 50.0))
    lat.add(Marker("BPM_2", is_bpm=True))
    lat.add(Drift("D4", 50.0))
    lat.add(Marker("BPM_3", is_bpm=True))
    lat.add(Drift("D5", 50.0))
    # ADJUST_STEERER 3 points at BPM_3.
    lat.add(AdjustSteerer("ADJ_3", diag_n=3, vmax=0.0, first_step=1e-4))
    lat.add(Steerer("STEER_3", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D6", 100.0))
    lat.add(Marker("BPM_4_decoy", is_bpm=True))  # would-be 4th BPM after the steerer

    res = run_correction_from_lattice(lat, _factory(), history=True)
    assert res["n_pairs"] == 1
    assert "STEER_3" in res["kicks"]


def test_diag_n_out_of_range_skipped():
    """Card with diag_n > #BPMs is skipped with a warning."""
    lat = Lattice()
    lat.add(Marker("BPM_1", is_bpm=True))
    lat.add(Drift("D1", 50.0))
    lat.add(AdjustSteerer("ADJ_99", diag_n=99, vmax=0.0, first_step=1e-4))
    lat.add(Steerer("STEER_X"))
    res = run_correction_from_lattice(lat, _factory(), history=True)
    assert res["n_pairs"] == 0


# ---------------------------------------------------------------------
# target_name override
# ---------------------------------------------------------------------
def test_target_name_overrides_next_steerer():
    """When ``target_name`` is set, the driver pairs with the named
    steerer rather than the default 'next-after-card' one."""
    lat = Lattice()
    lat.add(Drift("D0", 50.0))
    card = AdjustSteerer("ADJ_1", diag_n=1, vmax=0.0, first_step=1e-4)
    card.target_name = "STEER_FAR"
    lat.add(card)
    lat.add(Steerer("STEER_NEXT", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D1", 50.0))
    lat.add(Steerer("STEER_FAR", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D2", 200.0))
    lat.add(Marker("BPM_1", is_bpm=True))
    lat.add(Drift("D3", 50.0))
    res = run_correction_from_lattice(
        lat, _factory(), n_iter=1, history=True,
    )
    # Only STEER_FAR should appear in the kicks — STEER_NEXT was bypassed
    # because the card explicitly named STEER_FAR.
    assert "STEER_FAR" in res["kicks"]
    assert "STEER_NEXT" not in res["kicks"]


# ---------------------------------------------------------------------
# Plane masks (Bx/By subclasses)
# ---------------------------------------------------------------------
def test_adjust_steerer_bx_only_corrects_x():
    """ADJUST_STEERER_BX should only affect the x plane via by_l."""
    lat = Lattice()
    lat.add(Drift("D0", 100.0))
    lat.add(AdjustSteererBx("ADJX", diag_n=1, vmax=0.0,
                            first_step=1e-4))
    lat.add(Steerer("STEER", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Marker("BPM", is_bpm=True))
    lat.add(Drift("D2", 100.0))
    res = run_correction_from_lattice(lat, _factory(offset_x=2, offset_y=2),
                                      history=True)
    assert res["n_pairs"] == 1


# ---------------------------------------------------------------------
# vmax honoured from card
# ---------------------------------------------------------------------
def test_vmax_from_card_clips_kick():
    lat = _make_pair_lattice(1)
    # Set tiny vmax on the card.
    for e in lat.elements:
        if isinstance(e, AdjustSteerer):
            e.vmax = 1e-7
    res = run_correction_from_lattice(
        lat, _factory(offset_x=10.0, offset_y=10.0),
        n_iter=3, history=True,
    )
    kicks = res["kicks"]
    for k in kicks.values():
        assert abs(k["bx_l"]) <= 1e-7 + 1e-12
        assert abs(k["by_l"]) <= 1e-7 + 1e-12


def test_refuses_electric_partner_steerer():
    """Adversarial-review find (2026-07 round 2): the driver hard-codes
    the magnetic plane mapping (by_l→x′ / bx_l→y′) and T·m unit
    conversions — an ELECTRIC partner steerer (same-plane volt knobs,
    1/(βc·Bρ) response) must be refused, not silently mis-corrected."""
    lat = _make_pair_lattice(1)
    for e in lat.elements:
        if isinstance(e, Steerer):
            e.elec = True
    with pytest.raises(ValueError, match="ELECTRIC"):
        run_correction_from_lattice(lat, _factory(), history=True)
