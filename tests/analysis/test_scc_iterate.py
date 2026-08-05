"""Self-consistent SCC iterator (analysis/scc/iterate.py + apply.py).

The one-shot analysis computes f_c from sigma(z) tracked at a DIFFERENT
compensation state; the iterator closes the loop.  Placement semantics
in apply.py are the popup's Apply, lifted headless — the GUI test pins
the popup seam, these pin the shared function itself.
"""
from __future__ import annotations

import pytest

from linac_gen.analysis.scc.apply import apply_scc_cards
from linac_gen.analysis.scc.iterate import scc_self_consistent
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp


def _lebt():
    lat = Lattice()
    lat.add(Drift(name="D1", length=300.0, aperture=60.0))
    lat.add(Solenoid(name="S1", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D2", length=900.0, aperture=60.0))
    lat.add(Solenoid(name="S2", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D3", length=300.0, aperture=60.0))
    return lat


def _cfg(current=15.0, continuous=True):
    return BeamConfig(species="H-", energy=0.030, frequency=162.5,
                      current=current, duty_cycle=100.0, n_particles=200,
                      distribution="gaussian", cutoff=3.0,
                      emit_nx=0.2, alpha_x=0.0, beta_x=0.6,
                      emit_ny=0.2, alpha_y=0.0, beta_y=0.6,
                      emit_z=0.0, alpha_z=0.0, beta_z=1.0,
                      continuous=continuous)


# ---------------------------------------------------------------- apply --

def test_apply_places_at_nearest_boundary_and_merges():
    lat = _lebt()                     # boundaries: 0, .3, .55, 1.45, 1.7, 2.0
    inserted = apply_scc_cards(lat, [
        {"z_m": 0.0, "factor": 0.5},
        {"z_m": 0.29, "factor": 0.2},     # -> boundary 0.3 (idx 1)
        {"z_m": 0.31, "factor": 0.4},     # -> boundary 0.3: merge to 0.3
    ])
    sccs = [(i, e) for i, e in enumerate(lat.elements)
            if isinstance(e, SpaceChargeComp)]
    assert inserted == {0: 0.5, 1: pytest.approx(0.3)}
    assert [i for i, _ in sccs] == [0, 2]          # idx 1 shifts past card 0
    assert [e.factor for _, e in sccs] == [0.5, pytest.approx(0.3)]


def test_apply_is_idempotent_and_strips_existing():
    lat = _lebt()
    lat.insert(0, SpaceChargeComp(name="OLD", factor=0.9))
    cards = [{"z_m": 0.0, "factor": 0.5}, {"z_m": 2.0, "factor": 0.1}]
    apply_scc_cards(lat, cards)
    snapshot = [(type(e).__name__, e.name,
                 float(getattr(e, "factor", -1.0))) for e in lat.elements]
    assert not any(n == "OLD" for _, n, _f in snapshot)
    apply_scc_cards(lat, cards)                     # re-apply: no drift
    again = [(type(e).__name__, e.name,
              float(getattr(e, "factor", -1.0))) for e in lat.elements]
    assert again == snapshot


# -------------------------------------------------------------- iterate --

@pytest.mark.physics
def test_iterate_converges_and_leaves_lattice_untouched():
    lat = _lebt()
    a = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5)
    assert a["reason"] is None
    it = a["iterate"]
    assert it["converged"] and 1 <= it["n_iter"] <= 8
    assert len(it["history"]) == it["n_iter"]
    # iteration 1 measures the bare line: the first step must be large,
    # and the SECOND smaller but nonzero — proof the re-run with cards
    # actually changed the balance (the sigma coupling is live)
    assert it["history"][0]["max_delta"] > 0.5
    if it["n_iter"] > 1:
        assert it["history"][1]["max_delta"] < 0.1
    assert a["fc_source"].startswith("self-consistent")
    assert all(0.0 <= c["factor"] <= 1.0 for c in a["scc_cards"])
    # the advisory one-shot note must NOT appear on a self-consistent run
    assert not any("one-shot" in n for n in a["notes"])
    # the caller's lattice is never mutated
    assert len(lat.elements) == 5
    assert not any(isinstance(e, SpaceChargeComp) for e in lat.elements)


@pytest.mark.physics
def test_iterate_fixed_point_is_omega_independent():
    lat = _lebt()
    a1 = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5)
    a2 = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5,
                             omega=0.5, max_iter=12)
    assert a1["iterate"]["converged"] and a2["iterate"]["converged"]
    f1 = [c["factor"] for c in a1["scc_cards"]]
    f2 = [c["factor"] for c in a2["scc_cards"]]
    assert all(abs(x - y) <= 0.03 for x, y in zip(f1, f2))


@pytest.mark.physics
def test_iterate_zero_current_trivially_converges():
    a = scc_self_consistent(_lebt(), _cfg(current=0.0),
                            gas="N2", pressure_mbar=2e-5)
    assert a["reason"] is None
    assert a["iterate"]["converged"] and a["iterate"]["n_iter"] == 1
    assert all(c["factor"] == 0.0 for c in a["scc_cards"])


@pytest.mark.physics
def test_iterate_strips_preexisting_cards_with_note():
    lat = _lebt()
    lat.insert(0, SpaceChargeComp(name="OLD", factor=0.7))
    a = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5)
    assert a["reason"] is None
    assert any("set aside" in n for n in a["notes"])
    # caller's card untouched
    assert isinstance(lat.elements[0], SpaceChargeComp)
    assert lat.elements[0].factor == 0.7


def test_iterate_refuses_bunched_config():
    a = scc_self_consistent(_lebt(), _cfg(continuous=False),
                            gas="N2", pressure_mbar=2e-5)
    assert a["reason"] is not None and "DC" in a["reason"]
    assert a["iterate"]["converged"] is False


def test_iterate_abort_raises_verbatim():
    with pytest.raises(RuntimeError, match="^aborted$"):
        scc_self_consistent(_lebt(), _cfg(), gas="N2",
                            pressure_mbar=2e-5, should_stop=lambda: True)


def test_iterate_rejects_bad_loop_parameters():
    with pytest.raises(ValueError, match="omega"):
        scc_self_consistent(_lebt(), _cfg(), omega=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        scc_self_consistent(_lebt(), _cfg(), max_iter=0)
    # tol >= 1 would declare the all-zero first pass "converged"
    with pytest.raises(ValueError, match="tol"):
        scc_self_consistent(_lebt(), _cfg(), tol=10.0)


@pytest.mark.physics
def test_iterate_with_cleared_region_translates_indices():
    """Applied cards shift the work lattice's element indices each
    iteration — the cleared range must keep tracking the CALLER's
    elements, and the returned analysis reports the caller's indices."""
    import numpy as np
    lat = _lebt()
    a = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5,
                            cleared_regions=[[2, 2]], cleared_residual=0.0)
    assert a["reason"] is None and a["iterate"]["converged"]
    assert a["cleared_regions"] == [[2, 2]]
    mid = int(np.argmin(np.abs(np.asarray(a["z_m"]) - 1.0)))
    assert a["fc"][mid] <= 0.05
    inner = [c["factor"] for c in a["scc_cards"]
             if 0.7 <= c["z_m"] <= 1.2]
    assert inner and max(inner) < 0.4      # cards inherit the clearing


@pytest.mark.physics
def test_iterate_cleared_region_on_card_bearing_lattice():
    """Caller indices count the caller's OWN elements — including any
    pre-existing cards the iterator strips (adversarial finding: the
    missing caller->stripped translation silently cleared the WRONG
    element on card-bearing decks)."""
    import numpy as np
    lat = _lebt()
    lat.insert(0, SpaceChargeComp(name="OLD", factor=0.5))
    # with the card, D2 is caller-index 3 (bare index 2)
    a = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5,
                            cleared_regions=[[3, 3]], cleared_residual=0.0)
    assert a["reason"] is None and a["iterate"]["converged"]
    assert a["cleared_regions"] == [[3, 3]]        # caller's own indexing
    mid = int(np.argmin(np.abs(np.asarray(a["z_m"]) - 1.0)))
    assert a["fc"][mid] <= 0.05                    # D2 core IS cleared
    # last element (caller index 5) also addressable
    b = scc_self_consistent(lat, _cfg(), gas="N2", pressure_mbar=2e-5,
                            cleared_regions=[[5, 5]], cleared_residual=0.0)
    assert b["reason"] is None
    tail = int(np.argmin(np.abs(np.asarray(b["z_m"]) - 1.85)))
    assert b["fc"][tail] <= 0.30                   # D3 core (edge-tapered)


def test_iterate_cleared_bound_on_a_card_refuses():
    lat = _lebt()
    lat.insert(0, SpaceChargeComp(name="OLD", factor=0.5))
    with pytest.raises(ValueError, match="SPACE_CHARGE_COMP"):
        scc_self_consistent(lat, _cfg(), cleared_regions=[[0, 0]])
