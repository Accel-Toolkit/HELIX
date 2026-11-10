"""The ``gradient`` matcher algorithm — exact-Jacobian Levenberg–Marquardt
driven by the differentiable matrix-tracking engine.

The gradient algorithm must reproduce the ``least_squares`` result on the
problems it supports (SET_TWISS / SET_SIZE matching of quad / solenoid /
dipole knobs, no space charge), and must reject — with a clear error —
anything outside that subset.
"""
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.lattice_commands import (
    Adjust, SetSize, SetSizeMax, SetTwiss,
)
from linac_gen.matching import MATCH_ALGORITHMS, match


def _bcfg(**over) -> BeamConfig:
    base = dict(species="proton", energy=3.0, frequency=352.21,
                n_particles=10, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                emit_z=0.3, alpha_z=0.0, beta_z=10.0)
    base.update(over)
    return BeamConfig(**base)


def _fodo_set_size(k_init: float = 5.0) -> Lattice:
    """Single-quad cell: ADJUST on the quad gradient, SET_SIZE at the end."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0,
                    vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=k_init,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0, y_mm=0.0, phi_or_z=0.0))
    return lat


# ---------------------------------------------------------------------------
def test_gradient_in_match_algorithms():
    assert "gradient" in MATCH_ALGORITHMS


def _with_card_before_quad(card):
    lat = Lattice()
    src = _fodo_set_size()
    lat.add(src.elements[0])
    lat.add(card)
    for el in src.elements[1:]:
        lat.add(el)
    return lat


def _cards():
    from linac_gen.elements.lattice_commands import Freq, SetReferenceEnergy
    return {"FREQ": lambda: Freq("F1", frequency_mhz=704.42),
            "HELIX_SET_ENERGY": lambda: SetReferenceEnergy("SBE", energy_MeV=5.0),
            "HELIX_SHIFT_ENERGY": lambda: SetReferenceEnergy("SE", energy_MeV=2.0, shift=True)}


@pytest.mark.parametrize("card", ["FREQ", "HELIX_SET_ENERGY", "HELIX_SHIFT_ENERGY"])
def test_gradient_agrees_with_least_squares_across_a_card_that_sets_the_reference(card):
    """It used to refuse these: the torch mirror composed them as the identity, so it
    would have solved another lattice.  It now follows them, as the numpy walk does:
    the same quad gradient as ``least_squares``, and the match's own check of the torch
    residual against the numpy one at the start passes."""
    make = _cards()[card]
    grad = match(_with_card_before_quad(make()), _bcfg(), algorithm="gradient", max_iter=40)
    lsq = match(_with_card_before_quad(make()), _bcfg(), algorithm="least_squares", max_iter=200)
    assert grad.cost < 1e-6
    assert lsq.cost < 1e-6
    assert grad.x_final[0] == pytest.approx(lsq.x_final[0], rel=1e-3)
    plain = match(_fodo_set_size(), _bcfg(), algorithm="gradient", max_iter=40).x_final[0]
    if card == "FREQ":              # a clock does not focus
        assert grad.x_final[0] == pytest.approx(plain, rel=1e-6)
    else:                           # an energy does
        assert abs(grad.x_final[0] / plain - 1.0) > 1e-3


@pytest.mark.parametrize("card", ["FREQ", "HELIX_SET_ENERGY", "HELIX_SHIFT_ENERGY"])
def test_gradient_through_space_charge_runs_across_a_card(card):
    """Through PIC there is no numpy check at the start, so the step tracker's own
    agreement with the numpy tracker across a card is in its test file."""
    make = _cards()[card]
    r = match(_with_card_before_quad(make()), _bcfg(current=1.0), algorithm="gradient",
              space_charge=True, max_iter=3)
    assert r.cost >= 0.0


def test_gradient_accepts_ratio_one_freq_card():
    """The header FREQ that opens virtually every .dat deck is a no-op
    when it matches the beam frequency (jump ratio 1: the freq-jump D
    is the identity) — refusing it would kill the gradient algorithm
    for every imported lattice (adversarial-review finding)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.lattice_commands import Freq
    lat = Lattice()
    lat.add(Freq("F0", frequency_mhz=352.21))     # == _bcfg() frequency
    for el in _fodo_set_size().elements:
        lat.add(el)
    r = match(lat, _bcfg(), algorithm="gradient", max_iter=40)
    assert r.cost < 1e-6


def test_gradient_refuses_longitudinal_twiss_flags():
    """kaz/kbz used to become zero-gradient CONSTANTS in the torch
    residual (silent under SC; cryptic mismatch error without)."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0,
                    vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(SetTwiss("CTW", kax=1, kbx=1, kaz=1,
                     alpha_x=0.0, beta_x=2.0, alpha_z=0.0, beta_z=10.0))
    with pytest.raises(ValueError, match="kaz"):
        match(lat, _bcfg(), algorithm="gradient")


def test_gradient_accepts_passive_command_cards():
    """The passive SET_*/ADJUST_* markers must NOT be over-refused —
    identity is faithful for them."""
    r = match(_fodo_set_size(), _bcfg(), algorithm="gradient",
              max_iter=40)
    assert r.cost < 1e-6


@pytest.mark.parametrize("sc", [False, True])
def test_gradient_refuses_centroid_constraints(sc):
    """Since the envelope carries a real first moment (2026-07-19) the
    numpy SET_POSITION residual is no longer a zero stub — but the
    torch mirror propagates Σ only.  The gradient algorithm must refuse
    loudly in BOTH regimes: without SC it used to die with a cryptic
    residual-mismatch error, with SC it silently DROPPED the constraint
    and reported a cost that hid the violation."""
    from linac_gen.elements.lattice_commands import SetPosition
    lat = _fodo_set_size()
    lat.add(SetPosition("SP", k=2.0, x_mm=0.5, xp_mrad=0.1,
                        y_mm=-0.3, yp_mrad=0.0))
    with pytest.raises(ValueError, match="centroid constraint"):
        match(lat, _bcfg(centroid_x=1.0, centroid_y=-0.5),
              algorithm="gradient", space_charge=sc)


def test_gradient_matches_least_squares_set_size():
    """On a SET_SIZE problem the gradient algorithm lands on the same
    solution as least_squares (same local LM problem, exact vs FD jac)."""
    r_lsq = match(_fodo_set_size(), _bcfg(), algorithm="least_squares",
                  max_iter=80)
    r_grad = match(_fodo_set_size(), _bcfg(), algorithm="gradient",
                   max_iter=80)
    assert r_grad.success
    assert r_grad.cost < 1e-9
    assert r_grad.x_final[0] == pytest.approx(r_lsq.x_final[0], abs=1e-4)


def test_gradient_matches_least_squares_set_twiss():
    """SET_TWISS (end-of-lattice beta_x) matching via the gradient path."""
    def build():
        lat = Lattice()
        lat.add(Drift("D1", length=200.0, aperture=10.0))
        lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0,
                        vmin=-30, vmax=30, start_step=0.5))
        lat.add(Quadrupole("QUAD_001", length=100.0, gradient=6.0,
                           aperture=10.0))
        lat.add(Drift("D2", length=300.0, aperture=10.0))
        lat.add(SetTwiss("CT", family="END", beta_x=1.5, kbx=1))
        return lat

    r_lsq = match(build(), _bcfg(), algorithm="least_squares", max_iter=120)
    r_grad = match(build(), _bcfg(), algorithm="gradient", max_iter=120)
    assert r_grad.x_final[0] == pytest.approx(r_lsq.x_final[0], abs=1e-4)
    assert r_grad.cost == pytest.approx(r_lsq.cost, abs=1e-9)


def test_gradient_handles_linked_quads():
    """Two ADJUST cards sharing a link group collapse to one optimiser
    column; the gradient path handles that exactly as least_squares."""
    def build():
        lat = Lattice()
        lat.add(Drift("D1", length=100.0, aperture=10.0))
        lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=7,
                        vmin=-30, vmax=30))
        lat.add(Quadrupole("QUAD_001", length=50.0, gradient=10.0,
                           aperture=10.0))
        lat.add(Adjust("CMD2", target="QUAD", param_idx=2, link_group=7,
                        vmin=-30, vmax=30))
        lat.add(Quadrupole("QUAD_002", length=50.0, gradient=-10.0,
                           aperture=10.0))
        lat.add(Drift("D2", length=100.0, aperture=10.0))
        lat.add(SetSize("CSET", k=1.0, x_mm=3.0))
        return lat

    r_lsq = match(build(), _bcfg(), algorithm="least_squares", max_iter=80)
    r_grad = match(build(), _bcfg(), algorithm="gradient", max_iter=80)
    assert r_grad.x0.shape == (1,)        # one column for the linked pair
    assert r_grad.x_final[0] == pytest.approx(r_lsq.x_final[0], abs=1e-4)


def test_gradient_zero_crossing_quad():
    """Regression: a quad gradient driven through zero must not trap the
    optimiser — the differentiable quad matrix stays differentiable at 0."""
    res = match(_fodo_set_size(k_init=5.0), _bcfg(), algorithm="gradient",
                max_iter=80)
    assert res.success and res.cost < 1e-9


# ---------------------------------------------------------------------------
# Rejection of unsupported problems — must raise a clear ValueError.
# ---------------------------------------------------------------------------
def test_gradient_accepts_space_charge():
    """Gradient matching now supports space charge — the residual is a
    macro-particle bunch tracked through the differentiable PIC step
    tracker (see tests/matching/test_torch_sc_matching.py for the
    quantitative recovery test)."""
    result = match(_fodo_set_size(), _bcfg(current=5.0),
                   algorithm="gradient", space_charge=True, max_iter=40)
    assert result is not None
    assert result.x_final.shape == (1,)


def test_gradient_rejects_non_quad_variable():
    """An ADJUST on a Drift length is not a differentiable knob."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="D1", param_idx=1, link_group=0,
                    vmin=100, vmax=400))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0))
    with pytest.raises(ValueError, match="gradient.*algorithm tunes"):
        match(lat, _bcfg(), algorithm="gradient", max_iter=20)


def test_gradient_rejects_set_size_max():
    """SET_SIZE_MAX is a one-sided bound the gradient path does not model."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0,
                    vmin=-30, vmax=30))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSizeMax("CM", k=1.0, n_elems=1, x_mm=5.0))
    with pytest.raises(ValueError, match="SET_TWISS and SET_SIZE"):
        match(lat, _bcfg(), algorithm="gradient", max_iter=20)


def _foil():
    from linac_gen.elements.foil import Foil
    return Foil("F", material="C", thickness_ug_cm2=600.0, dedx_model="bethe")


def _gaining_matrix_element(gain=0.5):
    import numpy as np

    from linac_gen.elements.matrix_element import MatrixElement
    return MatrixElement("M", np.eye(6), ref_gain=gain)


@pytest.mark.parametrize("make, what", [
    (_foil, "foil"),
    (_gaining_matrix_element, "changes the reference energy by 0.5 MeV"),
    (lambda: _gaining_matrix_element(-0.5), "changes the reference energy by -0.5 MeV")],
    ids=["a foil", "a matrix element that gains energy", "one that loses energy"])
def test_an_element_the_mirror_would_ignore_is_refused(make, what):
    """Both have an identity map and passed as no-ops, but a foil scatters the beam and slows
    the reference, and the element moves it: with a quadrupole after them (3 MeV protons) the
    mirror gave sigma_x 0.608 mm where the envelope gives 2.300 behind a 600 ug/cm2 carbon foil,
    and 0.712 behind a +0.5 MeV gain (2026-11-08)."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching.torch_objective import check_gradient_supported
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(make())
    lat.add(Drift("D2", length=300.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    with pytest.raises(ValueError, match=f"{what}.*least_squares"):
        check_gradient_supported(lat, ref, [], [])


def test_a_foil_of_no_thickness_is_accepted():
    """It does nothing in the envelope either (no scattering, no loss)."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.foil import Foil
    from linac_gen.matching.torch_objective import check_gradient_supported
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Foil("F0", material="C", thickness_ug_cm2=0.0))
    lat.add(Drift("D2", length=300.0))
    check_gradient_supported(lat, ReferenceParticle(species=PROTON, w_kin=3.0,
                                                    frequency=352.21), [], [])


@pytest.mark.parametrize("gap_f, supported", [(704.42, True), (352.21, False)],
                         ids=["at the card\'s clock", "at the old clock"])
def test_the_support_check_judges_a_parked_gap_at_the_clock_it_sits_on(gap_f, supported):
    """At the card\'s clock such a gap is nothing.  At the old one it makes a jump back,
    which the mirror does not hold: with space charge no self-check would catch that."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.lattice_commands import Freq
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.matching.torch_objective import check_gradient_supported
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Freq("F", frequency_mhz=704.42))
    lat.add(RFGap("G", voltage=0.0, phase=-30.0, frequency=gap_f, ttf=1.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    if supported:
        check_gradient_supported(lat, ref, [], [])
    else:
        with pytest.raises(ValueError, match="all-linear"):
            check_gradient_supported(lat, ref, [], [])
