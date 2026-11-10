"""SET_BEAM_ENERGY and SET_BEAM_E0_P0 are TraceWin matching criteria; a HELIX card changes the reference energy.

The TraceWin manual (CEA, D. Uriot, 2021-12-31, p. 104, "Matching commands") defines ``SET_BEAM_ENERGY k Ei`` as the
criterion k(E - Ei)^2 and ``SET_BEAM_E0_P0 k dE dphi ke kp`` as one on the beam's energy and phase minus the linac's,
ke and kp switching the two terms; neither changes the beam.  HELIX applied both as reference-energy setters, so a
TraceWin deck that used ``SET_BEAM_ENERGY`` as a target had its energy forced.  The setter lives on as a HELIX card that
TraceWin reads as a comment: ``; HELIX_SET_ENERGY E`` and ``; HELIX_SHIFT_ENERGY dE``.
"""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    REFERENCE_COMMANDS,
    Adjust,
    SetBeamE0P0,
    SetBeamEnergy,
    SetReferenceEnergy,
)
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at
from linac_gen.tracking.tracker import Tracker

INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
            emit_z=0.2)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _line(card):
    """A cavity after the card, so that an energy the card set would show in the gain and the optics after it."""
    lat = Lattice()
    for e in (Drift("D1", 100.0), Quadrupole("Q", length=100.0, gradient=5.0), card, Drift("D2", 100.0),
              RFGap("G", voltage=0.5, phase=-30.0, frequency=352.21), Drift("D3", 100.0)):
        lat.add(e)
    return lat


def _runs(lat):
    """What the tracker, the envelope, the matrix walk and the reference walk give the line."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=50, current=0.0)
        beam.particles[:] = np.random.default_rng(1).normal(size=(50, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        Tracker(copy.deepcopy(lat), beam).run()
        env = EnvelopeSolver(copy.deepcopy(lat), _ref(), dict(INIT), current=0.0).run()
        return {"tracker": np.concatenate([beam.particles.ravel(), [beam.ref.w_kin, beam.ref.phi_s]]),
                "envelope": np.concatenate([np.asarray(env.sigma_matrix[-1]).ravel(), [env.ref_w_kin[-1]]]),
                "matrix": compute_transfer_matrix(copy.deepcopy(lat), _ref()),
                "reference": np.array([reference_at(copy.deepcopy(lat), _ref(), i).w_kin
                                       for i in range(len(lat.elements) + 1)])}


@pytest.mark.parametrize("card", [lambda: SetBeamEnergy("E", k=1, energy_MeV=4.0),
                                  lambda: SetBeamE0P0("E", k=1, dE_MeV=0.5, dphi_deg=10.0, ke=1, kp=1)],
                         ids=["SET_BEAM_ENERGY", "SET_BEAM_E0_P0"])
def test_a_tracewin_criterion_changes_no_walk(card):
    with_card, with_marker = _runs(_line(card())), _runs(_line(Marker("E")))
    for mode in with_card:
        np.testing.assert_array_equal(with_card[mode], with_marker[mode], err_msg=mode)
    assert not isinstance(card(), REFERENCE_COMMANDS)


@pytest.mark.parametrize("card, after", [(lambda: SetReferenceEnergy("E", energy_MeV=4.0), 4.0),
                                         (lambda: SetReferenceEnergy("E", energy_MeV=0.5, shift=True), 3.5)],
                         ids=["set", "shift"])
def test_the_helix_card_moves_the_reference_in_every_walk(card, after):
    lat = _line(card())
    got = _runs(lat)
    assert got["reference"][3] == after                                      # after the card, before the cavity
    gain = got["reference"][-1] - after
    assert got["tracker"][-2] == pytest.approx(after + gain, abs=1e-12)       # the tracker's reference at the end
    assert got["envelope"][-1] == pytest.approx(after + gain, abs=1e-12)
    assert isinstance(card(), REFERENCE_COMMANDS)


def test_the_helix_card_is_a_tracewin_comment_and_comes_back(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin
    deck = tmp_path / "e.dat"
    deck.write_text("DRIFT 100 20\n; HELIX_SET_ENERGY 4.25\nDRIFT 100 20\n;HELIX_SHIFT_ENERGY -0.5 ; a note\n"
                    "DRIFT 100 20\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_tracewin(str(deck))
        cards = [e for e in lat.elements if isinstance(e, SetReferenceEnergy)]
        assert [(c.energy_MeV, c.shift) for c in cards] == [(4.25, False), (-0.5, True)]
        out = tmp_path / "back.dat"
        write_tracewin(lat, out)
        text = out.read_text(encoding="latin-1")              # the deck encoding
        back, _ = parse_tracewin(str(out))
    lines = [ln for ln in text.splitlines() if "HELIX_" in ln and "ENERGY" in ln]
    assert len(lines) == 2 and all(ln.lstrip().startswith(";") for ln in lines)       # TraceWin skips them
    assert [(c.energy_MeV, c.shift) for c in back.elements if isinstance(c, SetReferenceEnergy)] == \
        [(4.25, False), (-0.5, True)]


def test_a_deck_with_the_tracewin_criteria_gets_one_notice(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "c.dat"
    deck.write_text("DRIFT 100 20\nSET_BEAM_ENERGY 1 120\nDRIFT 100 20\nSET_BEAM_E0_P0 1 0 0 1 1\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_tracewin(str(deck))
    notes = [w for w in meta["warnings"] if "HELIX_SET_ENERGY" in w]
    assert len(notes) == 1 and "matching" in notes[0]
    assert isinstance(lat.elements[1], SetBeamEnergy) and isinstance(lat.elements[3], SetBeamE0P0)


class _Results:
    """Two recorded rows: the input and the exit of the card (element 0)."""

    def __init__(self, w, dw, dphi, transmission=100.0):
        self.ref_w_kin = [3.0, w]
        self.ref_phi_s = [0.0, 50.0]
        self.centroid = [np.zeros(6), np.array([0, 0, 0, 0, dphi, dw], dtype=float)]
        self.element_exit_idx = [1]
        self.transmission = [100.0, transmission]
        self.s = [0.0, 0.0]


def _residuals(card, results):
    from linac_gen.matching.constraints import collect_constraints
    lat = Lattice()
    lat.add(card)
    cons = collect_constraints(lat)
    assert len(cons) == 1
    return cons[0].evaluate(results, lat)


def test_the_energy_criterion_is_the_beams_energy_against_the_target():
    """The beam's energy is the reference's and the centroid's offset; weighted by k, as HELIX's other SET_ cards."""
    r = _residuals(SetBeamEnergy("E", k=2, energy_MeV=4.0), _Results(w=4.1, dw=0.02, dphi=0.0))
    np.testing.assert_allclose(r, [2.0 * (4.12 - 4.0)], rtol=1e-12)


@pytest.mark.parametrize("ke, kp", [(1, 1), (1, 0), (0, 1)])
def test_the_energy_and_phase_criterion_is_the_beam_against_the_linac(ke, kp):
    """Beam minus linac is the centroid's offset from the reference at the card; ke and kp switch the terms."""
    r = _residuals(SetBeamE0P0("E", k=3, dE_MeV=0.05, dphi_deg=170.0, ke=ke, kp=kp),
                   _Results(w=4.0, dw=0.08, dphi=-175.0))
    np.testing.assert_allclose(r, [3.0 * ke * (0.08 - 0.05), 3.0 * kp * 15.0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("card", [lambda: SetBeamEnergy("E", k=1, energy_MeV=4.0),
                                  lambda: SetBeamE0P0("E", k=1, ke=1, kp=1)], ids=["energy", "energy and phase"])
def test_a_lost_beam_is_no_answer(card):
    """A beam lost at the card reads as a large residual, not as a beam on its target."""
    r = _residuals(card(), _Results(w=4.0, dw=0.0, dphi=0.0, transmission=0.0))
    assert np.all(r >= 1.0e3)


def test_cards_with_no_weight_make_no_criterion():
    from linac_gen.matching.constraints import collect_constraints
    lat = Lattice()
    for card in (SetBeamEnergy("A", k=0, energy_MeV=4.0), SetBeamE0P0("B", k=1, ke=0, kp=0)):
        lat.add(card)
    assert collect_constraints(lat) == []


def _cfg(**kw):
    return BeamConfig(**dict(dict(species="proton", energy=3.0, frequency=352.21, n_particles=200, current=0.0,
                                  distribution="waterbag", emit_nx=0.25, alpha_x=0.0, beta_x=2.0, emit_ny=0.25,
                                  alpha_y=0.0, beta_y=2.0, emit_z=0.3, alpha_z=0.0, beta_z=10.0), **kw))


def _energy_match(target):
    lat = Lattice()
    lat.add(Drift("D1", 100.0, aperture=20.0))
    lat.add(Adjust("CMD1", target="GAP", param_idx=1, link_group=0, vmin=0.0, vmax=2.0, start_step=0.05))
    lat.add(RFGap("GAP_1", voltage=0.5, phase=-30.0, frequency=352.21, aperture=20.0))
    lat.add(Drift("D2", 100.0, aperture=20.0))
    lat.add(SetBeamEnergy("TARGET", k=1, energy_MeV=target))
    return lat


@pytest.mark.parametrize("cost_solver", ["envelope", "mp"])
def test_the_matcher_reaches_the_energy_the_card_asks_for(cost_solver):
    from linac_gen.matching import match
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = match(_energy_match(3.6), _cfg(), algorithm="least_squares", cost_solver=cost_solver, max_iter=100)
    assert [c.label for c in r.constraints] == ["SET_BEAM_ENERGY:TARGET"]
    assert r.cost < 1e-10 and float(r.x_final[0]) != 0.5
    if cost_solver == "envelope":                     # a centred bunch: the beam's energy is the reference's
        done = _energy_match(3.6)
        done.elements[2].voltage = float(r.x_final[0])
        assert reference_at(done, _ref(), len(done.elements)).w_kin == pytest.approx(3.6, abs=1e-6)


@pytest.mark.parametrize("offset, card, off_target", [
    (dict(centroid_dphi=4.0), dict(dphi_deg=4.0, ke=1, kp=1), [0.0, 4.0]),       # a phase offset holds in a drift
    (dict(centroid_dw=0.05), dict(dE_MeV=0.05, ke=1, kp=0), [0.05, 0.0]),        # an energy offset holds, the phase slips
], ids=["phase", "energy"])
def test_the_energy_and_phase_criterion_sees_the_beam_off_the_linac(offset, card, off_target):
    """A bunch injected off the reference keeps the offset through a drift: the criterion asking for it is met, the one
    asking for none is not."""
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.matching.constraints import collect_constraints
    lat = Lattice()
    for e in (Drift("D1", 100.0), SetBeamE0P0("E", k=1, **card), Drift("D2", 100.0)):
        lat.add(e)
    res = run_envelope_sim(lat, _cfg(**offset))
    (con,) = collect_constraints(lat)
    np.testing.assert_allclose(con.evaluate(res, lat), [0.0, 0.0], atol=1e-9)
    lat.elements[1].dE_MeV, lat.elements[1].dphi_deg = 0.0, 0.0
    (con,) = collect_constraints(lat)
    np.testing.assert_allclose(con.evaluate(res, lat), off_target, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("card", [lambda: SetBeamEnergy("E", k=1, energy_MeV=3.6),
                                  lambda: SetBeamE0P0("E", k=1, ke=1, kp=0)], ids=["energy", "energy and phase"])
def test_the_gradient_algorithm_refuses_the_criteria(card):
    pytest.importorskip("torch")
    from linac_gen.matching import match
    lat = Lattice()
    lat.add(Drift("D1", 100.0, aperture=20.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0, vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_1", length=100.0, gradient=5.0, aperture=20.0))
    lat.add(card())
    with pytest.raises(ValueError, match="least_squares"):
        match(lat, _cfg(), algorithm="gradient", max_iter=10)


def test_a_mad_x_export_comments_the_card_as_the_deck_carries_it(tmp_path):
    from linac_gen.io.madx_writer import write_madx
    lat = Lattice()
    for e in (Drift("D1", 100.0), SetReferenceEnergy("E", energy_MeV=0.5, shift=True), Drift("D2", 100.0),
              SetBeamEnergy("C", k=1, energy_MeV=9.0)):
        lat.add(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        notes = write_madx(lat, tmp_path / "l.madx", _ref())
    text = (tmp_path / "l.madx").read_text(encoding="utf-8")
    assert "! HELIX: HELIX_SHIFT_ENERGY 0.5" in text and "! HELIX: SET_BEAM_ENERGY 1 9" in text
    assert len([w for w in notes if "one reference energy" in w]) == 1          # the HELIX card only


def test_a_card_with_no_weight_from_an_older_helix_deck_is_named(tmp_path):
    """HELIX wrote its energy setter as SET_BEAM_ENERGY 0 E until 2026-11 (the palette's E-set): a criterion with no
    weight, which changes nothing.  Each such card is named, with the HELIX card that keeps the old meaning; a strict
    parse refuses the deck.  A weighted criterion gets only the deck's one notice."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "old.dat"
    deck.write_text("DRIFT 100 20\nSET_BEAM_ENERGY 0 5\nDRIFT 100 20\nSET_BEAM_E0_P0 0 0.5 0 1 0\n"
                    "SET_BEAM_ENERGY 1 9\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_tracewin(str(deck))
        with pytest.raises(ValueError, match="HELIX_SET_ENERGY 5"):
            parse_tracewin(str(deck), strict=True)
    assert any(w.startswith("Line 2") and "; HELIX_SET_ENERGY 5" in w for w in meta["warnings"])
    assert any(w.startswith("Line 4") and "; HELIX_SHIFT_ENERGY 0.5" in w for w in meta["warnings"])
    assert not any(w.startswith("Line 5") and "k = 0" in w for w in meta["warnings"])


@pytest.mark.parametrize("line", ["; HELIX_SET_ENERGY 4.25 MeV", "; HELIX_SET_ENERGY", "; HELIX_SET_ENERGY =4.25",
                                  ";; HELIX_SHIFT_ENERGY x", "; helix_set_energy 4.25 # set",
                                  "; HELIX_SET_ENERGY nan", "; HELIX_SET_ENERGY -3", "; HELIX_SET_ENERGY 0",
                                  "; HELIX_SHIFT_ENERGY inf"])
def test_a_misread_energy_card_is_said_and_refused_in_strict(tmp_path, line):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "m.dat"
    deck.write_text(f"DRIFT 100 20\n{line}\nDRIFT 100 20\nEND\n")
    lat, meta = parse_tracewin(str(deck))
    assert not any(isinstance(e, SetReferenceEnergy) for e in lat.elements)
    assert any(w.startswith("Line 2") and "ENERGY" in w.upper() for w in meta["warnings"])
    with pytest.raises(ValueError, match="Line 2"):
        parse_tracewin(str(deck), strict=True)


@pytest.mark.parametrize("line, want", [("; helix_set_energy 4.25", (4.25, False)),
                                        (";;HELIX_SHIFT_ENERGY -0.5", (-0.5, True))])
def test_an_energy_card_reads_as_tracewin_comments_are_written(tmp_path, line, want):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "ok.dat"
    deck.write_text(f"DRIFT 100 20\n{line}\nEND\n")
    lat, meta = parse_tracewin(str(deck), strict=True)
    (card,) = [e for e in lat.elements if isinstance(e, SetReferenceEnergy)]
    assert (card.energy_MeV, card.shift) == want


def test_the_card_keeps_its_name_and_label_through_a_save(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin
    lat = Lattice()
    card = SetReferenceEnergy("MYCARD", energy_MeV=4.25)
    card.label = "LBL"
    for e in (Drift("D1", 100.0), card, Drift("D2", 100.0)):
        lat.add(e)
    out = tmp_path / "n.dat"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_tracewin(lat, out)
        back, _ = parse_tracewin(str(out), strict=True)
    (again,) = [e for e in back.elements if isinstance(e, SetReferenceEnergy)]
    assert (again.name, again.label, again.energy_MeV, again.shift) == ("MYCARD", "LBL", 4.25, False)


@pytest.mark.parametrize("card", [lambda: SetReferenceEnergy("E", energy_MeV=0.0),
                                  lambda: SetReferenceEnergy("E", energy_MeV=-3.5, shift=True)],
                         ids=["set to zero", "shifted below zero"])
def test_a_card_that_leaves_no_energy_is_refused_in_every_walk(card):
    lat = _line(card())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for run in (lambda: Tracker(copy.deepcopy(lat), Beam(_ref(), n_particles=5, current=0.0)).run(),
                    lambda: EnvelopeSolver(copy.deepcopy(lat), _ref(), dict(INIT), current=0.0).run(),
                    lambda: compute_transfer_matrix(copy.deepcopy(lat), _ref()),
                    lambda: reference_at(copy.deepcopy(lat), _ref(), len(lat.elements))):
            with pytest.raises(ValueError, match="no energy"):
                run()


def test_two_criteria_of_one_kind_are_told_apart():
    from linac_gen.matching.constraints import collect_constraints
    lat = Lattice()
    for card in (SetBeamE0P0("A", k=1, ke=1), SetBeamE0P0("B", k=1, kp=1), SetBeamEnergy("C", k=1, energy_MeV=4.0),
                 SetBeamEnergy("D", k=1, energy_MeV=4.0)):
        lat.add(card)
    labels = [c.label for c in collect_constraints(lat)]
    assert len(set(labels)) == 4


def test_a_weight_below_one_is_a_weight(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.matching.constraints import collect_constraints
    deck = tmp_path / "w.dat"
    deck.write_text("DRIFT 100 20\nSET_BEAM_ENERGY 0.5 120\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    (con,) = collect_constraints(lat)
    assert con.weight == 0.5


def test_a_lost_beam_scores_worse_than_any_beam_off_target():
    """Even a target 8 GeV away: the lost beam's residual is larger than the live one's."""
    live = _residuals(SetBeamEnergy("E", k=1, energy_MeV=8000.0), _Results(w=3.0, dw=0.0, dphi=0.0))
    lost = _residuals(SetBeamEnergy("E", k=1, energy_MeV=8000.0), _Results(w=3.0, dw=0.0, dphi=0.0, transmission=0.0))
    assert abs(lost[0]) > abs(live[0])

