"""The ninth value of an RFQ_CELL card (``dP``) does not move the reference clock.

Envelope, matrix and backtracking modes used to add it to ``ref.phi_s`` at each
cell exit; a multiparticle run never did.  So the two disagreed at an RFQ exit by
the sum of the column: 569 degrees on a deck where it is a phase, and 10 degrees on
a TraceWin 2.27 deck, where it holds values from 0 to 1 and is not a phase at all
(the manual's parameter table: "not used any more").  The value stays on the
element so that a deck round-trips.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking import tracker as T
from linac_gen.tracking.backtrack import build_replay_table
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import reference_at

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "lebt_plus_rfq"
#: not distributed: slot 9 is a phase in the first style, 0 to 1 in the second
_PHASE_STYLE = sorted(_EXAMPLES.glob("rfq_only_matched.dat"))
_NEW_STYLE = sorted(_EXAMPLES.glob("*/rfq.dat"))
_INITIAL = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
            "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2,
            "emit_z": 0.2}


def _cells(dp, n=4):
    return [RfqCell(name=f"C{i}", voltage_V=60000.0, r0_mm=5.0, A10=0.1,
                    modulation=1.2, length_mm=20.0, phi_s_deg=-60.0,
                    cell_type=2, dP_deg=dp) for i in range(n)]


def _lattice(dp, step=(100.0, 50.0)):
    lat = Lattice()
    for c in _cells(dp):
        lat.add(c)
    lat.add(Drift("D", length=10.0, aperture=20.0))
    lat.step_config = StepConfig(*step, rfq_solver="cells")
    return lat


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=0.1, frequency=162.5)


def _mp_phi(lat, ref, current=0.0, n=8):
    beam = Beam(ref=ref.copy(), n_particles=n, current=current)
    rng = np.random.default_rng(3)
    beam.particles[:] = rng.normal(size=(n, 6)) * [0.2, 0.2, 0.2, 0.2, 1.0, 1e-5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T.Tracker(lat, beam).run()
    return beam.ref.phi_s


def _env_phi(lat, ref, current=0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = EnvelopeSolver(lat, ref.copy(), dict(_INITIAL), current=current)
        s.run()
    return s._ref.phi_s


@pytest.mark.parametrize("method", ["advance_ref", "advance_ref_over"])
def test_the_element_does_not_add_dp(method):
    phases = []
    for dp in (0.0, -25.0):
        c = _cells(dp, n=1)[0]
        r = _ref()
        if method == "advance_ref":
            c.advance_ref(r)
        else:
            c.advance_ref_over(r, 0.0, c.length)
        phases.append(r.phi_s)
    assert phases[0] == phases[1]


def test_the_value_stays_on_the_element():
    assert _cells(-25.0, n=1)[0].dP_deg == -25.0


@pytest.mark.parametrize("current", [0.0, 5.0])
def test_envelope_clock_is_independent_of_dp(current):
    assert (_env_phi(_lattice(-25.0), _ref(), current)
            == _env_phi(_lattice(0.0), _ref(), current))


def test_matrix_and_replay_clocks_are_independent_of_dp():
    a, b = _lattice(-25.0), _lattice(0.0)
    assert reference_at(a, _ref(), len(a.elements)).phi_s == \
        reference_at(b, _ref(), len(b.elements)).phi_s
    end = len(a.elements) - 1
    assert (build_replay_table(a, _ref(), end=end)[-1].phi_s
            == build_replay_table(b, _ref(), end=end)[-1].phi_s)


def test_multiparticle_clock_was_already_independent_of_dp():
    assert _mp_phi(_lattice(-25.0), _ref()) == _mp_phi(_lattice(0.0), _ref())


def test_envelope_follows_the_multiparticle_clock():
    """Not bit for bit (an RfqCell is walked in full steps by the envelope, half
    steps by the tracker), but no longer by the sum of the column."""
    lat = _lattice(-25.0)
    assert abs(_env_phi(lat, _ref()) - _mp_phi(_lattice(-25.0), _ref())) < 1.0


def _real_deck(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(path))
    lat.step_config = StepConfig(100.0, 50.0, rfq_solver="cells")
    return lat


@pytest.mark.parametrize("current", [0.0, 5.0])
@pytest.mark.parametrize("decks", [_PHASE_STYLE, _NEW_STYLE], ids=["phase", "new"])
def test_real_decks_of_both_styles(decks, current):
    if not decks:
        pytest.skip("undistributed RFQ deck absent")
    ref = ReferenceParticle(species=PROTON, w_kin=0.03, frequency=162.5)
    env = _env_phi(_real_deck(decks[0]), ref, current)
    mp = _mp_phi(_real_deck(decks[0]), ref, current=0.0, n=4)
    assert abs(env - mp) < 1.0
