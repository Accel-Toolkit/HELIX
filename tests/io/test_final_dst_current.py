"""The current in a final ``.dst`` is the one its particles carry.

A macroparticle keeps its charge when others are lost (``pic/macrocharge.py``), so
the beam left after losses carries I × N_alive / N_launched.  TraceWin writes that
too: its own ``part_dtl1.dst`` after losses holds 5.036692 mA = 6.8 × 74069 / 100000,
and the reader applies the same rule to a file that flags its lost particles.
"""
import logging

import numpy as np
import pytest

from linac_gen.cli.common import write_final_dst
from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.distributions.factory import create_beam
from linac_gen.io.tracewin_dst import load_dst, write_dst
from linac_gen.pic.macrocharge import macro_charge_for
from linac_gen.pic.pic_solver import kick_continuous_2d
from tests.io.test_tracewin_dst_flagged import _ROWS, _flagged_file


def _beam(n=200, current=5.0, lost=0, continuous=False):
    cfg = BeamConfig(species="proton", energy=20.0, frequency=352.21, n_particles=n,
                     current=current, continuous=continuous)
    beam = create_beam(cfg, seed=5)
    beam.lost[:lost] = True
    return beam


def _file_config(path, current):
    return BeamConfig(species="proton", energy=20.0, frequency=352.21, current=current,
                      source="file", distribution_file=str(path))


# ---------------------------------------------------------------------------
# Beam.alive_current
# ---------------------------------------------------------------------------
def test_with_nothing_lost_the_current_is_untouched_to_the_last_bit():
    """I × N / N need not return I in floating point, and here it does not."""
    assert 0.1 * 3 / 3 != 0.1
    beam = Beam(ref=ReferenceParticle(species=PROTON, w_kin=20.0, frequency=352.21),
                n_particles=3, current=0.1)
    assert beam.alive_current == 0.1


def test_the_current_follows_the_particles_left():
    assert _beam(lost=17).alive_current == pytest.approx(5.0 * 183 / 200, rel=1e-15)


def test_every_particle_lost_leaves_no_current():
    assert _beam(lost=200).alive_current == 0.0


def test_a_beam_with_no_current_stays_at_zero():
    assert _beam(current=0.0, lost=17).alive_current == 0.0


def test_a_beam_with_no_particles_has_no_current():
    beam = Beam(ref=ReferenceParticle(species=PROTON, w_kin=20.0, frequency=352.21),
                n_particles=0, current=5.0)
    assert beam.alive_current == 0.0


def test_the_rule_is_the_one_tracewin_wrote_after_its_own_losses():
    """The numbers of TraceWin's ``part_dtl1.dst``; the file itself is read by
    ``test_genuine_classic_export_scales_the_current_as_the_reader_does``."""
    beam = Beam(ref=ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
                n_particles=100000, current=6.8)
    beam.lost[74069:] = True
    assert beam.alive_current == pytest.approx(5.036692, rel=1e-6)


def test_the_rule_is_the_one_the_reader_applies_to_a_flagged_file(tmp_path):
    _, head = load_dst(str(_flagged_file(tmp_path / "f.dst", _ROWS)))
    beam = Beam(ref=ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
                n_particles=head["n_in_file"], current=6.8)
    beam.lost[head["n_particles"]:] = True
    assert beam.alive_current == pytest.approx(head["current_mA"], rel=1e-15)


# ---------------------------------------------------------------------------
# write_final_dst
# ---------------------------------------------------------------------------
def test_a_run_with_no_losses_writes_the_same_file_as_before(tmp_path):
    beam = _beam(current=6.8)
    write_final_dst(beam, tmp_path / "new.dst")
    write_dst(str(tmp_path / "old.dst"), beam.alive_particles, current_mA=float(beam.current),
              frequency_MHz=float(beam.ref.frequency), mass_MeV=float(beam.ref.species.mass),
              w_kin_ref=float(beam.ref.w_kin))
    assert (tmp_path / "new.dst").read_bytes() == (tmp_path / "old.dst").read_bytes()


def test_the_header_holds_the_current_of_the_particles_written(tmp_path):
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=17), p)
    parts, head = load_dst(str(p))
    assert parts.shape[0] == 183
    assert head["current_mA"] == pytest.approx(5.0 * 183 / 200, rel=1e-12)


def test_a_beam_with_no_current_writes_zero(tmp_path):
    p = tmp_path / "final.dst"
    write_final_dst(_beam(current=0.0, lost=17), p)
    assert load_dst(str(p))[1]["current_mA"] == 0.0


def test_every_particle_lost_still_writes_an_empty_file(tmp_path):
    """As before the change: nothing in it, so nothing to keep a current for."""
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=200), p)
    assert p.stat().st_size > 0


def test_a_continuous_beam_gets_the_same_kick_through_a_file(tmp_path):
    """A DC beam's space charge scales the current by the fraction alive, so the
    same rule keeps it: the real kernel, on the same particles, before and after."""
    beam = _beam(lost=17, continuous=True)
    p = tmp_path / "final.dst"
    write_final_dst(beam, p)
    again = create_beam(_file_config(p, load_dst(str(p))[1]["current_mA"]), seed=1)
    x_prime = beam.alive_particles[:, 1].copy()
    x_prime_again = again.particles[:, 1].copy()
    np.testing.assert_allclose(x_prime_again, x_prime, rtol=1e-9)
    kick_continuous_2d(beam, 10.0)
    kick_continuous_2d(again, 10.0)
    kick = beam.alive_particles[:, 1] - x_prime
    assert np.abs(kick).max() > 1e-6
    np.testing.assert_allclose(again.particles[:, 1] - x_prime_again, kick, rtol=1e-6)


def test_each_particle_keeps_its_charge_through_a_file(tmp_path):
    """The point of the rule: re-imported at the file's current, the particles carry
    what they carried in the run that wrote them.  Written with the full current on
    fewer particles this comes out 200/183 too high."""
    beam = _beam(lost=17)
    p = tmp_path / "final.dst"
    write_final_dst(beam, p)
    header_current = load_dst(str(p))[1]["current_mA"]
    again = create_beam(_file_config(p, header_current), seed=1)
    assert again.n_particles == 183
    # a ratio: charges are about 1e-14 C, under approx's absolute tolerance of 1e-12
    assert macro_charge_for(again) / macro_charge_for(beam) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Re-import at another current
# ---------------------------------------------------------------------------
def _current_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "the beam is set to" in r.getMessage()]


def test_a_run_at_another_current_than_its_file_is_warned_about(tmp_path, caplog):
    """A file with nothing flagged, as HELIX writes: no reader's note says so."""
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=17), p)
    with caplog.at_level(logging.WARNING):
        create_beam(_file_config(p, 5.0), seed=1)
    messages = _current_warnings(caplog)
    assert len(messages) == 1
    assert "4.575" in messages[0] and "5 mA" in messages[0]


def test_the_note_comes_once_however_often_the_beam_is_built(tmp_path, caplog):
    """A match builds it at every evaluation."""
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=17), p)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            create_beam(_file_config(p, 5.0), seed=1)
    assert len(_current_warnings(caplog)) == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        create_beam(_file_config(p, 6.0), seed=1)       # another current is another note
    assert len(_current_warnings(caplog)) == 1


def test_a_run_at_its_file_current_is_not_warned_about(tmp_path, caplog):
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=17), p)
    with caplog.at_level(logging.WARNING):
        create_beam(_file_config(p, load_dst(str(p))[1]["current_mA"]), seed=1)
    assert not _current_warnings(caplog)


def test_a_deliberate_zero_current_is_not_warned_about(tmp_path, caplog):
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=17), p)
    with caplog.at_level(logging.WARNING):
        create_beam(_file_config(p, 0.0), seed=1)
    assert not _current_warnings(caplog)


def test_a_file_with_no_current_is_not_warned_about(tmp_path, caplog):
    p = tmp_path / "final.dst"
    write_final_dst(_beam(current=0.0), p)
    with caplog.at_level(logging.WARNING):
        create_beam(_file_config(p, 5.0), seed=1)
    assert not _current_warnings(caplog)


def test_the_run_keeps_the_current_it_was_given(tmp_path):
    p = tmp_path / "final.dst"
    write_final_dst(_beam(lost=17), p)
    assert create_beam(_file_config(p, 5.0), seed=1).current == 5.0


def test_a_flagged_file_warns_once_not_twice(tmp_path, caplog):
    path = _flagged_file(tmp_path / "f.dst", _ROWS)
    with caplog.at_level(logging.WARNING):
        create_beam(BeamConfig(species="H-", energy=2.1, frequency=162.5, current=6.8,
                               source="file", distribution_file=str(path)), seed=1)
    assert len(_current_warnings(caplog)) == 1
    assert np.isfinite(load_dst(str(path))[1]["current_mA"])
