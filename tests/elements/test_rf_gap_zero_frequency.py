"""An RF gap without a frequency.

With a voltage it is refused with a clear message: it used to reach the
reference particle and die there, half updated, with a bare ZeroDivisionError
(``wavelength = c / f``) — e.g. a MAD-X cavity with neither FREQ nor HARMON.
The error is still a ZeroDivisionError (:class:`RFFrequencyError`), so code
that handled that — or only ValueError — behaves as before.  Without a
voltage (a cavity whose RF settings live in another file) it does nothing and
the reference clock is kept, in envelope, matrix and multiparticle runs alike.
The checks go through the real entry points: an imported deck run by the
envelope solver and by the transfer-matrix code.
"""
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.rf_gap import RFFrequencyError, RFGap

DECK = """beam, particle=proton, energy=1.73827208816;
d: drift, l=1;
rf0: rfcavity, l=0, volt=0.01, lag=0.25;
r: line=(d, rf0, d); use, period=r;
"""


def _import(tmp_path, deck=DECK):
    from linac_gen.io.madx_parser import parse_madx
    p = tmp_path / "nofreq.madx"
    p.write_text(deck)
    return parse_madx(str(p))


def test_importer_reports_a_cavity_without_freq_or_harmon(tmp_path):
    lat, meta = _import(tmp_path)
    (gap,) = [e for e in lat.elements if isinstance(e, RFGap)]
    assert gap.frequency == 0.0
    assert any("1 RF cavity has a voltage but neither FREQ nor HARMON (rf0)" in w
               for w in meta["warnings"])
    # (the old catch called this a reference particle lost to deceleration)
    assert not any("decelerates below zero energy" in w for w in meta["warnings"])


def test_zero_frequency_gap_is_refused_by_an_envelope_run(tmp_path):
    from linac_gen.cli import common
    p = tmp_path / "nofreq.madx"
    p.write_text(DECK)
    lat, cfg, _ = common.load_input(str(p))
    with pytest.raises(RFFrequencyError, match="'rf0' has no frequency"):
        common.run_envelope_sim(lat, cfg)


def test_zero_frequency_gap_is_refused_by_the_transfer_matrix(tmp_path):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    lat, meta = _import(tmp_path)
    with pytest.raises(RFFrequencyError, match="'rf0' has no frequency"):
        compute_transfer_matrix(lat, meta["reference"].copy())


def test_refusal_is_a_zero_division_error_and_leaves_the_reference_untouched():
    ref = ReferenceParticle(PROTON, w_kin=800.0, frequency=44.7)
    gap = RFGap(name="rf0", voltage=0.01, phase=0.0, frequency=0.0)
    for call in (gap.advance_ref, gap.kick_matrix):
        with pytest.raises(ZeroDivisionError, match="'rf0' has no frequency"):
            call(ref)
    assert ref.w_kin == 800.0 and ref.frequency == 44.7
    # (not a ValueError: code that handles only those is unaffected)
    assert not issubclass(RFFrequencyError, ValueError)


@pytest.mark.parametrize("freq", [0.0, -5.0, float("nan")])
def test_a_gap_with_a_voltage_and_no_positive_frequency_is_refused(freq):
    ref = ReferenceParticle(PROTON, w_kin=800.0, frequency=44.7)
    gap = RFGap(name="rf0", voltage=0.01, phase=0.0, frequency=freq)
    with pytest.raises(RFFrequencyError):
        gap.kick_matrix(ref)


def test_gap_at_the_reference_frequency_is_unchanged():
    ref = ReferenceParticle(PROTON, w_kin=800.0, frequency=44.7)
    gap = RFGap(name="rf", voltage=0.01, phase=0.0, frequency=44.7)
    gap.advance_ref(ref)
    assert ref.frequency == 44.7
    assert ref.w_kin == pytest.approx(800.0 + 0.01, rel=1e-15)    # on crest: + q·V·T


IDLE = """beam, particle=proton, energy=1.73827208816;
d: drift, l=1; cav: rfcavity, l=4.114;
r: line=(d, cav, d); use, period=r;
"""


def test_idle_cavity_keeps_the_clock_in_every_run(tmp_path):
    # (CERN sequences keep their RF settings in another file: ACTCSE: RFCAVITY,
    # L := 4.114 with nothing else) — it does nothing in MAD-X; an envelope run
    # must not stop on it, and the reference keeps its frequency
    from linac_gen.cli import common
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    lat, meta = _import(tmp_path, IDLE)
    (gap,) = [e for e in lat.elements if isinstance(e, RFGap)]
    assert gap.voltage == 0.0 and gap.frequency == 0.0
    assert sum(e.length for e in lat.elements) == pytest.approx(6114.0)
    assert any("1 RF cavity has no VOLT, FREQ or HARMON (cav): they do nothing" in w
               for w in meta["warnings"])
    ref = meta["reference"].copy()
    f0, w0 = ref.frequency, ref.w_kin
    gap.advance_ref(ref)
    assert ref.frequency == f0 and ref.w_kin == w0
    compute_transfer_matrix(lat, meta["reference"].copy())
    p = tmp_path / "idle.madx"
    p.write_text(IDLE)
    lat2, cfg, _ = common.load_input(str(p))
    common.run_envelope_sim(lat2, cfg)
