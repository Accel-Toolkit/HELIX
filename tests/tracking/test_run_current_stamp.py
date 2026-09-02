"""Run-current provenance on multi-particle results.

The tune-depression / phase-advance popups compare the current a result
was produced at against the live beam config.  Only ``EnvelopeResults``
carried ``current_mA``; MP recorders did not, so every fresh MP run at
I > 0 read as "ran at 0 mA" (false STALE banner — skeptic log S1).

Contract under test (fix item tune-depression-stale-banner):
  * Tracker stamps ``recorder.current_mA = beam.current`` (the CONFIGURED
    current — an SC-off run at 5 mA records 5.0, not 0).
  * ``_Backtracker`` stamps the same on the backward recorder.
  * A bare ``DiagnosticRecorder`` carries ``current_mA is None`` (unknown);
    0.0 is data ("ran at 0 mA"), None is "unknown".
  * ``run_current_mA(results)`` is the single owner of that convention:
    finite ``results.current_mA`` -> finite ``results.beam.current`` -> None.
"""
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.diagnostics.recorder import DiagnosticRecorder, run_current_mA
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

from tests.helpers import stub_results


def _mini_lattice() -> Lattice:
    """5-element magnetic mini lattice (mm units, SC-off runs)."""
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="Q1", length=200.0, gradient=10.0, aperture=10.0))
    lat.add(Drift(name="D2", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="Q2", length=200.0, gradient=-10.0, aperture=10.0))
    lat.add(Drift(name="D3", length=100.0, aperture=10.0))
    return lat


def _cfg(current: float) -> BeamConfig:
    return BeamConfig(species="proton", energy=3.0, frequency=162.5,
                      current=current, n_particles=200)


@pytest.mark.parametrize("current", [5.0, 0.0])
def test_tracker_stamps_run_current_both_regimes(current):
    """MP regime: 5 mA records 5.0; 0 mA records 0.0 (data, not None)."""
    beam = create_beam(_cfg(current), seed=42)
    res = Simulation(_mini_lattice(), beam, space_charge="off").run()
    assert res.current_mA is not None
    assert res.current_mA == current
    assert run_current_mA(res) == current


def test_bare_recorder_run_current_is_unknown():
    rec = DiagnosticRecorder()
    assert rec.current_mA is None
    assert run_current_mA(rec) is None


def test_recorder_with_beam_only_resolves_via_helper():
    """The GUI attaches ``results.beam`` post-run; the helper falls back
    to ``beam.current`` when no explicit stamp exists."""
    rec = DiagnosticRecorder()
    rec.beam = stub_results(current=7.5)
    assert run_current_mA(rec) == 7.5


def test_run_current_helper_resolution_order():
    beam = stub_results(current=5.0)
    assert run_current_mA(stub_results(current_mA=3.5)) == 3.5
    # 0.0 is DATA — must win over the beam fallback.
    assert run_current_mA(stub_results(current_mA=0.0, beam=beam)) == 0.0
    # None means unknown -> fall through to the attached beam.
    assert run_current_mA(stub_results(current_mA=None, beam=beam)) == 5.0
    assert run_current_mA(stub_results(beam=beam)) == 5.0
    # No signal at all -> unknown.
    assert run_current_mA(stub_results()) is None
    # Non-finite / non-numeric values are unknown, never a crash.
    assert run_current_mA(stub_results(current_mA=float("nan"))) is None
    assert run_current_mA(stub_results(current_mA="bogus")) is None
    assert run_current_mA(stub_results(beam=stub_results(current=None))) is None


def test_backtracker_stamps_run_current():
    """MP backward walk (the GUI Backtrack path) carries the current."""
    lat = _mini_lattice()
    beam = create_beam(_cfg(5.0), seed=42)
    sim = Simulation(lat, beam, space_charge="off")
    sim.run()
    res = sim.run_backtrack(start=0, end=len(lat.elements) - 1)
    assert res.direction == "backward"
    assert res.current_mA == 5.0
