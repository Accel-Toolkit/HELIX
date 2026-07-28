# tests/tracking/test_no_sc_warning.py
"""The Simulation/Tracker no-space-charge footgun must be LOUD.

``Simulation(lattice, beam)`` without ``space_charge=`` tracks a bunched,
current-carrying beam with no SC kicks at all (tracker dispatch: "else no
kick").  The CLI/GUI always build a solver when current > 0, so only raw
API calls reach that state — these tests pin the warning that makes it
visible, including the DC->bunched mid-run transition where the built-in
continuous analytic kick silently stops applying.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.tracker import NoSpaceChargeWarning, Tracker


def _make_beam(n=50, current=60.0, continuous=False):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=current)
    beam.continuous = continuous
    rng = np.random.default_rng(42)
    beam.particles[:, 0] = rng.normal(0, 1.0, n)
    beam.particles[:, 2] = rng.normal(0, 1.0, n)
    return beam


def _drift_lattice():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    return lat


def test_bunched_current_no_solver_warns():
    with pytest.warns(NoSpaceChargeWarning, match="60 mA.*OMITTED"):
        Simulation(_drift_lattice(), _make_beam()).run()


def test_warning_fires_once_per_run():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Drift("D2", 100.0))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Simulation(lat, _make_beam()).run()
    hits = [w for w in rec if issubclass(w.category, NoSpaceChargeWarning)]
    assert len(hits) == 1


def test_explicit_config_does_not_warn():
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Simulation(_drift_lattice(), _make_beam(), space_charge=sc).run()
    assert not [w for w in rec
                if issubclass(w.category, NoSpaceChargeWarning)]


def test_off_sentinel_suppresses_warning_simulation():
    """space_charge="off" = declared no-SC intent — same physics as
    None, no warning (the permanent API design: explicit over magic)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Simulation(_drift_lattice(), _make_beam(),
                   space_charge="off").run()
    assert not [w for w in rec
                if issubclass(w.category, NoSpaceChargeWarning)]


def test_off_sentinel_suppresses_warning_tracker():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Tracker(_drift_lattice(), _make_beam(), pic_solver="off").run()
    assert not [w for w in rec
                if issubclass(w.category, NoSpaceChargeWarning)]


def test_off_sentinel_matches_none_physics():
    import numpy as np
    beams = []
    for sc in (None, "off"):
        beam = _make_beam()
        Simulation(_drift_lattice(), beam, space_charge=sc).run()
        beams.append(beam.particles.copy())
    assert np.array_equal(beams[0], beams[1])


def test_zero_current_does_not_warn():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Simulation(_drift_lattice(), _make_beam(current=0.0)).run()
    assert not [w for w in rec
                if issubclass(w.category, NoSpaceChargeWarning)]


def test_dc_beam_warns_at_bunching_transition():
    # A continuous beam gets the built-in 2-D analytic kick without a
    # solver — no warning while it stays DC.  The first bunching element
    # flips it to bunched, after which there is NO SC path at all.
    lat = Lattice()
    lat.add(Drift("D1", 50.0))
    lat.add(RFGap("G1", voltage=0.2, phase=-30.0, frequency=352.21))
    lat.add(Drift("D2", 50.0))
    with pytest.warns(NoSpaceChargeWarning, match="DC-to-bunched.*'G1'"):
        Tracker(lat, _make_beam(continuous=True)).run()


def test_dc_beam_without_buncher_does_not_warn():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Tracker(_drift_lattice(), _make_beam(continuous=True)).run()
    assert not [w for w in rec
                if issubclass(w.category, NoSpaceChargeWarning)]
