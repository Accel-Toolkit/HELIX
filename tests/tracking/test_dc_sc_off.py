# tests/tracking/test_dc_sc_off.py
"""Review round 2, claim 3: ``space_charge="off"`` must disable the DC
(continuous-beam) 2-D kick too — the sentinel used to be consulted only
for bunched beams, so a LEBT/DC run with "off" still got the full
line-charge defocus with no warning."""
import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift


def _lat():
    lat = Lattice()
    for i in range(10):
        lat.add(Drift(f"D{i}", length=200.0, aperture=50.0))
    return lat


def _dc_beam(current):
    ref = ReferenceParticle(species=PROTON, w_kin=0.03, frequency=162.5)
    b = Beam(ref=ref, n_particles=500, current=current)
    b.continuous = True
    rng = np.random.default_rng(7)
    b.particles[:, 0] = rng.normal(0, 2.0, 500)
    b.particles[:, 1] = rng.normal(0, 5.0, 500)
    b.particles[:, 2] = rng.normal(0, 2.0, 500)
    b.particles[:, 3] = rng.normal(0, 5.0, 500)
    return b


def test_dc_off_sentinel_disables_space_charge():
    rec_off = Simulation(_lat(), _dc_beam(10.0), space_charge="off").run()
    rec_i0 = Simulation(_lat(), _dc_beam(0.0)).run()
    # "off" ≡ zero current: the DC kick must NOT be applied.
    np.testing.assert_allclose(rec_off.sigma_x, rec_i0.sigma_x,
                               rtol=1e-12)


def test_dc_default_still_applies_space_charge():
    """No sentinel → the analytic DC kick stays default-on (bit-compat
    for every existing DC run)."""
    rec_def = Simulation(_lat(), _dc_beam(10.0)).run()
    rec_i0 = Simulation(_lat(), _dc_beam(0.0)).run()
    assert rec_def.sigma_x[-1] > rec_i0.sigma_x[-1] * 1.05


def test_backtrack_dc_off_sentinel():
    from linac_gen.tracking.backtrack import _Backtracker
    beam = _dc_beam(10.0)
    bt = _Backtracker(_lat(), beam, table=None, start=0, end=0,
                      pic_solver="off")
    assert bt._sc_explicitly_off is True and bt.pic_solver is None
    before = beam.particles.copy()
    bt._apply_sc_kick_negated(100.0)
    np.testing.assert_array_equal(beam.particles, before)


def test_run_backtrack_forwards_off_sentinel():
    """Adversarial-review find: Simulation.run_backtrack used to pass
    pic_solver=None, so a DC space_charge='off' run applied ZERO kick
    forward but the FULL DC kick on the backward walk (141 mm vs 1.9 mm
    reconstructed sigma_x in the repro).  The sentinel must reach the
    backtracker."""
    import warnings as _w
    lat = _lat()
    sim_off = Simulation(lat, _dc_beam(50.0), space_charge="off")
    sim_off.run()
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        rec_off = sim_off.run_backtrack()

    lat2 = _lat()
    sim_i0 = Simulation(lat2, _dc_beam(0.0))
    sim_i0.run()
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        rec_i0 = sim_i0.run_backtrack()
    np.testing.assert_allclose(rec_off.sigma_x, rec_i0.sigma_x,
                               rtol=1e-9)


def test_backtrack_csr_requires_approximate_opt_in():
    """CSR kicks have no backward model: a forward run with CSR enabled
    must REFUSE to backtrack unless approximate_backtracking=True (it
    used to warn-and-skip, passing a degraded result as an exact
    undo)."""
    import pytest
    import warnings as _w
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.beam import Beam

    def _bunched(current=5.0):
        ref = ReferenceParticle(species=PROTON, w_kin=3.0,
                                frequency=352.21)
        b = Beam(ref=ref, n_particles=200, current=current)
        rng = np.random.default_rng(3)
        b.particles[:, 0] = rng.normal(0, 1.0, 200)
        b.particles[:, 2] = rng.normal(0, 1.0, 200)
        return b

    sc = SpaceChargeConfig(nx=16, ny=16, nz=16, use_gpu="cpu",
                           grid_mode="adaptive", csr_enabled=True)
    sim = Simulation(_lat(), _bunched(), space_charge=sc)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        sim.run()
    with pytest.raises(ValueError, match="approximate_backtracking"):
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            sim.run_backtrack()
    # Opt-in path runs (with the loud approximate warning).
    sim2 = Simulation(_lat(), _bunched(), space_charge=sc)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        sim2.run()
    with _w.catch_warnings(record=True) as wl:
        _w.simplefilter("always")
        sim2.run_backtrack(approximate_backtracking=True)
    assert any("SKIPPED" in str(w.message) for w in wl)


def test_run_envelope_placeholder_warns():
    """External-API hardening: run_envelope without beam_envelope_params
    must WARN that a placeholder beam is used."""
    import warnings as _w
    from linac_gen.core.beam import Beam
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=8, current=0.0)
    sim = Simulation(_lat(), beam)
    with _w.catch_warnings(record=True) as wl:
        _w.simplefilter("always")
        sim.run_envelope()
    assert any("PLACEHOLDER" in str(w.message) for w in wl)
