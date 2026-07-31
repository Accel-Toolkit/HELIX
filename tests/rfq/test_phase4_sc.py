"""Phase-4: space charge through the RFQ — audit results as tests.

Audit findings (2026-07-30, PXIE dst input, 5 mA, 64³ IGF):
  * SC cadence: RfqCell rides the tracker's Strang bundles
    (RK4(ds/2) → SC(ds) → RK4(ds/2)) like every FieldMapElement; the
    LEBT solenoids' .scc neutralisation profiles (Ki=1) are active.
  * The DC→bunched flip fires at the FIRST RFQ cell — physically early
    (the beam stays quasi-DC through ~60 shaper cells), but a control
    experiment with the DC 2-D kick kept through the whole line gave
    transmission 65.8 % vs 65.9 % for the early flip: BENIGN at 5 mA.
    Neighbour-bunch periodic images (TW PICNIR practice) are deferred
    with that evidence.
  * SC raises line transmission (~+4 points) and lands the captured
    exit emittances at 0.170/0.159 π·mm·mrad vs the PIP2IT measurement
    0.17/0.16.
  * Exit-bunch metrics must use WRAPPED Δφ: barely-captured particles
    carry ±n·360° offsets that inflate the unwrapped rms (33° → ~5-7°
    wrapped).
  * No genuine SC envelope ground truth exists in-repo (the "ENV+SC"
    export is a byte-copy of the no-SC file — see conftest.env_sc).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl

DST = rl.PXIE_PROJECT / "part_rfq.dst"


@pytest.mark.slow
def test_sc_run_through_rfq_is_sane():
    """tw2term + slip + losses + 3-D PIC SC together (the interplay the
    adversarial review flagged as untested): must run clean and land on
    the measured beam quality."""
    import os
    if not DST.is_file() or not rl.PXIE_DECK.is_file():
        pytest.skip("PXIE dst / deck not present")
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.io.tracewin_parser import parse_tracewin
    cfg = BeamConfig(
        species="H-", energy=0.03, frequency=162.5,
        current=5.0, duty_cycle=100.0, n_particles=100000,
        distribution="gaussian", cutoff=4.0,
        source="file", distribution_file=str(DST),
        emit_nx=0.137, alpha_x=-6.1, beta_x=1.39,
        emit_ny=0.137, alpha_y=-6.1, beta_y=1.38,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0,
        continuous=True, dc_energy_spread_keV=0.0)
    beam = create_beam(cfg, seed=42)
    rng = np.random.default_rng(7)
    idx = rng.choice(beam.n_particles, 800, replace=False)
    beam.particles = beam.particles[idx].copy()
    beam.lost = beam.lost[idx].copy()
    cwd = os.getcwd()
    os.chdir(rl.PXIE_DECK.parent)
    try:
        lat, _ = parse_tracewin(rl.PXIE_DECK.name)
        for e in lat.elements:
            if isinstance(e, RfqCell):
                e.field_model = "tw2term"
        Simulation(lat, beam,
                   space_charge=SpaceChargeConfig(nx=32, ny=32, nz=32,
                                                  grid_extent=6.0),
                   record_substeps=False).run()
    finally:
        os.chdir(cwd)
    alive = ~beam.lost
    assert alive.sum() > 0
    W = beam.ref.w_kin + beam.particles[alive, 5]
    cap = W > 1.75
    # capture survives SC (achieved ~66 % of the line at 5 mA)
    assert cap.sum() / 800 > 0.45
    # This run is the FLAG-OFF baseline (BeamConfig.periodic_phase left
    # at its default), so Δφ is stored unwrapped and the bunch is tight
    # only in WRAPPED phase — the ±n·360° offsets of barely-captured
    # particles inflate the raw rms.  The manual wrap below is what
    # test_sc_run_with_periodic_phase_needs_no_manual_wrap shows becomes
    # unnecessary once the flag is on.
    dphi_raw = beam.particles[alive, 4][cap]
    dphi_w = (dphi_raw + 180) % 360 - 180
    assert dphi_w.std() < 20.0
    assert np.abs(dphi_raw).max() > 180.0, \
        "no out-of-bucket particles — the wrap below is not being tested"
    # captured emittances stay on the measured scale (0.17/0.16)
    bg = np.sqrt((1 + 1.9557 / 939.294308) ** 2 - 1)
    for c in (0, 2):
        x = beam.particles[alive, c][cap]
        xp = beam.particles[alive, c + 1][cap]
        x = x - x.mean()
        xp = xp - xp.mean()
        e_n = np.sqrt(max(np.mean(x * x) * np.mean(xp * xp)
                          - np.mean(x * xp) ** 2, 0)) * bg
        assert 0.08 < e_n < 0.30, (c, e_n)
    # energies stay physical (W<0 junk removed by the Phase-2 kill)
    assert W.min() > 0.0


@pytest.mark.slow
def test_sc_run_with_periodic_phase_needs_no_manual_wrap():
    """Same run with ``BeamConfig.periodic_phase=True``: the satellites
    never form, so the RAW Δφ is already single-bunch and the reported
    σ_φ / ε_z are single-bunch values — with no diagnostic, recorder or
    moment code involved.

    This is the SC-ON half of the evidence; the SC-OFF half (physics
    bit-equivalence) is in tests/tracking/test_periodic_phase.py.  The
    ~20 % of particles that cross a bucket boundary do so BECAUSE of
    space charge, so the flag can only be exercised here.
    """
    import os
    if not DST.is_file() or not rl.PXIE_DECK.is_file():
        pytest.skip("PXIE dst / deck not present")
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.io.tracewin_parser import parse_tracewin
    cfg = BeamConfig(
        species="H-", energy=0.03, frequency=162.5,
        current=5.0, duty_cycle=100.0, n_particles=100000,
        distribution="gaussian", cutoff=4.0,
        source="file", distribution_file=str(DST),
        emit_nx=0.137, alpha_x=-6.1, beta_x=1.39,
        emit_ny=0.137, alpha_y=-6.1, beta_y=1.38,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0,
        continuous=True, dc_energy_spread_keV=0.0,
        periodic_phase=True)
    beam = create_beam(cfg, seed=42)
    assert beam.periodic_phase is True
    rng = np.random.default_rng(7)
    idx = rng.choice(beam.n_particles, 800, replace=False)
    beam.particles = beam.particles[idx].copy()
    beam.lost = beam.lost[idx].copy()
    cwd = os.getcwd()
    os.chdir(rl.PXIE_DECK.parent)
    try:
        lat, _ = parse_tracewin(rl.PXIE_DECK.name)
        for e in lat.elements:
            if isinstance(e, RfqCell):
                e.field_model = "tw2term"
        res = Simulation(lat, beam,
                         space_charge=SpaceChargeConfig(nx=32, ny=32, nz=32,
                                                        grid_extent=6.0),
                         record_substeps=False).run()
    finally:
        os.chdir(cwd)
    alive = ~beam.lost
    assert alive.sum() > 0
    assert beam.bunch_train is True, "the DC→bunched flip never fired"
    W = beam.ref.w_kin + beam.particles[alive, 5]
    cap = W > 1.75
    # Same capture and beam quality as the unflagged run: folding is
    # bookkeeping, not physics.
    assert cap.sum() / 800 > 0.45
    assert W.min() > 0.0
    bg = np.sqrt((1 + 1.9557 / 939.294308) ** 2 - 1)
    for c in (0, 2):
        x = beam.particles[alive, c][cap]
        xp = beam.particles[alive, c + 1][cap]
        x = x - x.mean()
        xp = xp - xp.mean()
        e_n = np.sqrt(max(np.mean(x * x) * np.mean(xp * xp)
                          - np.mean(x * xp) ** 2, 0)) * bg
        assert 0.08 < e_n < 0.30, (c, e_n)
    # THE POINT: no manual wrap.  Every surviving particle is inside its
    # own bucket, and the RECORDED σ_φ is a single-bunch number.
    dphi_raw = beam.particles[alive, 4]
    assert np.abs(dphi_raw).max() <= 180.0
    assert float(np.asarray(res.sigma_phi)[-1]) < 20.0
