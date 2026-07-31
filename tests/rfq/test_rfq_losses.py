"""Phase-2 losses: vane apertures, .ouv scraping, and the dst benchmark.

Ground-truth anchors (2026-07-30):
  * vane_apertures() vs the actual PXIE ``pxie-rfq.vane`` tip table:
    0.03-0.14 % mean error, no empirical factor;
  * the LEBT solenoid ``.ouv`` bore (17-19 mm throats) reproduces
    TraceWin's LEBT scraping of the measured input distribution:
    HELIX 77.5 % vs TW partran 77.3 % — the deck's Ka flag had been
    transcribed as 0 (TW's own project decks carry Ka=1);
  * full-line dst run: captured-beam exit emittance 0.133/0.167
    π·mm·mrad vs the PIP2IT measurement 0.17/0.16;
  * KNOWN GAP: RFQ-segment transmission of LEBT survivors is ~82 %
    vs TW/Toutatis ~97 % — the excess loss sits in the last tight-bore
    cells (196-199, a_x ≈ 3.1 mm) where the residual y-halo/bunch heat
    of the card-level model clips; Phase-3 vane-table coefficients are
    the planned fix.  The lower gate below is set at 70 % so real
    regressions surface without freezing the known gap as "correct".
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl

DST = rl.PXIE_PROJECT / "part_rfq.dst"


def test_vane_apertures_match_vane_file(pxie_deck):
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.elements.rfq_coefficients import vane_apertures
    from linac_gen.io.tracewin_vane import parse_vane_file
    vane = rl.REPO_ROOT / "Fields" / "pxie-rfq.vane"
    if not vane.is_file():
        pytest.skip("pxie-rfq.vane not present")
    vg = parse_vane_file(str(vane))
    cells = [e for e in pxie_deck.elements
             if isinstance(e, RfqCell)]
    ent = np.concatenate([[0.0],
                          np.cumsum([c.length for c in cells])]) * 1e-3
    for ci in (50, 100, 150, 151, 190):     # both ±2 polarities
        c = cells[ci]
        m = (vg.z >= ent[ci]) & (vg.z <= ent[ci + 1])
        zl = (vg.z[m] - ent[ci]) * 1e3
        ax = vg.aperture_v1[m] * 1e3
        ay = vg.aperture_v2[m] * 1e3
        got = np.array([vane_apertures(c.r0_mm, c.A10, c.length,
                                       c.cell_type, z) for z in zl])
        errx = np.abs(got[:, 0] - ax) / ax
        erry = np.abs(got[:, 1] - ay) / ay
        assert errx.mean() < 0.01, (ci, errx.mean())
        assert erry.mean() < 0.01, (ci, erry.mean())


def test_lebt_ouv_flag_active(pxie_deck):
    """The three LEBT solenoids must carry the .ouv bore profile —
    TW's own decks have Ka=1; the transcription that dropped it hid
    22.7 % of LEBT scraping (found 2026-07-30)."""
    fmaps = [e for e in pxie_deck.elements
             if getattr(e, "field_data", None) is not None
             and getattr(e.field_data, "pipe_radius_profile", None)
             is not None]
    assert len(fmaps) >= 3


@pytest.mark.slow
def test_full_line_dst_benchmark():
    """TW's measured input through LEBT+RFQ with all losses active."""
    import os
    if not DST.is_file() or not rl.PXIE_DECK.is_file():
        pytest.skip("PXIE dst / deck not present")
    from linac_gen.core.config import BeamConfig
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
    idx = rng.choice(beam.n_particles, 1500, replace=False)
    beam.particles = beam.particles[idx].copy()
    beam.lost = beam.lost[idx].copy()
    cwd = os.getcwd()
    os.chdir(rl.PXIE_DECK.parent)
    try:
        lat, _ = parse_tracewin(rl.PXIE_DECK.name)
        for e in lat.elements:
            if isinstance(e, RfqCell):
                e.field_model = "tw2term"
        res = Simulation(lat, beam, space_charge="off",
                         record_substeps=False).run()
    finally:
        os.chdir(cwd)
    names = list(res.element_names)
    tr = np.asarray(res.transmission)
    i0 = next(i for i, n in enumerate(names) if n.startswith("RFQ_"))
    # LEBT scraping reproduces TW partran (77.3 %)
    assert 72.0 < tr[i0 - 1] < 83.0
    alive = ~beam.lost
    W = beam.ref.w_kin + beam.particles[alive, 5]
    # RFQ vanes remove the uncaptured junk: every survivor accelerated
    assert (W > 1.75).sum() / max(alive.sum(), 1) > 0.95
    # RFQ-segment transmission of the LEBT survivors (known-gap gate)
    surv = tr[i0 - 1] / 100.0 * 1500
    assert alive.sum() / surv > 0.70
    # captured-beam quality on the PIP2IT measurement (0.17/0.16)
    bg = np.sqrt((1 + 1.9557 / 939.294308) ** 2 - 1)
    cap = W > 1.75
    for cx, lo, hi in ((0, 0.08, 0.22), (2, 0.10, 0.26)):
        x = beam.particles[alive, cx][cap]
        xp = beam.particles[alive, cx + 1][cap]
        x = x - x.mean()
        xp = xp - xp.mean()
        e_n = np.sqrt(max(np.mean(x * x) * np.mean(xp * xp)
                          - np.mean(x * xp) ** 2, 0)) * bg
        assert lo < e_n < hi, (cx, e_n)
    assert np.mean(W[cap]) == pytest.approx(1.956, abs=0.010)


@pytest.mark.slow
def test_low_voltage_collapses_capture():
    """The S-curve mechanism: at 75 % vane voltage the bucket cannot
    hold the beam — capture collapses (measured PIP2IT behaviour:
    degradation already below ~58 kV of 60)."""
    import os
    if not rl.PXIE_DECK.is_file():
        pytest.skip("PXIE deck not present")
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.io.tracewin_parser import parse_tracewin
    cfg = BeamConfig(
        species="H-", energy=0.029999999, frequency=162.5,
        current=5.0, duty_cycle=100.0, n_particles=400,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.1370111, alpha_x=-6.1145021, beta_x=1.3870467,
        emit_ny=0.1370329, alpha_y=-6.07846, beta_y=1.3799354,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0,
        continuous=True, dc_energy_spread_keV=0.0)
    beam = create_beam(cfg, seed=42)
    cwd = os.getcwd()
    os.chdir(rl.PXIE_DECK.parent)
    try:
        lat, _ = parse_tracewin(rl.PXIE_DECK.name)
        for e in lat.elements:
            if isinstance(e, RfqCell):
                e.field_model = "tw2term"
                e.voltage_V *= 0.75
        Simulation(lat, beam, space_charge="off",
                   record_substeps=False).run()
    finally:
        os.chdir(cwd)
    alive = ~beam.lost
    W = beam.ref.w_kin + beam.particles[alive, 5]
    assert (W > 1.75).sum() / 400 < 0.10
