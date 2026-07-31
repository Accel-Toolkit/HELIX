"""tw2term vs TraceWin ground truth — the calibration record as tests.

IMPORTANT CONTEXT (2026-07-30 audit): the reference project's decks
carry ``RFQ_GEOM 1 pxie-rfq.vane`` — TW's matrices/ENV charts were
produced WITH the Toutatis vane-geometry coefficient tables, not the
pure card-driven model.  The levels pinned here are the card-physics
agreement against that vane-based reference; TW's OWN matrices
propagate the input Σ to only ~4 % of the same ENV chart, so the
envelope gate below is close to the reference's intrinsic floor.

Pinned achievements (vs the pre-2026-07 legacy model):
  * per-cell transverse matrices: median rel err 1.25 %, mean 2.1 %,
    200/203 within 10 % (legacy: mean ABSOLUTE error ~70 — opposite
    M[1,0] sign);
  * cumulative momentum invariant Π det(x-block) matches TW/physics
    (the cos-phased-K2 bug this invariant caught gave 4.5× overdamping);
  * synchronous ramp: exactly TW's 1.955717 MeV (legacy −0.12 %);
  * envelope σ through the whole RFQ: 1.1 % (x) / 1.7 % (y) mean vs
    the vane-based export with the smooth TW calibration (legacy:
    180-450 %; NOTE this is below the ~4 % with which TW's own
    matrices reproduce the same chart — part of the calibration
    absorbs chart-specific residuals, see rfq_coefficients cautions);
  * multiparticle capture: ~99 % vs legacy 24 % (the legacy track path
    had NO longitudinal phase slip — a DC beam could never bunch).

Remaining known residuals: a few shaper cells sit at 10-17 % on
near-cancelling elements (F/D halves nearly balance there); the
vane-field campaign showed these are Toutatis field-solution effects
whose smooth component the TW calibration captures.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl


@pytest.fixture(scope="module")
def tw2term_cells(pxie_deck):
    """PXIE cells flipped to tw2term — RESTORED to the parser default on
    teardown (pxie_deck is session-scoped; leaking the flip would make
    later tests' behavior depend on execution order)."""
    from linac_gen.elements.rfq_cell import RfqCell
    cells = [e for e in pxie_deck.elements if isinstance(e, RfqCell)]
    saved = [c.field_model for c in cells]
    for c in cells:
        c.field_model = "tw2term"
    yield cells
    for c, fm in zip(cells, saved):
        c.field_model = fm


@pytest.fixture(scope="module")
def cell_block_pairs(tw2term_cells, transfer_ref, env_nosc):
    """(cells, per-cell TW matrices, entrance gammas) aligned 1:1."""
    nums, s_m, cum = transfer_ref
    per = rl.per_element_matrices(cum)
    blk_len = np.diff(np.concatenate([[0.0], s_m]))
    pairs = []
    bi = int(np.searchsorted(s_m, 1.9561 + 1e-6))
    for ci in range(len(tw2term_cells)):
        while bi < len(s_m) and blk_len[bi] < 1e-9:
            bi += 1
        pairs.append((ci, bi))
        bi += 1
    s_ent = np.array([s_m[b] - blk_len[b] for _, b in pairs])
    gam_ent = 1.0 + np.interp(s_ent, env_nosc["s_m"], env_nosc["gam1"])
    return pairs, per, gam_ent


def test_per_cell_matrices_vs_tracewin(tw2term_cells, cell_block_pairs):
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    pairs, per, gam_ent = cell_block_pairs
    err = []
    det_mine = det_tw = 1.0
    for ci, bi in pairs:
        W = (gam_ent[ci] - 1.0) * H_MINUS.mass
        ref = ReferenceParticle(species=H_MINUS, w_kin=W, frequency=162.5)
        M = tw2term_cells[ci].fitted_matrix(ref)
        refM = per[bi]
        scale = np.maximum(np.abs(refM[:4, :4]), 1.0)
        err.append(np.max(np.abs(M[:4, :4] - refM[:4, :4]) / scale))
        det_mine *= np.linalg.det(M[0:2, 0:2])
        det_tw *= np.linalg.det(refM[0:2, 0:2])
    assert det_mine == pytest.approx(det_tw, rel=0.02)   # ≈ (βγ)in/(βγ)out
    err = np.array(err)
    # achieved 2026-07-30 (after the K2-phase, parser-neighbour,
    # ±4-sign and -3-cos³ fixes): median 0.0144, mean 0.0211,
    # 198/203 < 10 %, worst 0.165 (shaper cell 72 — near-cancelling
    # F/D halves; the per-slice vane-inversion experiment showed the
    # tip table is two-term to its own noise, so this is the card-
    # physics ceiling).  Also verify the momentum-bookkeeping
    # invariant that caught the K2 bug: Π det(x-block) must track
    # TW's (= physical (βγ)_in/(βγ)_out ≈ 0.1238).
    # 2026-07-30 vane-field campaign (smooth TW calibration in
    # step_kicks): median 0.0125, mean 0.021, 200/203 < 10 %.
    assert np.median(err) < 0.02
    assert err.mean() < 0.03
    assert (err < 0.10).sum() >= 197
    assert err.max() < 0.30


def test_sync_ramp_matches_tracewin_exactly(tw2term_cells, env_nosc):
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    g0 = 1.0 + np.interp(1.9561, env_nosc["s_m"], env_nosc["gam1"])
    ref = ReferenceParticle(species=H_MINUS, w_kin=(g0 - 1) * H_MINUS.mass,
                            frequency=162.5)
    for c in tw2term_cells:
        c.advance_ref(ref)
    # TW's own chart: gamma-1 = 2.082166e-3 -> 1.955717 MeV
    assert ref.w_kin == pytest.approx(1.955717, abs=5e-6)


def test_zero_a10_cell_gives_no_gain_and_pure_quad():
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.rfq_cell import RfqCell
    cell = RfqCell("c", voltage_V=60000.0, r0_mm=5.576, A10=0.0,
                   modulation=1.0, length_mm=7.4, phi_s_deg=-90.0,
                   cell_type=2, field_model="tw2term")
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.03, frequency=162.5)
    W0 = ref.w_kin
    M = cell.fitted_matrix(ref)
    cell.advance_ref(ref)
    assert ref.w_kin == pytest.approx(W0, abs=1e-15)   # m=1 ⇒ no gain
    assert M[1, 0] != 0.0                              # quad still acts
    assert M[5, 4] == pytest.approx(0.0, abs=1e-12)    # no bunching
    # exact mirror symmetry: flipping the cell type swaps the planes
    # (thick F and D lenses differ in |M21| — sin vs sinh — so the
    # naive x↔−y antisymmetry does NOT hold; the mirror identity does)
    flipped = RfqCell("c2", voltage_V=60000.0, r0_mm=5.576, A10=0.0,
                      modulation=1.0, length_mm=7.4, phi_s_deg=-90.0,
                      cell_type=-2, field_model="tw2term")
    ref2 = ReferenceParticle(species=H_MINUS, w_kin=0.03, frequency=162.5)
    M2 = flipped.fitted_matrix(ref2)
    assert np.allclose(M[0:2, 0:2], M2[2:4, 2:4], rtol=1e-12)
    assert np.allclose(M[2:4, 2:4], M2[0:2, 0:2], rtol=1e-12)


@pytest.mark.slow
def test_mp_capture_through_pxie_rfq():
    """The headline regression: legacy captured 24 % (frozen phases —
    no slip, no bunching, no losses).  With Phase-2 losses active this
    synthetic-Gaussian run scrapes its (LEBT-model-grown) halo on the
    real apertures, so the gates are: the bucket works (every survivor
    is accelerated — the DC junk is physically removed by the vanes)
    and total capture stays far above the legacy 24 %.  The
    measurement-grade benchmark with TW's own input distribution lives
    in test_rfq_losses.test_full_line_dst_benchmark."""
    import os
    if not rl.PXIE_DECK.is_file():
        pytest.skip("lebt_plus_rfq example deck not present")
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.io.tracewin_parser import parse_tracewin
    cfg = BeamConfig(
        species="H-", energy=0.029999999, frequency=162.5,
        current=5.0, duty_cycle=100.0, n_particles=800,
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
        Simulation(lat, beam, space_charge="off",
                   record_substeps=False).run()
    finally:
        os.chdir(cwd)
    alive = ~beam.lost
    W = beam.ref.w_kin + beam.particles[alive, 5]
    capture = (W > 1.75).sum() / cfg.n_particles
    assert capture > 0.50                                  # legacy: 0.24
    # every survivor is captured — the vanes remove the DC junk
    assert (W > 1.75).sum() / max(alive.sum(), 1) > 0.95
    # captured particles form a real bunch around TW's exit energy
    assert np.mean(W[W > 1.75]) == pytest.approx(1.956, abs=0.02)
    assert np.std(beam.particles[alive, 4][W > 1.75]) < 60.0  # deg


def test_envelope_vs_vane_based_reference(pxie_deck, env_nosc):
    """With the smooth TW calibration the envelope tracks the
    vane-based ENV export at 1.1 % (x) / 1.7 % (y) mean; gates at 5 %
    for BOTH planes (the y plane was previously ungated — adversarial
    finding).  The legacy model sat at 180-450 %."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.cli.common import build_ref, _envelope_initial
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.elements.rfq_cell import RfqCell
    cfg = BeamConfig(
        species="H-", energy=0.029999999, frequency=162.5,
        current=0.0, duty_cycle=100.0, n_particles=1000,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.1370111, alpha_x=-6.1145021, beta_x=1.3870467,
        emit_ny=0.1370329, alpha_y=-6.07846, beta_y=1.3799354,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0,
        continuous=True, dc_energy_spread_keV=0.0)
    cells = [e for e in pxie_deck.elements if isinstance(e, RfqCell)]
    saved = [c.field_model for c in cells]
    for e in cells:
        e.field_model = "tw2term"
    try:
        ref = build_ref(cfg)
        res = EnvelopeSolver(pxie_deck, ref, _envelope_initial(cfg, ref),
                             current=0.0).run()
    finally:
        for c, fm in zip(cells, saved):
            c.field_model = fm
    s = np.asarray(res.s) * 1e-3
    sx = np.asarray(res.sigma_x)
    tw_sx = np.interp(s, env_nosc["s_m"], env_nosc["sigma_x_mm"])
    sy = np.asarray(res.sigma_y)
    tw_sy = np.interp(s, env_nosc["s_m"], env_nosc["sigma_y_mm"])
    sel = (s > 1.9561) & (tw_sx > 0.05)
    dev_x = np.abs(sx[sel] / tw_sx[sel] - 1.0)
    dev_y = np.abs(sy[sel] / tw_sy[sel] - 1.0)
    # 2026-07-30 vane-field campaign: 1.1 % mean (x) / 1.7 % (y) with
    # the smooth TW calibration — BELOW the ~4 % with which TW's own
    # matrices reproduce the same chart (see the epistemic caution in
    # rfq_coefficients).  BOTH planes gated.
    assert dev_x.mean() < 0.05
    assert dev_y.mean() < 0.05
    assert np.asarray(res.ref_w_kin)[-1] == pytest.approx(1.955717,
                                                          abs=1e-5)
