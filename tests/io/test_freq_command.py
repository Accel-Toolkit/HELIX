"""FREQ card as an active lattice command (machine-clock switch at the card).

TraceWin semantics: the RF reference clock changes AT the FREQ card, not at
the first downstream RF element.  With beam frequency ≠ lattice FREQ (the
PIP-II HB650 example: 804.6 vs 804.96) the old first-cavity switch left the
clock ~0.5° behind TraceWin's over the entrance section — enough to seed a
spurious relative synchrotron oscillation (σ_φ correlation 0.929 → 0.940
after the fix, validated against the TraceWin reference run).
"""
import os
import tempfile

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.track_state import TrackState
from linac_gen.elements.lattice_commands import Freq
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin

DECK = """
FREQ 804.96
DRIFT 500 15 0
NCELLS 1 4 0.456630316 3.16E6 288.5 15.0 1.0 0.0 0.0 0.0 0.0
end
"""


def _parse(text):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "deck.dat")
        with open(p, "w") as f:
            f.write(text)
        return parse_tracewin(p)


def test_freq_card_materializes_at_its_position():
    lat, _ = _parse(DECK)
    assert isinstance(lat.elements[0], Freq)
    assert lat.elements[0].frequency_mhz == pytest.approx(804.96)


def test_freq_roundtrip_emits_single_card():
    lat, _ = _parse(DECK)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "rt.dat")
        write_tracewin(lat, out)
        txt = open(out).read()
        cards = [l for l in txt.splitlines()
                 if l.strip().upper().startswith("FREQ")]
        assert len(cards) == 1                      # no duplicate before NCELLS
        lat2, _ = _parse(txt)
    assert sum(isinstance(e, Freq) for e in lat2.elements) == 1


def test_apply_command_switches_clock_time_continuously():
    """Mid-lattice jump: same instant re-expressed in new-frequency degrees."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=116.1, frequency=100.0)
    ref.phi_s = 1000.0
    Freq("FREQ_X", frequency_mhz=200.0).apply_command(TrackState(ref=ref))
    assert ref.frequency == pytest.approx(200.0)
    assert ref.phi_s == pytest.approx(2000.0)       # ω₂t = (f₂/f₁)·ω₁t


def test_apply_command_noop_when_frequency_equal():
    """Same-frequency decks stay bit-identical (non-breaking guard)."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=116.1, frequency=804.96)
    ref.phi_s = 1234.5
    Freq("FREQ_X", frequency_mhz=804.96).apply_command(TrackState(ref=ref))
    assert ref.frequency == pytest.approx(804.96)
    assert ref.phi_s == pytest.approx(1234.5)


def test_apply_command_rescales_mp_dphi():
    from linac_gen.core.beam import Beam
    ref = ReferenceParticle(species=H_MINUS, w_kin=116.1, frequency=100.0)
    beam = Beam(ref=ref, n_particles=2, current=0.0)
    beam.particles[:] = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, -4.0, 0.1]])
    Freq("FREQ_X", frequency_mhz=200.0).apply_command(
        TrackState(ref=beam.ref, beam=beam))
    assert beam.particles[0, 4] == pytest.approx(20.0)
    assert beam.particles[1, 4] == pytest.approx(-8.0)
    assert beam.particles[1, 5] == pytest.approx(0.1)   # ΔW untouched


def test_envelope_clock_switches_at_card_not_first_cavity():
    """The reference runs at the FREQ value from s=0, before any RF element."""
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat, _ = _parse(DECK)
    ref = ReferenceParticle(species=H_MINUS, w_kin=116.1, frequency=804.6)
    ini = dict(alpha_x=0.0, beta_x=5.0, emit_x=0.2,
               alpha_y=0.0, beta_y=5.0, emit_y=0.2,
               alpha_z=0.0, beta_z=60.0, emit_z=0.9)
    res = EnvelopeSolver(lat, ref, ini, current=0.0).run()
    freqs = np.array(res.ref_frequency)
    # row 0 is the input record (before the FREQ card fires); every step
    # after it — including the entrance drift, i.e. BEFORE the cavity —
    # runs at the machine frequency.
    assert freqs[0] == pytest.approx(804.6)
    assert np.all(freqs[1:][freqs[1:] > 0] == pytest.approx(804.96))


def test_envelope_phase_probe_includes_freq_jump():
    """The probe's exact-tangent-map contract across a Freq command: the
    per-element map product must reconstruct the recorded σ, including the
    freq-jump D = diag(1,1,1,1,f2/f1,1) at the card (regression: the command
    branch used to rescale σ without pushing D into the probe)."""
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat, _ = _parse(DECK)          # FREQ 804.96 at index 0, beam at 804.6
    ref = ReferenceParticle(species=H_MINUS, w_kin=116.1, frequency=804.6)
    ini = dict(alpha_x=0.0, beta_x=5.0, emit_x=0.2,
               alpha_y=0.0, beta_y=5.0, emit_y=0.2,
               alpha_z=0.0, beta_z=60.0, emit_z=0.9)
    res = EnvelopeSolver(lat, ref, ini, current=0.0,
                         phase_probe=True).run()
    sigma0 = np.asarray(res.sigma_matrix[0], dtype=float)
    sigma_end = np.asarray(res.sigma_matrix[-1], dtype=float)
    M = np.eye(6)
    for Mj in res.element_maps_dep:
        M = np.asarray(Mj, dtype=float) @ M
    recon = M @ sigma0 @ M.T
    np.testing.assert_allclose(recon[4, 4], sigma_end[4, 4],
                               rtol=1e-9, err_msg="probe missed the freq-jump D")
    np.testing.assert_allclose(recon[0, 0], sigma_end[0, 0], rtol=1e-9)


def test_backtrack_closes_dphi_through_freq():
    """Exact-closure across a Freq command: the forward Δφ *= f2/f1 rescale
    must be undone on the backward walk (regression: it wasn't, giving a
    ×(f2/f1) Δφ error — factor 2 across a 162.5→325 boundary)."""
    from linac_gen.core.beam import Beam
    from linac_gen.tracking.tracker import Tracker
    from linac_gen.tracking.backtrack import backtrack_distribution
    deck = """FREQ 325.0
DRIFT 200 15 0
DRIFT 300 15 0
end
"""
    lat, _ = _parse(deck)
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    beam = Beam(ref=ref, n_particles=8, current=0.0)
    rng = np.random.default_rng(7)
    beam.particles[:, :] = 0.0
    beam.particles[:, 4] = rng.uniform(-10.0, 10.0, 8)   # Δφ [deg @162.5]
    beam.particles[:, 5] = rng.uniform(-0.01, 0.01, 8)   # ΔW [MeV]
    initial = beam.particles.copy()
    Tracker(lat, beam).run()
    entrance = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    backtrack_distribution(lat, beam, entrance,
                           end=len(lat.elements) - 1)
    np.testing.assert_allclose(beam.particles[:, 4], initial[:, 4],
                               atol=1e-9,
                               err_msg="Freq Δφ rescale not undone backward")
