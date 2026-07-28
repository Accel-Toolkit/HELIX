"""Round-trip + real-file tests for the TraceWin .dst reader/writer.

Format spec: TraceWin user manual ("Files" section).  Layout:

    2*CHAR + INT(Np) + DOUBLE(Ib) + DOUBLE(freq) + CHAR
    + Np × 6×DOUBLE [x cm, x' rad, y cm, y' rad, phi rad, W MeV]
    + DOUBLE(mc² MeV)

Total: 31 + 48*Np bytes.
"""
import struct
import numpy as np
import pytest

from linac_gen.io.tracewin_dst import load_dst, write_dst


def test_round_trip_preserves_particles(tmp_path):
    """Write+read of LG-format particles must give back the same array."""
    rng = np.random.default_rng(42)
    n = 1000
    particles = np.column_stack([
        rng.normal(0, 1.0, n),       # x mm
        rng.normal(0, 0.5, n),       # xp mrad
        rng.normal(0, 1.0, n),
        rng.normal(0, 0.5, n),
        rng.uniform(-10, 10, n),     # dphi deg
        rng.normal(0, 0.001, n),     # dW MeV
    ])
    # Subtract sample mean from longitudinal so dphi/dW are pure deviations
    # (mirrors the convention load_dst applies on read).
    particles[:, 4] -= particles[:, 4].mean()
    particles[:, 5] -= particles[:, 5].mean()

    path = tmp_path / "test.dst"
    write_dst(str(path), particles,
              current_mA=5.0, frequency_MHz=162.5,
              mass_MeV=939.294, w_kin_ref=2.5, phi_ref_rad=0.0)

    loaded, header = load_dst(str(path))

    np.testing.assert_allclose(loaded, particles, rtol=1e-12, atol=1e-12)
    assert header["n_particles"] == n
    assert header["current_mA"] == pytest.approx(5.0)
    assert header["frequency_MHz"] == pytest.approx(162.5)
    assert header["mass_MeV"] == pytest.approx(939.294)
    assert header["w_kin_ref"] == pytest.approx(2.5, abs=1e-3)
    assert header["phi_ref_rad"] == pytest.approx(0.0, abs=1e-3)


def test_round_trip_with_nonzero_phi_ref(tmp_path):
    """phi_ref_rad on write must be reproducible on read (centroid recovery)."""
    rng = np.random.default_rng(11)
    n = 500
    particles = np.column_stack([
        rng.normal(0, 1.0, n), rng.normal(0, 0.5, n),
        rng.normal(0, 1.0, n), rng.normal(0, 0.5, n),
        rng.uniform(-5, 5, n), rng.normal(0, 0.01, n),
    ])
    particles[:, 4] -= particles[:, 4].mean()
    particles[:, 5] -= particles[:, 5].mean()

    path = tmp_path / "shift.dst"
    write_dst(str(path), particles,
              current_mA=10.0, frequency_MHz=80.5,
              mass_MeV=938.272, w_kin_ref=10.0, phi_ref_rad=1.234)

    loaded, header = load_dst(str(path))
    np.testing.assert_allclose(loaded, particles, rtol=1e-12, atol=1e-12)
    assert header["w_kin_ref"] == pytest.approx(10.0, abs=1e-3)
    assert header["phi_ref_rad"] == pytest.approx(1.234, abs=1e-6)


def test_truncated_body_rejected(tmp_path):
    """File whose size doesn't match 31 + 48*N + 8 must raise."""
    path = tmp_path / "trunc.dst"
    raw = struct.pack("<2sIddB", b"\x7d\x64", 10, 5.0, 162.5, 0x7d)
    raw += np.zeros(3, dtype="<f8").tobytes()    # only 3 floats
    raw += struct.pack("<d", 939.294)             # trailer
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="File size"):
        load_dst(str(path))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dst("/nonexistent/path/foo.dst")


# ---------------------------------------------------------------------------
# Real-file test — opt-in via the HELIX_REAL_DST env var (the assertions
# below are specific to the PIP-II SCL+BTL input.dst).
# ---------------------------------------------------------------------------
import os as _os
import os.path as _osp
_REAL_DST = _os.environ.get("HELIX_REAL_DST", "")


@pytest.mark.skipif(not (_REAL_DST and _osp.isfile(_REAL_DST)),
                    reason="set HELIX_REAL_DST=<path to the PIP-II SCL+BTL "
                           "input.dst> to run this real-file check")
def test_real_tracewin_dst_loads():
    """Load a real TraceWin-produced .dst and sanity-check the header."""
    particles, header = load_dst(_REAL_DST)

    # Empirically known PIP-II SCL+BTL input.dst values:
    assert header["n_particles"] == 10000
    assert header["current_mA"] == pytest.approx(4.84235, abs=1e-4)
    assert header["frequency_MHz"] == pytest.approx(162.5, abs=1e-3)
    assert header["mass_MeV"] == pytest.approx(939.294, abs=1e-2)

    assert particles.shape == (10000, 6)
    # Centroid of returned (dphi, dW) must be ≈ 0 since load_dst subtracts
    # the bunch centroid.
    assert abs(np.mean(particles[:, 4])) < 1e-9   # dphi deg
    assert abs(np.mean(particles[:, 5])) < 1e-9   # dW MeV

    # Reasonable transverse rms for a MEBT-entrance beam (1-3 mm scale)
    assert 0.5 < np.std(particles[:, 0]) < 5.0
    assert 0.5 < np.std(particles[:, 2]) < 5.0

    # Sample-derived Twiss + emittance must be present and physical.
    for k in ("emit_x", "emit_y", "emit_z",
              "emit_nx", "emit_ny",
              "alpha_x", "beta_x",
              "alpha_y", "beta_y",
              "alpha_z", "beta_z"):
        assert k in header, f"missing header key {k}"
        assert isinstance(header[k], float), f"{k} not float"
    # Normalised emittance for PIP-II MEBT entrance: ε_n ~ 0.2 mm·mrad
    assert 0.1 < header["emit_nx"] < 0.4
    assert 0.1 < header["emit_ny"] < 0.4
    # β_x must be positive (positive-definite σ-matrix).
    assert header["beta_x"] > 0
    assert header["beta_y"] > 0


def test_twiss_round_trip_through_factory(tmp_path):
    """Generate a beam from Twiss, write it to .dst, reload it via the
    distributions factory and check that the loaded ε/Twiss match the
    originals to sample-statistics tolerance.
    """
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_dst import write_dst

    cfg = BeamConfig(
        species="H-", energy=2.1226695, frequency=162.5,
        current=4.84235, duty_cycle=100.0, n_particles=20000,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.21, alpha_x=1.228, beta_x=0.316,
        emit_ny=0.22, alpha_y=-0.095, beta_y=0.113,
        emit_z=0.066, alpha_z=0.0, beta_z=819.05,
    )
    beam0 = create_beam(cfg, seed=42)
    path = tmp_path / "rt.dst"
    write_dst(str(path), beam0.particles,
              current_mA=cfg.current, frequency_MHz=cfg.frequency,
              mass_MeV=beam0.ref.species.mass,
              w_kin_ref=cfg.energy, phi_ref_rad=0.0)

    # Load via factory in source="file" mode
    cfg.source = "file"
    cfg.distribution_file = str(path)
    beam1 = create_beam(cfg)
    # Particles round-trip exactly in (x, x', y, y'); the longitudinal pair
    # (dphi, dW) loses any residual centroid because load_dst subtracts the
    # bunch mean as the synchronous reference (matches TraceWin convention).
    beam0_ref = beam0.particles.copy()
    beam0_ref[:, 4] -= beam0_ref[:, 4].mean()
    beam0_ref[:, 5] -= beam0_ref[:, 5].mean()
    np.testing.assert_allclose(beam1.particles, beam0_ref,
                               rtol=1e-12, atol=1e-12)
    # Config emittance & Twiss should now reflect the file's values
    assert cfg.emit_nx == pytest.approx(0.21, abs=2e-3)
    assert cfg.emit_ny == pytest.approx(0.22, abs=2e-3)
    assert cfg.alpha_x == pytest.approx(1.228, abs=2e-2)
    assert cfg.beta_x  == pytest.approx(0.316, abs=2e-3)
