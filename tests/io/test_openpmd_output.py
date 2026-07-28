"""openPMD writer / reader: openPMD-standard 1.1 compliance and round-trip.

Verifies:
* Root attributes match the openPMD-1.1 spec (``openPMD``, ``basePath``,
  ``meshesPath``, ``particlesPath``, ``iterationEncoding``,
  ``iterationFormat``, ``openPMDextension``).
* Particle group layout (``position/{x,y,z}``, ``momentum/{x,y,z}``,
  ``mass``, ``charge``, ``weighting``, ``particleStatus``,
  ``positionOffset``) with correct ``unitSI`` and ``unitDimension``.
* Round-trip via :func:`load_results_openpmd` returns the same dict
  shape as :func:`linac_gen.io.hdf5_output.load_results_hdf5`.
* Envelope-only runs (no particle snapshots) produce a valid file with
  only the HELIX-extension ``envelope/`` group.
"""
import numpy as np
import pytest
import h5py

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.diagnostics.recorder import DiagnosticRecorder
from linac_gen.io.openpmd_output import (
    save_results_openpmd, load_results_openpmd, is_openpmd_file,
)


def _make_recorder_with_snapshot(n=100, w_kin_MeV=800.0):
    """Build a DiagnosticRecorder populated with one envelope step and one
    particle snapshot — enough to exercise both code paths."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=w_kin_MeV, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=4.84)
    rng = np.random.default_rng(0)
    beam.particles[:, 0] = rng.normal(0.0, 2.0, n)    # x mm
    beam.particles[:, 1] = rng.normal(0.0, 1.0, n)    # xp mrad
    beam.particles[:, 2] = rng.normal(0.0, 2.0, n)    # y mm
    beam.particles[:, 3] = rng.normal(0.0, 1.0, n)    # yp mrad
    beam.particles[:, 4] = rng.normal(0.0, 1.0, n)    # dphi deg
    beam.particles[:, 5] = rng.normal(0.0, 0.05, n)   # dw MeV

    rec = DiagnosticRecorder()
    rec.record(beam, s_position=0.0, element_name="INPUT")
    rec.record(beam, s_position=100.0, element_name="DRIFT_1")
    # Take a particle snapshot at s=100mm.
    rec._snapshots[100.0] = (beam.particles.copy(), beam.ref.copy())
    return rec


# ── openPMD spec compliance ──────────────────────────────────────────────────

def test_root_attributes_match_openpmd_spec(tmp_path):
    rec = _make_recorder_with_snapshot(n=50)
    out = tmp_path / "round1.opmd.h5"
    save_results_openpmd(rec, out)

    with h5py.File(out, "r") as f:
        assert f.attrs["openPMD"] == "1.1.0"
        assert int(f.attrs["openPMDextension"]) == 0
        assert f.attrs["basePath"] == "/data/%T/"
        assert f.attrs["iterationEncoding"] == "groupBased"
        assert f.attrs["iterationFormat"] == "/data/%T/"
        assert f.attrs["particlesPath"] == "particles/"
        assert f.attrs["meshesPath"] == "meshes/"


def test_particle_group_layout_complete(tmp_path):
    rec = _make_recorder_with_snapshot(n=30)
    out = tmp_path / "round2.opmd.h5"
    save_results_openpmd(rec, out)

    with h5py.File(out, "r") as f:
        spec = f["data/0/particles/H-"]
        # Required sub-records
        for sub in ("position", "positionOffset", "momentum"):
            assert sub in spec, f"missing {sub}"
            for axis in ("x", "y", "z"):
                ds = spec[f"{sub}/{axis}"]
                assert ds.shape == (30,)
                assert "unitSI" in ds.attrs
                assert "unitDimension" in ds.attrs
        # mass / charge / weighting / particleStatus must be length-N arrays
        for sub in ("mass", "charge", "weighting", "particleStatus"):
            assert sub in spec, f"missing {sub}"
            assert spec[sub].shape == (30,)


def test_unit_dimensions_are_correct(tmp_path):
    """Length records → [1,0,0,0,0,0,0], momentum → [1,1,-1,0,0,0,0]."""
    rec = _make_recorder_with_snapshot(n=10)
    out = tmp_path / "unit_dim.opmd.h5"
    save_results_openpmd(rec, out)

    with h5py.File(out, "r") as f:
        x_dim = f["data/0/particles/H-/position/x"].attrs["unitDimension"]
        px_dim = f["data/0/particles/H-/momentum/x"].attrs["unitDimension"]
        assert np.allclose(x_dim, [1, 0, 0, 0, 0, 0, 0])
        assert np.allclose(px_dim, [1, 1, -1, 0, 0, 0, 0])


def test_species_label_propagates(tmp_path):
    """Species name from the reference particle becomes the openPMD species
    group name and the speciesType attribute."""
    rec = _make_recorder_with_snapshot(n=10)
    # Override the snapshot's ref species to a proton — both have same mass
    # but different name.
    ref_p = ReferenceParticle(species=PROTON, w_kin=800.0, frequency=162.5)
    rec._snapshots[100.0] = (rec._snapshots[100.0][0], ref_p)

    out = tmp_path / "species.opmd.h5"
    save_results_openpmd(rec, out)
    with h5py.File(out, "r") as f:
        assert "proton" in f["data/0/particles"]
        spec_attr = f["data/0/particles/proton"].attrs["speciesType"]
        assert spec_attr == "proton"


# ── round-trip ───────────────────────────────────────────────────────────────

def test_loader_returns_envelope_dict(tmp_path):
    rec = _make_recorder_with_snapshot(n=20)
    out = tmp_path / "rt.opmd.h5"
    save_results_openpmd(rec, out)

    loaded = load_results_openpmd(out)
    # Same dict shape as HDF5 loader (HELIX_FEATURE_GAPS S1 contract).
    assert "s" in loaded
    assert "sigma_x" in loaded
    assert "ref_w_kin" in loaded
    assert len(loaded["s"]) == 2   # two record() calls
    assert np.array_equal(loaded["s"], np.asarray(rec.s, dtype=np.float64))


def test_is_openpmd_file_sentinel(tmp_path):
    rec = _make_recorder_with_snapshot(n=10)
    opmd = tmp_path / "is_opmd.opmd.h5"
    save_results_openpmd(rec, opmd)
    assert is_openpmd_file(opmd) is True

    # A plain (HELIX-native) HDF5 file should sniff False.
    from linac_gen.io.hdf5_output import save_results_hdf5
    helix = tmp_path / "is_helix.h5"
    save_results_hdf5(rec, str(helix))
    assert is_openpmd_file(helix) is False


# ── envelope-only path ───────────────────────────────────────────────────────

def test_envelope_only_run_produces_valid_file(tmp_path):
    """No particle snapshots → file still has /data/0/envelope/ group, no
    particles."""
    rec = DiagnosticRecorder()
    # Populate a couple of envelope steps without recording any snapshot.
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=100, current=4.84)
    rec.record(beam, s_position=0.0, element_name="INPUT")
    rec.record(beam, s_position=10.0, element_name="DRIFT_1")
    assert not rec._snapshots

    out = tmp_path / "env_only.opmd.h5"
    save_results_openpmd(rec, out)
    with h5py.File(out, "r") as f:
        assert "envelope" in f["data/0"]
        assert "particles" not in f["data/0"]


def test_hdf5_writer_handles_envelope_results_without_snapshots_attr(tmp_path):
    """EnvelopeResults has no ``_snapshots`` attribute; the HDF5 writer's
    defensive ``getattr(..., None)`` must let it through cleanly.

    Regression test: prior to the fix the writer crashed with
    AttributeError, silently failing every envelope auto-dump in the GUI.
    """
    from linac_gen.io.hdf5_output import save_results_hdf5
    from linac_gen.tracking.envelope import EnvelopeResults

    results = EnvelopeResults()
    results.s = [0.0, 100.0]
    results.sigma_x = [1.0, 1.0]
    results.sigma_y = [1.0, 1.0]
    results.ref_w_kin = [800.0, 800.0]

    out = tmp_path / "env_only.h5"
    # Must NOT raise AttributeError
    save_results_hdf5(results, str(out))
    assert out.exists()
    with h5py.File(out, "r") as f:
        # envelope arrays land in /envelope/
        assert f["envelope/s"][:].tolist() == [0.0, 100.0]
        # No particles group when results have no snapshots
        assert "particles" not in f


# ── coordinate conversion physics ────────────────────────────────────────────

def test_zero_offset_particle_produces_zero_lab_coords(tmp_path):
    """A particle exactly on the reference orbit (HELIX zeros) → x=y=z=0,
    px=py=0, pz=p_ref."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    rec = DiagnosticRecorder()
    rec.s = [0.0]
    rec.sigma_x = [0.0]
    # Build a snapshot with one particle at (0,0,0,0,0,0).
    particles = np.zeros((1, 6))
    rec._snapshots[0.0] = (particles, ref.copy())

    out = tmp_path / "zero.opmd.h5"

    class _BC:
        frequency = 162.5
        current = 4.84
        n_particles = 1
    save_results_openpmd(rec, out, beam_config=_BC())
    with h5py.File(out, "r") as f:
        x = f["data/0/particles/H-/position/x"][:]
        y = f["data/0/particles/H-/position/y"][:]
        z = f["data/0/particles/H-/position/z"][:]
        px = f["data/0/particles/H-/momentum/x"][:]
        py = f["data/0/particles/H-/momentum/y"][:]
        pz = f["data/0/particles/H-/momentum/z"][:]
    assert x[0] == 0.0 and y[0] == 0.0 and z[0] == 0.0
    assert px[0] == 0.0 and py[0] == 0.0
    # pz should equal p_ref (in SI).
    from linac_gen.core.constants import C_LIGHT, E_CHARGE
    p_ref_si = ref.bg * ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
    assert pz[0] == pytest.approx(p_ref_si, rel=1e-9)
