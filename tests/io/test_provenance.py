# tests/io/test_provenance.py
"""Result-file provenance (honesty round, phase 3).

A results file used to store only beam-config attrs — nothing pinned
WHICH code, WHICH deck or WHICH numerical configuration produced it
(PRAB review finding).  HDF5 output now always carries a provenance/
group; openPMD carries the standard software attributes; the
LINAC_GEN_USE_GPU env override is loud and recorded.
"""
import hashlib
import logging

import h5py
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift


@pytest.fixture(scope="module")
def recorder():
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    return Simulation(lat, beam).run()


@pytest.fixture()
def deck(tmp_path):
    p = tmp_path / "prov.dat"
    p.write_text("DRIFT 100 15 0\nEND\n")
    return str(p)


def test_hdf5_provenance_always_present(recorder, tmp_path):
    from linac_gen.io.hdf5_output import save_results_hdf5
    out = tmp_path / "r.h5"
    save_results_hdf5(recorder, str(out))          # no optional kwargs
    with h5py.File(out, "r") as f:
        prov = f["provenance"].attrs
        assert prov["linac_gen_version"]
        assert prov["git_commit"]                  # short hash or "unknown"
        assert prov["numpy_version"] == np.__version__
        assert prov["written"]


def test_hdf5_provenance_full(recorder, tmp_path, deck):
    from linac_gen.io.hdf5_output import save_results_hdf5
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_mode="adaptive")
    out = tmp_path / "r.h5"
    save_results_hdf5(recorder, str(out), lattice_path=deck, seed=42,
                      sc_config=sc)
    expect_hash = hashlib.sha256(open(deck, "rb").read()).hexdigest()
    with h5py.File(out, "r") as f:
        prov = f["provenance"].attrs
        assert prov["lattice_sha256"] == expect_hash
        assert prov["lattice_path"] == deck
        assert prov["beam_seed"] == 42
        assert prov["sc_grid_mode"] == "adaptive"
        assert prov["sc_nx"] == 32
        assert "backend_resolved_mode" in prov


def test_hdf5_provenance_referenced_inputs(recorder, tmp_path, deck):
    """2026-07 completeness round: the provenance group hashes the
    files the lattice_sha256 does NOT cover — field-map data, the
    imported beam file — and records the parse-downgrade ledger,
    cadence, FP precision and the OpenMP schedule."""
    from linac_gen.io.hdf5_output import save_results_hdf5

    # A lattice carrying a resolved field-file prefix + a parse ledger.
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    fm_data = tmp_path / "cav.edz"
    fm_data.write_bytes(b"0 1\n1 2\n")
    lat.elements[0].field_file = str(tmp_path / "cav")   # prefix
    lat.parse_warnings = ["Line 3: THIN_STEERING elec dropped"]

    # SUPERPOSE container: field_file=None on the container, real files
    # on the children ((z0, map) tuples) — the loop must descend.
    child_data = tmp_path / "sup_child.edz"
    child_data.write_bytes(b"9 9\n8 8\n")
    child = type("FakeMap", (), {})()
    child.field_file = str(tmp_path / "sup_child")
    container = Drift("SUP", length=0.0)
    container.children = [(0.0, child)]
    lat.add(container)

    dst = tmp_path / "beam.dst"
    dst.write_bytes(b"\x00" * 64)

    out = tmp_path / "r2.h5"
    save_results_hdf5(recorder, str(out), lattice=lat,
                      input_beam_path=str(dst))
    with h5py.File(out, "r") as f:
        prov = f["provenance"].attrs
        assert prov["fp_dtype"] == "float64"
        # C++ kernels report the OpenMP schedule; pure-Python CI
        # builds (LINAC_GEN_DISABLE_CPP=1) honestly report n/a
        assert (prov["omp_schedule"] == "static"
                or str(prov["omp_schedule"]).startswith("n/a"))
        assert prov["integration_steps_per_metre"] == 100.0
        assert prov["sc_steps_per_metre"] == 50.0
        assert "THIN_STEERING" in prov["parse_downgrades"]
        expect = hashlib.sha256(fm_data.read_bytes()).hexdigest()
        assert f"cav.edz:{expect}" in prov["field_map_sha256"]
        expect_child = hashlib.sha256(child_data.read_bytes()).hexdigest()
        assert f"sup_child.edz:{expect_child}" in prov["field_map_sha256"]
        assert prov["input_beam_sha256"] == hashlib.sha256(
            dst.read_bytes()).hexdigest()
        assert prov["input_beam_path"] == str(dst)


def test_hdf5_provenance_surrogate_manifest(recorder, tmp_path):
    """Registered surrogates appear in the provenance manifest."""
    import types

    from linac_gen.io.hdf5_output import save_results_hdf5
    from linac_gen.surrogates import registry

    registry.clear()
    wdir = tmp_path / "weights_cav1"
    wdir.mkdir()
    (wdir / "weights.pt").write_bytes(b"\x01\x02weights")
    # metadata-level hash (the route-independent source) takes
    # precedence; weights_dir remains the fallback for older objects.
    fake_meta = types.SimpleNamespace(
        element_class="FieldMap", training_seed=7, val_mape=0.42,
        weights_sha256=hashlib.sha256(b"\x01\x02weights").hexdigest())
    fake_surr = types.SimpleNamespace(
        metadata=fake_meta,
        wrapped=types.SimpleNamespace(name="CAV1"))
    try:
        registry._REGISTRY[("abc123def456", "CAV1")] = fake_surr
        registry._BY_NAME["CAV1"] = fake_surr
        out = tmp_path / "r3.h5"
        save_results_hdf5(recorder, str(out))
        with h5py.File(out, "r") as f:
            prov = f["provenance"].attrs
            assert "CAV1" in prov.get("surrogates_registered", "")
            assert "seed=7" in prov["surrogates_registered"]
            expect = hashlib.sha256(b"\x01\x02weights").hexdigest()[:16]
            assert f"weights={expect}" in prov["surrogates_registered"]
    finally:
        registry.clear()


def test_hdf5_provenance_weights_dir_fallback(recorder, tmp_path):
    """Older surrogate objects without the metadata-level hash still
    get their weights checksummed via the retained weights_dir (the
    fallback branch must stay covered — adversarial-review finding)."""
    import types

    from linac_gen.io.hdf5_output import save_results_hdf5
    from linac_gen.surrogates import registry

    registry.clear()
    wdir = tmp_path / "weights_legacy"
    wdir.mkdir()
    (wdir / "weights.pt").write_bytes(b"legacy-bytes")
    fake_meta = types.SimpleNamespace(
        element_class="FieldMap", training_seed=3, val_mape=0.1,
        weights_sha256="")                     # no metadata-level hash
    fake_surr = types.SimpleNamespace(
        metadata=fake_meta, weights_dir=str(wdir),
        wrapped=types.SimpleNamespace(name="CAV9"))
    try:
        registry._REGISTRY[("ffff00001111", "CAV9")] = fake_surr
        registry._BY_NAME["CAV9"] = fake_surr
        out = tmp_path / "r4.h5"
        save_results_hdf5(recorder, str(out))
        with h5py.File(out, "r") as f:
            manifest = f["provenance"].attrs["surrogates_registered"]
            expect = hashlib.sha256(b"legacy-bytes").hexdigest()[:16]
            assert f"weights={expect}" in manifest
    finally:
        registry.clear()


def test_write_results_hashes_imported_beam(recorder, tmp_path):
    """The CLI/GUI entry path derives input_beam_path from a
    file-sourced BeamConfig — the kwarg must not be dead wiring
    (adversarial review 2026-07)."""
    from linac_gen.cli.common import write_results
    from linac_gen.core.config import BeamConfig

    dst = tmp_path / "in.dst"
    dst.write_bytes(b"beamdata" * 8)
    cfg = BeamConfig()
    cfg.source = "file"
    cfg.distribution_file = str(dst)
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    out = tmp_path / "cli.h5"
    write_results(recorder, out, "hdf5", cfg, lat)
    with h5py.File(out, "r") as f:
        prov = f["provenance"].attrs
        assert prov["input_beam_sha256"] == hashlib.sha256(
            dst.read_bytes()).hexdigest()
        assert prov["input_beam_path"] == str(dst)
    # generate-sourced beams record nothing
    out2 = tmp_path / "cli2.h5"
    write_results(recorder, out2, "hdf5", BeamConfig(), lat)
    with h5py.File(out2, "r") as f:
        assert "input_beam_sha256" not in f["provenance"].attrs


def test_hdf5_legacy_signature_unchanged(recorder, tmp_path):
    """Old positional callers must keep working (kwargs are optional)."""
    from linac_gen.io.hdf5_output import save_results_hdf5
    out = tmp_path / "legacy.h5"
    save_results_hdf5(recorder, str(out), None, None)
    with h5py.File(out, "r") as f:
        assert "envelope" in f and "provenance" in f


def test_openpmd_software_attrs(recorder, tmp_path):
    from linac_gen import __version__
    from linac_gen.io.openpmd_output import save_results_openpmd
    out = tmp_path / "r.opmd.h5"
    save_results_openpmd(recorder, str(out))
    with h5py.File(out, "r") as f:
        assert f.attrs["software"] == "HELIX (linac_gen)"
        assert f.attrs["softwareVersion"] == __version__
        assert f.attrs["date"]


def test_effective_backend_info(monkeypatch):
    from linac_gen.pic import gpu_backend
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    info = gpu_backend.effective_backend_info("cpu")
    assert info["requested"] == "cpu"
    assert info["resolved_mode"] == "cpu"
    assert info["env_override"] == ""
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "cpu")
    info = gpu_backend.effective_backend_info("gpu")
    assert info["resolved_mode"] == "cpu"          # env wins
    assert info["env_override"] == "cpu"


def test_env_override_logs_warning_once(monkeypatch, caplog):
    from linac_gen.pic import gpu_backend
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "cpu")
    monkeypatch.setattr(gpu_backend, "_ENV_OVERRIDE_WARNED", False)
    with caplog.at_level(logging.WARNING, logger=gpu_backend._log.name):
        assert gpu_backend._resolve_mode("gpu") == "cpu"
        assert gpu_backend._resolve_mode("gpu") == "cpu"   # second call quiet
    over = [r for r in caplog.records if "OVERRIDES" in r.getMessage()]
    assert len(over) == 1, [r.getMessage() for r in caplog.records]


def test_env_override_matching_value_is_silent(monkeypatch, caplog):
    from linac_gen.pic import gpu_backend
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "cpu")
    monkeypatch.setattr(gpu_backend, "_ENV_OVERRIDE_WARNED", False)
    with caplog.at_level(logging.WARNING, logger=gpu_backend._log.name):
        assert gpu_backend._resolve_mode("cpu") == "cpu"
    assert not [r for r in caplog.records if "OVERRIDES" in r.getMessage()]
