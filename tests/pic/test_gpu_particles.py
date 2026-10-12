"""The default engine's deposit and gather on the GPU
(linac_gen.pic.gpu_particles): the same code runs here on the CPU (numpy for
cupy, torch's CPU tensors) and must agree with the CPU kernels to rounding
and repeat bit for bit; the solve on the device equals the host solve; the
switch is honoured only with the CUDA backend.  The GPU itself is
exercised only where there is one (the CUDA test at the end;
scripts/verify_cuda.py on a CUDA machine)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.beam import Beam  # noqa: E402
from linac_gen.core.config import SpaceChargeConfig  # noqa: E402
from linac_gen.core.particle import PROTON  # noqa: E402
from linac_gen.core.reference import ReferenceParticle  # noqa: E402
from linac_gen.pic import gpu_particles  # noqa: E402
from linac_gen.pic.pic_solver import PicSolver  # noqa: E402


def _beam(n=6000, seed=3):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    b = Beam(ref=ref, n_particles=n, current=20.0)
    b.particles[:] = (np.random.default_rng(seed).normal(size=(n, 6))
                      * [1.0, 0.3, 1.0, 0.3, 3.0, 0.005])
    return b


def _ready_solver(kind, mode="fixed", **sc):
    """A PicSolver whose grid is set for the beam, and its rest-frame
    coordinates and charges."""
    beam = _beam()
    pic = PicSolver(SpaceChargeConfig(nx=24, ny=24, nz=24, grid_extent=4.0,
                                      kernel=kind, grid_mode=mode,
                                      use_gpu="cpu", **sc))
    coords = np.ascontiguousarray(beam.particles[:, [0, 2, 4]]) * [1, 1, 0.3]
    pic._setup_grid(coords)
    return pic, coords, pic._macro_charges(beam)


@pytest.mark.parametrize("kind", ["cic", "tsc"])
def test_the_device_solve_is_the_host_solve(kind):
    """solve_device on the CPU backend: solve's numbers, bit for bit."""
    pic, coords, q = _ready_solver(kind)
    from linac_gen.pic.pic_solver import _select_kernel
    rho = _select_kernel(kind)[0](coords, q, pic._grid_min, pic._grid_max,
                                  pic._n_grid)
    host = pic._solver.solve(rho)
    device = pic._solver.solve_device(rho.copy())
    for a, b in zip(host, device):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("kind", ["cic", "tsc"])
@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_the_gpu_path_agrees_with_the_cpu_kernels(kind, mode):
    """The torch deposit and gather add in another order than the CPU
    kernels: equal to rounding, and the same bits every time."""
    pic, coords, q = _ready_solver(kind, mode)
    cpu = pic._rest_frame_field(coords, _beam())
    args = (pic._solver, coords, q, pic._grid_min, pic._grid_max,
            pic._n_grid, kind)
    gpu = gpu_particles.rest_frame_field(*args, device="cpu")
    again = gpu_particles.rest_frame_field(*args, device="cpu")
    scale = np.max(np.abs(cpu))
    assert np.max(np.abs(gpu - cpu)) <= 1e-12 * scale
    assert np.array_equal(gpu, again)


def test_the_switch(monkeypatch):
    """The environment decides when set, either way; else the config."""
    monkeypatch.delenv(gpu_particles.ENV, raising=False)
    off = SpaceChargeConfig()
    on = SpaceChargeConfig(gpu_particles=True)
    assert not gpu_particles.requested(off)
    assert gpu_particles.requested(on)
    for value in ("1", "true", "on"):
        monkeypatch.setenv(gpu_particles.ENV, value)
        assert gpu_particles.requested(off)
    for value in ("0", "false", "off"):
        monkeypatch.setenv(gpu_particles.ENV, value)
        assert not gpu_particles.requested(on)


def test_the_setting_from_every_source():
    """Config (numpy bools too), project files with and without a named
    method, the CLI's text."""
    from linac_gen.pic.methods import resolve_sc
    assert SpaceChargeConfig(gpu_particles=np.bool_(True)).gpu_particles
    for bad in ("true", 2, 1.5, None):
        with pytest.raises(ValueError):
            SpaceChargeConfig(gpu_particles=bad)
    assert resolve_sc({"gpu_particles": True}).gpu_particles
    assert resolve_sc({"grid_nx": 32, "gpu_particles": "true"}).gpu_particles
    assert resolve_sc({"sc_method": "tsc_fixed",
                       "gpu_particles": True}).gpu_particles
    assert resolve_sc({}, {"sc": {"gpu_particles": "true"}}).gpu_particles
    assert not resolve_sc({}).gpu_particles


def test_a_request_that_cannot_be_met_is_said_once(monkeypatch, caplog):
    import logging
    monkeypatch.delenv(gpu_particles.ENV, raising=False)
    monkeypatch.setattr(gpu_particles, "_warned", set())
    with caplog.at_level(logging.WARNING, logger=gpu_particles.__name__):
        for _ in range(2):
            beam = _beam()
            pic = PicSolver(SpaceChargeConfig(nx=16, ny=16, nz=16,
                                              grid_extent=4.0, use_gpu="cpu",
                                              gpu_particles=True))
            pic.kick(beam, 10.0)
            pic.kick(beam, 10.0)
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("stay on the CPU" in m for m in msgs) == 1
    assert any("PIC backend is cpu" in m for m in msgs)
    assert "gpu_particle_kicks" not in pic.stats()


def test_field_reuse_says_it_keeps_the_cpu(monkeypatch, caplog):
    import logging

    from linac_gen.pic.field_reuse import FieldReusePicSolver
    monkeypatch.delenv(gpu_particles.ENV, raising=False)
    monkeypatch.setattr(gpu_particles, "_warned", set())
    with caplog.at_level(logging.WARNING, logger=gpu_particles.__name__):
        FieldReusePicSolver(SpaceChargeConfig(
            nx=16, ny=16, nz=16, grid_mode="adaptive", reuse_every=2,
            use_gpu="cpu", gpu_particles=True))
    assert any("reuse_every=2" in r.getMessage() for r in caplog.records)


def test_bunch_train_kicks_are_counted():
    """The companion solver's GPU kicks count too."""
    pic, _, _ = _ready_solver("cic")
    pic._gpu_particle_kicks = 3
    pic._train_solver = PicSolver(SpaceChargeConfig(use_gpu="cpu"))
    pic._train_solver._gpu_particle_kicks = 5
    assert pic.stats()["gpu_particle_kicks"] == 8


def test_cuda_chosen_for_the_torch_engine_without_cupy(monkeypatch):
    """A torch run's backward walk (and the GUI's continuous-beam fallback)
    hands the numpy engine the torch engine's use_gpu: without cupy,
    cuda/gpu become auto instead of failing (or, for gpu, float32 MPS)."""
    from linac_gen.pic import factory, gpu_backend
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    assert gpu_backend.numpy_engine_use_gpu("cuda") == "auto"
    assert gpu_backend.numpy_engine_use_gpu("gpu") == "auto"
    assert gpu_backend.numpy_engine_use_gpu("mps") == "mps"
    assert gpu_backend.numpy_engine_use_gpu("cpu") == "cpu"
    back = factory.make_sc_solver(
        SpaceChargeConfig(sc_backend="torch", grid_mode="adaptive",
                          use_gpu="cuda"), purpose="backward")
    assert (back.config.sc_backend, back.config.use_gpu) == ("numpy", "auto")
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: object())
    assert gpu_backend.numpy_engine_use_gpu("cuda") == "cuda"


def test_only_the_cuda_backend_with_a_cuda_device(monkeypatch):
    from linac_gen.pic.torch import devices

    class Backend:
        name = "gpu"

    class Solver:
        _backend = Backend()
    pic, _, _ = _ready_solver("cic")
    assert not gpu_particles.available(pic._solver)          # CPU backend
    monkeypatch.setattr(devices, "cuda_available", lambda: False)
    assert not gpu_particles.available(Solver())
    monkeypatch.setattr(devices, "cuda_available", lambda: True)
    assert gpu_particles.available(Solver())
    Backend.name = "mps"                                      # float32
    assert not gpu_particles.available(Solver())


@pytest.mark.parametrize("kind", ["cic", "tsc"])
def test_asking_without_a_gpu_changes_nothing(kind):
    """gpu_particles=True on the CPU backend: the CPU kernels, same bits."""
    outs = []
    for flag in (False, True):
        beam = _beam()
        pic = PicSolver(SpaceChargeConfig(nx=24, ny=24, nz=24,
                                          grid_extent=4.0, kernel=kind,
                                          use_gpu="cpu", gpu_particles=flag))
        pic.kick(beam, 10.0)
        outs.append(beam.particles.copy())
    assert np.array_equal(outs[0], outs[1])


def _cupy_cuda():
    try:
        import cupy
        return (cupy.cuda.runtime.getDeviceCount() > 0
                and torch.cuda.is_available())
    except Exception:                               # noqa: BLE001
        return False


@pytest.mark.skipif(not _cupy_cuda(), reason="needs cupy and CUDA")
@pytest.mark.parametrize("kind", ["cic", "tsc"])
def test_on_a_gpu_it_agrees_with_the_cpu_and_repeats(kind, monkeypatch):
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.delenv(gpu_particles.ENV, raising=False)
    outs, kicks = {}, {}
    for name, sc in (("cpu", dict(use_gpu="cpu")),
                     ("gpu", dict(use_gpu="cuda", gpu_particles=True)),
                     ("again", dict(use_gpu="cuda", gpu_particles=True))):
        beam = _beam(20_000)
        before = beam.particles.copy()
        pic = PicSolver(SpaceChargeConfig(nx=32, ny=32, nz=32,
                                          grid_extent=4.0, kernel=kind, **sc))
        pic.kick(beam, 10.0)
        outs[name] = (beam.particles - before)[:, [1, 3, 5]]  # the kick
        kicks[name] = pic.stats().get("gpu_particle_kicks", 0)
    assert kicks == {"cpu": 0, "gpu": 1, "again": 1}
    scale = np.max(np.abs(outs["cpu"]), axis=0)
    assert np.all(np.max(np.abs(outs["gpu"] - outs["cpu"]), axis=0)
                  <= 1e-10 * scale)
    assert np.array_equal(outs["gpu"], outs["again"])
