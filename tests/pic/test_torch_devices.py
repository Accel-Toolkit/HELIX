"""The torch space-charge engine's device (linac_gen.pic.torch.devices):
an NVIDIA GPU only when asked for and usable, never MPS (no float64),
"auto" on the CPU; a CUDA device torch cannot run on counts as none; the
kick leaves torch's global switches alone.  The GPU itself is exercised
only where there is one (the CUDA test at the end; scripts/verify_cuda.py
on a CUDA machine)."""
from __future__ import annotations

import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.config import SpaceChargeConfig  # noqa: E402
from linac_gen.pic.torch import devices  # noqa: E402

needs_cuda = pytest.mark.skipif(not devices.cuda_available(),
                                reason="no CUDA device")


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(devices, "_warned", set())
    return devices


@pytest.mark.parametrize("use_gpu, cuda, want", [
    ("auto", True, "cpu"), ("auto", False, "cpu"),
    ("cpu", True, "cpu"),
    ("cuda", True, "cuda"), ("gpu", True, "cuda"),
    ("cuda", False, "cpu"), ("gpu", False, "cpu"),
    ("mps", True, "cpu"), ("mps", False, "cpu"),
])
def test_the_device_for_each_setting(fresh, monkeypatch, use_gpu, cuda, want):
    monkeypatch.setattr(fresh, "cuda_available", lambda: cuda)
    assert fresh.torch_device(use_gpu).type == want


def test_a_gpu_asked_for_but_absent_is_said_once(fresh, monkeypatch, caplog):
    monkeypatch.setattr(fresh, "cuda_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=fresh.__name__):
        for _ in range(3):
            fresh.torch_device("cuda")
        fresh.torch_device("mps")
    messages = [r.getMessage() for r in caplog.records]
    assert sum("no usable CUDA device" in m for m in messages) == 1
    assert sum("float64" in m for m in messages) == 1


def test_the_environment_variable_decides(fresh, monkeypatch):
    monkeypatch.setattr(fresh, "cuda_available", lambda: True)
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "cuda")
    assert fresh.torch_device("auto").type == "cuda"
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "cpu")
    assert fresh.torch_device("cuda").type == "cpu"
    with pytest.raises(ValueError):
        fresh.torch_device("junk")


def _flags():
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


def test_a_cuda_device_torch_cannot_run_on_counts_as_none(monkeypatch,
                                                          caplog):
    """torch reports the device but its first kernel fails (a wheel without
    kernels for the card): not used, said once."""
    monkeypatch.setattr(devices, "_usable", None)
    monkeypatch.setattr(devices, "_warned", set())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_ones = torch.ones

    def ones(*a, **k):
        if str(k.get("device", "")).startswith("cuda"):
            raise RuntimeError("no kernel image is available")
        return real_ones(*a, **k)
    monkeypatch.setattr(torch, "ones", ones)
    with caplog.at_level(logging.WARNING, logger=devices.__name__):
        assert devices.cuda_available() is False
        assert devices.cuda_available() is False    # tried once
    assert sum("cannot run on it" in r.getMessage()
               for r in caplog.records) == 1
    monkeypatch.setattr(devices, "_usable", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert devices.cuda_available() is False


def test_a_kick_leaves_torch_s_switches_alone():
    """No process-wide switch is flipped around a kick (torch's CUDA
    deposit sums in sorted order without one)."""
    from linac_gen.pic.torch.sc_kick import torch_pic_sc_kick
    before = _flags()
    p = torch.as_tensor(np.random.default_rng(2).normal(size=(500, 6)))
    torch_pic_sc_kick(p, SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                                           grid_mode="adaptive",
                                           sc_backend="torch"),
                      ds_mm=10.0, gamma=1.0027, beta=0.0735,
                      mass_mev=938.272, wavelength_mm=923.0,
                      macro_charge=1e-15)
    assert _flags() == before


def test_the_solver_runs_on_the_cpu_unless_a_gpu_is_there(monkeypatch):
    """use_gpu="cuda" without a CUDA device: the CPU, and the same bits as
    an explicit CPU run."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.pic.torch.solver import TorchPicSolver
    monkeypatch.setattr(devices, "cuda_available", lambda: False)
    monkeypatch.setattr(devices, "_warned", set())
    outs = []
    for use_gpu in ("cpu", "cuda"):
        sc = TorchPicSolver(SpaceChargeConfig(
            nx=16, ny=16, nz=16, grid_extent=4.0, grid_mode="adaptive",
            sc_backend="torch", use_gpu=use_gpu))
        assert sc.device.type == "cpu"
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=2000, current=20.0)
        beam.particles[:] = (np.random.default_rng(1).normal(size=(2000, 6))
                             * [1.0, 0.3, 1.0, 0.3, 3.0, 0.005])
        sc.kick(beam, 10.0)
        outs.append(beam.particles.copy())
    assert np.array_equal(outs[0], outs[1])


@needs_cuda
@pytest.mark.parametrize("kind", ["cic", "tsc"])
def test_a_gpu_kick_agrees_with_the_cpu_and_repeats(kind, monkeypatch):
    from linac_gen.pic.torch.sc_kick import (clear_solver_cache,
                                             torch_pic_sc_kick)
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    p = (np.random.default_rng(9).normal(size=(50_000, 6))
         * [1.0, 1.0, 1.2, 1.0, 3.0, 0.01])
    cfg = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=4.0,
                            grid_mode="adaptive", sc_backend="torch",
                            kernel=kind)
    kw = dict(ds_mm=10.0, gamma=1.0027, beta=0.0735, mass_mev=938.272,
              wavelength_mm=923.0, macro_charge=1e-15)

    def kick(device):
        clear_solver_cache()
        P = torch.as_tensor(p, dtype=torch.float64, device=device)
        out = torch_pic_sc_kick(P, cfg, device=device, **kw)
        assert out.device.type == device
        return (out - P).cpu().numpy()[:, [1, 3, 5]]  # the kick itself
    cpu, gpu, again = kick("cpu"), kick("cuda"), kick("cuda")
    scale = np.max(np.abs(cpu), axis=0)
    assert np.all(np.max(np.abs(gpu - cpu), axis=0) <= 1e-10 * scale)
    assert np.array_equal(gpu, again)
