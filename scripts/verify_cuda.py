"""verify_cuda.py -- check HELIX's NVIDIA-GPU space-charge paths on a CUDA
machine against the CPU and against themselves, and time them.

    python scripts/verify_cuda.py            (from the HELIX directory)
    python scripts/verify_cuda.py --device cpu   (dry run of part 1 on the
                                                  CPU, for checking the script)

It checks, and prints PASS / FAIL / SKIP for each:

1. the torch engine on the GPU (sc_backend="torch", use_gpu="cuda"): it
   runs there; one kick and a whole FODO run agree with the CPU to
   rounding; two GPU runs give the same bits;
2. the default engine's deposit and gather on the GPU
   (SpaceChargeConfig.gpu_particles with the cupy backend), fixed and
   adaptive grids: the same.

LINAC_GEN_USE_GPU and the other device switches in the environment are
cleared first (they would override the settings the checks compare).

Paste the output back to whoever asked you to run it.  Nothing is written
to disk.  Needs torch with CUDA; part 2 also needs cupy.
"""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# agreement with the CPU: the GPU adds in another order and its FFT rounds
# differently, so results agree to rounding, not bit for bit
RTOL = 1e-10

results = []

#: The device part 1 tests ("cuda"; "cpu" only to dry-run the script).
DEVICE = "cuda"


def report(name, ok, detail):
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    results.append((tag, name))
    print(f"[{tag}] {name}: {detail}", flush=True)


def rel(a, b):
    """Largest difference relative to the size of its column (or row of
    moments), so a small column is not judged against a large one."""
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(
        np.asarray(b, float))
    axis = 0 if a.shape[0] > a.shape[1] else 1
    scale = np.maximum(np.max(np.abs(b), axis=axis, keepdims=True), 1e-300)
    return float(np.max(np.abs(a - b) / scale))


def same_bits(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint8),
                                                 b.view(np.uint8))


def timed(fn, repeat=3):
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


#: Settings from the environment that would decide the device behind the
#: script's back (LINAC_GEN_USE_GPU overrides use_gpu): cleared here.
ENV_VARS = ("LINAC_GEN_USE_GPU", "LINAC_GEN_GPU_PARTICLES",
            "LINAC_GEN_TORCH_SC_KERNELS")


def environment():
    import scipy
    for name in ENV_VARS:
        if name in os.environ:
            print(f"cleared {name}={os.environ.pop(name)!r} for this check")
    print("python", platform.python_version(), "|", platform.platform())
    print("numpy", np.__version__, "| scipy", scipy.__version__)
    try:
        import torch
        cuda = torch.cuda.is_available()
        print("torch", torch.__version__, "| CUDA available:", cuda,
              "| torch CUDA", torch.version.cuda)
        if cuda:
            p = torch.cuda.get_device_properties(0)
            print("GPU:", p.name, f"| {p.total_memory / 2**30:.1f} GiB",
                  f"| compute capability {p.major}.{p.minor}")
    except ImportError:
        print("torch: not installed")
    try:
        import cupy
        print("cupy", cupy.__version__, "| devices:",
              cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:                        # noqa: BLE001
        print("cupy: not usable --", type(exc).__name__, exc)
    import linac_gen
    from linac_gen.parallel import threads
    print("HELIX", getattr(linac_gen, "__version__", "?"), "| CPU threads:",
          threads.get_threads())
    print()


def fodo():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    lat.add(Drift("D_IN", length=50.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=+8.0))
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-8.0))
    lat.add(Drift("D2", length=200.0))
    return lat


def beam(n, seed=4):
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    b = Beam(ref=ref, n_particles=n, current=20.0)
    rng = np.random.default_rng(seed)
    b.particles[:] = rng.normal(size=(n, 6)) * [1.0, 0.3, 1.0, 0.3, 3.0,
                                                0.005]
    return b


def whole_run(n, **sc):
    """A FODO run; returns (beam sizes and emittances along the line, final
    particles, what ran: the torch engine's device or the default engine's
    FFT backend and its GPU deposit/gather kicks)."""
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    b = beam(n)
    sim = Simulation(fodo(), b, space_charge=SpaceChargeConfig(
        nx=32, ny=32, nz=32, grid_extent=4.0, **sc))
    rec = sim.run()
    moments = np.array([rec.sigma_x, rec.sigma_y, rec.emit_x, rec.emit_y],
                       float)
    pic = sim._pic_solver
    ran = {"device": str(getattr(pic, "device", "")),
           "backend": getattr(getattr(getattr(pic, "_solver", None),
                                      "_backend", None), "name", ""),
           "gpu_kicks": int((pic.stats() if hasattr(pic, "stats") else {})
                            .get("gpu_particle_kicks", 0))}
    return moments, b.particles.copy(), ran


def torch_engine():
    import torch
    if DEVICE == "cuda" and not torch.cuda.is_available():
        report("torch engine on the GPU", None, "torch sees no CUDA device")
        return
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.torch.sc_kick import (clear_solver_cache,
                                             torch_pic_sc_kick)
    rng = np.random.default_rng(9)
    p = rng.normal(size=(200_000, 6)) * [1.0, 1.0, 1.2, 1.0, 3.0, 0.01]
    kw = dict(ds_mm=10.0, gamma=1.0027, beta=0.0735, mass_mev=938.272,
              wavelength_mm=923.0, macro_charge=1e-15)
    for kind in ("cic", "tsc"):
        cfg = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=4.0,
                                grid_mode="adaptive", sc_backend="torch",
                                kernel=kind)

        def kick(device):
            """The kick (particles out minus in), computed on ``device``."""
            clear_solver_cache()
            P = torch.as_tensor(p, dtype=torch.float64, device=device)
            out = torch_pic_sc_kick(P, cfg, device=device, **kw)
            assert out.device.type == torch.device(device).type, out.device
            if device == "cuda":
                torch.cuda.synchronize()
            return (out - P).cpu().numpy()[:, [1, 3, 5]]
        t_cpu, cpu = timed(lambda: kick("cpu"))
        kick(DEVICE)                                # warm-up
        t_gpu, gpu = timed(lambda: kick(DEVICE))
        again = kick(DEVICE)
        r = rel(gpu, cpu)
        report(f"torch engine, one {kind.upper()} kick, GPU vs CPU",
               r < RTOL, f"max relative difference {r:.1e} "
               f"(CPU {t_cpu * 1e3:.1f} ms, GPU {t_gpu * 1e3:.1f} ms; "
               f"200k particles, 32^3)")
        report(f"torch engine, {kind.upper()} kick repeats on the GPU",
               same_bits(gpu, again), "two GPU kicks, same bits")
    t_cpu, (m_cpu, _, ran_cpu) = timed(lambda: whole_run(
        20_000, sc_backend="torch", grid_mode="adaptive", use_gpu="cpu"), 1)
    t_gpu, (m_gpu, p_gpu, ran_gpu) = timed(lambda: whole_run(
        20_000, sc_backend="torch", grid_mode="adaptive", use_gpu=DEVICE), 1)
    _, p_again, _ = whole_run(20_000, sc_backend="torch",
                              grid_mode="adaptive", use_gpu=DEVICE)
    want = "cuda" if DEVICE == "cuda" else "cpu"
    report("torch engine, whole run on the device asked for",
           ran_cpu["device"] == "cpu" and ran_gpu["device"] == want,
           f"use_gpu='cpu' ran on {ran_cpu['device']}, use_gpu={DEVICE!r} "
           f"on {ran_gpu['device']}")
    r = rel(m_gpu, m_cpu)
    report("torch engine, whole FODO run, GPU vs CPU", r < 1e-8,
           f"max relative difference of the beam sizes and emittances "
           f"{r:.1e} (CPU {t_cpu:.2f} s, GPU {t_gpu:.2f} s; 20k particles)")
    report("torch engine, whole run repeats on the GPU",
           same_bits(p_gpu, p_again), "final particles, same bits")


def default_engine():
    import torch
    try:
        import cupy  # noqa: F401
        has_cupy = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:                               # noqa: BLE001
        has_cupy = False
    if not (has_cupy and torch.cuda.is_available()):
        report("default engine, deposit and gather on the GPU", None,
               "needs cupy and torch with CUDA")
        return
    for kind in ("cic", "tsc"):
        for mode in ("fixed", "adaptive"):
            common = dict(kernel=kind, grid_mode=mode)
            name = f"default engine ({kind.upper()}, {mode} grid)"
            t_cpu, (m_cpu, _, ran_cpu) = timed(lambda: whole_run(
                20_000, use_gpu="cpu", **common), 1)
            t_fft, (m_fft, _, _) = timed(lambda: whole_run(
                20_000, use_gpu="cuda", **common), 1)
            t_gpu, (m_gpu, p_gpu, ran_gpu) = timed(lambda: whole_run(
                20_000, use_gpu="cuda", gpu_particles=True, **common), 1)
            _, p_again, _ = whole_run(20_000, use_gpu="cuda",
                                      gpu_particles=True, **common)
            report(f"{name}, the GPU path ran",
                   ran_cpu["backend"] == "cpu" and ran_cpu["gpu_kicks"] == 0
                   and ran_gpu["backend"] == "gpu"
                   and ran_gpu["gpu_kicks"] > 0,
                   f"CPU run: backend {ran_cpu['backend']}; GPU run: "
                   f"backend {ran_gpu['backend']}, "
                   f"{ran_gpu['gpu_kicks']} kicks with deposit+gather on "
                   f"the GPU")
            r = rel(m_gpu, m_cpu)
            report(f"{name}, deposit and gather on the GPU vs all on the "
                   f"CPU", r < 1e-8,
                   f"max relative difference {r:.1e} (CPU {t_cpu:.2f} s, "
                   f"GPU FFT only {t_fft:.2f} s, GPU deposit+FFT+gather "
                   f"{t_gpu:.2f} s; 20k particles)")
            report(f"{name}, GPU run repeats",
                   same_bits(p_gpu, p_again), "final particles, same bits")


def main():
    global DEVICE
    if "--device" in sys.argv:
        DEVICE = sys.argv[sys.argv.index("--device") + 1]
    environment()
    try:
        torch_engine()
    except Exception as exc:                        # noqa: BLE001
        report("torch engine on the GPU", False,
               f"raised {type(exc).__name__}: {exc}")
    try:
        default_engine()
    except Exception as exc:                        # noqa: BLE001
        report("default engine on the GPU", False,
               f"raised {type(exc).__name__}: {exc}")
    print()
    counts = {t: sum(1 for r, _ in results if r == t)
              for t in ("PASS", "FAIL", "SKIP")}
    print("summary:", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
