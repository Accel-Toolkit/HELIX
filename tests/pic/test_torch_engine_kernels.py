"""The torch engine's deposit, gather, gather sums and Green's function on
HELIX's threads (linac_gen.pic.torch.cpu_kernels) give torch's own numbers
bit for bit, at any thread count, whenever no gradient is needed; with a
gradient, with inputs the kernels do not take, outside the kernels'
envelope, or with the kill switch, torch's code runs."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.config import SpaceChargeConfig  # noqa: E402
from linac_gen.pic.torch import cpu_kernels  # noqa: E402
from linac_gen.pic.torch.deposition import (  # noqa: E402
    deposit_cic_torch, deposit_tsc_torch,
)
from linac_gen.pic.torch.interpolation import (  # noqa: E402
    interpolate_cic_torch, interpolate_tsc_torch,
)
from linac_gen.pic.torch.sc_kick import (  # noqa: E402
    clear_solver_cache, torch_pic_sc_kick,
)
from tests.helpers import kept_thread_setting  # noqa: E402

needs_kernels = pytest.mark.skipif(
    cpu_kernels._deposit is None,
    reason="linac_gen._tsc_kernels without the torch-engine kernels")

DEPOSIT = {"cic": deposit_cic_torch, "tsc": deposit_tsc_torch}
GATHER = {"cic": interpolate_cic_torch, "tsc": interpolate_tsc_torch}
GRIDS = [(16, 16, 16), (32, 24, 20), (3, 4, 5), (5, 5, 5)]
COUNTS = (1, 2, 3, 7, 14)


def _bits(t):
    return np.ascontiguousarray(t.detach().numpy()).view(np.uint64)


def _beam(n, seed=11):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)) * [1.0, 1.3, 0.7]
    pos[: max(1, n // 50)] *= 6.0                   # some beyond the grid
    q = rng.uniform(0.5, 1.5, size=n) * 1e-15
    return (torch.as_tensor(pos), torch.as_tensor(q),
            torch.tensor([-3.0, -3.5, -2.0], dtype=torch.float64),
            torch.tensor([3.1, 3.2, 2.4], dtype=torch.float64))


def _torch_code(monkeypatch, fn, *args):
    monkeypatch.setenv(cpu_kernels.ENV, "0")
    try:
        return fn(*args)
    finally:
        monkeypatch.delenv(cpu_kernels.ENV)


def _count_calls(monkeypatch, name):
    calls = []
    real = getattr(cpu_kernels, name)

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(cpu_kernels, name, spy)
    return calls


@needs_kernels
@pytest.mark.parametrize("kind", ["cic", "tsc"])
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("n", [1, 7, 5000])
def test_deposit_is_torch_s_at_any_thread_count(kind, grid, n, monkeypatch):
    pos, q, gmin, gmax = _beam(n)
    ref = _torch_code(monkeypatch, DEPOSIT[kind], pos, q, gmin, gmax, grid)
    calls = _count_calls(monkeypatch, "_deposit")
    with kept_thread_setting() as threads:
        for count in COUNTS:
            threads.set_threads(count)
            out = DEPOSIT[kind](pos, q, gmin, gmax, grid)
            assert np.array_equal(_bits(out), _bits(ref)), count
    assert len(calls) == len(COUNTS)                # the kernel ran


@needs_kernels
@pytest.mark.parametrize("kind", ["cic", "tsc"])
@pytest.mark.parametrize("grid", GRIDS)
def test_gather_is_torch_s_at_any_thread_count(kind, grid, monkeypatch):
    pos, _, gmin, gmax = _beam(3000)
    rng = np.random.default_rng(3)
    fields = [torch.as_tensor(rng.normal(size=grid)) for _ in range(3)]
    ref = _torch_code(monkeypatch, GATHER[kind], *fields, pos, gmin, gmax,
                      grid)
    calls = _count_calls(monkeypatch, "_gather_products")
    with kept_thread_setting() as threads:
        for count in COUNTS:
            threads.set_threads(count)
            out = GATHER[kind](*fields, pos, gmin, gmax, grid)
            assert np.array_equal(_bits(out), _bits(ref)), count
    assert len(calls) == len(COUNTS)


@needs_kernels
@pytest.mark.parametrize("kind", ["cic", "tsc"])
def test_outside_the_envelope_runs_torch_s_code(kind, monkeypatch):
    """A non-finite coordinate: the kernels decline (torch's float-to-int
    cast differs by platform there) and torch's code gives the answer."""
    pos, q, gmin, gmax = _beam(500)
    pos[17, 1] = float("nan")
    ref = _torch_code(monkeypatch, DEPOSIT[kind], pos, q, gmin, gmax,
                      (8, 8, 8))
    out = DEPOSIT[kind](pos, q, gmin, gmax, (8, 8, 8))
    assert np.array_equal(_bits(out), _bits(ref))
    fields = [torch.ones((8, 8, 8), dtype=torch.float64)] * 3
    ref = _torch_code(monkeypatch, GATHER[kind], *fields, pos, gmin, gmax,
                      (8, 8, 8))
    out = GATHER[kind](*fields, pos, gmin, gmax, (8, 8, 8))
    assert np.array_equal(_bits(out), _bits(ref))


@needs_kernels
def test_a_gradient_runs_torch_s_code(monkeypatch):
    pos, q, gmin, gmax = _beam(400)
    pos.requires_grad_(True)
    dep = _count_calls(monkeypatch, "_deposit")
    gat = _count_calls(monkeypatch, "_gather_products")
    rho = deposit_cic_torch(pos, q, gmin, gmax, (8, 8, 8))
    e = interpolate_cic_torch(rho, rho, rho, pos, gmin, gmax, (8, 8, 8))
    e.sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    assert not dep and not gat
    with torch.no_grad():                           # no gradient wanted
        deposit_cic_torch(pos, q, gmin, gmax, (8, 8, 8))
    assert len(dep) == 1


def test_a_build_without_the_kernels_runs_torch_s_code(monkeypatch):
    pos, q, gmin, gmax = _beam(300)
    ref = _torch_code(monkeypatch, deposit_tsc_torch, pos, q, gmin, gmax,
                      (8, 8, 8))
    monkeypatch.setattr(cpu_kernels, "_deposit", None)
    out = deposit_tsc_torch(pos, q, gmin, gmax, (8, 8, 8))
    assert np.array_equal(_bits(out), _bits(ref))


def _kick(kind, n):
    rng = np.random.default_rng(9)
    p = torch.as_tensor(rng.normal(size=(n, 6))
                        * [1.0, 1.0, 1.2, 1.0, 3.0, 0.01])
    cfg = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=4.0,
                            grid_mode="adaptive", sc_backend="torch",
                            kernel=kind)
    clear_solver_cache()
    return torch_pic_sc_kick(p, cfg, ds_mm=10.0, gamma=1.0027, beta=0.0735,
                             mass_mev=938.272, wavelength_mm=923.0,
                             macro_charge=1e-15)


@needs_kernels
@pytest.mark.parametrize("kind", ["cic", "tsc"])
def test_a_kick_is_torch_s_at_any_thread_count(kind, monkeypatch):
    """The whole kick (deposit, Green's function in slabs, torch's FFTs,
    gather, sums in blocks) equals torch's own code, bit for bit, at any
    thread count."""
    ref = _torch_code(monkeypatch, _kick, kind, 6000)
    with kept_thread_setting() as threads:
        for count in (1, 4, "all"):
            threads.set_threads(count)
            assert np.array_equal(_bits(_kick(kind, 6000)), _bits(ref)), count


@needs_kernels
@pytest.mark.parametrize("n", [(16, 16, 16), (32, 24, 20), (33, 17, 16),
                               (40, 40, 40)])
def test_green_s_function_is_torch_s_at_any_thread_count(n, monkeypatch):
    """The antiderivative in fixed slabs of 16 vertex planes (each starting
    on a multiple of 16 points; the last one short) and the cell differences
    in slabs: torch's single call, bit for bit, for any thread count."""
    from linac_gen.pic.torch.greens import build_integrated_greens_torch
    dx = torch.tensor([0.031, 0.027, 0.11], dtype=torch.float64)
    ref = _torch_code(monkeypatch, build_integrated_greens_torch, n, dx)
    calls = _count_calls(monkeypatch, "parallel")
    with kept_thread_setting() as threads:
        for count in (1, 2, 5, 14):
            threads.set_threads(count)
            G = build_integrated_greens_torch(n, dx)
            assert np.array_equal(_bits(G), _bits(ref)), count
    assert len(calls) == 2 * 4                      # the slab path ran


@needs_kernels
def test_gather_sums_in_blocks_are_one_call_s(monkeypatch):
    """The products summed in blocks of rows equal one torch.sum call."""
    monkeypatch.setattr(cpu_kernels, "SUM_ROWS", 1000)   # several blocks
    pos, _, gmin, gmax = _beam(4321)
    rng = np.random.default_rng(8)
    fields = [torch.as_tensor(rng.normal(size=(12, 12, 12)))
              for _ in range(3)]
    for kind in ("cic", "tsc"):
        ref = _torch_code(monkeypatch, GATHER[kind], *fields, pos, gmin,
                          gmax, (12, 12, 12))
        with kept_thread_setting() as threads:
            threads.set_threads(3)
            out = GATHER[kind](*fields, pos, gmin, gmax, (12, 12, 12))
        assert out.is_contiguous() and out.shape == (4321, 3)
        assert np.array_equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("helix_threads", [1, 4])
def test_every_piece_runs_on_one_torch_thread(helix_threads):
    """Every piece -- also when the caller's torch count is higher (outside
    a run) -- runs on one torch thread, each piece exactly once, and torch's
    default for new threads is left as it was."""
    import threading
    seen = {}
    lock = threading.Lock()
    with kept_thread_setting() as threads:
        threads.set_threads(helix_threads)
        torch.set_num_threads(3)                    # the caller's count

        def piece(a, b):
            with lock:
                seen[(a, b)] = seen.get((a, b), []) + [torch.get_num_threads()]
        cpu_kernels.parallel(piece, 40, 5)
        assert torch.get_num_threads() == 3
        new = []
        t = threading.Thread(target=lambda: new.append(torch.get_num_threads()))
        t.start()
        t.join()
    assert sorted(seen) == [(a, a + 5) for a in range(0, 40, 5)]
    assert all(v == [1] for v in seen.values()), seen
    assert new == [3]


def test_no_pieces_is_nothing_to_do():
    cpu_kernels.parallel(lambda a, b: 1 / 0, 0, 16)


@needs_kernels
def test_inputs_the_kernels_do_not_take_run_torch_s_code(monkeypatch):
    """Scalar grid bounds, broadcast charges and a TSC axis of two nodes
    (SpaceChargeConfig allows it) are torch's code's business: the same
    answer as with the kill switch, no error."""
    pos, q, gmin, gmax = _beam(300)
    cases = [
        ("cic", (pos, q, torch.tensor(-3.0, dtype=torch.float64),
                 torch.tensor(3.0, dtype=torch.float64), (8, 8, 8))),
        ("cic", (pos, q[:1].clone(), gmin, gmax, (8, 8, 8))),
        ("tsc", (pos, q, gmin, gmax, (8, 2, 8))),
    ]
    dep = _count_calls(monkeypatch, "_deposit")
    for kind, args in cases:
        ref = _torch_code(monkeypatch, DEPOSIT[kind], *args)
        out = DEPOSIT[kind](*args)
        assert np.array_equal(_bits(out), _bits(ref))
    assert not dep


@needs_kernels
def test_another_default_device_is_left_alone(monkeypatch):
    """The CPU path makes its own arrays on the CPU even when torch's
    default device is another one."""
    from linac_gen.pic.torch.greens import build_integrated_greens_torch
    dx = torch.tensor([0.031, 0.027, 0.11], dtype=torch.float64)
    pos, _, gmin, gmax = _beam(3000)
    fields = [torch.as_tensor(np.random.default_rng(2).normal(size=(16,) * 3))
              for _ in range(3)]
    ref_g = build_integrated_greens_torch((16, 16, 16), dx, device="cpu")
    ref_e = interpolate_cic_torch(*fields, pos, gmin, gmax, (16, 16, 16),
                                  device="cpu")
    torch.set_default_device("meta")
    try:
        g = build_integrated_greens_torch((16, 16, 16), dx, device="cpu")
        e = interpolate_cic_torch(*fields, pos, gmin, gmax, (16, 16, 16),
                                  device="cpu")
    finally:
        torch.set_default_device("cpu")
    assert g.device.type == e.device.type == "cpu"
    assert np.array_equal(_bits(g), _bits(ref_g))
    assert np.array_equal(_bits(e), _bits(ref_e))


# ---------------------------------------------------------------------------
# The FFTs: split on HELIX's threads only where torch's FFT is pocketfft
# ---------------------------------------------------------------------------

POCKETFFT = cpu_kernels._pocketfft()
FFT_SHAPES = [(64, 64, 64), (96, 96, 96), (66, 70, 74), (90, 94, 98),
              (128, 64, 96), (26, 26, 202)]


def _fft_inputs(shape, seed=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float64, generator=g)
    return x, torch.fft.rfftn(x) * (1.0 + 0.5j)


@pytest.fixture
def fft_threads_from_setting(monkeypatch):
    """The FFT follows the thread setting, not LINAC_GEN_FFT_WORKERS."""
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)


@pytest.mark.usefixtures("fft_threads_from_setting")
@pytest.mark.parametrize("shape", FFT_SHAPES)
def test_the_ffts_are_torch_s_at_any_thread_count(shape):
    """With pocketfft (macOS) the split transforms are torch's own, bit for
    bit, at any thread count; with Intel MKL (Linux, Windows) they are not
    used at all."""
    x, X = _fft_inputs(shape)
    ref_f, ref_i = torch.fft.rfftn(x), torch.fft.irfftn(X, s=shape)
    with kept_thread_setting() as threads:
        for count in (2, 3, 7, 14):
            threads.set_threads(count)
            f, i = cpu_kernels.rfftn(x), cpu_kernels.irfftn(X, shape)
            if not POCKETFFT:
                assert f is None and i is None
                continue
            assert f is not None and i is not None
            assert np.array_equal(_bits(torch.view_as_real(f)),
                                  _bits(torch.view_as_real(ref_f))), count
            assert np.array_equal(_bits(i), _bits(ref_i)), count
    assert np.array_equal(_bits(torch.view_as_real(X)),
                          _bits(torch.view_as_real(ref_f * (1.0 + 0.5j))))


def test_the_split_passes_are_the_transform_on_every_build():
    """The three passes are the 3-D transform whatever the FFT library
    (to rounding: MKL is not pocketfft)."""
    shape = (40, 36, 52)
    x, X = _fft_inputs(shape)
    with kept_thread_setting() as threads:
        threads.set_threads(3)
        f = cpu_kernels._rfftn_split(x)
        i = cpu_kernels._irfftn_split(X, shape)
    assert torch.allclose(f, torch.fft.rfftn(x), rtol=1e-12, atol=1e-12)
    assert torch.allclose(i, torch.fft.irfftn(X, s=shape), rtol=1e-12,
                          atol=1e-12)


@pytest.mark.usefixtures("fft_threads_from_setting")
def test_a_split_that_is_not_torch_s_is_not_used(monkeypatch, caplog):
    """The split is used for a shape only after it reproduced torch's own
    transform: one ulp off and torch's call runs, checked once a shape."""
    monkeypatch.setattr(cpu_kernels, "_pocketfft", lambda: True)
    monkeypatch.setattr(cpu_kernels, "_fft_checked", {})
    real = cpu_kernels._rfftn_split
    calls = []

    def off_by_one_ulp(x):
        calls.append(1)
        out = real(x)
        r = torch.view_as_real(out)
        r[3, 2, 1, 0] = torch.nextafter(r[3, 2, 1, 0],
                                        torch.tensor(np.inf, dtype=r.dtype))
        return out
    monkeypatch.setattr(cpu_kernels, "_rfftn_split", off_by_one_ulp)
    x, _ = _fft_inputs((64, 64, 64))
    with kept_thread_setting() as threads, \
            caplog.at_level("INFO", logger=cpu_kernels.__name__):
        threads.set_threads(4)
        assert cpu_kernels.rfftn(x) is None
        assert cpu_kernels.rfftn(x) is None
    assert len(calls) == 1                          # one check per shape
    assert cpu_kernels._fft_checked == {("r2c", (64, 64, 64)): False}
    assert "split differs" in caplog.text


@pytest.mark.usefixtures("fft_threads_from_setting")
def test_ffts_the_split_does_not_take_are_torch_s(monkeypatch):
    """Small transforms, one thread, a gradient, a lazily conjugated or
    single-precision input, a 2-D input, a mismatched size, the kill
    switch: torch's call runs."""
    monkeypatch.setattr(cpu_kernels, "_pocketfft", lambda: True)
    x, X = _fft_inputs((64, 64, 64))
    s = (64, 64, 64)
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        assert cpu_kernels.rfftn(x[:32, :32, :32].contiguous()) is None
        assert cpu_kernels.rfftn(x.float()) is None
        assert cpu_kernels.rfftn(x.reshape(64, 4096)) is None
        assert cpu_kernels.rfftn(x.clone().requires_grad_()) is None
        assert cpu_kernels.irfftn(X.conj(), s) is None
        assert cpu_kernels.irfftn(X, (64, 64, 66)) is None
        assert cpu_kernels.irfftn(X.to(torch.complex64), s) is None
        monkeypatch.setenv(cpu_kernels.ENV, "0")
        assert cpu_kernels.rfftn(x) is None
        assert cpu_kernels.irfftn(X, s) is None
        monkeypatch.delenv(cpu_kernels.ENV)
        threads.set_threads(1)
        assert cpu_kernels.rfftn(x) is None
        assert cpu_kernels.irfftn(X, s) is None


@pytest.mark.usefixtures("fft_threads_from_setting")
def test_without_pocketfft_the_ffts_are_never_split(monkeypatch):
    monkeypatch.setattr(cpu_kernels, "_pocketfft", lambda: False)
    split = _count_calls(monkeypatch, "_rfftn_split")
    x, X = _fft_inputs((64, 64, 64))
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        assert cpu_kernels.rfftn(x) is None
        assert cpu_kernels.irfftn(X, (64, 64, 64)) is None
    assert not split


@pytest.mark.usefixtures("fft_threads_from_setting")
@pytest.mark.skipif(not POCKETFFT, reason="torch's FFT is Intel MKL's")
def test_the_solver_uses_the_split(monkeypatch):
    """The torch engine's Poisson solve and Green's function transform go
    through the split -- every transform, once the shape is checked -- and
    give torch's numbers."""
    from linac_gen.pic.torch.poisson import TorchPoissonSolverFFT
    lo = torch.tensor([-1.0, -1.2, -0.8], dtype=torch.float64)
    hi = torch.tensor([1.1, 1.3, 0.9], dtype=torch.float64)
    rho = torch.as_tensor(np.random.default_rng(3).normal(size=(32, 32, 32)))
    monkeypatch.setenv(cpu_kernels.ENV, "0")
    ref_solver = TorchPoissonSolverFFT(lo, hi, (32, 32, 32))
    ref = ref_solver.solve(rho)
    monkeypatch.delenv(cpu_kernels.ENV)
    monkeypatch.setattr(cpu_kernels, "_fft_checked", {})
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        TorchPoissonSolverFFT(lo, hi, (32, 32, 32)).solve(rho)  # the checks
        assert cpu_kernels._fft_checked == {("r2c", (64, 64, 64)): True,
                                            ("c2r", (64, 64, 64)): True}
        f = _count_calls(monkeypatch, "_rfftn_split")
        i = _count_calls(monkeypatch, "_irfftn_split")
        solver = TorchPoissonSolverFFT(lo, hi, (32, 32, 32))
        assert (len(f), len(i)) == (1, 0)           # the Green's function
        out = solver.solve(rho)
        assert (len(f), len(i)) == (2, 1)           # the solve
    assert np.array_equal(_bits(torch.view_as_real(solver._G_fft)),
                          _bits(torch.view_as_real(ref_solver._G_fft)))
    for a, b in zip(out, ref):
        assert np.array_equal(_bits(a), _bits(b))


def test_the_pieces_start_on_multiples_of_8_lines(monkeypatch):
    """Every pass cuts its planes into pieces holding a multiple of 8 lines
    (pocketfft batches 2, 4 or 8 lines per vector step), counted with the
    lines each plane really holds."""
    for lines in range(1, 70):
        for points in (1, 50, 4704, 9216, 1 << 16, 1 << 20):
            step = cpu_kernels._fft_step(lines, points)
            assert step >= 1 and (step * lines) % 8 == 0, (lines, points)
    seen = []
    real_step, real_parallel = cpu_kernels._fft_step, cpu_kernels.parallel

    def step_spy(lines, points):
        step = real_step(lines, points)
        seen.append(("step", lines, points, step))
        return step

    def parallel_spy(fn, n, step, threads=None):
        seen.append(("parallel", n, step))
        return real_parallel(fn, n, step, threads)
    monkeypatch.setattr(cpu_kernels, "_fft_step", step_spy)
    monkeypatch.setattr(cpu_kernels, "parallel", parallel_spy)
    nx, ny, nz = 18, 22, 26
    m = nz // 2 + 1
    x, X = _fft_inputs((nx, ny, nz))
    cpu_kernels._rfftn_split(x)
    cpu_kernels._irfftn_split(X, (nx, ny, nz))
    steps = [e for e in seen if e[0] == "step"]
    calls = [e for e in seen if e[0] == "parallel"]
    # (lines per plane, points per plane) and planes cut, pass by pass
    assert [e[1:3] for e in steps] == [(ny, ny * nz), (m, nx * m),
                                       (m, ny * m), (m, nx * m),
                                       (m, ny * m), (ny, ny * m)]
    assert [c[1] for c in calls] == [nx, ny, nx, ny, nx, nx]
    assert [c[2] for c in calls] == [e[3] for e in steps]
