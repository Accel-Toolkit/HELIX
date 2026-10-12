"""HELIX's threaded CPU code standing in for torch's in the torch engine.

When no gradient is needed (a normal run: the multi-particle tracker's
``TorchPicSolver``), the torch engine's costly steps run on HELIX's threads
instead of torch's -- which stays on one thread during runs
(linac_gen.parallel.threads) -- with torch's numbers, bit for bit:

* the charge deposit and the field gather: ``linac_gen._tsc_kernels``
  (``deposit_torch``, ``gather_products_torch``) compute them in torch's
  arithmetic, each cell's charge summed in particle order as torch's
  float64 ``index_put_(accumulate=True)`` does; the gather hands back the
  per-corner products and torch sums them, in its own order;
* the sums of those products and the Green's function's antiderivative and
  cell differences stay torch's code, run in fixed pieces on HELIX's threads
  (:func:`parallel`, torch on one thread in every piece): blocks of rows,
  slabs of 16 grid planes.  The pieces do not depend on the thread count,
  and each starts on a multiple of 16 elements -- the most torch's
  element-wise loops take in one vector step (2 x 8 doubles with AVX-512)
  -- so torch processes every element as in a single call.

* the 3-D FFTs (:func:`rfftn`, :func:`irfftn`), where torch's FFT is
  pocketfft (builds without Intel MKL: macOS, ARM machines): three passes
  of 1-D torch transforms on pieces of whole planes, each starting on a
  multiple of 8 lines, so pocketfft treats every line as in the one-call
  transform -- the method of linac_gen.pic.fft_split, with torch's own
  calls.  A shape is split only after the split of a fixed random array has
  reproduced torch's one-call transform bit for bit in this process.  With
  Intel MKL (x86 Linux and Windows) the FFTs stay torch's, on one thread:
  MKL's results can depend on memory alignment, so no such check could hold
  for every array.

The results do not depend on the thread count.  With a gradient (the
differentiable step tracker, the gradient matcher), on another device or
dtype, or with inputs the kernels do not take (scalar grid bounds,
broadcast charges, axes too short for the stencil), torch's code runs.
``LINAC_GEN_TORCH_SC_KERNELS=0`` runs torch's code always.  Pinned by
tests/pic/test_torch_engine_kernels.py.
"""
from __future__ import annotations

import contextlib
import logging
import os
from math import gcd

import numpy as np
import torch

from linac_gen.parallel import threads as _threads
from linac_gen.parallel.threads import sync_torch as _sync_torch

# a run that loads torch here keeps its CPU work on one thread
_sync_torch()

try:
    from linac_gen._tsc_kernels import (
        deposit_torch as _deposit,
        gather_products_torch as _gather_products,
    )
except ImportError:             # a build before these kernels, or none
    _deposit = _gather_products = None
else:
    _threads.sync_kernels()     # the module's loops take the thread setting

#: Kill switch: "0" runs torch's own code for every step.
ENV = "LINAC_GEN_TORCH_SC_KERNELS"

_log = logging.getLogger(__name__)

_ORDER = {"cic": 1, "tsc": 2}


def enabled() -> bool:
    return os.environ.get(ENV, "1") != "0"


def plain(*tensors, dtype=torch.float64) -> bool:
    """True when HELIX's code may stand in for torch's: CPU tensors of
    ``dtype`` that need no gradient, under no torch.func transform, not
    traced or compiled."""
    for t in tensors:
        if t.device.type != "cpu" or t.dtype != dtype:
            return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return False
    if torch.jit.is_tracing():
        return False
    compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if compiling is not None and compiling():
        return False
    from torch.autograd import forward_ad
    if any(forward_ad.unpack_dual(t).tangent is not None for t in tensors):
        return False
    wrapped = getattr(getattr(torch._C, "_functorch", None),
                      "is_functorch_wrapped_tensor", None)
    if wrapped is None:                 # cannot tell: leave it to torch
        return False
    return not any(wrapped(t) for t in tensors)


#: Rows per block when the gather's products are summed in pieces.
SUM_ROWS = 8192


def parallel(fn, n, step, threads=None):
    """``fn(a, b)`` for the fixed pieces ``[0, step), [step, 2 step), ...``
    of ``range(n)``, on at most ``threads`` HELIX threads (default: the
    setting), each taking the next piece as it finishes one, with torch on
    one thread in every piece and in the caller's gradient and inference
    modes (torch keeps all three per thread); returns when all have
    finished.  The pieces do not depend on the thread count, and every piece
    runs on one torch thread, so neither does the result."""
    pieces = [(a, min(n, a + step)) for a in range(0, n, step)]
    if not pieces:
        return
    if threads is None:
        threads = _threads.get_threads()
    from linac_gen.pic import fft_split
    modes = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    fft_split.run_pieces(
        fn, pieces, threads, enter=lambda: _one_torch_thread(modes),
        # the caller takes pieces when its torch is on one thread
        caller=torch.get_num_threads() == 1)


@contextlib.contextmanager
def _one_torch_thread(modes):
    # torch keeps a thread count per thread: a pool thread that first used
    # torch outside a run would split every piece again
    _threads.hold_torch_at_one_here()
    grad, inference = modes
    with torch.inference_mode(inference), torch.set_grad_enabled(grad):
        yield


def _grid_args(grid_min, grid_max, n_grid):
    return (grid_min.numpy(), grid_max.numpy(),
            np.asarray([int(n) for n in n_grid], dtype=np.int64))


def _takes(kind, positions, grid_min, grid_max, n_grid, charges=None,
           fields=()):
    """True when the kernels take these shapes, which torch's code may also
    broadcast: (N, 3) positions, (3,) bounds, (N,) charges, fields of shape
    n_grid, at least 2 (CIC) or 3 (TSC) nodes per axis."""
    if positions.dim() != 2 or positions.shape[1] != 3:
        return False
    if tuple(grid_min.shape) != (3,) or tuple(grid_max.shape) != (3,):
        return False
    if charges is not None and tuple(charges.shape) != (positions.shape[0],):
        return False
    try:
        ng = tuple(int(v) for v in n_grid)
    except (TypeError, ValueError):
        return False
    if len(ng) != 3 or min(ng) < _ORDER[kind] + 1:
        return False
    return all(tuple(f.shape) == ng for f in fields)


def deposit(kind, positions, charges, grid_min, grid_max, n_grid):
    """Each cell's summed charge, (nx, ny, nz) -- not yet divided by the
    cell volume -- or None when the torch code must run (no kernel, a
    gradient, a particle outside the kernel's envelope)."""
    if (_deposit is None or not enabled()
            or not plain(positions, charges, grid_min, grid_max)
            or not _takes(kind, positions, grid_min, grid_max, n_grid,
                          charges=charges)):
        return None
    out = _deposit(positions.numpy(), charges.numpy(),
                   *_grid_args(grid_min, grid_max, n_grid), _ORDER[kind])
    return None if out is None else torch.from_numpy(out)


def gather(kind, field_x, field_y, field_z, positions, grid_min, grid_max,
           n_grid):
    """``(N, 3)`` fields at the particles, summed by torch from the
    kernel's per-corner products, or None when the torch code must run."""
    if (_gather_products is None or not enabled()
            or not plain(field_x, field_y, field_z, positions, grid_min,
                         grid_max)
            or not _takes(kind, positions, grid_min, grid_max, n_grid,
                          fields=(field_x, field_y, field_z))):
        return None
    prod = _gather_products(field_x.numpy(), field_y.numpy(),
                            field_z.numpy(), positions.numpy(),
                            *_grid_args(grid_min, grid_max, n_grid),
                            _ORDER[kind])
    if prod is None:
        return None
    p = torch.from_numpy(prod)                       # (3, N, corners)
    n = p.shape[1]
    out = torch.empty((3, n), dtype=p.dtype, device=p.device)

    def rows(a, b):
        for f in range(3):
            torch.sum(p[f, a:b], dim=1, out=out[f, a:b])
    parallel(rows, n, SUM_ROWS)
    return out.T.contiguous()                        # (N, 3), as torch.stack


# ---------------------------------------------------------------------------
# The FFTs
# ---------------------------------------------------------------------------

#: Transforms with fewer points run as one torch call (as in fft_split).
FFT_MIN_POINTS = 1 << 17

#: Points per piece of a pass (whole planes; boundaries on multiples of
#: 8 lines, a multiple of every batch width pocketfft uses for doubles).
FFT_PIECE_POINTS = 1 << 16
_ALIGN = 8

#: (kind, shape) -> whether the split reproduced torch's transform.
_fft_checked: dict = {}


def _pocketfft() -> bool:
    """True when torch's CPU FFT is pocketfft: torch uses it exactly when it
    is built without Intel MKL (cmake/Dependencies.cmake)."""
    mkl = getattr(getattr(torch.backends, "mkl", None), "is_available", None)
    return mkl is not None and not mkl()


def _fft_step(lines: int, plane_points: int) -> int:
    """Planes per piece: whole aligned units (planes holding a multiple of
    8 lines between them), about FFT_PIECE_POINTS points."""
    unit = _ALIGN // gcd(int(lines), _ALIGN)
    return unit * max(1, -(-FFT_PIECE_POINTS // (unit * max(1, plane_points))))


def _rfftn_split(x):
    nx, ny, nz = x.shape
    nt = _threads.fft_threads()
    m = nz // 2 + 1
    out = torch.empty((nx, ny, m), dtype=torch.complex128, device=x.device)

    def last(a, b):                     # r2c along axis 2
        out[a:b].copy_(torch.fft.rfft(x[a:b], dim=2))

    def first(a, b):                    # c2c along axis 0
        v = out[:, a:b]
        v.copy_(torch.fft.fft(v, dim=0))

    def middle(a, b):                   # c2c along axis 1
        v = out[a:b]
        v.copy_(torch.fft.fft(v, dim=1))

    parallel(last, nx, _fft_step(ny, ny * nz), nt)  # plane i: lines (i, j)
    parallel(first, ny, _fft_step(m, nx * m), nt)   # plane j: lines (j, k)
    parallel(middle, nx, _fft_step(m, ny * m), nt)  # plane i: lines (i, k)
    return out


def _irfftn_split(X, s):
    nx, ny, nz = s
    nt = _threads.fft_threads()
    m = nz // 2 + 1
    tmp = torch.empty((nx, ny, m), dtype=torch.complex128, device=X.device)
    tmp.copy_(X)                        # X stays as it was
    out = torch.empty(s, dtype=torch.float64, device=X.device)
    fct = 1.0 / float(nx * ny * nz)     # torch's factor: 1 / (double) N

    def first(a, b):                    # c2c along axis 0, unscaled
        v = tmp[:, a:b]
        v.copy_(torch.fft.ifft(v, dim=0, norm="forward"))

    def middle(a, b):                   # c2c along axis 1, unscaled
        v = tmp[a:b]
        v.copy_(torch.fft.ifft(v, dim=1, norm="forward"))

    def last(a, b):                     # c2r along axis 2, then the 1/N
        r = torch.fft.irfft(tmp[a:b], n=nz, dim=2, norm="forward")
        torch.mul(r, fct, out=out[a:b])

    parallel(first, ny, _fft_step(m, nx * m), nt)
    parallel(middle, nx, _fft_step(m, ny * m), nt)
    parallel(last, nx, _fft_step(ny, ny * m), nt)
    return out


def _same_bits(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    return torch.equal(a.contiguous().view(torch.int64),
                       b.contiguous().view(torch.int64))


def _split_checked(kind: str, shape: tuple) -> bool:
    """Whether the split of a fixed random array of this shape reproduces
    torch's one-call transform bit for bit (tried once per shape).  The
    pieces depend on the shape alone, so one check holds for every thread
    count; pocketfft's work does not depend on the values."""
    ok = _fft_checked.get((kind, shape))
    if ok is None:
        gen = torch.Generator().manual_seed(20261012)
        probe = torch.rand(shape, dtype=torch.float64, generator=gen,
                           device="cpu") - 0.5
        with torch.no_grad():
            if kind == "r2c":
                ok = _same_bits(_rfftn_split(probe), torch.fft.rfftn(probe))
            else:
                spec = torch.fft.rfftn(probe)
                ok = _same_bits(_irfftn_split(spec, shape),
                                torch.fft.irfftn(spec, s=shape))
        _fft_checked[(kind, shape)] = ok
        if not ok:
            _log.info("torch FFT %s %s: split differs from torch's own "
                      "transform; it runs as one torch call", kind, shape)
    return ok


def _fft_may_split(t, points, dtype) -> bool:
    return (enabled() and t.dim() == 3 and points >= FFT_MIN_POINTS
            and plain(t, dtype=dtype) and t.is_contiguous()
            and not t.is_conj() and not t.is_neg()
            and _threads.fft_threads() > 1 and _pocketfft())


def rfftn(x):
    """``torch.fft.rfftn(x)`` of a 3-D float64 CPU tensor on HELIX's
    threads, bit for bit, or None when torch's call must run."""
    if not _fft_may_split(x, x.numel(), torch.float64):
        return None
    if not _split_checked("r2c", tuple(x.shape)):
        return None
    return _rfftn_split(x)


def irfftn(X, s):
    """``torch.fft.irfftn(X, s=s)`` of a 3-D complex128 CPU tensor on
    HELIX's threads, bit for bit, or None when torch's call must run."""
    s = tuple(int(v) for v in s)
    if (len(s) != 3 or tuple(X.shape) != (s[0], s[1], s[2] // 2 + 1)
            or not _fft_may_split(X, s[0] * s[1] * s[2], torch.complex128)):
        return None
    if not _split_checked("c2r", s):
        return None
    return _irfftn_split(X, s)
