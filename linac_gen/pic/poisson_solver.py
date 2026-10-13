"""3D FFT Poisson solver with open boundary conditions (Hockney's method).

Solves nabla^2 phi = -rho/eps_0 on a 3D grid using FFT-based convolution
with a free-space Green's function on a doubled (zero-padded) grid.

Two Green's-function variants are supported:

* ``green_kind="igf"`` (default) — Integrated Green Function (Qiang et al.,
  PRSTAB 9, 044204 (2006)).  The cell-averaged 1/r kernel is computed in
  closed form via an analytic antiderivative.  Converges much faster with
  grid resolution than the point-source variant; standard in OPAL,
  IMPACT, Cheetah.

* ``green_kind="point"`` — Point-source 1/(4πε₀ r) sampled at cell
  centres, with a self-potential regulariser at r=0.  Cheaper to build
  and matches the legacy LG behaviour bit-for-bit when needed for
  regression checks.

The solver precomputes the FFT of the Green's function at construction
time. Each call to :meth:`solve` zero-pads the charge density, performs
a convolution via FFT, and returns the three components of E = -grad(phi)
computed with finite differences.

FFTs are dispatched through :mod:`linac_gen.pic.gpu_backend`, which picks
between a CPU path (``scipy.fft`` on the HELIX thread setting, see
:mod:`linac_gen.parallel.threads`, or ``numpy.fft`` if
SciPy is unavailable) and a GPU path (``cupyx.scipy.fft`` in FP64).  The
selection is controlled by ``SpaceChargeConfig.use_gpu`` and overridable by
the ``LINAC_GEN_USE_GPU`` env var.  The FFT of the Green's function is
cached on the chosen device, so only ρ and Ex/Ey/Ez cross the host/device
boundary per solve.
"""
import os
import sys
import threading

import numpy as np

from linac_gen.core.constants import EPSILON_0, PI
from linac_gen.pic.gpu_backend import (
    _CpuBackend,
    _GpuBackend,
    _MpsBackend,
    _maybe_emit_banner,
    _pick_gpu_backend_class,
    select_backend,
)


def _build_backend(use_gpu: str, grid_cells: int):
    """Pick CPU/cupy/MPS backend for the given use_gpu mode and grid size."""
    name = select_backend(use_gpu, grid_cells=grid_cells)
    if name == "gpu":
        backend = _pick_gpu_backend_class(use_gpu)()
    else:
        backend = _CpuBackend()
    _maybe_emit_banner(backend.name)
    return backend


# ---------------------------------------------------------------------------
# Closed-form antiderivative for the integrated Green's function.
#
# Qiang et al. (PRSTAB 9, 044204, 2006) showed that the integral of the free-
# space 1/r kernel over a rectangular cell admits a closed analytic form.
# Define
#
#     F(x, y, z) = - (z²/2) atan2(xy, z·r) - (y²/2) atan2(xz, y·r)
#                  - (x²/2) atan2(yz, x·r)
#                  + y·z log(x + r) + x·z log(y + r) + x·y log(z + r)
#
# with r = sqrt(x² + y² + z²).  Then for a cell centred at displacement
# (i hₓ, j h_y, k h_z) with extents (hₓ, h_y, h_z),
#
#     ∫_cell 1/r dV = 8-point inclusion–exclusion of F at the cell corners.
#
# The integrated Green's function is this divided by the cell volume,
# carrying the 1/(4πε₀) prefactor of the Coulomb kernel.  The antideriv
# is well-defined for all corner positions we evaluate (corners sit at
# ½-integer multiples of the grid spacing, so x, y, z are never simultaneously
# zero); the atan2 form keeps the formula numerically stable when one of the
# coordinates is small.
# ---------------------------------------------------------------------------
def _antideriv_igf(xp, x, y, z):
    """Closed-form antiderivative F where ∂³F/(∂x∂y∂z) = 1/r.

    Uses Cheetah's atan + asinh formulation, which is odd under
    (x, y, z) → (−x, −y, −z) and therefore produces a kernel symmetric
    under cell-displacement reflection — required for the Hockney
    convolution to yield a physically correct (even) Coulomb interaction.
    The atan2 / log form is also a valid antiderivative but picks up
    branch shifts that break this symmetry.

    ``xp`` is the array module (``numpy`` for CPU backend, ``cupy`` for
    GPU); the math is identical, only the dispatch changes.
    """
    r = xp.sqrt(x * x + y * y + z * z)
    # Corners sit at ±½-integer multiples of (dx, dy, dz), so x, y, z are
    # never simultaneously zero and the ``small denominator`` guards below
    # are only there as a defensive epsilon.
    eps = 1e-300
    F = (
        -0.5 * z * z * xp.arctan(x * y / (z * r + eps))
        - 0.5 * y * y * xp.arctan(x * z / (y * r + eps))
        - 0.5 * x * x * xp.arctan(y * z / (x * r + eps))
        + y * z * xp.arcsinh(x / (xp.sqrt(y * y + z * z) + eps))
        + x * z * xp.arcsinh(y / (xp.sqrt(x * x + z * z) + eps))
        + x * y * xp.arcsinh(z / (xp.sqrt(x * x + y * y) + eps))
    )
    return F


# ---------------------------------------------------------------------------
# The integrated Green's function on the CPU, on HELIX's threads.
#
# An adaptive grid rebuilds the kernel at every kick, and the antiderivative
# costs most of a rebuild (~35 of 40 ms at 48^3 on one thread).  On the CPU
# both builds run in pieces of whole x planes on HELIX's threads
# (linac_gen.parallel.threads), each thread taking the next piece as it
# finishes one: the antiderivative at the vertices, then the 8-corner
# difference of each cell written straight into the Hockney layout.  Every
# element goes through the same numpy operations on the same doubles, in the
# same order, as in one whole-grid call.  They are element-wise, so an
# element's result does not depend on where it sits in an array; +, -, *, /
# and sqrt are correctly rounded on every inner loop, and arctan and arcsinh
# -- whose last digit can depend on which loop numpy picks (SVML or libm,
# chosen by a memory-overlap test) -- run exactly in place, which numpy never
# counts as an overlap.  So the kernel is the one-call kernel bit for bit,
# whatever the pieces or the thread count (pinned by
# tests/pic/test_greens_threads.py).
# ---------------------------------------------------------------------------

#: A phase of the build (vertices, cells) is cut into about this many
#: pieces of whole x planes, each between these numbers of points (at least
#: one plane).  Every numpy call holds Python's GIL for a few microseconds,
#: so many small pieces scale worse than a few large ones: about six pieces
#: was fastest from 32^3 to 128^3 on a 14-core Mac.
GREEN_PIECES = 6
GREEN_PIECE_POINTS = (1 << 14, 1 << 18)


def _step(planes: int, plane: int) -> int:
    """Planes per piece for ``planes`` planes of ``plane`` points."""
    least, most = GREEN_PIECE_POINTS
    target = min(max(planes * plane // GREEN_PIECES, least), most)
    return max(1, -(-target // max(1, plane)))


def _pieces(lo: int, hi: int, step: int) -> list[tuple[int, int]]:
    """``range(lo, hi)`` in runs of ``step`` planes."""
    return [(a, min(hi, a + step)) for a in range(lo, hi, step)]


def _on_threads(fn, pieces) -> None:
    """``fn(a, b)`` for every piece, on up to the setting's threads, under
    the caller's numpy error handling (numpy keeps it per thread)."""
    from linac_gen.parallel.threads import get_threads
    from linac_gen.pic import fft_split
    err = np.geterr()
    fft_split.run_pieces(fn, pieces, get_threads(),
                         enter=lambda: np.errstate(**err))


def _pair(shape):
    """Two scratch arrays of ``shape`` in one allocation, apart."""
    size = int(np.prod(shape))
    buf = np.empty(2 * size + 8, dtype=np.float64)
    return (buf[:size].reshape(shape),
            buf[size + 8:2 * size + 8].reshape(shape))


def _antideriv_into(out, x, y, z, yz, ryz, r, s) -> None:
    """``out[...] = _antideriv_igf(np, x, y, z)``: the same operations on
    the same values in the same order, into the scratch arrays ``r`` and
    ``s`` (``out``'s shape) instead of a fresh array per operation --
    allocating the large temporaries costs more than the arithmetic and
    does not scale across threads.  ``yz`` and ``ryz`` are ``y * z`` and
    ``np.sqrt(y * y + z * z) + 1e-300``, the same for every piece."""
    eps = 1e-300
    np.add(x * x + y * y, z * z, out=r)
    np.sqrt(r, out=r)
    # -0.5 z^2 atan(xy / (z r + eps))
    np.multiply(z, r, out=s)
    np.add(s, eps, out=s)
    np.divide(x * y, s, out=s)
    np.arctan(s, out=s)
    np.multiply(-0.5 * z * z, s, out=out)
    # - 0.5 y^2 atan(xz / (y r + eps))
    np.multiply(y, r, out=s)
    np.add(s, eps, out=s)
    np.divide(x * z, s, out=s)
    np.arctan(s, out=s)
    np.multiply(0.5 * y * y, s, out=s)
    np.subtract(out, s, out=out)
    # - 0.5 x^2 atan(yz / (x r + eps))
    np.multiply(x, r, out=s)
    np.add(s, eps, out=s)
    np.divide(yz, s, out=s)
    np.arctan(s, out=s)
    np.multiply(0.5 * x * x, s, out=s)
    np.subtract(out, s, out=out)
    # + yz asinh(x / (sqrt(y^2 + z^2) + eps)), and the two like it
    np.divide(x, ryz, out=s)
    np.arcsinh(s, out=s)
    np.multiply(yz, s, out=s)
    np.add(out, s, out=out)
    np.divide(y, np.sqrt(x * x + z * z) + eps, out=s)
    np.arcsinh(s, out=s)
    np.multiply(x * z, s, out=s)
    np.add(out, s, out=out)
    np.divide(z, np.sqrt(x * x + y * y) + eps, out=s)
    np.arcsinh(s, out=s)
    np.multiply(x * y, s, out=s)
    np.add(out, s, out=out)


def _vertex_antideriv(vx, vy, vz) -> np.ndarray:
    """``_antideriv_igf`` at every vertex ``(vx[i], vy[j], vz[k])``."""
    F = np.empty((len(vx), len(vy), len(vz)), dtype=np.float64)
    plane = len(vy) * len(vz)
    pieces = _pieces(0, len(vx), _step(len(vx), plane))
    rows = max(b - a for a, b in pieces)
    y, z = vy[None, :, None], vz[None, None, :]
    yz, ryz = y * z, np.sqrt(y * y + z * z) + 1e-300
    scratch = {}                        # thread -> its two scratch arrays

    def piece(a, b):
        pair = scratch.get(threading.get_ident())
        if pair is None:
            pair = scratch[threading.get_ident()] = _pair(
                (rows, len(vy), len(vz)))
        _antideriv_into(F[a:b], vx[a:b, None, None], y, z, yz, ryz,
                        pair[0][:b - a], pair[1][:b - a])

    _on_threads(piece, pieces)
    return F


def _cell_average(F, a, b, cell_vol):
    """Cells ``a … b−1`` along x: the 8-corner difference of ``F`` (cell
    ``i`` between vertices ``i`` and ``i + 1`` on each axis), in the
    per-corner term order HHH − LHH − HLH − HHL + LLH + LHL + HLL − LLL,
    over the cell volume, times 1/(4πε₀) -- the one-call expression's
    operations, in place in one array."""
    Hx, Lx = slice(a + 1, b + 1), slice(a, b)
    H, L = slice(1, None), slice(None, -1)
    g = np.subtract(F[Hx, H, H], F[Lx, H, H])
    np.subtract(g, F[Hx, L, H], out=g)
    np.subtract(g, F[Hx, H, L], out=g)
    np.add(g, F[Lx, L, H], out=g)
    np.add(g, F[Lx, H, L], out=g)
    np.add(g, F[Hx, L, L], out=g)
    np.subtract(g, F[Lx, L, L], out=g)
    np.divide(g, cell_vol, out=g)
    g *= 1.0 / (4.0 * PI * EPSILON_0)
    return g


def _greens_exact_cpu(nx, ny, nz, vx, vy, vz, cell_vol) -> np.ndarray:
    """The exact build's kernel, in the Hockney layout: cell ``p`` (natural
    order, displacement ``p − n``) lands at index ``(p + n) mod 2n`` on each
    axis -- what ``np.roll(G, -n)`` does."""
    F = _vertex_antideriv(vx, vy, vz)
    G = np.empty((2 * nx, 2 * ny, 2 * nz), dtype=np.float64)

    def piece(a, b):                    # never straddles nx
        g = _cell_average(F, a, b, cell_vol)
        m = a + nx if a < nx else a - nx
        dst = G[m:m + (b - a)]
        dst[:, :ny, :nz] = g[:, ny:, nz:]
        dst[:, :ny, nz:] = g[:, ny:, :nz]
        dst[:, ny:, :nz] = g[:, :ny, nz:]
        dst[:, ny:, nz:] = g[:, :ny, :nz]

    step = _step(2 * nx, (2 * ny) * (2 * nz))
    _on_threads(piece, _pieces(0, nx, step) + _pieces(nx, 2 * nx, step))
    return G


def _greens_symmetric_cpu(nx, ny, nz, vx, vy, vz, cell_vol) -> np.ndarray:
    """The symmetric build's kernel, in the Hockney layout: octant cell
    ``i`` (``0 … n``) at index ``i`` (``i < n``) and ``2n − i``
    (``i ≥ 1``) on each axis -- the copies ``G[ix_(m, m, m)]`` makes."""
    F = _vertex_antideriv(vx, vy, vz)
    G = np.empty((2 * nx, 2 * ny, 2 * nz), dtype=np.float64)

    def put(dst, g):                    # mirror along y and z
        dst[:, :ny, :nz] = g[:, :ny, :nz]
        dst[:, :ny, nz:] = g[:, :ny, nz:0:-1]
        dst[:, ny:, :nz] = g[:, ny:0:-1, :nz]
        dst[:, ny:, nz:] = g[:, ny:0:-1, nz:0:-1]

    def piece(a, b):
        g = _cell_average(F, a, b, cell_vol)
        lo, hi = a, min(b, nx)          # octant planes i < n at i
        if hi > lo:
            put(G[lo:hi], g[lo - a:hi - a])
        lo, hi = max(a, 1), b           # planes 1 … n at 2n − i
        if hi > lo:
            put(G[2 * nx - hi + 1:2 * nx - lo + 1], g[lo - a:hi - a][::-1])

    step = _step(nx + 1, (ny + 1) * (nz + 1))
    _on_threads(piece, _pieces(0, nx + 1, step))
    return G


def _guarded_copy(a: np.ndarray) -> np.ndarray:
    """Copy of ``a`` (same shape, dtype, values and memory layout) placed
    inside a 1-D allocation one element larger on each side, so no other
    array can ever start or end exactly where its data does.

    The layout is rebuilt by sorting the axes from slowest to fastest
    stride; for the dense arrays FFT libraries return (C order, or a
    permutation of it) the copy's strides equal ``a``'s.
    """
    order = sorted(range(a.ndim), key=lambda i: -abs(a.strides[i]))
    buf = np.empty(a.size + 2, dtype=a.dtype)
    inner = buf[1:-1].reshape(tuple(a.shape[i] for i in order))
    view = inner.transpose(np.argsort(order))
    view[...] = a
    return view


def _lean_solve() -> bool:
    """The solve without its redundant work, the same bits: the padded charge
    buffer is not zeroed again (only its charge block is ever written -- in
    solve and solve_device -- and no backend's forward FFT writes its input,
    so the rest keeps the +0.0 it was allocated with), the inverse FFT works
    in place on the spectrum, which solve gives up, and on the CPU the FFTs
    are pruned (fft_split.rfftn_padded, irfftn_corner) and the field
    differences run in the compiled kernel (_field_kernel).
    LINAC_GEN_LEAN_SOLVE=0 restores the old path; LINAC_GEN_PRUNED_FFT=0
    alone keeps the full transforms."""
    return os.environ.get("LINAC_GEN_LEAN_SOLVE", "1") != "0"


_FIELD_KERNEL = None        # linac_gen._tsc_kernels.field_from_potential


def _field_kernel():
    """The compiled field differences (bit for bit the numpy code below, on
    HELIX's threads), looked up at first use, or None: no module, or one
    built before it.  torch is never imported for it: on macOS the kernel
    binds the OpenMP runtime torch loads, so until something has imported
    torch (a PIC run's solver does) the numpy code runs and the lookup is
    tried again at the next call."""
    global _FIELD_KERNEL
    kernel = _FIELD_KERNEL
    if kernel is None:
        if sys.platform == "darwin" and "torch" not in sys.modules:
            return None
        try:
            from linac_gen._tsc_kernels import field_from_potential as kernel
        except ImportError:
            kernel = False
        else:
            from linac_gen.parallel.threads import sync_kernels
            sync_kernels()
        _FIELD_KERNEL = kernel
    return kernel or None


def _field_from_potential(phi, dx, dy, dz, xp):
    """E = -grad(phi): central differences inside, one-sided on the
    boundary planes (``xp``: numpy, or the device's array module).  On the
    CPU the compiled kernel does it (see _field_kernel and _lean_solve)."""
    if xp is np and _lean_solve():
        kernel = _field_kernel()
        if (kernel is not None and isinstance(phi, np.ndarray)
                and phi.dtype == np.float64 and phi.ndim == 3):
            fields = kernel(phi, float(dx), float(dy), float(dz))
            if fields is not None:
                return fields
    Ex = xp.zeros_like(phi)
    Ey = xp.zeros_like(phi)
    Ez = xp.zeros_like(phi)

    # Central differences (interior)
    Ex[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
    Ey[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
    Ez[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)

    # One-sided differences at boundaries
    Ex[0, :, :] = -(phi[1, :, :] - phi[0, :, :]) / dx
    Ex[-1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / dx
    Ey[:, 0, :] = -(phi[:, 1, :] - phi[:, 0, :]) / dy
    Ey[:, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / dy
    Ez[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / dz
    Ez[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dz
    return Ex, Ey, Ez


class PoissonSolverFFT:
    """3D FFT Poisson solver with open boundary conditions (Hockney's method)."""

    def __init__(self, grid_min: np.ndarray, grid_max: np.ndarray,
                 n_grid: np.ndarray, use_gpu: str = "auto",
                 green_kind: str = "igf", green_build: str = "exact"):
        """Precompute Green's function FFT for the given grid.

        Parameters
        ----------
        grid_min : np.ndarray
            (3,) grid lower bounds.
        grid_max : np.ndarray
            (3,) grid upper bounds.
        n_grid : np.ndarray
            (3,) number of grid points per dimension (integer array).
        use_gpu : {"auto", "cpu", "gpu", "cuda", "mps"}
            Backend selection — see ``linac_gen.pic.gpu_backend.select_backend``.
        green_kind : {"igf", "point"}
            Green's function flavour.  ``"igf"`` (default) uses the
            cell-integrated kernel of Qiang 2006; ``"point"`` uses the
            sampled 1/r kernel and the legacy self-potential regulariser.
        green_build : {"exact", "symmetric"}
            How the ``"igf"`` kernel is built.  ``"exact"`` (default)
            evaluates every displacement; ``"symmetric"`` evaluates the
            non-negative displacements only and mirrors them (the kernel is
            even in each axis), about 7x fewer evaluations at 48^3.  It
            equals the exact build within that build's own rounding: the
            8-corner difference loses digits in the far cells (~5e-13 of the
            kernel's peak at 48^3, growing with the grid, 3e-12 at 96^3), so
            the exact build is not exactly even either.  Ignored for
            ``"point"``.
        """
        self.grid_min = np.asarray(grid_min, dtype=np.float64)
        self.grid_max = np.asarray(grid_max, dtype=np.float64)
        self.n_grid = np.asarray(n_grid, dtype=np.int64)
        self.dx = (self.grid_max - self.grid_min) / (self.n_grid - 1).astype(
            np.float64
        )
        if green_kind not in ("igf", "point"):
            raise ValueError(
                f"green_kind must be 'igf' or 'point', got {green_kind!r}"
            )
        self.green_kind = green_kind
        if green_build not in ("exact", "symmetric"):
            raise ValueError(
                "green_build must be 'exact' or 'symmetric', got "
                f"{green_build!r}"
            )
        self.green_build = green_build
        grid_cells = int(self.n_grid.prod())
        self._backend = _build_backend(use_gpu, grid_cells)
        # Lazy-allocated reusable charge buffer on the device (filled & FFT'd
        # per solve).  Avoids a per-solve host alloc + H→D copy in adaptive
        # mode (~5–10 ms saved per kick).  Shape never changes.
        self._rho_padded = None
        #: Green's-function builds (the kernel is rebuilt whenever the cell
        #: size changes, i.e. at every kick of an adaptive grid)
        self.n_green_builds = 0
        self._precompute_greens_fft()

    # ------------------------------------------------------------------
    def update_grid(self, grid_min, grid_max) -> None:
        """Re-target the solver to a new grid extent without rebuilding the
        backend, FFT plan cache, or device buffers.  Used by adaptive grid
        mode to avoid re-instantiating the solver every PIC kick.

        ``n_grid``, ``green_kind``, ``green_build``, ``use_gpu`` are
        unchanged; only the grid bounds and per-axis spacing are updated,
        then the Green's function FFT is rebuilt in place on the same device.
        When the spacing is exactly the one the stored Green's function was
        built for, it is kept: it depends only on (n, dx, kind, build), and
        :meth:`solve` never reads the grid bounds.
        """
        grid_min = np.asarray(grid_min, dtype=np.float64)
        grid_max = np.asarray(grid_max, dtype=np.float64)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.dx = (grid_max - grid_min) / (self.n_grid - 1).astype(np.float64)
        if not np.array_equal(self.dx, getattr(self, "_greens_dx", None)):
            self._precompute_greens_fft()

    # ------------------------------------------------------------------
    def _precompute_greens_fft(self) -> None:
        """Compute FFT of the doubled-grid Green's function (on backend device).

        The Green's function is built directly on the backend's array module
        (cupy on GPU, numpy on CPU), so no host→device transfer is needed.
        """
        if self.green_kind == "igf" and self.green_build == "symmetric":
            G = self._build_integrated_greens_symmetric()
        elif self.green_kind == "igf":
            G = self._build_integrated_greens()
        else:
            G = self._build_point_greens()
        G_fft = self._backend.rfftn(G)
        if isinstance(G_fft, np.ndarray):
            # Keep the host spectrum inside an allocation one element larger
            # on each side.  numpy multiplies complex arrays with a vectorised
            # loop unless the output's memory touches an input's end to end;
            # then it falls back to a scalar loop that rounds both the real
            # and imaginary parts differently (no fused multiply-add).  A
            # freshly allocated FFT array could land flush against a bare
            # spectrum -- on macOS it did for grids whose spectra are a
            # whole number of allocator blocks (4^3, 32^3 = 66 x 32 KiB,
            # 64^3) -- and runs then differed at the 1e-13 level.  With the
            # margins that placement cannot happen.  The copy keeps the FFT
            # output's own memory layout (numpy.fft returns permuted
            # strides) so the in-place product in solve() pairs like with
            # like; the values are copied unchanged.
            G_fft = _guarded_copy(G_fft)
        self._G_fft = G_fft
        # The spacing this Green's function belongs to.  Set only after a
        # successful build, so update_grid retries a rebuild that raised.
        self._greens_dx = self.dx.copy()
        self.n_green_builds += 1

    # ------------------------------------------------------------------
    def _build_point_greens(self):
        """Sampled 1/(4πε₀ r) Green's function with self-potential regulariser.

        Built on the backend's array module (numpy on CPU, cupy on GPU) so
        the result is already on-device for the FFT.
        """
        xp = self._backend.xp
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        ii = xp.arange(2 * nx, dtype=xp.float64)
        jj = xp.arange(2 * ny, dtype=xp.float64)
        kk = xp.arange(2 * nz, dtype=xp.float64)
        ii = xp.where(ii < nx, ii, ii - 2 * nx)
        jj = xp.where(jj < ny, jj, jj - 2 * ny)
        kk = xp.where(kk < nz, kk, kk - 2 * nz)

        II, JJ, KK = xp.meshgrid(ii, jj, kk, indexing="ij")
        r = xp.sqrt((II * dx) ** 2 + (JJ * dy) ** 2 + (KK * dz) ** 2)

        G = xp.zeros_like(r)
        mask = r > 0
        G[mask] = 1.0 / (4.0 * PI * EPSILON_0 * r[mask])
        # Self-potential regularisation at r=0 (use Python sqrt on host scalars)
        import math
        self_pot = 1.0 / (
            4.0 * PI * EPSILON_0 * 0.5 * math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        )
        G[~mask] = self_pot
        return G

    # ------------------------------------------------------------------
    def _build_integrated_greens(self):
        """Cell-integrated Green's function (Qiang 2006).

        Each cell at displacement (i, j, k)·(dx, dy, dz) is approximated by
        the volume average of 1/(4πε₀r) over a box of size (dx, dy, dz)
        centred on the cell, computed via the 8-point inclusion–exclusion of
        the analytic antiderivative ``_antideriv_igf``.

        Neighbouring cells share corners, so the antiderivative is evaluated
        once per vertex of the (2n+1)³ corner lattice and differenced by
        slicing: 8× fewer transcendental evaluations than one call per cell
        corner.  The vertex coordinates ``(v − 0.5)·d`` are the same doubles
        as the per-cell corners ``(ii ± 0.5)·d`` and the eight terms are
        combined in the same order, so the kernel is bit-for-bit identical
        to the per-corner construction (pinned by
        tests/pic/test_greens_vertex.py).

        Built on the backend's array module (numpy on CPU, cupy on GPU) so
        the result is already on-device for the FFT; with numpy, on HELIX's
        threads (:func:`_greens_exact_cpu`, the same numbers).
        """
        xp = self._backend.xp
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        # Vertex lattice in natural order: vertex v sits at (v − 0.5)·d for
        # v = −n … n — the low corner of cell v and the high corner of v − 1.
        vx = (xp.arange(-nx, nx + 1, dtype=xp.float64) - 0.5) * dx
        vy = (xp.arange(-ny, ny + 1, dtype=xp.float64) - 0.5) * dy
        vz = (xp.arange(-nz, nz + 1, dtype=xp.float64) - 0.5) * dz
        if xp is np:
            return _greens_exact_cpu(nx, ny, nz, vx, vy, vz, dx * dy * dz)
        # Broadcastable axes, not full meshes: every element still goes
        # through the same IEEE operations on the same doubles, while the
        # sub-expressions that depend on one or two axes are evaluated once
        # per line or plane.
        F = _antideriv_igf(xp, vx[:, None, None], vy[None, :, None],
                           vz[None, None, :])
        H, L = slice(1, None), slice(None, -1)

        # Inclusion–exclusion over the 8 corners, in the per-corner term
        # order HHH − LHH − HLH − HHL + LLH + LHL + HLL − LLL.
        cell_vol = dx * dy * dz
        G = (
            F[H, H, H] - F[L, H, H] - F[H, L, H] - F[H, H, L]
            + F[L, L, H] + F[L, H, L] + F[H, L, L] - F[L, L, L]
        ) / cell_vol
        G *= 1.0 / (4.0 * PI * EPSILON_0)
        # Natural displacement order (−n … n−1) → Hockney wrap (0 … n−1, −n … −1).
        return xp.roll(G, (-nx, -ny, -nz), axis=(0, 1, 2))

    # ------------------------------------------------------------------
    def _build_integrated_greens_symmetric(self):
        """The cell-integrated Green's function from the non-negative
        displacements only.

        The cell average of 1/r over a box centred at displacement
        (i, j, k)·d is even in each of i, j, k, so the cells 0 … n per axis
        determine the whole kernel: the antiderivative is evaluated on the
        (n+2)³ vertices (v − 0.5)·d, v = 0 … n+1, instead of the (2n+1)³ of
        :meth:`_build_integrated_greens`, and the result is mirrored into
        the Hockney layout (index m holds displacement m for m < n and
        displacement m − 2n, i.e. |i| = 2n − m, for m ≥ n).

        Cells with i, j, k ≥ 0 are computed exactly as in the exact build,
        from the same vertex doubles and the same term order, so they are
        bit-identical to it.  A mirrored cell takes its partner's value,
        where the exact build evaluates the negative-side corners; the two
        differ by the rounding of the 8-corner difference, which loses
        digits in the far cells (~5e-13 of the kernel's peak at 48³,
        growing with the grid).  The kernel is exactly even (pinned by
        tests/pic/test_greens_symmetric.py).  With numpy it is built on
        HELIX's threads (:func:`_greens_symmetric_cpu`, the same numbers).
        """
        xp = self._backend.xp
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        vx = (xp.arange(0, nx + 2, dtype=xp.float64) - 0.5) * dx
        vy = (xp.arange(0, ny + 2, dtype=xp.float64) - 0.5) * dy
        vz = (xp.arange(0, nz + 2, dtype=xp.float64) - 0.5) * dz
        if xp is np:
            return _greens_symmetric_cpu(nx, ny, nz, vx, vy, vz,
                                         dx * dy * dz)
        F = _antideriv_igf(xp, vx[:, None, None], vy[None, :, None],
                           vz[None, None, :])
        H, L = slice(1, None), slice(None, -1)
        cell_vol = dx * dy * dz
        G = (
            F[H, H, H] - F[L, H, H] - F[H, L, H] - F[H, H, L]
            + F[L, L, H] + F[L, H, L] + F[H, L, L] - F[L, L, L]
        ) / cell_vol
        G *= 1.0 / (4.0 * PI * EPSILON_0)
        # G[i, j, k] for i = 0 … n (one octant); mirror into 2n per axis.
        mx = xp.concatenate([xp.arange(nx), xp.arange(nx, 0, -1)])
        my = xp.concatenate([xp.arange(ny), xp.arange(ny, 0, -1)])
        mz = xp.concatenate([xp.arange(nz), xp.arange(nz, 0, -1)])
        return G[xp.ix_(mx, my, mz)]

    # ------------------------------------------------------------------
    def solve(
        self, rho: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve Poisson equation and return (Ex, Ey, Ez) on the original grid.

        Parameters
        ----------
        rho : np.ndarray
            (nx, ny, nz) charge density array.

        Returns
        -------
        Ex, Ey, Ez : np.ndarray
            (nx, ny, nz) electric field components, always on the host as
            NumPy arrays regardless of backend.
        """
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx

        # Cell volume (needed to turn density into charge for the convolution)
        cell_vol = dx * dy * dz

        # Zero-pad rho into a reusable on-device buffer.  Allocated lazily on
        # first solve so the shape is known; reused on subsequent solves to
        # avoid the per-kick host alloc + H→D copy.  Shape is fixed by config.
        xp = self._backend.xp
        shape = (2 * nx, 2 * ny, 2 * nz)
        lean = _lean_solve()
        if (self._rho_padded is None
                or tuple(self._rho_padded.shape) != shape):
            self._rho_padded = xp.zeros(shape, dtype=xp.float64)
        elif not lean:
            self._rho_padded.fill(0.0)
        rho_dev = self._backend.to_device(rho * cell_vol)  # (nx, ny, nz) → device
        self._rho_padded[:nx, :ny, :nz] = rho_dev

        # Convolution via FFT on the chosen backend.  Green's-function FFT
        # is pre-cached on device so only rho / phi cross the H↔D boundary.
        # On the CPU the pair is pruned where it applies (the lines of zeros
        # and the unread lines skipped, the same bits: fft_split), the
        # inverse then giving an array whose corner is the potential.
        corner = (nx, ny, nz)
        spectrum = None
        if lean and hasattr(self._backend, "rfftn_padded"):
            spectrum = self._backend.rfftn_padded(self._rho_padded, corner)
        pruned = spectrum is not None
        if not pruned:
            spectrum = self._backend.rfftn(self._rho_padded)
        if isinstance(spectrum, np.ndarray):
            # In place: no third array is allocated beside the two inputs,
            # and the guarded spectrum (see _precompute_greens_fft) keeps the
            # product on numpy's vectorised loop every time.
            np.multiply(spectrum, self._G_fft, out=spectrum)
        else:
            spectrum = spectrum * self._G_fft
        phi_dev = None
        if pruned:
            phi_dev = self._backend.irfftn_corner(spectrum, shape, corner,
                                                  overwrite_x=True)
        if phi_dev is None:
            phi_dev = self._backend.irfftn(spectrum, s=shape, overwrite_x=lean)
        del spectrum
        phi_padded = self._backend.to_host(phi_dev)

        # Extract potential on original grid
        phi = phi_padded[:nx, :ny, :nz]
        return _field_from_potential(phi, dx, dy, dz, np)

    def solve_device(self, rho_dev):
        """:meth:`solve` for a charge density already on the backend's device
        (an array of its ``xp``), returning the three fields there too: the
        same arithmetic, the field differences on the device.  For the GPU
        deposit and gather (:mod:`linac_gen.pic.gpu_particles`)."""
        nx, ny, nz = (int(self.n_grid[0]), int(self.n_grid[1]),
                      int(self.n_grid[2]))
        dx, dy, dz = self.dx
        cell_vol = dx * dy * dz
        xp = self._backend.xp
        shape = (2 * nx, 2 * ny, 2 * nz)
        lean = _lean_solve()
        if (self._rho_padded is None
                or tuple(self._rho_padded.shape) != shape):
            self._rho_padded = xp.zeros(shape, dtype=xp.float64)
        elif not lean:
            self._rho_padded.fill(0.0)
        self._rho_padded[:nx, :ny, :nz] = rho_dev * cell_vol
        spectrum = self._backend.rfftn(self._rho_padded)
        if isinstance(spectrum, np.ndarray):
            np.multiply(spectrum, self._G_fft, out=spectrum)
        else:
            spectrum = spectrum * self._G_fft
        phi_padded = self._backend.irfftn(spectrum, s=shape, overwrite_x=lean)
        del spectrum
        return _field_from_potential(phi_padded[:nx, :ny, :nz], dx, dy, dz,
                                     xp)

    # ------------------------------------------------------------------
    def free_memory(self) -> None:
        """Release any device-side memory the backend is holding (no-op on CPU)."""
        self._backend.free_memory()
