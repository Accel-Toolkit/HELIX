// TSC (triangular-shaped cloud, 27-cell quadratic) charge deposition and field
// gather for HELIX — C++ twins of linac_gen/pic/charge_deposition.deposit_tsc and
// linac_gen/pic/field_interpolation.interpolate_tsc.
// Also the torch space-charge engine's CIC and TSC deposit and gather in
// torch's arithmetic (deposit_torch, gather_products_torch; see their section
// below).  Both kinds need what this module is built for: no FMA contraction.
//
// BIT-IDENTITY CONTRACT with the Python functions (pinned by
// tests/pic/test_tsc_kernels_cpp.py, which every CI build job runs):
//   * the same floating-point expressions, evaluated in the same order:
//       dx = (gmax - gmin) / (n - 1)            s  = (p - gmin) / dx
//       i  = floor(s) clamped to [1, n-2]       f  = clamp(s - i, 0, 1)
//       w- = 0.5*(1-f)*(1-f)   w0 = 0.75-(f-0.5)*(f-0.5)   w+ = 0.5*f*f
//       contribution = ((wx*wy)*wz)*q ;  rho /= cell_vol at the end
//   * the deposit sums each cell in np.add.at order: 27 offset passes
//     (di, dj, dk each -1, 0, +1, nested as in Python), particles in index
//     order inside a pass.  deposit_tsc does it in one loop on one thread;
//     deposit_tsc_planes on HELIX's threads, each owning whole x-planes and
//     adding every cell's terms in that same order, so the result does not
//     depend on the thread count (deposit_tsc stays as its reference);
//   * the gather sums the 27 terms per particle from 0.0 in the same order
//     (parallel over particles: disjoint writes, so bit-safe);
//   * no FMA contraction: built with -ffp-contract=off (Unix) or /fp:precise
//     (MSVC), and the pragma below for compilers that honour it.
// Envelope: linac_gen.pic.pic_solver sends only float64 (N, 3) positions with
// N > 0, float64 bounds and n >= 3 per axis.  The kernels themselves return None
// (and the dispatcher runs the Python implementation) when any particle maps to
// a grid coordinate s that is not finite or has |s| >= 2**62 -- non-finite
// positions or bounds, zero spacing, absurdly distant particles -- because
// numpy's float-to-int cast is platform-dependent there.  Shapes are validated,
// so a direct call with malformed arrays raises ValueError instead of reading
// out of bounds.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "helix_threads.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#pragma STDC FP_CONTRACT OFF

namespace py = pybind11;
using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

struct Axis {
    double gmin, dx;
    int64_t n;
};

// Index and fraction exactly as numpy: floor -> int64 -> clip(1, n-2), then
// clip(s - i, 0, 1).  Returns false (i, f unset) when s is not finite or
// |s| >= 2**62, where numpy's cast is platform-dependent; below that floor(s)
// converts exactly, so clamping the floored double equals numpy's clip.
constexpr double kMaxCells = 4611686018427387904.0;   // 2**62

inline bool locate(double p, const Axis& a, int64_t& i, double& f) {
    const double s = (p - a.gmin) / a.dx;
    if (!(std::fabs(s) < kMaxCells)) return false;     // also catches NaN
    const double fl = std::floor(s);
    const double lo = 1.0, hi = static_cast<double>(a.n - 2);
    if (fl < lo) i = 1;
    else if (fl > hi) i = a.n - 2;
    else i = static_cast<int64_t>(fl);
    double t = s - static_cast<double>(i);
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    f = t;
    return true;
}

inline void weights(double f, double w[3]) {
    const double one_m = 1.0 - f;
    const double c = f - 0.5;
    w[0] = 0.5 * one_m * one_m;      // offset -1
    w[1] = 0.75 - c * c;             // offset  0
    w[2] = 0.5 * f * f;              // offset +1
}

Axis axis(const double* gmin, const double* gmax, const int64_t* n, int k) {
    Axis a;
    a.gmin = gmin[k];
    a.n = n[k];
    a.dx = (gmax[k] - gmin[k]) / static_cast<double>(n[k] - 1);
    return a;
}

void check_grid(const arr_d& positions, const arr_d& grid_min, const arr_d& grid_max,
                const arr_i& n_grid) {
    if (positions.ndim() != 2 || positions.shape(1) != 3)
        throw std::invalid_argument("positions must have shape (N, 3)");
    if (grid_min.size() != 3 || grid_max.size() != 3 || n_grid.size() != 3)
        throw std::invalid_argument("grid_min, grid_max and n_grid must have 3 entries");
    for (int k = 0; k < 3; ++k)
        if (n_grid.data()[k] < 3)
            throw std::invalid_argument("TSC needs at least 3 grid nodes per axis");
}

}  // namespace


py::object deposit_tsc_cpp(arr_d positions, arr_d charges, arr_d grid_min,
                           arr_d grid_max, arr_i n_grid) {
    check_grid(positions, grid_min, grid_max, n_grid);
    if (charges.ndim() != 1 || charges.shape(0) != positions.shape(0))
        throw std::invalid_argument("charges must have shape (N,)");
    auto pos = positions.unchecked<2>();
    auto q = charges.unchecked<1>();
    const double* gmin = grid_min.data();
    const double* gmax = grid_max.data();
    const int64_t* ng = n_grid.data();
    const Axis ax = axis(gmin, gmax, ng, 0), ay = axis(gmin, gmax, ng, 1), az = axis(gmin, gmax, ng, 2);
    const int64_t nx = ng[0], ny = ng[1], nz = ng[2];
    const int64_t N = pos.shape(0);
    const double cell_vol = ax.dx * ay.dx * az.dx;

    // Per-particle indices and weights, computed once.
    std::vector<int64_t> ix(N), iy(N), iz(N);
    std::vector<double> wx(3 * N), wy(3 * N), wz(3 * N);
    for (int64_t p = 0; p < N; ++p) {
        double fx, fy, fz;
        if (!locate(pos(p, 0), ax, ix[p], fx) || !locate(pos(p, 1), ay, iy[p], fy)
                || !locate(pos(p, 2), az, iz[p], fz))
            return py::none();                         // outside the envelope
        weights(fx, &wx[3 * p]);
        weights(fy, &wy[3 * p]);
        weights(fz, &wz[3 * p]);
    }

    py::array_t<double> result({nx, ny, nz});
    double* rho = result.mutable_data();
    const size_t cells = static_cast<size_t>(nx) * ny * nz;
    for (size_t c = 0; c < cells; ++c) rho[c] = 0.0;
    // 27 passes in np.add.at order.
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int d = 0; d < 3; ++d)
                for (int64_t p = 0; p < N; ++p) {
                    const double w = wx[3 * p + a] * wy[3 * p + b] * wz[3 * p + d] * q(p);
                    const size_t idx = (static_cast<size_t>(ix[p] - 1 + a) * ny
                                        + static_cast<size_t>(iy[p] - 1 + b)) * nz
                                       + static_cast<size_t>(iz[p] - 1 + d);
                    rho[idx] += w;
                }
    for (size_t c = 0; c < cells; ++c) rho[c] /= cell_vol;
    return result;
}


// deposit_tsc_planes_cpp's team: about sqrt(N / kTeamParticles) threads
// (helix::team_for; the best teams measured on a 14-core Mac, 2026-10-12:
// 2 threads at 2k particles, 4 at 5k-20k, 8 at 50k, 14 at 200k).  Any team
// gives the same bits; this only sets the speed.
constexpr double kTeamParticles = 700.0;

// One particle's deposit inputs, in bucket order (deposit_tsc_planes_cpp).
struct Sorted {
    double wx[3], wy[3], wz[3], q;
    int64_t iy, iz;
};

// deposit_tsc_cpp on HELIX's threads, with its bits.  Every cell gets the same
// terms in the same order: pass (a, b, d) outermost in np.add.at order, the
// particles of a pass in index order.  A particle with base x-cell i reaches
// plane i - 1 + a in pass a only, so plane X takes, for a = 0, 1, 2, the
// particles of bucket X + 1 - a (a stable counting sort keeps index order in a
// bucket), then b, then d, then those particles.  Planes are owned by one
// thread each: no cell is shared, and no thread count changes the result.
// Memory: one Sorted (96 bytes) per particle -- what deposit_tsc_cpp's arrays
// take -- left uninitialised until the parallel loops fill it; a particle's
// cells and weights are computed twice (to count it, then to place it), by
// the same functions, so with the same bits.  Every allocation happens
// before the GIL is released (a failure is a MemoryError, not a terminate),
// and every thread computes in the caller's floating-point environment.
// The team: helix::team_for(N, team_particles), the whole setting for
// team_particles <= 0; one thread runs deposit_tsc_cpp itself, which is
// faster there.
py::object deposit_tsc_planes_cpp(arr_d positions, arr_d charges, arr_d grid_min,
                                  arr_d grid_max, arr_i n_grid,
                                  double team_particles) {
    check_grid(positions, grid_min, grid_max, n_grid);
    if (charges.ndim() != 1 || charges.shape(0) != positions.shape(0))
        throw std::invalid_argument("charges must have shape (N,)");
    const int nc = helix::team_for(static_cast<double>(positions.shape(0)),
                                   team_particles);
    if (nc <= 1)
        return deposit_tsc_cpp(positions, charges, grid_min, grid_max, n_grid);
    auto pos = positions.unchecked<2>();
    auto q = charges.unchecked<1>();
    const double* gmin = grid_min.data();
    const double* gmax = grid_max.data();
    const int64_t* ng = n_grid.data();
    const Axis ax = axis(gmin, gmax, ng, 0), ay = axis(gmin, gmax, ng, 1), az = axis(gmin, gmax, ng, 2);
    const int64_t nx = ng[0], ny = ng[1], nz = ng[2];
    const int64_t N = pos.shape(0);
    const double cell_vol = ax.dx * ay.dx * az.dx;
    const size_t plane = static_cast<size_t>(ny) * nz;

    py::array_t<double> result({nx, ny, nz});
    double* rho = result.mutable_data();
    std::unique_ptr<Sorted[]> ps(new Sorted[static_cast<size_t>(N)]);
    std::vector<int64_t> start(static_cast<size_t>(nx) + 1, 0);
    // per chunk and base x-cell: the count, then the next free slot; and the
    // end of that chunk's part of the bucket
    std::vector<int64_t> slot(static_cast<size_t>(nc) * nx, 0);
    std::vector<int64_t> limit(static_cast<size_t>(nc) * nx, 0);
    // planes for each thread, heaviest first to the least loaded (any
    // assignment gives the same bits: a plane's sums are its own)
    std::vector<int64_t> work(static_cast<size_t>(nx), 0);
    std::vector<int64_t> plane_order(static_cast<size_t>(nx));
    std::vector<int64_t> load(static_cast<size_t>(nc), 0);
    std::vector<int> owner(static_cast<size_t>(nx), 0);
    const auto heavier = [&work](int64_t u, int64_t v) {
        return work[u] > work[v] || (work[u] == work[v] && u < v);
    };
    const helix::FpEnv fpenv;           // the caller's, for every thread
    int declined = 0;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel num_threads(nc)
#endif
        {
            const helix::FpEnv::Use use_fpenv(fpenv);
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
            const int team = omp_get_num_threads();
#else
            const int tid = 0, team = 1;
#endif
            // Chunk t (particles N*t/team .. N*(t+1)/team - 1) counts its
            // particles per base x-cell, and finds any outside the envelope.
#ifdef _OPENMP
            #pragma omp for schedule(static) reduction(|:declined)
#endif
            for (int t = 0; t < team; ++t) {
                const int64_t a = N * t / team, b = N * (t + 1) / team;
                int64_t* cnt = &slot[static_cast<size_t>(t) * nx];
                for (int64_t p = a; p < b; ++p) {
                    int64_t i, j, k;
                    double fx, fy, fz;
                    if (!locate(pos(p, 0), ax, i, fx) || !locate(pos(p, 1), ay, j, fy)
                            || !locate(pos(p, 2), az, k, fz)) {
                        declined |= 1;                 // outside the envelope
                        break;
                    }
                    cnt[i]++;
                }
            }
            // (the loop's barrier makes `declined` final for every thread)
            if (!declined) {
                // Buckets laid out cell-major then chunk-minor, so a bucket
                // holds its particles in index order (as deposit_torch_cpp).
#ifdef _OPENMP
                #pragma omp single
#endif
                {
                    int64_t run = 0;
                    for (int64_t x = 0; x < nx; ++x) {
                        start[x] = run;
                        for (int t = 0; t < team; ++t) {
                            int64_t& sl = slot[static_cast<size_t>(t) * nx + x];
                            const int64_t c = sl;
                            sl = run;
                            run += c;
                        }
                    }
                    start[nx] = run;
                    for (int t = 0; t < team; ++t)
                        for (int64_t x = 0; x < nx; ++x)
                            limit[static_cast<size_t>(t) * nx + x] =
                                t + 1 < team ? slot[static_cast<size_t>(t + 1) * nx + x]
                                             : start[x + 1];
                    // plane X reads buckets X-1, X, X+1 (bucket = base cell)
                    for (int64_t x = 0; x < nx; ++x) {
                        for (int a = 0; a < 3; ++a) {
                            const int64_t base = x + 1 - a;
                            if (base >= 0 && base < nx)
                                work[x] += start[base + 1] - start[base];
                        }
                        plane_order[x] = x;
                    }
                    std::sort(plane_order.begin(), plane_order.end(), heavier);
                    for (int64_t k = 0; k < nx; ++k) {
                        const int64_t x = plane_order[k];
                        int best = 0;
                        for (int t = 1; t < team; ++t)
                            if (load[t] < load[best]) best = t;
                        owner[x] = best;
                        load[best] += work[x] + 1;
                    }
                }
                // Chunk t places its particles' weights, cells and charges in
                // their slots, so each plane reads its buckets from contiguous
                // memory (the values, and each bucket's index order, as is).
                // The checks only fail if the positions changed since they
                // were counted (another thread writing them while the GIL is
                // released): the call then returns None, as for a decline,
                // instead of writing outside a bucket.
#ifdef _OPENMP
                #pragma omp for schedule(static) reduction(|:declined)
#endif
                for (int t = 0; t < team; ++t) {
                    const int64_t a = N * t / team, b = N * (t + 1) / team;
                    int64_t* sl = &slot[static_cast<size_t>(t) * nx];
                    const int64_t* lim = &limit[static_cast<size_t>(t) * nx];
                    for (int64_t p = a; p < b; ++p) {
                        int64_t i, j, k;
                        double fx, fy, fz;
                        if (!locate(pos(p, 0), ax, i, fx) || !locate(pos(p, 1), ay, j, fy)
                                || !locate(pos(p, 2), az, k, fz) || sl[i] >= lim[i]) {
                            declined |= 1;
                            break;
                        }
                        Sorted& o = ps[static_cast<size_t>(sl[i]++)];
                        weights(fx, o.wx);
                        weights(fy, o.wy);
                        weights(fz, o.wz);
                        o.q = q(p);
                        o.iy = j;
                        o.iz = k;
                    }
                }
            }
            // (again final for every thread: every slot is filled unless set)
            if (!declined) {
                // Each thread its own planes; no cell is shared.
                for (int64_t X = 0; X < nx; ++X) {
                    if (owner[X] != tid) continue;
                    double* pl = rho + static_cast<size_t>(X) * plane;
                    for (size_t c = 0; c < plane; ++c) pl[c] = 0.0;
                    for (int a = 0; a < 3; ++a) {
                        const int64_t base = X + 1 - a;
                        if (base < 0 || base >= nx) continue;
                        const Sorted* b0 = ps.get() + start[base];
                        const Sorted* b1 = ps.get() + start[base + 1];
                        for (int b = 0; b < 3; ++b)
                            for (int d = 0; d < 3; ++d)
                                for (const Sorted* o = b0; o < b1; ++o) {
                                    const double w = o->wx[a] * o->wy[b] * o->wz[d] * o->q;
                                    const size_t idx = static_cast<size_t>(o->iy - 1 + b) * nz
                                                       + static_cast<size_t>(o->iz - 1 + d);
                                    pl[idx] += w;
                                }
                    }
                    for (size_t c = 0; c < plane; ++c) pl[c] /= cell_vol;
                }
            }
        }
    }
    if (declined) return py::none();
    return result;
}


// E = -grad(phi) as linac_gen.pic.poisson_solver._field_from_potential
// computes it, bit for bit: along each axis -(phi[i+1] - phi[i-1]) / (2.0 * d)
// inside, -(phi[1] - phi[0]) / d and -(phi[n-1] - phi[n-2]) / d on the two
// boundary planes -- the difference, its negation, then the division, as
// numpy does them (an axis of 2 nodes has only the boundary planes).  phi may
// be a strided view (solve passes the corner of the padded potential); the
// three results are fresh C-contiguous arrays with every element written.
// None when an axis has fewer than 2 nodes (numpy raises IndexError there)
// or when any thread raised a floating-point exception that numpy reports
// (invalid, divide-by-zero, overflow, underflow): numpy then recomputes the
// same values and warns or raises as the caller's numpy.errstate says.
// Each element is computed on its own, so threading over x-planes gives the
// same bits at any thread count; every thread computes in the caller's
// floating-point environment.
py::object field_from_potential_cpp(py::array_t<double> phi_in, double dx,
                                    double dy, double dz, double team_points) {
    if (phi_in.ndim() != 3)
        throw std::invalid_argument("phi must be a 3-D array");
    const int64_t nx = phi_in.shape(0), ny = phi_in.shape(1), nz = phi_in.shape(2);
    if (nx < 2 || ny < 2 || nz < 2) return py::none();
    auto phi = phi_in.unchecked<3>();
    py::array_t<double> ex({nx, ny, nz}), ey({nx, ny, nz}), ez({nx, ny, nz});
    double* Ex = ex.mutable_data();
    double* Ey = ey.mutable_data();
    double* Ez = ez.mutable_data();
    const double tdx = 2.0 * dx, tdy = 2.0 * dy, tdz = 2.0 * dz;
    const helix::FpEnv fpenv;           // the caller's, for every thread
    // about sqrt(points / team_points) threads (helix::team_for; with the
    // default 7000 the best teams measured on a 14-core Mac: 1 at 24^3, 2 at
    // 32^3, 4 at 40^3-48^3, 8 at 64^3)
    const int team = helix::team_for(static_cast<double>(nx) * ny * nz,
                                     team_points);
    int raised = 0;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel num_threads(team) if(team > 1) reduction(|:raised)
#endif
        {
        const helix::FpEnv::Use use_fpenv(fpenv);
        std::feclearexcept(FE_ALL_EXCEPT);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int64_t i = 0; i < nx; ++i) {
            // the two nodes and the divisor of this x-plane
            const bool xb = (i == 0 || i == nx - 1);
            const int64_t ip = i == 0 ? 1 : (i == nx - 1 ? nx - 1 : i + 1);
            const int64_t im = i == 0 ? 0 : (i == nx - 1 ? nx - 2 : i - 1);
            const double divx = xb ? dx : tdx;
            for (int64_t j = 0; j < ny; ++j) {
                const bool yb = (j == 0 || j == ny - 1);
                const int64_t jp = j == 0 ? 1 : (j == ny - 1 ? ny - 1 : j + 1);
                const int64_t jm = j == 0 ? 0 : (j == ny - 1 ? ny - 2 : j - 1);
                const double divy = yb ? dy : tdy;
                const size_t row = (static_cast<size_t>(i) * ny + j) * nz;
                double* ox = Ex + row;
                double* oy = Ey + row;
                double* oz = Ez + row;
                for (int64_t k = 0; k < nz; ++k)
                    ox[k] = -(phi(ip, j, k) - phi(im, j, k)) / divx;
                for (int64_t k = 0; k < nz; ++k)
                    oy[k] = -(phi(i, jp, k) - phi(i, jm, k)) / divy;
                oz[0] = -(phi(i, j, 1) - phi(i, j, 0)) / dz;
                for (int64_t k = 1; k < nz - 1; ++k)
                    oz[k] = -(phi(i, j, k + 1) - phi(i, j, k - 1)) / tdz;
                oz[nz - 1] = -(phi(i, j, nz - 1) - phi(i, j, nz - 2)) / dz;
            }
        }
        if (std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW
                              | FE_UNDERFLOW))
            raised |= 1;
        }
    }
    // numpy reports these (a warning, or an error under numpy.errstate):
    // leave them to its code, whose values are these
    if (raised) return py::none();
    return py::make_tuple(ex, ey, ez);
}


py::object interpolate_tsc_cpp(arr_d field_x, arr_d field_y, arr_d field_z,
                               arr_d positions, arr_d grid_min, arr_d grid_max,
                               arr_i n_grid) {
    check_grid(positions, grid_min, grid_max, n_grid);
    {
        const int64_t* g = n_grid.data();
        for (const arr_d* f : {&field_x, &field_y, &field_z})
            if (f->ndim() != 3 || f->shape(0) != g[0] || f->shape(1) != g[1]
                    || f->shape(2) != g[2])
                throw std::invalid_argument("field arrays must have shape n_grid");
    }
    auto pos = positions.unchecked<2>();
    const double* Fx = field_x.data();
    const double* Fy = field_y.data();
    const double* Fz = field_z.data();
    const double* gmin = grid_min.data();
    const double* gmax = grid_max.data();
    const int64_t* ng = n_grid.data();
    const Axis ax = axis(gmin, gmax, ng, 0), ay = axis(gmin, gmax, ng, 1), az = axis(gmin, gmax, ng, 2);
    const int64_t ny = ng[1], nz = ng[2];
    const int64_t N = pos.shape(0);

    py::array_t<double> result({N, static_cast<int64_t>(3)});
    double* out = result.mutable_data();
    int declined = 0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(|:declined) \
        num_threads(helix::num_threads())
#endif
    for (int64_t p = 0; p < N; ++p) {
        int64_t i, j, k;
        double fx, fy, fz, wxp[3], wyp[3], wzp[3];
        if (!locate(pos(p, 0), ax, i, fx) || !locate(pos(p, 1), ay, j, fy)
                || !locate(pos(p, 2), az, k, fz)) {
            declined |= 1;                             // outside the envelope
            continue;
        }
        weights(fx, wxp);
        weights(fy, wyp);
        weights(fz, wzp);
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int d = 0; d < 3; ++d) {
                    const double w = wxp[a] * wyp[b] * wzp[d];
                    const size_t idx = (static_cast<size_t>(i - 1 + a) * ny
                                        + static_cast<size_t>(j - 1 + b)) * nz
                                       + static_cast<size_t>(k - 1 + d);
                    s0 += w * Fx[idx];
                    s1 += w * Fy[idx];
                    s2 += w * Fz[idx];
                }
        out[3 * p] = s0;
        out[3 * p + 1] = s1;
        out[3 * p + 2] = s2;
    }
    if (declined) return py::none();
    return result;
}



// ---------------------------------------------------------------------------
// The torch space-charge engine's deposit and gather (linac_gen/pic/torch/
// deposition.py and interpolation.py) for its float64 CPU path, in torch's
// arithmetic -- bit for bit what the torch code computes, on any number of
// threads (pinned by tests/pic/test_torch_engine_kernels.py):
//   s = (p - gmin) / dx, dx = (gmax - gmin) / (n - 1);
//   i = floor(s) clamped to [lo, n-2] (CIC lo = 0, TSC lo = 1);
//   f = clamp(s - i, 0, 1);
//   CIC weights 1-f, f;  TSC weights 0.5*((1-f)*(1-f)), 0.75-(f-0.5)*(f-0.5),
//   (0.5*f)*f;  a corner's weight (wx*wy)*wz, its charge weight*q;
//   corners in torch's order (_CIC_OFFSETS, _TSC_OFFSETS: x, then y, then z).
// The deposit sums every cell in particle order, as torch's
// index_put_(accumulate=True) does for float64 on the CPU (one thread, along
// the flattened (N, corners) values): each thread owns whole x-planes and
// merges the particles that reach its plane in index order.  It returns the
// sums; torch divides them by the cell volume.
// The gather returns each particle's products weight * field, (3, N,
// corners), and torch sums them over the corners -- in its own order, which
// depends on the CPU's vector width.
// Both return None when a particle maps to a coordinate that is not finite
// or has |s| >= 2**62, where torch's float-to-int cast differs by platform;
// the caller then runs the torch code.
// ---------------------------------------------------------------------------

namespace {

inline bool locate_torch(double p, const Axis& a, int64_t lo, int64_t& i,
                         double& f) {
    const double s = (p - a.gmin) / a.dx;
    if (!(std::fabs(s) < kMaxCells)) return false;     // also catches NaN
    const double fl = std::floor(s);
    if (fl < static_cast<double>(lo)) i = lo;
    else if (fl > static_cast<double>(a.n - 2)) i = a.n - 2;
    else i = static_cast<int64_t>(fl);
    double t = s - static_cast<double>(i);
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    f = t;
    return true;
}

// The weights of one axis, w[0 .. width-1], as torch builds them.
inline void weights_torch(int order, double f, double* w) {
    if (order == 1) {
        w[0] = 1.0 - f;
        w[1] = f;
    } else {
        const double u = 1.0 - f;
        const double c = f - 0.5;
        w[0] = 0.5 * (u * u);
        w[1] = 0.75 - c * c;
        w[2] = 0.5 * f * f;
    }
}

struct TorchGrid {
    Axis ax, ay, az;
    int64_t nx, ny, nz;
    int order, width, lo, off0;    // off0: offset of the first corner
};

TorchGrid torch_grid(const arr_d& positions, const arr_d& grid_min,
                     const arr_d& grid_max, const arr_i& n_grid, int order) {
    if (order != 1 && order != 2)
        throw std::invalid_argument("order must be 1 (CIC) or 2 (TSC)");
    if (positions.ndim() != 2 || positions.shape(1) != 3)
        throw std::invalid_argument("positions must have shape (N, 3)");
    if (grid_min.size() != 3 || grid_max.size() != 3 || n_grid.size() != 3)
        throw std::invalid_argument("grid_min, grid_max and n_grid must have 3 entries");
    const int64_t* ng = n_grid.data();
    for (int k = 0; k < 3; ++k)
        if (ng[k] < order + 1)
            throw std::invalid_argument(order == 1
                ? "CIC needs at least 2 grid nodes per axis"
                : "TSC needs at least 3 grid nodes per axis");
    TorchGrid g;
    g.ax = axis(grid_min.data(), grid_max.data(), ng, 0);
    g.ay = axis(grid_min.data(), grid_max.data(), ng, 1);
    g.az = axis(grid_min.data(), grid_max.data(), ng, 2);
    g.nx = ng[0];
    g.ny = ng[1];
    g.nz = ng[2];
    g.order = order;
    g.width = order + 1;
    g.lo = order == 1 ? 0 : 1;
    g.off0 = order == 1 ? 0 : -1;
    return g;
}

// One particle's base cells and weights (wx, wy, wz: width entries each).
inline bool particle_torch(const TorchGrid& g, double px, double py, double pz,
                           int64_t& i, int64_t& j, int64_t& k, double* wx,
                           double* wy, double* wz) {
    double fx, fy, fz;
    if (!locate_torch(px, g.ax, g.lo, i, fx) || !locate_torch(py, g.ay, g.lo, j, fy)
            || !locate_torch(pz, g.az, g.lo, k, fz))
        return false;
    weights_torch(g.order, fx, wx);
    weights_torch(g.order, fy, wy);
    weights_torch(g.order, fz, wz);
    return true;
}

}  // namespace


py::object deposit_torch_cpp(arr_d positions, arr_d charges, arr_d grid_min,
                             arr_d grid_max, arr_i n_grid, int order) {
    const TorchGrid g = torch_grid(positions, grid_min, grid_max, n_grid, order);
    if (charges.ndim() != 1 || charges.shape(0) != positions.shape(0))
        throw std::invalid_argument("charges must have shape (N,)");
    const int64_t N = positions.shape(0);
    const double* pos = positions.data();
    const double* q = charges.data();
    const int W = g.width;
    const int64_t nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t plane = static_cast<size_t>(ny) * nz;

    std::vector<int64_t> ci(N), cj(N), ck(N);
    std::vector<double> w(static_cast<size_t>(3 * W) * N);
    py::array_t<double> result({nx, ny, nz});
    double* rho = result.mutable_data();
    const int nt = helix::num_threads();
    int declined = 0;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(|:declined) \
            num_threads(nt)
#endif
        for (int64_t p = 0; p < N; ++p) {
            double* wp = &w[static_cast<size_t>(3 * W) * p];
            if (!particle_torch(g, pos[3 * p], pos[3 * p + 1], pos[3 * p + 2],
                                ci[p], cj[p], ck[p], wp, wp + W, wp + 2 * W))
                declined |= 1;
        }
        if (!declined) {
            // Corner (a, b, d) of particle p: cell (i+off0+a, j+off0+b,
            // k+off0+d), charge ((wx[a]*wy[b])*wz[d])*q.  Each particle
            // reaches a cell at most once, so a cell's sum in particle order
            // is torch's.
            auto add_plane = [&](int64_t p, int a, double* pl) {
                const double* wp = &w[static_cast<size_t>(3 * W) * p];
                const double* wy = wp + W;
                const double* wz = wp + 2 * W;
                const double wxa = wp[a];
                const int64_t j0 = cj[p] + g.off0, k0 = ck[p] + g.off0;
                for (int b = 0; b < W; ++b)
                    for (int d = 0; d < W; ++d) {
                        const double c = wxa * wy[b] * wz[d] * q[p];
                        pl[static_cast<size_t>(j0 + b) * nz
                           + static_cast<size_t>(k0 + d)] += c;
                    }
            };
            if (nt <= 1) {
                std::fill(rho, rho + static_cast<size_t>(nx) * plane, 0.0);
                for (int64_t p = 0; p < N; ++p)
                    for (int a = 0; a < W; ++a)
                        add_plane(p, a, rho + static_cast<size_t>(ci[p] + g.off0 + a) * plane);
            } else {
                // Particles bucketed by base x-cell, in index order inside a
                // bucket (a stable counting sort: chunk t counts, buckets are
                // laid out cell-major, chunk-minor, chunk t places).
                std::vector<int64_t> start(static_cast<size_t>(nx) + 1, 0);
                std::vector<int64_t> slot(static_cast<size_t>(nt) * nx, 0);
                std::vector<int64_t> sorted(N);
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) num_threads(nt)
#endif
                for (int t = 0; t < nt; ++t) {
                    const int64_t a = N * t / nt, b = N * (t + 1) / nt;
                    int64_t* cnt = &slot[static_cast<size_t>(t) * nx];
                    for (int64_t p = a; p < b; ++p) cnt[ci[p]]++;
                }
                int64_t run = 0;
                for (int64_t x = 0; x < nx; ++x) {
                    start[x] = run;
                    for (int t = 0; t < nt; ++t) {
                        int64_t& s = slot[static_cast<size_t>(t) * nx + x];
                        const int64_t c = s;
                        s = run;
                        run += c;
                    }
                }
                start[nx] = run;
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) num_threads(nt)
#endif
                for (int t = 0; t < nt; ++t) {
                    const int64_t a = N * t / nt, b = N * (t + 1) / nt;
                    int64_t* s = &slot[static_cast<size_t>(t) * nx];
                    for (int64_t p = a; p < b; ++p) sorted[s[ci[p]]++] = p;
                }
                // Plane x takes corner a of the particles with base cell
                // x - off0 - a, merged in index order; planes dealt
                // round-robin so the crowded core planes land on different
                // threads (static: see the CIC kernel in pic_kernels.cpp).
#ifdef _OPENMP
                #pragma omp parallel for schedule(static, 1) num_threads(nt)
#endif
                for (int64_t x = 0; x < nx; ++x) {
                    double* pl = rho + static_cast<size_t>(x) * plane;
                    std::fill(pl, pl + plane, 0.0);
                    int64_t at[3], end[3];
                    for (int a = 0; a < W; ++a) {
                        const int64_t base = x - g.off0 - a;
                        if (base >= 0 && base < nx) {
                            at[a] = start[base];
                            end[a] = start[base + 1];
                        } else {
                            at[a] = end[a] = 0;
                        }
                    }
                    for (;;) {
                        int pick = -1;
                        for (int a = 0; a < W; ++a)
                            if (at[a] < end[a]
                                    && (pick < 0 || sorted[at[a]] < sorted[at[pick]]))
                                pick = a;
                        if (pick < 0) break;
                        add_plane(sorted[at[pick]++], pick, pl);
                    }
                }
            }
        }
    }
    if (declined) return py::none();
    return result;
}


py::object gather_products_torch_cpp(arr_d field_x, arr_d field_y, arr_d field_z,
                                     arr_d positions, arr_d grid_min,
                                     arr_d grid_max, arr_i n_grid, int order) {
    const TorchGrid g = torch_grid(positions, grid_min, grid_max, n_grid, order);
    for (const arr_d* f : {&field_x, &field_y, &field_z})
        if (f->ndim() != 3 || f->shape(0) != g.nx || f->shape(1) != g.ny
                || f->shape(2) != g.nz)
            throw std::invalid_argument("field arrays must have shape n_grid");
    const int64_t N = positions.shape(0);
    const int W = g.width;
    const int64_t K = static_cast<int64_t>(W) * W * W;
    const double* pos = positions.data();
    const double* F[3] = {field_x.data(), field_y.data(), field_z.data()};
    const int64_t ny = g.ny, nz = g.nz;
    py::array_t<double> result({static_cast<int64_t>(3), N, K});
    double* out = result.mutable_data();
    const size_t per_field = static_cast<size_t>(N) * K;
    const int nt = helix::num_threads();
    int declined = 0;
    {
        py::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(|:declined) \
            num_threads(nt)
#endif
        for (int64_t p = 0; p < N; ++p) {
            int64_t i, j, k;
            double wx[3], wy[3], wz[3];
            if (!particle_torch(g, pos[3 * p], pos[3 * p + 1], pos[3 * p + 2],
                                i, j, k, wx, wy, wz)) {
                declined |= 1;
                continue;
            }
            size_t c = static_cast<size_t>(p) * K;
            for (int a = 0; a < W; ++a)
                for (int b = 0; b < W; ++b)
                    for (int d = 0; d < W; ++d, ++c) {
                        const double wc = wx[a] * wy[b] * wz[d];
                        const size_t idx =
                            (static_cast<size_t>(i + g.off0 + a) * ny
                             + static_cast<size_t>(j + g.off0 + b)) * nz
                            + static_cast<size_t>(k + g.off0 + d);
                        out[c] = wc * F[0][idx];
                        out[per_field + c] = wc * F[1][idx];
                        out[2 * per_field + c] = wc * F[2][idx];
                    }
        }
    }
    if (declined) return py::none();
    return result;
}

PYBIND11_MODULE(_tsc_kernels, m) {
    m.doc() = "C++ TSC deposit/gather for HELIX, bit-identical to the Python kernels, "
              "and the torch engine's CIC/TSC deposit and gather in torch's arithmetic";
    m.def("deposit_tsc", &deposit_tsc_cpp,
          "TSC charge deposition (bit-identical to Python); None outside the envelope",
          py::arg("positions"), py::arg("charges"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("n_grid"));
    m.def("deposit_tsc_planes", &deposit_tsc_planes_cpp,
          "deposit_tsc on HELIX's threads (about sqrt(N / team_particles) of "
          "them; 0: the whole setting), the same bits at any thread count; "
          "None outside the envelope",
          py::arg("positions"), py::arg("charges"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("n_grid"), py::arg("team_particles") = kTeamParticles);
    m.def("field_from_potential", &field_from_potential_cpp,
          "(Ex, Ey, Ez) = -grad(phi) as poisson_solver._field_from_potential "
          "computes it, bit for bit, on HELIX's threads; None when an axis has "
          "fewer than 2 nodes",
          py::arg("phi"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
          py::arg("team_points") = 7000.0);
    m.def("interpolate_tsc", &interpolate_tsc_cpp,
          "TSC field gather (bit-identical to Python); None outside the envelope",
          py::arg("field_x"), py::arg("field_y"), py::arg("field_z"), py::arg("positions"),
          py::arg("grid_min"), py::arg("grid_max"), py::arg("n_grid"));
    m.def("deposit_torch", &deposit_torch_cpp,
          "The torch engine's CIC (order 1) or TSC (order 2) deposit in torch's "
          "arithmetic: each cell's charge sum in particle order (not yet divided "
          "by the cell volume); None outside the envelope",
          py::arg("positions"), py::arg("charges"), py::arg("grid_min"),
          py::arg("grid_max"), py::arg("n_grid"), py::arg("order"));
    m.def("gather_products_torch", &gather_products_torch_cpp,
          "The torch engine's CIC/TSC gather in torch's arithmetic: the products "
          "weight * field, shape (3, N, corners), for torch to sum; None outside "
          "the envelope",
          py::arg("field_x"), py::arg("field_y"), py::arg("field_z"),
          py::arg("positions"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("n_grid"), py::arg("order"));
    m.def("set_num_threads", &helix::set_num_threads, py::arg("n"),
          "Threads for this module's OpenMP loops (0 = the OpenMP default)");
    m.def("get_num_threads", &helix::num_threads,
          "Threads this module's OpenMP loops use");
    m.def("_team_size", &helix::team_size,
          "Team a parallel region gets from the calling thread (tests)");
#ifdef _OPENMP
    m.attr("_openmp_enabled") = true;
#else
    m.attr("_openmp_enabled") = false;
#endif
}
