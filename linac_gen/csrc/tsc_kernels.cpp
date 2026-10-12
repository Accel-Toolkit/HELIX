// TSC (triangular-shaped cloud, 27-cell quadratic) charge deposition and field
// gather for HELIX — C++ twins of linac_gen/pic/charge_deposition.deposit_tsc and
// linac_gen/pic/field_interpolation.interpolate_tsc.
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
//     order inside a pass.  Single-threaded by design, so the result does not
//     depend on the OpenMP thread count;
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
#include <cmath>
#include <cstdint>
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


PYBIND11_MODULE(_tsc_kernels, m) {
    m.doc() = "C++ TSC deposit/gather for HELIX, bit-identical to the Python kernels";
    m.def("deposit_tsc", &deposit_tsc_cpp,
          "TSC charge deposition (bit-identical to Python); None outside the envelope",
          py::arg("positions"), py::arg("charges"), py::arg("grid_min"), py::arg("grid_max"),
          py::arg("n_grid"));
    m.def("interpolate_tsc", &interpolate_tsc_cpp,
          "TSC field gather (bit-identical to Python); None outside the envelope",
          py::arg("field_x"), py::arg("field_y"), py::arg("field_z"), py::arg("positions"),
          py::arg("grid_min"), py::arg("grid_max"), py::arg("n_grid"));
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
