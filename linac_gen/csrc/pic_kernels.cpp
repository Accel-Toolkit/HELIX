#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "helix_threads.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// CIC charge deposition: scatter particle charges to a 3D grid.
//
// Every cell sums its particles' contributions in particle order -- the
// order of a single-threaded loop -- whatever the thread count, so 1 thread
// or 64 give the same bits: those of the per-thread-grid kernel this
// replaces when it ran on one thread.  (That kernel summed per-thread
// partial grids, so its last bits depended on the thread count: the GUI,
// capped at 1 OpenMP thread, and the CLI on 10 differed.)
//
// One thread runs the particle loop itself.  Several threads cut the grid
// into x-planes: plane i takes the particles of x-cell i (their low-x
// corners) and of x-cell i-1 (their high-x corners), merged by particle
// index after a stable counting sort by x-cell.  Planes share no cells, so
// they run in parallel with no per-thread grids.
//
// The arithmetic is the old kernel's, statement for statement, so a
// compiler that contracts `acc += a * b` into an FMA (the clang and gcc
// default; this module is NOT built with -ffp-contract=off) contracts it
// the same way.  The old kernel's last step on one thread added every
// cell's sum to 0.0 (its reduce over threads), which turns a -0.0 -- an FMA
// that underflows -- into +0.0; so does this one.
// ---------------------------------------------------------------------------
py::array_t<double> deposit_cic_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,  // (N, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> charges,    // (N,)
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_min,   // (3,)
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_max,   // (3,)
    py::array_t<int,    py::array::c_style | py::array::forcecast> n_grid      // (3,)
) {
    auto pos = positions.unchecked<2>();
    auto q = charges.unchecked<1>();
    auto gmin = grid_min.unchecked<1>();
    auto gmax = grid_max.unchecked<1>();
    auto ng = n_grid.unchecked<1>();

    // The old kernel trusted these and read or wrote out of bounds.
    if (pos.shape(1) != 3)
        throw std::invalid_argument("positions must have shape (N, 3)");
    if (q.shape(0) != pos.shape(0))
        throw std::invalid_argument("charges must have one value per particle");
    if (gmin.shape(0) != 3 || gmax.shape(0) != 3 || ng.shape(0) != 3)
        throw std::invalid_argument("grid_min, grid_max and n_grid must have 3 entries");
    if (ng(0) < 2 || ng(1) < 2 || ng(2) < 2)
        throw std::invalid_argument("n_grid needs at least 2 nodes per axis");
    if (pos.shape(0) > std::numeric_limits<int>::max())
        throw std::invalid_argument("more particles than a 32-bit index holds");

    const int N = static_cast<int>(pos.shape(0));
    const int nx = ng(0), ny = ng(1), nz = ng(2);
    const double dx = (gmax(0) - gmin(0)) / (nx - 1);
    const double dy = (gmax(1) - gmin(1)) / (ny - 1);
    const double dz = (gmax(2) - gmin(2)) / (nz - 1);
    const double cell_vol = dx * dy * dz;
    const double gmin0 = gmin(0), gmin1 = gmin(1), gmin2 = gmin(2);
    const size_t plane = static_cast<size_t>(ny) * nz;
    const size_t cells = static_cast<size_t>(nx) * plane;

    auto result = py::array_t<double>({nx, ny, nz});
    double* rho = result.mutable_data();
    const int nt = helix::num_threads();

    // The one-thread kernel's last step: + 0.0, then the density.
    auto finish = [cell_vol](double* c, size_t n) {
        for (size_t k = 0; k < n; k++)
            c[k] = 0.0 + c[k];
        if (cell_vol > 0)
            for (size_t k = 0; k < n; k++)
                c[k] /= cell_vol;
    };

    if (nt <= 1) {
        py::gil_scoped_release release;
        std::fill(rho, rho + cells, 0.0);
        for (int p = 0; p < N; p++) {
            double px = pos(p, 0), py_val = pos(p, 1), pz = pos(p, 2);
            double fx = (px - gmin0) / dx;
            double fy = (py_val - gmin1) / dy;
            double fz = (pz - gmin2) / dz;
            int ix = std::max(0, std::min((int)fx, nx - 2));
            int iy = std::max(0, std::min((int)fy, ny - 2));
            int iz = std::max(0, std::min((int)fz, nz - 2));
            double wx1 = fx - ix; wx1 = std::max(0.0, std::min(1.0, wx1));
            double wy1 = fy - iy; wy1 = std::max(0.0, std::min(1.0, wy1));
            double wz1 = fz - iz; wz1 = std::max(0.0, std::min(1.0, wz1));
            double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1, wz0 = 1.0 - wz1;
            double charge = q(p);
            // Flat index helpers: idx = ((i*ny)+j)*nz + k
            const size_t s0 = (static_cast<size_t>(ix)   * ny + iy  ) * nz + iz;
            const size_t s1 = (static_cast<size_t>(ix+1) * ny + iy  ) * nz + iz;
            const size_t s2 = (static_cast<size_t>(ix)   * ny + iy+1) * nz + iz;
            const size_t s3 = (static_cast<size_t>(ix+1) * ny + iy+1) * nz + iz;
            rho[s0]     += charge * wx0 * wy0 * wz0;
            rho[s1]     += charge * wx1 * wy0 * wz0;
            rho[s2]     += charge * wx0 * wy1 * wz0;
            rho[s3]     += charge * wx1 * wy1 * wz0;
            rho[s0 + 1] += charge * wx0 * wy0 * wz1;
            rho[s1 + 1] += charge * wx1 * wy0 * wz1;
            rho[s2 + 1] += charge * wx0 * wy1 * wz1;
            rho[s3 + 1] += charge * wx1 * wy1 * wz1;
        }
        finish(rho, cells);
    } else {
        // Each particle's cell and weights, computed once (same
        // expressions as above), and the particles bucketed by x-cell in
        // particle order within a bucket (a stable counting sort, the same
        // order for any thread count): chunk t counts its particles per
        // cell, the buckets are laid out cell-major, chunk-minor, and each
        // chunk writes its particles into its slots.
        std::vector<int> cell_x(N), cell_y(N), cell_z(N), order(N);
        std::vector<double> w1(3 * static_cast<size_t>(N));
        std::vector<int> start(static_cast<size_t>(nx) + 1);
        std::vector<int> slot(static_cast<size_t>(nt) * nx, 0);
        py::gil_scoped_release release;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nt)
#endif
        for (int t = 0; t < nt; t++) {
            const int lo = static_cast<int>(static_cast<int64_t>(N) * t / nt);
            const int hi = static_cast<int>(static_cast<int64_t>(N) * (t + 1) / nt);
            int* cnt = &slot[static_cast<size_t>(t) * nx];
            for (int p = lo; p < hi; p++) {
                double px = pos(p, 0), py_val = pos(p, 1), pz = pos(p, 2);
                double fx = (px - gmin0) / dx;
                double fy = (py_val - gmin1) / dy;
                double fz = (pz - gmin2) / dz;
                int ix = std::max(0, std::min((int)fx, nx - 2));
                int iy = std::max(0, std::min((int)fy, ny - 2));
                int iz = std::max(0, std::min((int)fz, nz - 2));
                double wx1 = fx - ix; wx1 = std::max(0.0, std::min(1.0, wx1));
                double wy1 = fy - iy; wy1 = std::max(0.0, std::min(1.0, wy1));
                double wz1 = fz - iz; wz1 = std::max(0.0, std::min(1.0, wz1));
                cell_x[p] = ix;
                cell_y[p] = iy;
                cell_z[p] = iz;
                w1[3 * static_cast<size_t>(p)] = wx1;
                w1[3 * static_cast<size_t>(p) + 1] = wy1;
                w1[3 * static_cast<size_t>(p) + 2] = wz1;
                cnt[ix]++;
            }
        }
        int run = 0;
        for (int i = 0; i < nx; i++) {
            start[i] = run;
            for (int t = 0; t < nt; t++) {
                int& s = slot[static_cast<size_t>(t) * nx + i];
                const int c = s;
                s = run;
                run += c;
            }
        }
        start[nx] = run;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nt)
#endif
        for (int t = 0; t < nt; t++) {
            const int lo = static_cast<int>(static_cast<int64_t>(N) * t / nt);
            const int hi = static_cast<int>(static_cast<int64_t>(N) * (t + 1) / nt);
            int* s = &slot[static_cast<size_t>(t) * nx];
            for (int p = lo; p < hi; p++)
                order[s[cell_x[p]]++] = p;
        }

        // Planes in parallel, dealt round-robin so the crowded planes of
        // the beam core land on different threads.  (Not schedule(dynamic):
        // it needs __kmpc_dispatch_deinit, which older libomp builds -- an
        // older torch's -- lack, and the module would then fail to load.)
        // Bucket nx-1 is always empty (ix <= nx-2).
#ifdef _OPENMP
        #pragma omp parallel for schedule(static, 1) num_threads(nt)
#endif
        for (int i = 0; i < nx; i++) {
            double* pl = rho + static_cast<size_t>(i) * plane;
            std::fill(pl, pl + plane, 0.0);
            int a = start[i];                                // x-cell i
            const int a_end = start[i + 1];
            int b = (i > 0) ? start[i - 1] : 0;              // x-cell i-1
            const int b_end = (i > 0) ? start[i] : 0;
            while (a < a_end || b < b_end) {
                const bool low = (b >= b_end
                                  || (a < a_end && order[a] < order[b]));
                const int p = low ? order[a++] : order[b++];
                const int iy = cell_y[p], iz = cell_z[p];
                const double* w = &w1[3 * static_cast<size_t>(p)];
                double wx1 = w[0], wy1 = w[1], wz1 = w[2];
                double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1, wz0 = 1.0 - wz1;
                double charge = q(p);
                // low-x corners for x-cell i, high-x corners for i-1
                const double wx = low ? wx0 : wx1;
                const size_t c0 = static_cast<size_t>(iy) * nz + iz;
                const size_t c2 = static_cast<size_t>(iy + 1) * nz + iz;
                pl[c0]     += charge * wx * wy0 * wz0;
                pl[c2]     += charge * wx * wy1 * wz0;
                pl[c0 + 1] += charge * wx * wy0 * wz1;
                pl[c2 + 1] += charge * wx * wy1 * wz1;
            }
            finish(pl, plane);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// CIC field interpolation: gather field values from grid to particles.
// Each particle writes to its own unique row in `out`, so a plain
// `#pragma omp parallel for` is race-free and bit-identical.
// ---------------------------------------------------------------------------
py::array_t<double> interpolate_cic_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> field_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> field_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> field_z,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_min,
    py::array_t<double, py::array::c_style | py::array::forcecast> grid_max,
    py::array_t<int,    py::array::c_style | py::array::forcecast> n_grid
) {
    auto fx_grid = field_x.unchecked<3>();
    auto fy_grid = field_y.unchecked<3>();
    auto fz_grid = field_z.unchecked<3>();
    auto pos = positions.unchecked<2>();
    auto gmin = grid_min.unchecked<1>();
    auto gmax = grid_max.unchecked<1>();
    auto ng = n_grid.unchecked<1>();

    const int N = pos.shape(0);
    const int nx = ng(0), ny = ng(1), nz = ng(2);
    const double dx = (gmax(0) - gmin(0)) / (nx - 1);
    const double dy = (gmax(1) - gmin(1)) / (ny - 1);
    const double dz = (gmax(2) - gmin(2)) / (nz - 1);
    const double gmin0 = gmin(0), gmin1 = gmin(1), gmin2 = gmin(2);

    auto result = py::array_t<double>({N, 3});
    auto out = result.mutable_unchecked<2>();

#ifdef _OPENMP
    // Each particle writes its own row: the same bits for any thread
    // count and any schedule.
    #pragma omp parallel for schedule(static) num_threads(helix::num_threads())
#endif
    for (int p = 0; p < N; p++) {
        double px = pos(p, 0), py_val = pos(p, 1), pz = pos(p, 2);
        double ffx = (px - gmin0) / dx;
        double ffy = (py_val - gmin1) / dy;
        double ffz = (pz - gmin2) / dz;
        int ix = std::max(0, std::min((int)ffx, nx - 2));
        int iy = std::max(0, std::min((int)ffy, ny - 2));
        int iz = std::max(0, std::min((int)ffz, nz - 2));
        double wx1 = std::max(0.0, std::min(1.0, ffx - ix));
        double wy1 = std::max(0.0, std::min(1.0, ffy - iy));
        double wz1 = std::max(0.0, std::min(1.0, ffz - iz));
        double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1, wz0 = 1.0 - wz1;

        out(p, 0) = fx_grid(ix,iy,iz)*wx0*wy0*wz0 + fx_grid(ix+1,iy,iz)*wx1*wy0*wz0
                  + fx_grid(ix,iy+1,iz)*wx0*wy1*wz0 + fx_grid(ix+1,iy+1,iz)*wx1*wy1*wz0
                  + fx_grid(ix,iy,iz+1)*wx0*wy0*wz1 + fx_grid(ix+1,iy,iz+1)*wx1*wy0*wz1
                  + fx_grid(ix,iy+1,iz+1)*wx0*wy1*wz1 + fx_grid(ix+1,iy+1,iz+1)*wx1*wy1*wz1;

        out(p, 1) = fy_grid(ix,iy,iz)*wx0*wy0*wz0 + fy_grid(ix+1,iy,iz)*wx1*wy0*wz0
                  + fy_grid(ix,iy+1,iz)*wx0*wy1*wz0 + fy_grid(ix+1,iy+1,iz)*wx1*wy1*wz0
                  + fy_grid(ix,iy,iz+1)*wx0*wy0*wz1 + fy_grid(ix+1,iy,iz+1)*wx1*wy0*wz1
                  + fy_grid(ix,iy+1,iz+1)*wx0*wy1*wz1 + fy_grid(ix+1,iy+1,iz+1)*wx1*wy1*wz1;

        out(p, 2) = fz_grid(ix,iy,iz)*wx0*wy0*wz0 + fz_grid(ix+1,iy,iz)*wx1*wy0*wz0
                  + fz_grid(ix,iy+1,iz)*wx0*wy1*wz0 + fz_grid(ix+1,iy+1,iz)*wx1*wy1*wz0
                  + fz_grid(ix,iy,iz+1)*wx0*wy0*wz1 + fz_grid(ix+1,iy,iz+1)*wx1*wy0*wz1
                  + fz_grid(ix,iy+1,iz+1)*wx0*wy1*wz1 + fz_grid(ix+1,iy+1,iz+1)*wx1*wy1*wz1;
    }
    return result;
}


PYBIND11_MODULE(_pic_kernels, m) {
    m.doc() = "C++ PIC kernels for Linac_Gen (OpenMP-enabled)";
    m.def("deposit_cic", &deposit_cic_cpp, "CIC charge deposition",
          py::arg("positions"), py::arg("charges"),
          py::arg("grid_min"), py::arg("grid_max"), py::arg("n_grid"));
    m.def("interpolate_cic", &interpolate_cic_cpp, "CIC field interpolation",
          py::arg("field_x"), py::arg("field_y"), py::arg("field_z"),
          py::arg("positions"), py::arg("grid_min"), py::arg("grid_max"), py::arg("n_grid"));
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
    // Provenance reads this: the deposit sums every cell in particle
    // order, so its result does not depend on the thread count (a binary
    // built before 2026-10-11 lacks the attribute).
    m.attr("_deposit_order") = "particle";
}
