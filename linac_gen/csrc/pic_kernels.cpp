#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// CIC charge deposition: scatter particle charges to 3D grid.
//
// Parallel strategy: each thread accumulates into a private flat buffer,
// then we reduce across threads cell-by-cell in a deterministic order.
// This avoids any data race and keeps the final result bitwise identical
// across runs (per-thread partial sums are deterministic once the
// particle partition is fixed; the final sum-over-threads for each cell
// is evaluated in a fixed thread-index order).
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

    const int N = pos.shape(0);
    const int nx = ng(0), ny = ng(1), nz = ng(2);
    const double dx = (gmax(0) - gmin(0)) / (nx - 1);
    const double dy = (gmax(1) - gmin(1)) / (ny - 1);
    const double dz = (gmax(2) - gmin(2)) / (nz - 1);
    const double cell_vol = dx * dy * dz;
    const double gmin0 = gmin(0), gmin1 = gmin(1), gmin2 = gmin(2);
    const size_t cells = static_cast<size_t>(nx) * ny * nz;

    auto result = py::array_t<double>({nx, ny, nz});
    auto rho = result.mutable_unchecked<3>();
    // Zero init (OMP-parallel over grid cells)
#ifdef _OPENMP
    #pragma omp parallel for collapse(3)
#endif
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                rho(i, j, k) = 0.0;

    // --- Particle loop --------------------------------------------------
    // Without OpenMP: original in-place accumulation into rho.
    // With OpenMP: each thread accumulates into its own flat buffer; after
    // the parallel region we sum the buffers into rho in a deterministic
    // cell-major + thread-index-major order.
#ifdef _OPENMP
    const int T = omp_get_max_threads();
    std::vector<std::vector<double>> per_thread_rho(T);
    #pragma omp parallel
    {
        const int t = omp_get_thread_num();
        per_thread_rho[t].assign(cells, 0.0);
        double* my = per_thread_rho[t].data();
        // schedule(static): the particle->thread partition must be a
        // pure function of (N, thread count) for the per-thread partial
        // sums to be reproducible -- with a dynamic/runtime schedule the
        // partition (and thus the FP summation order) could vary run to
        // run.  Previously this rested on the implementation-default
        // schedule; now it is contractual.
        #pragma omp for schedule(static)
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
            my[s0]             += charge * wx0 * wy0 * wz0;
            my[s1]             += charge * wx1 * wy0 * wz0;
            my[s2]             += charge * wx0 * wy1 * wz0;
            my[s3]             += charge * wx1 * wy1 * wz0;
            my[s0 + 1]         += charge * wx0 * wy0 * wz1;
            my[s1 + 1]         += charge * wx1 * wy0 * wz1;
            my[s2 + 1]         += charge * wx0 * wy1 * wz1;
            my[s3 + 1]         += charge * wx1 * wy1 * wz1;
        }
    }
    // Deterministic reduce: for each cell, sum thread 0..T-1 in order.
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                const size_t idx = (static_cast<size_t>(i) * ny + j) * nz + k;
                double s = 0.0;
                for (int t = 0; t < T; t++) s += per_thread_rho[t][idx];
                rho(i, j, k) = s;
            }
#else
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
        rho(ix,   iy,   iz)   += charge * wx0 * wy0 * wz0;
        rho(ix+1, iy,   iz)   += charge * wx1 * wy0 * wz0;
        rho(ix,   iy+1, iz)   += charge * wx0 * wy1 * wz0;
        rho(ix+1, iy+1, iz)   += charge * wx1 * wy1 * wz0;
        rho(ix,   iy,   iz+1) += charge * wx0 * wy0 * wz1;
        rho(ix+1, iy,   iz+1) += charge * wx1 * wy0 * wz1;
        rho(ix,   iy+1, iz+1) += charge * wx0 * wy1 * wz1;
        rho(ix+1, iy+1, iz+1) += charge * wx1 * wy1 * wz1;
    }
#endif

    // Normalise by cell volume → density.
    if (cell_vol > 0) {
#ifdef _OPENMP
        #pragma omp parallel for collapse(3)
#endif
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    rho(i, j, k) /= cell_vol;
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
    // schedule(static) is contractual here too (symmetry with the
    // deposit loop); the gather is bit-identical under any schedule
    // (disjoint writes, shared reads), so this only pins the contract.
    #pragma omp parallel for schedule(static)
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
#ifdef _OPENMP
    m.attr("_openmp_enabled") = true;
    // Contractual since 2026-07: the deposit/gather loops carry an
    // explicit schedule(static) clause.  Provenance reads this attr —
    // a binary built from older source lacks it, so the HDF5 record
    // honestly reports "unknown" instead of asserting the contract.
    m.attr("_omp_schedule") = "static";
#else
    m.attr("_openmp_enabled") = false;
#endif
}
