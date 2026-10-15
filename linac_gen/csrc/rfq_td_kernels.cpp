// Time-domain RFQ solver kernels (linac_gen.rfq_td).
//
// field_gather: the static electrode field E = -grad(phi) at particle
// positions from the stored near-axis potential blocks of an RFQ field map
// (linac_gen/rfq_td/fieldmap.py).  Tricubic Catmull-Rom interpolation of
// phi on the quadrant grid x, y = 0, h, 2h ..., one uniform block per z
// window; the gradient is taken analytically from the interpolant; mirror
// symmetry (phi even in x and y) supplies the nodes at x, y < 0.
//
// The arithmetic mirrors RfqFieldMap.field (the numpy reference) term by
// term -- same weights, same loop order over (a, c, e), same products --
// and the build disables FP contraction, so the two agree to rounding
// (tests/rfq_td/test_fieldmap.py).
//
// push: one leapfrog step (half kick, drift, gather, half kick) of every
// active particle -- the numpy step of linac_gen/rfq_td/tracker.py and
// field_gather fused into one pass, operation for operation, so the two
// agree bit for bit (tests/rfq_td/test_tracker.py).
//
// Threading: OpenMP over particles, each computed on its own, so the result
// is bitwise the same for any thread count.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "helix_threads.h"

namespace py = pybind11;

namespace {

inline void catmull_rom(double t, double w[4], double d[4]) {
    const double t2 = t * t, t3 = t2 * t;
    w[0] = 0.5 * (-t3 + 2.0 * t2 - t);
    w[1] = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0);
    w[2] = 0.5 * (-3.0 * t3 + 4.0 * t2 + t);
    w[3] = 0.5 * (t3 - t2);
    d[0] = 0.5 * (-3.0 * t2 + 4.0 * t - 1.0);
    d[1] = 0.5 * (9.0 * t2 - 10.0 * t);
    d[2] = 0.5 * (-9.0 * t2 + 8.0 * t + 1.0);
    d[3] = 0.5 * (3.0 * t2 - 2.0 * t);
}

using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using larr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using farr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using barr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// A field map's arrays, for the per-particle gather below.
struct MapView {
    double h;
    const double* c0;
    long nb;
    double z_last;
    const double* z0;
    const double* hz;
    const int64_t* nz;
    const int64_t* nx;
    const int64_t* off;
    const float* phi;
};

// E at one point; false (and zero field) outside the stored region.
// Upstream of the first block (before the entrance plate, where particles
// start) the field is zero and the point is not counted as outside; a
// non-finite position is outside, as in the numpy reference.
inline bool gather_one(const MapView& m, double xp, double yp, double zp,
                       double& exo, double& eyo, double& ezo) {
    exo = 0.0; eyo = 0.0; ezo = 0.0;
    if (!std::isfinite(zp)) return false;
    // block: last core0 <= zp (searchsorted right - 1)
    long b = static_cast<long>(std::upper_bound(m.c0, m.c0 + m.nb, zp)
                               - m.c0) - 1;
    if (b < 0) return true;
    if (zp > m.z_last) return false;
    const double ax = std::fabs(xp) / m.h;
    const double ay = std::fabs(yp) / m.h;
    const long nxb = static_cast<long>(m.nx[b]);
    if (!(ax < nxb - 2) || !(ay < nxb - 2)) return false;
    const long ix = static_cast<long>(ax);
    const long iy = static_cast<long>(ay);
    const double tx = ax - ix, ty = ay - iy;
    const double fz = (zp - m.z0[b]) / m.hz[b];
    const long nzb = static_cast<long>(m.nz[b]);
    long iz = static_cast<long>(fz);
    if (iz < 1) iz = 1;
    if (iz > nzb - 3) iz = nzb - 3;
    const double tz = fz - iz;
    double wx[4], dwx[4], wy[4], dwy[4], wz[4], dwz[4];
    catmull_rom(tx, wx, dwx);
    catmull_rom(ty, wy, dwy);
    catmull_rom(tz, wz, dwz);
    double gx = 0.0, gy = 0.0, gz = 0.0;
    const int64_t base = m.off[b];
    for (int a = 0; a < 4; ++a) {
        long ii = ix - 1 + a;
        if (ii < 0) ii = -ii;                  // mirror: -1 -> 1
        for (int c = 0; c < 4; ++c) {
            long jj = iy - 1 + c;
            if (jj < 0) jj = -jj;
            const int64_t row = base + (ii * nxb + jj) * nzb + iz - 1;
            const double wxy = wx[a] * wy[c];
            const double dxy = dwx[a] * wy[c];
            const double ydx = wx[a] * dwy[c];
            for (int e = 0; e < 4; ++e) {
                const double v = static_cast<double>(m.phi[row + e]);
                gx += dxy * wz[e] * v;
                gy += ydx * wz[e] * v;
                gz += wxy * dwz[e] * v;
            }
        }
    }
    exo = -gx / m.h * (xp >= 0.0 ? 1.0 : -1.0);
    eyo = -gy / m.h * (yp >= 0.0 ? 1.0 : -1.0);
    ezo = -gz / m.hz[b];
    return true;
}

MapView map_view(double h, darr& core0, double z_last, darr& z0, darr& hz,
                 larr& nz, larr& nx, larr& off, farr& phi) {
    return MapView{h, core0.data(), static_cast<long>(core0.size()), z_last,
                   z0.data(), hz.data(), nz.data(), nx.data(), off.data(),
                   phi.data()};
}

// Returns the number of active particles outside the stored region (they
// get zero field).
long field_gather(darr x, darr y, darr z, barr active, double h,
                  darr core0, double z_last, darr z0, darr hz, larr nz,
                  larr nx, larr off, farr phi, darr Ex, darr Ey, darr Ez) {
    const long n = static_cast<long>(x.size());
    const MapView m = map_view(h, core0, z_last, z0, hz, nz, nx, off, phi);
    const double* px = x.data();
    const double* py_ = y.data();
    const double* pz = z.data();
    const uint8_t* pa = active.data();
    double* ex = Ex.mutable_data();
    double* ey = Ey.mutable_data();
    double* ez = Ez.mutable_data();
    long n_out = 0;
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static) reduction(+:n_out) \
        num_threads(helix::num_threads())
    for (long p = 0; p < n; ++p) {
        ex[p] = 0.0; ey[p] = 0.0; ez[p] = 0.0;
        if (!pa[p]) continue;
        if (!gather_one(m, px[p], py_[p], pz[p], ex[p], ey[p], ez[p]))
            ++n_out;
    }
    return n_out;
}

// One leapfrog step of every active particle, in place: half kick with the
// field it holds (c1 = dt/2 * sin(w t) * charge factor), drift over dt
// (cdt = c * dt), the field at the new position, half kick with it (c2, at
// t + dt).  The same operations, in the same order, as the numpy step in
// linac_gen/rfq_td/tracker.py followed by field_gather; inactive particles
// are left where they are with zero field.  Returns the active particles
// outside the stored region.
long push(darr X, darr Y, darr Z, darr UX, darr UY, darr UZ, barr active,
          darr EX, darr EY, darr EZ, double c1, double c2, double cdt,
          double h, darr core0, double z_last, darr z0, darr hz, larr nz,
          larr nx, larr off, farr phi) {
    const long n = static_cast<long>(X.size());
    const MapView m = map_view(h, core0, z_last, z0, hz, nz, nx, off, phi);
    double* px = X.mutable_data();
    double* py_ = Y.mutable_data();
    double* pz = Z.mutable_data();
    double* pux = UX.mutable_data();
    double* puy = UY.mutable_data();
    double* puz = UZ.mutable_data();
    const uint8_t* pa = active.data();
    double* ex = EX.mutable_data();
    double* ey = EY.mutable_data();
    double* ez = EZ.mutable_data();
    long n_out = 0;
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static) reduction(+:n_out) \
        num_threads(helix::num_threads())
    for (long p = 0; p < n; ++p) {
        if (!pa[p]) {
            ex[p] = 0.0; ey[p] = 0.0; ez[p] = 0.0;
            continue;
        }
        const double ux = pux[p] + c1 * ex[p];
        const double uy = puy[p] + c1 * ey[p];
        const double uz = puz[p] + c1 * ez[p];
        const double g = std::sqrt(1.0 + ux * ux + uy * uy + uz * uz);
        const double f = cdt / g;
        const double xn = px[p] + ux * f;
        const double yn = py_[p] + uy * f;
        const double zn = pz[p] + uz * f;
        px[p] = xn; py_[p] = yn; pz[p] = zn;
        double fx, fy, fz;
        if (!gather_one(m, xn, yn, zn, fx, fy, fz)) ++n_out;
        ex[p] = fx; ey[p] = fy; ez[p] = fz;
        pux[p] = ux + c2 * fx;
        puy[p] = uy + c2 * fy;
        puz[p] = uz + c2 * fz;
    }
    return n_out;
}

}  // namespace

PYBIND11_MODULE(_rfq_td_kernels, m) {
    m.doc() = "Time-domain RFQ solver kernels (linac_gen.rfq_td)";
    m.def("field_gather", &field_gather,
          "Static electrode field at particles from an RFQ field map",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("active"),
          py::arg("h"), py::arg("core0"), py::arg("z_last"), py::arg("z0"),
          py::arg("hz"), py::arg("nz"), py::arg("nx"), py::arg("off"),
          py::arg("phi"), py::arg("Ex"), py::arg("Ey"), py::arg("Ez"));
    m.def("push", &push,
          "One leapfrog step of the active particles through an RFQ field map",
          py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("UX"),
          py::arg("UY"), py::arg("UZ"), py::arg("active"), py::arg("EX"),
          py::arg("EY"), py::arg("EZ"), py::arg("c1"), py::arg("c2"),
          py::arg("cdt"), py::arg("h"), py::arg("core0"), py::arg("z_last"),
          py::arg("z0"), py::arg("hz"), py::arg("nz"), py::arg("nx"),
          py::arg("off"), py::arg("phi"));
    m.def("set_num_threads", &helix::set_num_threads, py::arg("n"),
          "Thread count for this module's OpenMP loops (0 = OpenMP default)");
    m.def("get_num_threads", &helix::num_threads,
          "Thread count this module's OpenMP loops use");
#ifdef _OPENMP
    m.attr("_openmp_enabled") = true;
#else
    m.attr("_openmp_enabled") = false;
#endif
}
