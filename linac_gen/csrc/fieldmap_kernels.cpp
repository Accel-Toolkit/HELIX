// Fused multi-component trilinear field-map sampler.
//
// Replaces the per-component scipy RegularGridInterpolator calls in
// FieldMap3D field sampling: ONE pass computes the cell indices and
// weights per particle and applies them to all M field components of a
// channel (scipy recomputes them per component, plus generic-ND
// overhead — measured ~800 ns/particle vs ~tens of ns here).
//
// BIT-IDENTITY CONTRACT (pinned by tests/elements/test_fieldmap_kernels.py):
// this kernel reproduces scipy 1.x RGI(method="linear", bounds_error=False,
// fill_value=0.0) BITWISE. That requires mirroring its arithmetic exactly:
//   * index: scipy's compiled find_indices uses RIGHT-side search
//     semantics — i = searchsorted_right(grid, v) - 1, clipped to
//     [0, n-2] (an exact interior node v == g[k] lands in cell k with
//     norm 0, NOT cell k-1 with norm 1; identical for finite fields but
//     it decides which neighbor multiplies the zero weight, which
//     matters when a map value is non-finite: 0*inf = NaN)
//   * norm distance: y = (v - g[i]) / (g[i+1] - g[i])
//   * corner order: itertools.product over (low, high) per axis
//     -> (x lo/hi outermost, z lo/hi innermost)
//   * weight product order: ((1 * wx) * wy) * wz
//   * accumulation: out = out + f * w in corner order, starting from 0
//   * out of bounds on ANY axis -> exactly 0.0, EXCEPT that a NaN in any
//     coordinate wins over out-of-bounds: scipy fills OOB first and then
//     overwrites rows with any-NaN coordinates with NaN
// The build disables FP contraction (-ffp-contract=off in CMakeLists) so
// the compiler cannot re-round f*w + out into an FMA.
//
// Threading: OpenMP over particles. Each particle's result is computed
// independently (no cross-particle reduction), so the output is bitwise
// identical for any thread count.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

// scipy find_indices: right-side search, i in [0, n-2].
// std::upper_bound = first element GREATER than v == searchsorted right;
// an exact node v == g[k] therefore selects cell k (norm distance 0).
inline long cell_index(const double* g, long n, double v) {
    const double* it = std::upper_bound(g, g + n, v);
    long i = static_cast<long>(it - g) - 1;
    if (i < 0) i = 0;
    if (i > n - 2) i = n - 2;
    return i;
}

}  // namespace

// xs, ys, zs           : (N,) particle coordinates (same units as axes)
// gx, gy, gz           : (nx,), (ny,), (nz,) strictly ascending axes
// fields               : (M, nx, ny, nz) C-contiguous stacked components
// returns              : (M, N) sampled values (0.0 outside the box)
py::array_t<double> interp3_multi(
    py::array_t<double, py::array::c_style | py::array::forcecast> xs,
    py::array_t<double, py::array::c_style | py::array::forcecast> ys,
    py::array_t<double, py::array::c_style | py::array::forcecast> zs,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gy,
    py::array_t<double, py::array::c_style | py::array::forcecast> gz,
    py::array_t<double, py::array::c_style | py::array::forcecast> fields)
{
    auto x = xs.unchecked<1>();
    auto y = ys.unchecked<1>();
    auto z = zs.unchecked<1>();
    auto ax = gx.unchecked<1>();
    auto ay = gy.unchecked<1>();
    auto az = gz.unchecked<1>();
    auto f = fields.unchecked<4>();

    const long N = x.shape(0);
    const long M = f.shape(0);
    const long nx = ax.shape(0), ny = ay.shape(0), nz = az.shape(0);
    if (f.shape(1) != nx || f.shape(2) != ny || f.shape(3) != nz)
        throw std::invalid_argument("fields shape does not match axes");
    if (nx < 2 || ny < 2 || nz < 2)
        throw std::invalid_argument("each axis needs >= 2 points");

    py::array_t<double> out({M, N});
    auto o = out.mutable_unchecked<2>();

    const double* gxp = ax.data(0);
    const double* gyp = ay.data(0);
    const double* gzp = az.data(0);
    const double x_lo = gxp[0], x_hi = gxp[nx - 1];
    const double y_lo = gyp[0], y_hi = gyp[ny - 1];
    const double z_lo = gzp[0], z_hi = gzp[nz - 1];

    py::gil_scoped_release release;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long p = 0; p < N; ++p) {
        const double xp = x(p), yp = y(p), zp = z(p);
        // scipy order: OOB rows get fill_value FIRST, then rows with any
        // NaN coordinate are overwritten with NaN — so NaN wins over OOB.
        if (std::isnan(xp) || std::isnan(yp) || std::isnan(zp)) {
            const double qnan = std::numeric_limits<double>::quiet_NaN();
            for (long m = 0; m < M; ++m) o(m, p) = qnan;
            continue;
        }
        if (xp < x_lo || xp > x_hi || yp < y_lo || yp > y_hi
            || zp < z_lo || zp > z_hi) {
            for (long m = 0; m < M; ++m) o(m, p) = 0.0;
            continue;
        }
        const long ix = cell_index(gxp, nx, xp);
        const long iy = cell_index(gyp, ny, yp);
        const long iz = cell_index(gzp, nz, zp);
        const double yx = (xp - gxp[ix]) / (gxp[ix + 1] - gxp[ix]);
        const double yy = (yp - gyp[iy]) / (gyp[iy + 1] - gyp[iy]);
        const double yz = (zp - gzp[iz]) / (gzp[iz + 1] - gzp[iz]);

        // corner weights in scipy's product order (x outer .. z inner),
        // each built as ((1 * wx) * wy) * wz
        const double wx0 = 1.0 - yx, wx1 = yx;
        const double wy0 = 1.0 - yy, wy1 = yy;
        const double wz0 = 1.0 - yz, wz1 = yz;
        double w[8];
        w[0] = ((1.0 * wx0) * wy0) * wz0;
        w[1] = ((1.0 * wx0) * wy0) * wz1;
        w[2] = ((1.0 * wx0) * wy1) * wz0;
        w[3] = ((1.0 * wx0) * wy1) * wz1;
        w[4] = ((1.0 * wx1) * wy0) * wz0;
        w[5] = ((1.0 * wx1) * wy0) * wz1;
        w[6] = ((1.0 * wx1) * wy1) * wz0;
        w[7] = ((1.0 * wx1) * wy1) * wz1;

        for (long m = 0; m < M; ++m) {
            // corner values in the same order
            const double f000 = f(m, ix,     iy,     iz);
            const double f001 = f(m, ix,     iy,     iz + 1);
            const double f010 = f(m, ix,     iy + 1, iz);
            const double f011 = f(m, ix,     iy + 1, iz + 1);
            const double f100 = f(m, ix + 1, iy,     iz);
            const double f101 = f(m, ix + 1, iy,     iz + 1);
            const double f110 = f(m, ix + 1, iy + 1, iz);
            const double f111 = f(m, ix + 1, iy + 1, iz + 1);
            // accumulate in scipy's corner order starting from 0
            double v = 0.0;
            v = v + f000 * w[0];
            v = v + f001 * w[1];
            v = v + f010 * w[2];
            v = v + f011 * w[3];
            v = v + f100 * w[4];
            v = v + f101 * w[5];
            v = v + f110 * w[6];
            v = v + f111 * w[7];
            o(m, p) = v;
        }
    }
    return out;
}

PYBIND11_MODULE(_fieldmap_kernels, m) {
    m.doc() = "Fused trilinear field-map sampling (bitwise-mirrors scipy "
              "RGI linear with fill_value=0)";
    m.def("interp3_multi", &interp3_multi,
          py::arg("xs"), py::arg("ys"), py::arg("zs"),
          py::arg("gx"), py::arg("gy"), py::arg("gz"), py::arg("fields"));
}
