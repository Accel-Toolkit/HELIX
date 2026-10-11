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
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

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

// One map: strictly ascending axes and M stacked C-contiguous components
// of shape (nx, ny, nz).
struct Map3 {
    const double* gx; const double* gy; const double* gz;
    std::int64_t nx, ny, nz;  // 64-bit: flat offsets below never overflow
    const double* f;          // (M, nx, ny, nz)
    std::int64_t M;
};

// The z part of one sample: NaN flag, out-of-box flag, cell and weights.
struct ZCell { bool nan; bool out; std::int64_t iz; double wz0, wz1; };

inline ZCell z_cell(const Map3& m, double zp) {
    ZCell c{std::isnan(zp), false, 0, 0.0, 0.0};
    if (c.nan) return c;
    if (zp < m.gz[0] || zp > m.gz[m.nz - 1]) { c.out = true; return c; }
    c.iz = cell_index(m.gz, static_cast<long>(m.nz), zp);
    const double yz = (zp - m.gz[c.iz]) / (m.gz[c.iz + 1] - m.gz[c.iz]);
    c.wz0 = 1.0 - yz;
    c.wz1 = yz;
    return c;
}

// Sample the M components of `m` at (xp, yp, z of `zc`) into out[0..M-1],
// with the arithmetic of scipy RGI(linear, bounds_error=False,
// fill_value=0) documented above.  NaN in any coordinate wins over out
// of the box.
inline void sample_xy(const Map3& m, const ZCell& zc, double xp, double yp,
                      double* out) {
    if (std::isnan(xp) || std::isnan(yp) || zc.nan) {
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        for (std::int64_t c = 0; c < m.M; ++c) out[c] = qnan;
        return;
    }
    if (xp < m.gx[0] || xp > m.gx[m.nx - 1] || yp < m.gy[0]
        || yp > m.gy[m.ny - 1] || zc.out) {
        for (std::int64_t c = 0; c < m.M; ++c) out[c] = 0.0;
        return;
    }
    const std::int64_t ix = cell_index(m.gx, static_cast<long>(m.nx), xp);
    const std::int64_t iy = cell_index(m.gy, static_cast<long>(m.ny), yp);
    const std::int64_t iz = zc.iz;
    const double yx = (xp - m.gx[ix]) / (m.gx[ix + 1] - m.gx[ix]);
    const double yy = (yp - m.gy[iy]) / (m.gy[iy + 1] - m.gy[iy]);

    // corner weights in scipy's product order (x outer .. z inner),
    // each built as ((1 * wx) * wy) * wz
    const double wx0 = 1.0 - yx, wx1 = yx;
    const double wy0 = 1.0 - yy, wy1 = yy;
    const double wz0 = zc.wz0, wz1 = zc.wz1;
    double w[8];
    w[0] = ((1.0 * wx0) * wy0) * wz0;
    w[1] = ((1.0 * wx0) * wy0) * wz1;
    w[2] = ((1.0 * wx0) * wy1) * wz0;
    w[3] = ((1.0 * wx0) * wy1) * wz1;
    w[4] = ((1.0 * wx1) * wy0) * wz0;
    w[5] = ((1.0 * wx1) * wy0) * wz1;
    w[6] = ((1.0 * wx1) * wy1) * wz0;
    w[7] = ((1.0 * wx1) * wy1) * wz1;

    const std::int64_t sz = m.nz, sy = m.ny * m.nz, sm = m.nx * m.ny * m.nz;
    for (std::int64_t c = 0; c < m.M; ++c) {
        const double* f = m.f + c * sm;
        // corner values in the same order
        const double f000 = f[ix * sy + iy * sz + iz];
        const double f001 = f[ix * sy + iy * sz + iz + 1];
        const double f010 = f[ix * sy + (iy + 1) * sz + iz];
        const double f011 = f[ix * sy + (iy + 1) * sz + iz + 1];
        const double f100 = f[(ix + 1) * sy + iy * sz + iz];
        const double f101 = f[(ix + 1) * sy + iy * sz + iz + 1];
        const double f110 = f[(ix + 1) * sy + (iy + 1) * sz + iz];
        const double f111 = f[(ix + 1) * sy + (iy + 1) * sz + iz + 1];
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
        out[c] = v;
    }
}

// One full sample at (xp, yp, zp).
inline void sample_point(const Map3& m, double xp, double yp, double zp,
                         double* out) {
    sample_xy(m, z_cell(m, zp), xp, yp, out);
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
    if (y.shape(0) != N || z.shape(0) != N)
        throw std::invalid_argument("xs, ys and zs must have the same length");
    const long nx = ax.shape(0), ny = ay.shape(0), nz = az.shape(0);
    if (f.shape(1) != nx || f.shape(2) != ny || f.shape(3) != nz)
        throw std::invalid_argument("fields shape does not match axes");
    if (nx < 2 || ny < 2 || nz < 2)
        throw std::invalid_argument("each axis needs >= 2 points");

    py::array_t<double> out({M, N});
    auto o = out.mutable_unchecked<2>();
    const Map3 map{ax.data(0), ay.data(0), az.data(0), nx, ny, nz,
                   f.data(0, 0, 0, 0), M};

    py::gil_scoped_release release;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long p = 0; p < N; ++p) {
        double v[64];
        double* dst = (M <= 64) ? v : nullptr;
        if (dst == nullptr) {
            // more components than the stack buffer: sample one at a time
            for (long c = 0; c < M; ++c) {
                const Map3 one{map.gx, map.gy, map.gz, nx, ny, nz,
                               map.f + c * nx * ny * nz, 1};
                double vc;
                sample_point(one, x(p), y(p), z(p), &vc);
                o(c, p) = vc;
            }
            continue;
        }
        sample_point(map, x(p), y(p), z(p), dst);
        for (long c = 0; c < M; ++c) o(c, p) = dst[c];
    }
    return out;
}

// ---------------------------------------------------------------------------
// One kick-then-drift step of FieldMap3D._track_kd for the particles in
// `idx`, in place.  BIT-IDENTICAL to the numpy path: the same expressions
// in the same order, each particle on its own (no cross-particle sums);
// identical for every non-NaN value (a NaN stays NaN, but which operand's
// NaN payload or sign survives a NaN*NaN product is up to the compiler):
//   fields   F = sample(channel map at x, y, z_pos)   (0 for absent comps)
//            s = (F * amp) * phasor                    (phasor 1: static)
//            E += s (electric channels) / B += s (magnetic), from 0
//   energy   dW = (charge * Ez) * ds_m;  W = W + (dW - dW_ref)
//   damping  d = 1 / (1 + dW / damp_denom);  x' = x' * d;  y' = y' * d
//   slip     phi = phi + slip * W               (W after the energy kick)
//   kick     x' = x' + charge * (fE*Ex + fB*(y'0*1e-3*Bz - By)) * 1e3
//            y' = y' + charge * (fE*Ey + fB*(Bx - x'0*1e-3*Bz)) * 1e3
//                                       (x'0, y'0: angles before the step)
//   drift    x = x + x' * ds_m;  y = y + y' * ds_m
// The phasors (cos / sin of each particle's phase) come in from numpy, so
// no transcendental function is evaluated here.  Built without FP
// contraction (-ffp-contract=off / MSVC /fp:precise), like interp3_multi.
//
// particles : (N, 6) float64 C-contiguous, writeable -- updated in place
// idx       : (n,) int64 strictly increasing indices of the particles
// channels  : list of (gx, gy, gz, fields (M,nx,ny,nz), slots (3,) int64
//             -- the row of Fx, Fy, Fz in `fields` or -1 --, amp,
//             is_electric, phasor (n,) float64 or None for static)
void kd_step(py::array particles, py::array idx_in,
             double z_pos, py::list channels,
             double charge, double ds_m, double dW_ref,
             bool do_damp, double damp_denom,
             bool do_slip, double slip,
             bool do_kick, double factor_E, double factor_B)
{
    if (particles.ndim() != 2 || particles.shape(1) != 6
        || !py::isinstance<py::array_t<double>>(particles)
        || !(particles.flags() & py::array::c_style)
        || !particles.writeable())
        throw std::invalid_argument(
            "particles must be a writeable C-contiguous float64 (N, 6) array");
    double* P = static_cast<double*>(particles.mutable_data());
    const long N = static_cast<long>(particles.shape(0));
    // indices: int64, 1-D, strictly increasing (so each particle once --
    // the threads write disjoint rows)
    if (!py::isinstance<py::array_t<std::int64_t>>(idx_in)
        || idx_in.ndim() != 1)
        throw std::invalid_argument("idx must be a 1-D int64 array");
    auto idx = py::array_t<std::int64_t, py::array::c_style>::ensure(idx_in);
    auto ix = idx.unchecked<1>();
    const long n = static_cast<long>(ix.shape(0));

    struct Chan {
        Map3 map; ZCell zc; long slot[3]; double amp; bool electric;
        const double* phasor;          // nullptr: static (phasor 1)
    };
    std::vector<Chan> chans;
    std::vector<py::object> keep;      // holds converted arrays alive
    for (auto item : channels) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(item);
        if (t.size() != 8)
            throw std::invalid_argument("channel tuple needs 8 entries");
        using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
        arr gx = arr::ensure(t[0]), gy = arr::ensure(t[1]),
            gz = arr::ensure(t[2]), fl = arr::ensure(t[3]);
        auto sl = py::array_t<int64_t, py::array::c_style
                              | py::array::forcecast>::ensure(t[4]);
        if (!gx || !gy || !gz || !fl || !sl || gx.ndim() != 1
            || gy.ndim() != 1 || gz.ndim() != 1 || fl.ndim() != 4
            || sl.ndim() != 1 || sl.shape(0) != 3)
            throw std::invalid_argument("malformed channel arrays");
        const long nx = gx.shape(0), ny = gy.shape(0), nz = gz.shape(0);
        const long M = fl.shape(0);
        if (fl.shape(1) != nx || fl.shape(2) != ny || fl.shape(3) != nz
            || nx < 2 || ny < 2 || nz < 2 || M < 1 || M > 3)
            throw std::invalid_argument("fields shape does not match axes");
        Chan c;
        c.map = Map3{gx.data(), gy.data(), gz.data(), nx, ny, nz,
                     fl.data(), M};
        c.zc = z_cell(c.map, z_pos);   // every particle sits at z_pos
        for (const auto* g : {&gx, &gy, &gz}) {
            const double* d = g->data();
            for (py::ssize_t k = 1; k < g->shape(0); ++k)
                if (!(d[k] > d[k - 1]))
                    throw std::invalid_argument(
                        "map axes must be strictly ascending");
        }
        for (int a = 0; a < 3; ++a) {
            c.slot[a] = static_cast<long>(sl.at(a));
            if (c.slot[a] < -1 || c.slot[a] >= M)
                throw std::invalid_argument("component slot out of range");
        }
        c.amp = t[5].cast<double>();
        c.electric = t[6].cast<bool>();
        c.phasor = nullptr;
        if (!t[7].is_none()) {
            arr ph = arr::ensure(t[7]);
            if (!ph || ph.ndim() != 1 || ph.shape(0) != n)
                throw std::invalid_argument("phasor must have shape (n,)");
            c.phasor = ph.data();
            keep.push_back(ph);
        }
        keep.push_back(gx); keep.push_back(gy); keep.push_back(gz);
        keep.push_back(fl);
        chans.push_back(c);
    }
    for (long k = 0; k < n; ++k) {
        if (ix(k) < 0 || ix(k) >= N)
            throw std::invalid_argument("particle index out of range");
        if (k > 0 && ix(k) <= ix(k - 1))
            throw std::invalid_argument(
                "particle indices must be strictly increasing");
    }

    py::gil_scoped_release release;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long k = 0; k < n; ++k) {
        double* p = P + 6 * ix(k);
        const double x = p[0], y = p[2];
        const double xp_rad = p[1] * 1e-3;
        const double yp_rad = p[3] * 1e-3;
        double Ex = 0.0, Ey = 0.0, Ez = 0.0;
        double Bx = 0.0, By = 0.0, Bz = 0.0;
        for (const Chan& c : chans) {
            double v[3];
            sample_xy(c.map, c.zc, x, y, v);
            const double Fx = c.slot[0] >= 0 ? v[c.slot[0]] : 0.0;
            const double Fy = c.slot[1] >= 0 ? v[c.slot[1]] : 0.0;
            const double Fz = c.slot[2] >= 0 ? v[c.slot[2]] : 0.0;
            const double ph = c.phasor ? c.phasor[k] : 1.0;
            const double sx = (Fx * c.amp) * ph;
            const double sy = (Fy * c.amp) * ph;
            const double sz = (Fz * c.amp) * ph;
            if (c.electric) { Ex = Ex + sx; Ey = Ey + sy; Ez = Ez + sz; }
            else            { Bx = Bx + sx; By = By + sy; Bz = Bz + sz; }
        }
        const double dW = (charge * Ez) * ds_m;
        p[5] = p[5] + (dW - dW_ref);
        if (do_damp) {
            const double d = 1.0 / (1.0 + dW / damp_denom);
            p[1] = p[1] * d;
            p[3] = p[3] * d;
        }
        if (do_slip) p[4] = p[4] + slip * p[5];
        if (do_kick) {
            const double dxp = charge * (factor_E * Ex
                                         + factor_B * (yp_rad * Bz - By));
            const double dyp = charge * (factor_E * Ey
                                         + factor_B * (Bx - xp_rad * Bz));
            p[1] = p[1] + dxp * 1e3;
            p[3] = p[3] + dyp * 1e3;
        }
        p[0] = p[0] + p[1] * ds_m;
        p[2] = p[2] + p[3] * ds_m;
    }
}

PYBIND11_MODULE(_fieldmap_kernels, m) {
    m.doc() = "Fused trilinear field-map sampling (bitwise-mirrors scipy "
              "RGI linear with fill_value=0) and the fused KD step";
    m.def("interp3_multi", &interp3_multi,
          py::arg("xs"), py::arg("ys"), py::arg("zs"),
          py::arg("gx"), py::arg("gy"), py::arg("gz"), py::arg("fields"));
    m.def("kd_step", &kd_step,
          py::arg("particles"), py::arg("idx"), py::arg("z_pos"),
          py::arg("channels"), py::arg("charge"), py::arg("ds_m"),
          py::arg("dW_ref"), py::arg("do_damp"), py::arg("damp_denom"),
          py::arg("do_slip"), py::arg("slip"), py::arg("do_kick"),
          py::arg("factor_E"), py::arg("factor_B"));
#ifdef _OPENMP
    m.attr("_openmp_enabled") = true;
#else
    m.attr("_openmp_enabled") = false;
#endif
}
