"""Tests for the TraceWin-compatible output writers.

Coverage:

* ``write_partran_out`` emits the documented column count and header per
  PDF p. 43-44 (Partran/Toutatis output).
* ``write_envelope_txt`` produces the exact 26-column layout used in the
  ``Tracewin_code/*.txt`` reference fixtures.
* Number formatting matches TraceWin's ``%+.6e`` with 3-digit exponents.
* Round-trip through ``np.loadtxt`` recovers position / σ_x / σ_y to
  better than 1 ppm.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.io.tracewin_outputs import (
    _fmt,
    write_envelope_txt,
    write_partran_out,
)


REPO = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------- #
def _fake_envelope_results(n: int = 5) -> SimpleNamespace:
    """Hand-rolled minimal results object — avoids the full solver run.

    ``s`` is in mm to match the internal solver units; the writers
    convert to metres on emit.
    """
    return SimpleNamespace(
        s            = list(np.linspace(0.0, 1000.0, n)),  # mm → 0..1.0 m
        sigma_x      = [1.0 + 0.1 * i for i in range(n)],   # mm
        sigma_y      = [0.5 + 0.05 * i for i in range(n)],
        sigma_phi    = [3.0 + 0.2 * i for i in range(n)],   # deg
        sigma_w      = [0.01 * (i + 1) for i in range(n)],  # MeV
        emit_x       = [0.25] * n,
        emit_y       = [0.25] * n,
        emit_z       = [0.30] * n,
        emit_z_mmmrad= [0.10] * n,
        beta_x       = [1.5] * n,
        beta_y       = [1.5] * n,
        alpha_x      = [0.0] * n,
        alpha_y      = [0.0] * n,
        ref_w_kin    = [3.0 + 0.01 * i for i in range(n)],
        ref_phi_s    = [0.0] * n,
        ref_beta     = [0.08] * n,
        ref_gamma    = [1.003] * n,
        ref_bg       = [0.08] * n,
        element_names= ["INPUT"] + [f"E{j}" for j in range(n - 1)],
        transmission = [100.0] * n,
        centroid     = [np.zeros(6) for _ in range(n)],
        halo_x       = [0.1] * n,
        halo_y       = [0.1] * n,
        x_max        = [0.005] * n,
        y_max        = [0.005] * n,
        emit_4d      = [0.0625] * n,
        sigma_matrix = [np.eye(6) for _ in range(n)],
        current_mA   = 0.0,
        continuous   = False,
    )


def _beam_cfg(freq_MHz: float = 352.21, current: float = 0.0,
              energy: float = 3.0, n_particles: int = 1000) -> SimpleNamespace:
    return SimpleNamespace(
        frequency=freq_MHz, current=current,
        energy=energy, n_particles=n_particles,
    )


# --------------------------------------------------------------------- #
class TestNumberFormat:
    def test_three_digit_exponent_padding(self):
        # Python default is 2-digit exponent; TraceWin wants 3.
        assert _fmt(1.234e0)  == "+1.234000e+000"
        assert _fmt(-2.5e3)   == "-2.500000e+003"
        assert _fmt(0.0)      == "+0.000000e+000"

    def test_six_significand_digits(self):
        s = _fmt(np.pi)
        # mantissa is exactly six fractional digits.
        mantissa = s.split("e")[0]
        assert "." in mantissa
        assert len(mantissa.split(".")[1]) == 6


# --------------------------------------------------------------------- #
class TestEnvelopeTxt:
    """The 26-column save-data format used in Tracewin_code/*.txt."""

    def test_header_matches_reference_fixture_verbatim(self, tmp_path):
        out = write_envelope_txt(_fake_envelope_results(), _beam_cfg(),
                                 tmp_path / "env.txt")
        first = out.read_text().splitlines()[0]
        # Header taken directly from Tracewin_code/MEBT_spacechargeenvelope_envelope.txt
        expected = (
            "position\tgam-1"
            "\tcentroid position(x,x',y,y',z,dp/p,z',phase,time,energy)"
            "\trms_size(x,x',y,y',z,dp/p,z',phase,time,energy)"
            "\t(dispX,dispY,betX,betY)"
            "\tunit(m,rad,deg,s,MeV)"
        )
        assert first == expected

    def test_data_rows_have_26_numeric_columns(self, tmp_path):
        n = 7
        out = write_envelope_txt(_fake_envelope_results(n), _beam_cfg(),
                                 tmp_path / "env.txt")
        # Header (1) + blank (1) + n data rows
        rows = [l for l in out.read_text().splitlines() if "\t" in l
                and not l.startswith("position")]
        assert len(rows) == n
        for r in rows:
            cols = r.split("\t")
            assert len(cols) == 26, (
                f"expected 26 columns, got {len(cols)}: {r[:80]}…"
            )

    def test_position_round_trips_via_loadtxt(self, tmp_path):
        res = _fake_envelope_results(8)
        out = write_envelope_txt(res, _beam_cfg(), tmp_path / "env.txt")
        # np.loadtxt skipping the 2 header lines (header + blank)
        data = np.loadtxt(out, skiprows=2, delimiter="\t")
        assert data.shape == (8, 26)
        # Internal `s` is in mm; the writer emits metres.
        expected_m = np.asarray(res.s) * 1e-3
        np.testing.assert_allclose(data[:, 0], expected_m, rtol=1e-6)

    def test_sigma_x_in_metres_not_millimetres(self, tmp_path):
        # Internally σ_x is stored in mm; TraceWin's column is m.  Verify
        # the writer does the conversion (column 13 is rms x).
        res = _fake_envelope_results(3)
        res.sigma_x = [1.0, 2.0, 3.0]   # mm
        out = write_envelope_txt(res, _beam_cfg(), tmp_path / "env.txt")
        data = np.loadtxt(out, skiprows=2, delimiter="\t")
        # Column index 12 = first rms column (after position, gam-1, 10
        # centroid).  rms ordering: x, x', y, y', z, dp/p, z', φ, t, W.
        np.testing.assert_allclose(data[:, 12], [1e-3, 2e-3, 3e-3], rtol=1e-9)

    def test_centroid_columns_zero_without_first_moment(self, tmp_path):
        # A results object with no ``centroid`` field (or an on-axis
        # one) keeps the block at zero — legacy behaviour.
        res = _fake_envelope_results(4)
        out = write_envelope_txt(res, _beam_cfg(), tmp_path / "env.txt")
        data = np.loadtxt(out, skiprows=2, delimiter="\t")
        # Centroid block: cols 2-11 (0-indexed)
        np.testing.assert_array_equal(data[:, 2:12], 0.0)

    def test_centroid_columns_emit_real_first_moment(self, tmp_path):
        # Envelope results now carry a centroid — the TW ENV block must
        # emit it with the rms block's unit conventions (mm→m, mrad→rad,
        # Δφ→z via −βλΔφ/360, dp/p via ΔW/(β²γmc²)).
        res = _fake_envelope_results(3)
        res.centroid = [np.array([1.0, 2.0, -3.0, 0.5, 10.0, 0.02])] * 3
        res.mass_mev = 938.272            # dp/p needs the rest mass
        out = write_envelope_txt(res, _beam_cfg(), tmp_path / "env.txt")
        data = np.loadtxt(out, skiprows=2, delimiter="\t")
        np.testing.assert_allclose(data[:, 2], 1.0e-3, rtol=1e-9)   # x m
        np.testing.assert_allclose(data[:, 3], 2.0e-3, rtol=1e-9)   # x' rad
        np.testing.assert_allclose(data[:, 4], -3.0e-3, rtol=1e-9)  # y m
        np.testing.assert_allclose(data[:, 9], 10.0, rtol=1e-9)     # φ deg
        assert np.all(data[:, 6] < 0)          # z = −βλΔφ/360 with Δφ>0
        assert np.all(data[:, 7] > 0)          # dp/p from ΔW>0

    def test_position_emitted_in_metres_not_millimetres(self, tmp_path):
        # ref.s is mm internally; this fixture uses 0..1000 mm so the
        # output should be 0..1.0 m.
        res = _fake_envelope_results(5)
        out = write_envelope_txt(res, _beam_cfg(), tmp_path / "env.txt")
        data = np.loadtxt(out, skiprows=2, delimiter="\t")
        np.testing.assert_allclose(data[:, 0],
                                   [0.0, 0.25, 0.5, 0.75, 1.0],
                                   rtol=1e-9)


# --------------------------------------------------------------------- #
class TestPartranOut:
    """Per-element schema from PDF p. 43-44."""

    def test_emits_one_data_row_per_step(self, tmp_path):
        n = 6
        out = write_partran_out(_fake_envelope_results(n), lattice=None,
                                beam_cfg=_beam_cfg(),
                                path=tmp_path / "partran1.out")
        rows = [l for l in out.read_text().splitlines()
                if not l.startswith("#") and "\t" in l]
        assert len(rows) == n

    def test_first_row_is_input_with_element_zero(self, tmp_path):
        out = write_partran_out(_fake_envelope_results(3), lattice=None,
                                beam_cfg=_beam_cfg(),
                                path=tmp_path / "partran1.out")
        data_rows = [l for l in out.read_text().splitlines()
                     if not l.startswith("#") and "\t" in l]
        first = data_rows[0].split("\t")
        # Element# is the first column; INPUT → 0.
        assert int(round(float(first[0]))) == 0

    def test_column_count_matches_documented_schema(self, tmp_path):
        # PDF p. 43-44 column tally (each unpacked vector element counted
        # separately):
        #   Element#(1) Position(1) γ-1(1) centroid(6) σx,y,φ(3)
        #   covariances <xx'>,<yy'>,<φW>(3) norm-emit(3) halo(3)
        #   N_alive(1) phase-adv σx,y,z(3) ε99(3)
        #   φ_s(1) W_s(1) Ib(1) Ap(1)
        #   4D-ε²(1) εrr'(1) σ_r(1) Plost(1)
        #   Xmax,Ymax(2) εzpp_n(1) σz(1) zpp(1)
        #   D_h,D_v(2) D'_h,D'_v(2) E6D(1) σxy-cross(4) = 50 columns —
        #   matches a genuine TraceWin partran1.out (50 cols; the old
        #   49-column layout lacked E6D and mislabeled cols 12-14).
        out = write_partran_out(_fake_envelope_results(2), lattice=None,
                                beam_cfg=_beam_cfg(),
                                path=tmp_path / "partran1.out")
        data_rows = [l for l in out.read_text().splitlines()
                     if not l.startswith("#") and "\t" in l]
        cols = data_rows[0].split("\t")
        assert len(cols) == 50, (
            f"expected 50 columns (genuine-TW-audited schema), got {len(cols)}"
        )

    def test_normalised_emit_is_geometric_times_betagamma(self, tmp_path):
        res = _fake_envelope_results(3)
        res.emit_x = [0.25] * 3        # mm.mrad geometric
        res.ref_bg = [2.0] * 3         # so norm = 0.5
        out = write_partran_out(res, lattice=None, beam_cfg=_beam_cfg(),
                                path=tmp_path / "partran1.out")
        data_rows = [l for l in out.read_text().splitlines()
                     if not l.startswith("#") and "\t" in l]
        cols = data_rows[0].split("\t")
        # Column 12 is the covariance <xx'> per the genuine TW schema
        # (the fixture has no σ-matrix, so it is 0 here — NOT the
        # geometric emittance the old writer put there).
        assert float(cols[12]) == pytest.approx(0.0, abs=1e-12)
        # normalised ε_xx' is index 15
        assert float(cols[15]) == pytest.approx(0.50, rel=1e-9)

    def test_position_emitted_in_metres_not_millimetres(self, tmp_path):
        # Same conversion check for the partran writer.
        res = _fake_envelope_results(4)   # s = 0..1000 mm
        out = write_partran_out(res, lattice=None, beam_cfg=_beam_cfg(),
                                path=tmp_path / "partran1.out")
        rows = [l for l in out.read_text().splitlines()
                if not l.startswith("#") and "\t" in l]
        positions = [float(r.split("\t")[1]) for r in rows]   # col 1
        np.testing.assert_allclose(positions,
                                   [0.0, 1/3, 2/3, 1.0], rtol=1e-6)

    def test_beam_current_propagates_from_beam_cfg(self, tmp_path):
        out = write_partran_out(_fake_envelope_results(2), lattice=None,
                                beam_cfg=_beam_cfg(current=12.5),
                                path=tmp_path / "partran1.out")
        data_rows = [l for l in out.read_text().splitlines()
                     if not l.startswith("#") and "\t" in l]
        cols = data_rows[0].split("\t")
        # Current column (Ibeam) — 0-based index 30 per the 49-column schema.
        assert float(cols[30]) == pytest.approx(12.5, rel=1e-9)


# --------------------------------------------------------------------- #
class TestRoundTripWithRealEnvelope:
    """End-to-end: run the actual solver on a tiny lattice and confirm
    the writer produces a parseable file."""

    def test_envelope_writer_consumes_real_solver_output(self, tmp_path):
        from linac_gen.core.config import BeamConfig
        from linac_gen.core.lattice import Lattice
        from linac_gen.elements.drift import Drift
        from linac_gen.distributions.factory import create_beam
        from linac_gen.tracking.envelope import EnvelopeSolver

        cfg = BeamConfig(
            species="proton", energy=3.0, frequency=352.21,
            current=0.0, duty_cycle=100.0,
            n_particles=10, distribution="waterbag", cutoff=3.0,
            emit_nx=0.25, alpha_x=0.0, beta_x=1.5,
            emit_ny=0.25, alpha_y=0.0, beta_y=1.5,
            emit_z=0.30, alpha_z=0.0, beta_z=10.0,
        )
        lat = Lattice()
        lat.add(Drift(name="D1", length=100.0, aperture=20.0))
        beam = create_beam(cfg, seed=42)
        bg = max(beam.ref.bg, 1e-9)
        initial = dict(
            alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
            alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
            alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
        )
        results = EnvelopeSolver(lat, beam.ref, initial,
                                 current=cfg.current).run()
        env_path = write_envelope_txt(results, cfg, tmp_path / "env.txt")
        par_path = write_partran_out(results, lat, cfg,
                                     tmp_path / "partran1.out")
        # Both files exist and are non-empty.
        assert env_path.stat().st_size > 0
        assert par_path.stat().st_size > 0
        # Envelope file: exactly 26 columns per data row.
        env_rows = [l for l in env_path.read_text().splitlines()
                    if l.strip() and not l.startswith("position")]
        for r in env_rows:
            assert len(r.split("\t")) == 26
        # Partran file: 50 columns per data row (TW-audited schema).
        par_rows = [l for l in par_path.read_text().splitlines()
                    if "\t" in l and not l.startswith("#")]
        for r in par_rows:
            assert len(r.split("\t")) == 50


def test_envelope_txt_dpp_column_formula(tmp_path):
    """dp/p (rms col index 17) must be dW/(β²·γ·mc²) — the E·dE = c²·p·dp
    identity.  Regression: the old code divided by (βγ)² with no mass at all
    (dimensionally wrong, off by γ·mc²); the fixture-based tests only
    exercised the mass-less → 0.0 fallback, so BOTH regimes are pinned."""
    res = _fake_envelope_results()
    res.mass_mev = 939.2940880                      # physical H⁻ ion
    out = write_envelope_txt(res, _beam_cfg(), tmp_path / "env_dpp.txt")
    rows = [l.split("\t") for l in out.read_text().splitlines()
            if "\t" in l and not l.startswith("position")]
    data = np.array([[float(x) for x in r] for r in rows])
    i = 2                                            # an interior row
    bg = res.ref_bg[i]
    gamma = np.sqrt(1.0 + bg * bg)
    beta = bg / gamma
    expected = res.sigma_w[i] / (beta * beta * gamma * res.mass_mev)
    # file stores 9 significant digits
    assert data[i, 17] == pytest.approx(expected, rel=1e-6)
    # mass-less regime: falls back to 0.0, never a wrong number
    res2 = _fake_envelope_results()
    out2 = write_envelope_txt(res2, _beam_cfg(), tmp_path / "env_dpp0.txt")
    rows2 = [l.split("\t") for l in out2.read_text().splitlines()
             if "\t" in l and not l.startswith("position")]
    data2 = np.array([[float(x) for x in r] for r in rows2])
    assert np.all(data2[:, 17] == 0.0)
