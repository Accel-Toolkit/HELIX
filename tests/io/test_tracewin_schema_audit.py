"""TraceWin .out schema audit — pinned against a GENUINE partran1.out.

The genuine file (tests/analysis/fixtures/partran1_subset.out, 50
columns) settles several defects of the old writer/reader:

* cols 12-14 are the covariances <xx'>, <yy'>, <φW> — the old writer
  put geometric emittances there and the reader mislabeled real TW
  covariances as ε;
* cols 15-17 are NORMALIZED emittances (ε_geo derives via βγ);
* col 31 is the aperture (the reader used to read col 32 = TW's e4D);
* TW has an E6D column before the σ-cross terms (old layout: 49 cols,
  misaligned tail);
* MP centroids/Xmax are recorded in mm/mrad already — the old ×1e3
  inflated them 1000×;
* kx/ky/kz are the element-average Δμ/Δs (deg/mm) with a 1e-5 sentinel
  on thin rows — the old writer hardwired zeros;
* σ_z must use the per-record RF frequency (FREQ-jump lattices).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.tracewin_outputs import read_partran_out, write_partran_out

_FIXTURE = (Path(__file__).resolve().parents[1]
            / "analysis" / "fixtures" / "partran1_subset.out")


class TestReaderAgainstGenuineFile:
    def test_geometric_emittance_derived_from_normalized(self):
        out = read_partran_out(_FIXTURE)
        # Select the fixture data row at s = 0.1774220 m (the permissive
        # parser also swallows the numeric mc2 header line, shifting raw
        # indices): ε_n = 0.205664 π·mm·mrad at γ-1 = 2.2598e-3 →
        # βγ = 0.067266 → ε_geo = 3.05747 mm·mrad.
        i = int(np.argmin(np.abs(out["s_m"] - 0.177422)))
        assert out["emit_x_mm_mrad"][i] == pytest.approx(3.05747, rel=1e-4)

    def test_aperture_column_is_31_not_32(self):
        out = read_partran_out(_FIXTURE)
        # Fixture Aper = 15 mm; TW's e4D (col 32) is 0.0406 — the old
        # off-by-one read returned that instead.
        i = int(np.argmin(np.abs(out["s_m"] - 0.177422)))
        assert out["aperture_mm"][i] == pytest.approx(15.0, abs=1e-6)

    def test_n_alive(self):
        out = read_partran_out(_FIXTURE)
        i = int(np.argmin(np.abs(out["s_m"] - 0.177422)))
        assert out["n_alive"][i] == 10000


def _results(n, *, with_sigma=True, centroid_mm=None):
    from linac_gen.tracking.envelope import EnvelopeResults
    beta = 2.0
    res = EnvelopeResults(
        s=[100.0 * i for i in range(n)],
        sigma_x=[1.0] * n, sigma_y=[1.0] * n,
        sigma_phi=[5.0] * n, sigma_w=[0.01] * n,
        emit_x=[0.5] * n, emit_y=[0.5] * n,
        emit_z=[0.1] * n, emit_z_mmmrad=[0.2] * n,
        alpha_x=[0.0] * n, beta_x=[beta] * n,
        alpha_y=[0.0] * n, beta_y=[beta] * n,
        ref_w_kin=[2.5] * n, ref_beta=[0.07] * n,
        ref_gamma=[1.0027] * n, ref_frequency=[162.5] * n,
        element_names=["INPUT"] + [f"E{i}" for i in range(1, n)],
        mass_mev=938.272,
    )
    if with_sigma:
        S = np.zeros((6, 6))
        S[0, 0] = S[2, 2] = 4.0
        S[1, 1] = S[3, 3] = 1.0
        S[0, 1] = S[1, 0] = 1.5          # <xx'>
        S[2, 3] = S[3, 2] = -0.75        # <yy'>
        S[4, 4], S[5, 5] = 25.0, 1e-4
        S[4, 5] = S[5, 4] = 0.02         # <φW>
        S[0, 5] = S[5, 0] = 3e-4         # dispersion correlation
        res.sigma_matrix = [S.copy() for _ in range(n)]
    if centroid_mm is not None:
        res.centroid = [np.asarray(centroid_mm, float)] * n
    return res


class TestWriterSchema:
    def test_covariance_columns_from_sigma_matrix(self, tmp_path):
        out = write_partran_out(_results(3), lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        row = [l for l in out.read_text().splitlines()
               if "\t" in l and not l.startswith("#")][1].split("\t")
        assert float(row[12]) == pytest.approx(1.5, rel=1e-9)    # <xx'>
        assert float(row[13]) == pytest.approx(-0.75, rel=1e-9)  # <yy'>
        assert float(row[14]) == pytest.approx(0.02, rel=1e-9)   # <φW>

    def test_e6d_column_present_and_positive(self, tmp_path):
        out = write_partran_out(_results(3), lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        row = [l for l in out.read_text().splitlines()
               if "\t" in l and not l.startswith("#")][1].split("\t")
        assert len(row) == 50
        # E6D is column 45 (after Dh, Dv, Dhp, Dvp — matches the
        # genuine fixture's Hdisp..Vdisp/dz then E6D layout).
        S_det_sqrt = float(row[45])
        assert S_det_sqrt > 0.0

    def test_centroid_and_xmax_not_inflated_1000x(self, tmp_path):
        res = _results(3, centroid_mm=[1.25, -0.5, 0.75, 0.1, 3.0, 0.001])
        res.x_max = [2.5] * 3
        res.y_max = [1.5] * 3
        out = write_partran_out(res, lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        row = [l for l in out.read_text().splitlines()
               if "\t" in l and not l.startswith("#")][1].split("\t")
        assert float(row[3]) == pytest.approx(1.25, rel=1e-9)   # x (mm)
        assert float(row[6]) == pytest.approx(-0.5, rel=1e-9)   # x' (mrad)
        assert float(row[36]) == pytest.approx(2.5, rel=1e-9)   # Xmax (mm)
        assert float(row[37]) == pytest.approx(1.5, rel=1e-9)   # Ymax (mm)

    def test_phase_advance_columns_element_average(self, tmp_path):
        # Constant β = 2 mm/mrad over 100 mm elements:
        # k = (180/π)·1e-3 / β deg/mm.
        out = write_partran_out(_results(4), lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        rows = [l for l in out.read_text().splitlines()
                if "\t" in l and not l.startswith("#")]
        k_expected = (180.0 / math.pi) * 1e-3 / 2.0
        first = rows[0].split("\t")
        # Genuine TW writes 0.0 on the s=0 INPUT row (its 1e-5 sentinel
        # appears only on skipped/thin rows further in).
        assert float(first[22]) == pytest.approx(0.0, abs=1e-12)
        second = rows[1].split("\t")
        assert float(second[22]) == pytest.approx(k_expected, rel=1e-6)
        assert float(second[23]) == pytest.approx(k_expected, rel=1e-6)

    def test_one_row_per_element_with_substep_records(self, tmp_path):
        res = _results(3)
        # Simulate substep recording: interleave interior rows and set
        # the exit mapping (INPUT=0, exits at 2 and 4).
        for attr in ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                     "emit_x", "emit_y", "emit_z", "emit_z_mmmrad",
                     "alpha_x", "beta_x", "alpha_y", "beta_y",
                     "ref_w_kin", "ref_beta", "ref_gamma",
                     "ref_frequency", "element_names", "sigma_matrix"):
            seq = list(getattr(res, attr))
            expanded = [seq[0], seq[1], seq[1], seq[2], seq[2]]
            if attr == "s":
                expanded = [0.0, 50.0, 100.0, 150.0, 200.0]
            setattr(res, attr, expanded)
        res.element_exit_idx = [2, 4]
        out = write_partran_out(res, lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        rows = [l for l in out.read_text().splitlines()
                if "\t" in l and not l.startswith("#")]
        assert len(rows) == 3                       # INPUT + 2 elements
        assert int(float(rows[1].split("\t")[0])) == 1
        assert int(float(rows[2].split("\t")[0])) == 2
        # Positions are the ELEMENT-EXIT rows, not interior ones.
        assert float(rows[1].split("\t")[1]) == pytest.approx(0.1)
        assert float(rows[2].split("\t")[1]) == pytest.approx(0.2)

    def test_sigma_z_uses_per_record_frequency(self, tmp_path):
        res = _results(3)
        res.ref_frequency = [162.5, 162.5, 325.0]
        out = write_partran_out(res, lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        rows = [l for l in out.read_text().splitlines()
                if "\t" in l and not l.startswith("#")]
        sz_low = float(rows[1].split("\t")[39])
        sz_high = float(rows[2].split("\t")[39])
        # Same σ_φ and β, doubled frequency → halved σ_z.
        # rel tolerance bounded by the file's 7-significand formatting.
        assert sz_high == pytest.approx(sz_low / 2.0, rel=1e-6)


class TestValueLevelColumnsAgainstFixture:
    """Value-level semantics proven by internal identities of the genuine
    fixture (post-review fixes):

    * ep (col 17) is the RAW (φ,W) emittance — ep = (360·mc²[GeV]/λ[mm])
      ·ezdp holds to 7 digits, so NO ×βγ;
    * e4D (col 32) = ε_nx·ε_ny (the header's (π.mm.mrad)² is a unit);
    * E6D (col 45) = ε_nx·ε_ny·ε_n,zdp;
    * W0 (col 8) is the centroid ΔW OFFSET (absolute energy lives in
      gam-1, col 2);
    * a numeric parameter line (mc² freq charge current npart) follows
      the header.
    """

    def test_fixture_identities_hold(self):
        # Independent arithmetic pin of the three identities on the
        # genuine row at s = 0.1774220 m.
        raw = [l.split() for l in _FIXTURE.read_text().splitlines()
               if l and not l.startswith("#")]
        data = []
        for r in raw:
            if len(r) < 50:
                continue
            try:                                  # text header rows out
                float(r[1])
            except ValueError:
                continue
            data.append(r)
        row = next(r for r in data
                   if abs(float(r[1]) - 0.177422) < 1e-6)
        ex_n, ey_n, ep = (float(row[15]), float(row[16]), float(row[17]))
        ezdp = float(row[38])
        e4d = float(row[32])
        e6d = float(row[45])
        mc2_GeV = 0.939294308
        lam_mm = 299792458.0 / 162.5e6 * 1e3
        assert ep == pytest.approx(360.0 * mc2_GeV / lam_mm * ezdp,
                                   rel=1e-5)
        assert e4d == pytest.approx(ex_n * ey_n, rel=1e-5)
        assert e6d == pytest.approx(ex_n * ey_n * ezdp, rel=1e-5)
        # W0 is an offset (~1e-6 MeV), not the absolute ~2.12 MeV.
        assert abs(float(row[8])) < 1e-3

    def test_writer_ep_raw_e4d_e6d_w0(self, tmp_path):
        res = _results(3, centroid_mm=[0.1, 0.2, 0.3, 0.4, 0.5, 0.006])
        res.emit_4d = [0.25] * 3
        out = write_partran_out(res, lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        row = [l for l in out.read_text().splitlines()
               if "\t" in l and not l.startswith("#")][1].split("\t")
        bg = 0.07 * 1.0027
        assert float(row[17]) == pytest.approx(0.1, rel=1e-9)     # ep RAW
        assert float(row[32]) == pytest.approx(0.25 * bg * bg,
                                               rel=1e-6)          # e4D
        assert float(row[45]) == pytest.approx(
            (0.5 * bg) * (0.5 * bg) * (0.2 * bg), rel=1e-6)       # E6D
        assert float(row[8]) == pytest.approx(0.006, rel=1e-9)    # ΔW offset
        assert float(row[34]) == pytest.approx(0.0, abs=1e-12)    # kr absent

    def test_writer_emits_parameter_line(self, tmp_path):
        out = write_partran_out(_results(2), lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        lines = [l for l in out.read_text().splitlines()
                 if l and not l.startswith("#")]
        param = lines[0].split()
        assert "\t" not in lines[0] and 3 <= len(param) <= 8
        assert float(param[0]) == pytest.approx(938.272, rel=1e-6)  # mc²
        assert float(param[1]) == pytest.approx(162.5, rel=1e-9)    # f MHz

    def test_reader_genuine_ep_and_absolute_energy(self):
        out = read_partran_out(_FIXTURE)
        i = int(np.argmin(np.abs(out["s_m"] - 0.177422)))
        # ep read RAW (no βγ division): fixture row value 0.06528121.
        assert out["emit_z_deg_MeV"][i] == pytest.approx(0.0652812,
                                                         rel=1e-5)
        # Absolute energy reconstructed from gam-1 × mc² (parameter
        # line), NOT read from the ΔW-offset column 8.
        assert out["ref_W_MeV"][i] == pytest.approx(2.1226, rel=1e-3)

    def test_reader_skips_parameter_line_as_data(self):
        out = read_partran_out(_FIXTURE)
        # The old permissive parser swallowed the 5-token parameter line
        # as a bogus s = 162.5 m data row.
        assert np.all(out["s_m"] < 100.0)
        assert np.all(np.isfinite(out["s_m"]))

    def test_reader_roundtrips_writer(self, tmp_path):
        res = _results(3)
        out = write_partran_out(res, lattice=None, beam_cfg=None,
                                path=tmp_path / "p.out")
        back = read_partran_out(out)
        assert back["s_m"] == pytest.approx([0.0, 0.1, 0.2], rel=1e-9)
        # ep round-trips raw; ref_W comes back via gam-1·mc².
        assert back["emit_z_deg_MeV"] == pytest.approx([0.1] * 3, rel=1e-9)
        # gam-1 in the file comes from βγ (the synthetic β, γ are not
        # exactly consistent): γ' = √(1+(βγ)²).
        bg = 0.07 * 1.0027
        assert back["ref_W_MeV"][0] == pytest.approx(
            (np.sqrt(1.0 + bg * bg) - 1.0) * 938.272, rel=1e-3)
