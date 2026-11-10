"""The ``.dst`` layout with seven values per particle (seen from TraceWin 2.27).

Same 23-byte header (magic 0x7e 0x65), then x, x', y, y', phi, W and a seventh
value that is 0 for a particle still in the beam, then mc²: 31 + 56·N bytes.  A
lost particle stays in the file, so it must not reach any statistic.  Measured on
genuine files, which the last tests read when present.
"""
import logging
import struct
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.tracewin_dst import load_dst, write_dst

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "lebt_plus_rfq"
#: not distributed: a TraceWin 2.27 run with its Toutatis files
_GENUINE = sorted(_EXAMPLES.glob("*/toutatis.dst"))
_needs_genuine = pytest.mark.skipif(not _GENUINE,
                                    reason="undistributed TraceWin 2.27 files absent")


def _flagged_file(path, rows, *, current=6.8, freq=162.5, mass=939.294):
    """``rows``: x cm, x' rad, y cm, y' rad, phi rad, W MeV, flag."""
    arr = np.asarray(rows, dtype="<f8")
    head = struct.pack("<2sIddB", b"\x7e\x65", len(arr), current, freq, 0x65)
    path.write_bytes(head + arr.tobytes() + struct.pack("<d", mass))
    return path


_ROWS = [
    # in the beam
    [0.1, 0.001, 0.2, 0.002, 0.5, 2.1, 0.0],
    [-0.1, -0.001, -0.2, -0.002, 0.7, 2.2, 0.0],
    [0.3, 0.003, -0.4, 0.004, 0.6, 2.0, 0.0],
    # lost: only the flag says so.  Coordinates may be zero, or kept
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.0],
    [5.0, 0.05, -5.0, 0.05, 9.0, 0.03, 14.17],
]


def test_only_the_particles_in_the_beam_are_read(tmp_path):
    particles, head = load_dst(str(_flagged_file(tmp_path / "f.dst", _ROWS)))
    assert particles.shape == (3, 6)
    assert head["n_particles"] == 3
    assert head["n_in_file"] == 5
    assert head["n_lost"] == 2
    assert head["layout"] == "flagged"
    # the same conversion as a classic file: cm → mm, rad → mrad, deviations
    assert particles[0, :4] == pytest.approx([1.0, 1.0, 2.0, 2.0])
    # the reference is the mean of the beam: the lost rows do not pull it away
    assert head["w_kin_ref"] == pytest.approx(2.1)
    assert head["phi_ref_rad"] == pytest.approx(0.6)
    assert np.mean(particles[:, 5]) == pytest.approx(0.0, abs=1e-12)
    assert head["mass_MeV"] == pytest.approx(939.294)
    assert head["frequency_MHz"] == pytest.approx(162.5)


def test_current_follows_the_particles_in_the_beam(tmp_path):
    """The macrocharge stays: I / N is the file's.  TraceWin's own classic export
    after losses writes I × N_alive / N_start (see the genuine test below)."""
    _, head = load_dst(str(_flagged_file(tmp_path / "f.dst", _ROWS)))
    assert head["current_mA"] == pytest.approx(6.8 * 3 / 5)


def test_statistics_equal_those_of_a_classic_file_of_the_same_beam(tmp_path):
    flagged = _flagged_file(tmp_path / "f.dst", _ROWS)
    arr = np.asarray(_ROWS, dtype="<f8")[:3, :6]
    raw = (struct.pack("<2sIddB", b"\x7d\x64", 3, 6.8 * 3 / 5, 162.5, 0x7d)
           + arr.tobytes() + struct.pack("<d", 939.294))
    classic = tmp_path / "c.dst"
    classic.write_bytes(raw)
    p_f, h_f = load_dst(str(flagged))
    p_c, h_c = load_dst(str(classic))
    assert np.array_equal(p_f, p_c)
    for k in h_c:
        if k not in ("n_in_file", "n_lost", "layout"):
            assert h_f[k] == h_c[k], k


def test_a_classic_file_reports_no_loss(tmp_path):
    p = tmp_path / "c.dst"
    write_dst(str(p), np.zeros((4, 6)), current_mA=5.0, frequency_MHz=162.5,
              mass_MeV=939.294, w_kin_ref=2.1)
    _, head = load_dst(str(p))
    assert (head["n_in_file"], head["n_lost"], head["layout"]) == (4, 0, "classic")
    assert head["current_mA"] == 5.0


def test_a_file_with_no_particle_in_the_beam_is_refused(tmp_path):
    rows = [[0.0] * 6 + [29.0]] * 2
    with pytest.raises(ValueError, match="in the beam"):
        load_dst(str(_flagged_file(tmp_path / "f.dst", rows)))


@pytest.mark.parametrize("extra", [8, 16, 48])
def test_a_size_of_neither_layout_is_refused(tmp_path, extra):
    p = _flagged_file(tmp_path / "f.dst", _ROWS)
    p.write_bytes(p.read_bytes() + b"\0" * extra)
    with pytest.raises(ValueError, match="File size"):
        load_dst(str(p))


def _columns(path):
    """The numeric rows of a TraceWin ``.out`` table, by the names of its header."""
    lines = Path(path).read_text(errors="replace").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith("##"))
    names = lines[i].split()
    rows = [[float(x) for x in ln.split()] for ln in lines[i + 1:]
            if len(ln.split()) == len(names)]
    return {n: np.array(c) for n, c in zip(names, zip(*rows, strict=True), strict=True)}


def _config(path, **kw):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="H-", energy=2.1, frequency=162.5, source="file",
                      distribution_file=str(path), **kw)


def test_a_beam_starts_from_the_particles_in_the_beam(tmp_path):
    from linac_gen.distributions.factory import create_beam
    cfg = _config(_flagged_file(tmp_path / "f.dst", _ROWS), current=6.8 * 3 / 5)
    beam = create_beam(cfg, seed=1)
    assert beam.n_particles == 3
    assert beam.particles.shape == (3, 6)


def test_the_run_warns_when_the_project_current_ignores_the_losses(tmp_path, caplog):
    """A project keeps its own current (the file does not override it): 6.8 mA on
    the 60 % left in the file would be two-thirds too much space charge."""
    from linac_gen.distributions.factory import create_beam
    path = _flagged_file(tmp_path / "f.dst", _ROWS)
    with caplog.at_level(logging.WARNING):
        create_beam(_config(path, current=6.8), seed=1)
    assert any("2 of 5" in r.getMessage() and "4.08" in r.getMessage()
               for r in caplog.records), caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        create_beam(_config(path, current=6.8 * 3 / 5), seed=1)
    assert not any("2 of 5" in r.getMessage() for r in caplog.records), caplog.text


def test_a_deliberate_zero_current_run_is_not_warned_about(tmp_path, caplog):
    from linac_gen.distributions.factory import create_beam
    path = _flagged_file(tmp_path / "f.dst", _ROWS)
    with caplog.at_level(logging.WARNING):
        create_beam(_config(path, current=0.0), seed=1)
    assert not any("flagged lost" in r.getMessage() for r in caplog.records)


def _beam_rows(w, phi, flag=0.0):
    return [[0.1 * (-1) ** i, 0.001, 0.1, 0.001, phi[i], w[i], flag]
            for i in range(len(w))]


def test_particles_far_from_the_median_energy_are_counted_and_kept(tmp_path, caplog):
    """As an RFQ-exit file: TraceWin counts them in the beam and its current."""
    from linac_gen.distributions.factory import create_beam
    rows = _beam_rows([2.1, 2.11, 2.12, 2.13, 0.03], [0.0, 0.01, 0.02, 0.03, 1.0])
    path = _flagged_file(tmp_path / "f.dst", rows)
    particles, head = load_dst(str(path))
    assert particles.shape[0] == 5
    assert head["n_off_energy"] == 1
    with caplog.at_level(logging.WARNING):
        create_beam(_config(path, current=6.8), seed=1)
    assert any("1 of 5" in r.getMessage() and "median energy" in r.getMessage()
               for r in caplog.records), caplog.text


def test_a_high_energy_tail_is_not_far_from_the_median(tmp_path):
    """Measured at 800 MeV: tails of several MeV, which are not unaccelerated."""
    rows = _beam_rows([800.0, 800.1, 799.9, 806.0, 794.0], [0.0] * 5)
    _, head = load_dst(str(_flagged_file(tmp_path / "f.dst", rows)))
    assert head["n_off_energy"] == 0


@pytest.mark.parametrize("phi_rad,dc", [
    (np.linspace(-np.pi, np.pi, 41), True),         # uniform: rms 103.9 deg
    (np.linspace(-0.2, 0.2, 41), False),            # a bunch
])
def test_a_phase_spanning_the_rf_period_is_a_dc_beam(tmp_path, phi_rad, dc):
    rows = _beam_rows([2.1] * 41, list(phi_rad))
    _, head = load_dst(str(_flagged_file(tmp_path / "f.dst", rows)))
    assert head["dc_like"] is dc


@_needs_genuine
def test_genuine_toutatis_file_agrees_with_tracewins_own_outputs():
    """The file's beam against two independent outputs at that plane.  The first
    row of ``toutatis.out`` gives the same statistics to its printed digits;
    ``partran1.out`` the same count and its space-charge current."""
    from linac_gen.io.tracewin_outputs import read_partran_out
    folder = _GENUINE[0].parent
    particles, head = load_dst(str(_GENUINE[0]))
    assert head["n_in_file"] == 100000
    assert head["n_particles"] == 74891

    t = _columns(folder / "toutatis.out")
    assert t["npart"][0] == 74891
    assert np.std(particles[:, 0]) == pytest.approx(t["SizeX"][0], rel=1e-5)
    assert np.std(particles[:, 2]) == pytest.approx(t["SizeY"][0], rel=1e-5)

    pt = read_partran_out(str(folder / "partran1.out"))
    p = _columns(folder / "partran1.out")
    i = int(np.searchsorted(np.asarray(pt["s_m"]), 1.9612, side="right")) - 1
    assert int(pt["n_alive"][i]) == 74891
    assert head["current_mA"] == pytest.approx(p["SCCurrent"][i], abs=1e-5)
    # the entrance of an RFQ: a DC beam
    assert head["dc_like"] is True
    assert head["n_off_energy"] == 0


@_needs_genuine
def test_genuine_rfq_exit_file_counts_its_unaccelerated_particles():
    """They stay in (TraceWin's count and current include them) and are reported.
    The rest give the longitudinal emittance of TraceWin's table."""
    from linac_gen.io.tracewin_dst import _twiss_from_particles
    folder = _GENUINE[0].parent
    particles, head = load_dst(str(folder / "part_rfq.dst"))
    t = _columns(folder / "toutatis.out")
    assert head["n_particles"] == t["npart"][-1] == 74148
    assert head["current_mA"] == pytest.approx(t["SCCurrent"][-1], abs=1e-5)
    assert head["dc_like"] is False
    assert head["n_off_energy"] == 77
    assert head["emit_z"] > 0.4
    w = head["w_kin_ref"] + particles[:, 5]
    core = particles[w > 1.0]
    tw = _twiss_from_particles(core, head["mass_MeV"], float(np.mean(w[w > 1.0])))
    assert tw["emit_z"] == pytest.approx(0.0479, rel=0.01)


@_needs_genuine
def test_genuine_classic_export_scales_the_current_as_the_reader_does():
    """TraceWin's own file after the losses: I = I0 × N / N0.  The rule the reader
    applies to a flagged file comes from here."""
    _, head = load_dst(str(_GENUINE[0].parent / "part_dtl1.dst"))
    assert head["layout"] == "classic"
    assert head["n_particles"] == 74069
    assert head["current_mA"] == pytest.approx(6.8 * 74069 / 100000, rel=1e-6)


def test_ring_injection_reads_the_particles_in_the_beam(tmp_path):
    from types import SimpleNamespace

    from linac_gen.ring.injection.source import _file_pool
    cfg = SimpleNamespace(path=str(_flagged_file(tmp_path / "f.dst", _ROWS)),
                          species="H-", kinetic_mev=None, frequency_mhz=None)
    assert _file_pool(cfg, "helix").X.shape == (6, 3)
