"""The `python -m linac_gen backtrack` subcommand."""

from tests.dataguard import needs, require  # noqa: E402
from pathlib import Path

import numpy as np
import pytest

from linac_gen.__main__ import main

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "pipii" / "mebt" / "mebt.dat"

# Real PIP-II MEBT entrance Twiss (examples/pipii/mebt/mebt.lgproj) —
# unmatched defaults debunch the beam and push the bunchers far outside
# their linear range, so a matched beam is load-bearing here.
_TW = ["--species", "H-", "--energy", "2.1", "--freq", "162.5",
       "--beam", "emit_nx=0.21", "--beam", "alpha_x=1.228",
       "--beam", "beta_x=0.316", "--beam", "emit_ny=0.21",
       "--beam", "alpha_y=-0.095394", "--beam", "beta_y=0.113",
       "--beam", "emit_z=0.06231832", "--beam", "alpha_z=0.0",
       "--beam", "beta_z=819.05492"]


def test_backtrack_registered():
    with pytest.raises(SystemExit) as exc:
        main(["backtrack", "--help"])
    assert exc.value.code == 0


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_design_mode(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--n-particles", "300", "--seed", "5",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_backtrack.h5"))


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_envelope_mode(tmp_path):
    rc = main(["backtrack", str(_DAT), "--mode", "envelope", *_TW,
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_backtrack.h5"))


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_dst_roundtrip_with_validate(tmp_path):
    """Forward run → .dst → backtrack → --validate closure must pass."""
    rc = main(["run", str(_DAT), "--mode", "mp", *_TW,
               "--n-particles", "500", "--seed", "11",
               "--write-dst", "--out", str(tmp_path), "-q"])
    assert rc == 0
    dst = tmp_path / "mebt_final.dst"
    assert dst.is_file()
    out_dst = tmp_path / "entrance.dst"
    rc = main(["backtrack", str(_DAT), "--dst", str(dst), *_TW,
               "--write-dst", str(out_dst), "--validate",
               "--out", str(tmp_path), "-q"])
    assert rc == 0                      # validate exits 1 on closure FAIL
    assert out_dst.is_file()

    # The reconstructed entrance must resemble the design entrance beam
    # (same Twiss family): sigma_x within the buncher fidelity floor.
    from linac_gen.io.tracewin_dst import load_dst
    particles, header = load_dst(str(out_dst))
    assert header["w_kin_ref"] == pytest.approx(2.1, rel=1e-3)
    sig_x = float(np.std(particles[:, 0]))
    assert 0.2 < sig_x < 2.0            # sane physical scale, not blown up


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_subrange(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--n-particles", "200", "--from-element", "2",
               "--to-element", "30", "--out", str(tmp_path), "-q"])
    assert rc == 0


def test_backtrack_bad_range_exits_2(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--from-element", "50", "--to-element", "10",
               "--out", str(tmp_path), "-q"])
    assert rc == 2


def test_backtrack_missing_dst_exits_2(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--dst", str(tmp_path / "nope.dst"),
               "--out", str(tmp_path), "-q"])
    assert rc == 2


def _drift_deck_and_exit_dst(tmp_path, n_outside):
    """A 100 mm drift with a 20 mm aperture and an exit beam of 100
    particles, ``n_outside`` of them beyond the aperture: the backward walk
    applies no cuts, the forward re-track loses exactly those."""
    from linac_gen.core.particle import H_MINUS
    from linac_gen.io.tracewin_dst import write_dst
    deck = tmp_path / "drift.dat"
    deck.write_text("DRIFT 100 20\nEND\n")
    rng = np.random.default_rng(4)
    p = rng.normal(size=(100, 6)) * [1.0, 0.5, 1.0, 0.5, 2.0, 1e-3]
    p[:n_outside, 0] = 30.0 + rng.uniform(0.0, 5.0, n_outside)
    dst = tmp_path / "exit.dst"
    write_dst(str(dst), p, current_mA=0.0, frequency_MHz=162.5,
              mass_MeV=H_MINUS.mass, w_kin_ref=2.1)
    return deck, dst, p


def test_validate_compares_the_same_particles(tmp_path, capsys):
    """One particle in a hundred is lost in the re-track: the closure is
    measured on the 99 that survive (round-off for a drift), and the loss
    is reported, not mixed into the sigmas (sigma_x of the survivors against
    that of all 100 differed by ~3x)."""
    deck, dst, p = _drift_deck_and_exit_dst(tmp_path, 1)
    rc = main(["backtrack", str(deck), "--dst", str(dst), *_TW[:6],
               "--validate", "--out", str(tmp_path), "-q"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "99 of 100 exit particles survive the re-track" in out.out
    assert "field-map inverse: rk4" in out.out
    rels = [float(line.rsplit("rel=", 1)[1]) for line in out.out.splitlines()
            if "rel=" in line]
    assert len(rels) == 4 and max(rels) < 1e-9
    # particle by particle too, dW included: the loader references dW to the
    # bunch mean, the backward walk to the design exit energy
    res = next(line for line in out.out.splitlines() if "per-particle" in line)
    assert [float(v.split("=")[1]) < 1e-9 for v in res.split(":")[1].split()] == [True] * 4
    assert abs(np.std(p[1:, 0]) - np.std(p[:, 0])) > 1.0       # what the old check compared
    assert "1 of 100 lost in the re-track" in out.out


def test_validate_fails_when_the_re_track_loses_particles(tmp_path, capsys):
    deck, dst, _ = _drift_deck_and_exit_dst(tmp_path, 10)
    rc = main(["backtrack", str(deck), "--dst", str(dst), *_TW[:6],
               "--validate", "--fieldmap-inverse", "linear",
               "--out", str(tmp_path), "-q"])
    out = capsys.readouterr()
    assert rc == 1
    assert "10 of 100 exit particles (10.0 %) are lost in the re-track" in out.err
    assert "worst relative sigma residual" not in out.err      # the survivors close
    assert "field-map inverse: linear" in out.err
