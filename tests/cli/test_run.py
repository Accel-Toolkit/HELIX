"""The `python -m linac_gen run` subcommand."""
from pathlib import Path

from linac_gen.__main__ import main

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "csr_chicane.dat"
_LGPROJ = _REPO / "examples" / "csr_chicane.lgproj"


def test_run_envelope_from_lattice(tmp_path):
    rc = main(["run", str(_DAT), "--mode", "envelope",
               "--energy", "800", "--species", "H-",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert len(list(tmp_path.glob("*_results.h5"))) == 1


def test_run_envelope_from_project(tmp_path):
    rc = main(["run", str(_LGPROJ), "--mode", "envelope",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_results.h5"))


def test_run_mp_writes_dst(tmp_path):
    rc = main(["run", str(_DAT), "--mode", "mp",
               "--energy", "800", "--species", "H-", "--current", "2",
               "--n-particles", "800", "--nx", "12",
               "--write-dst", "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_results.h5"))
    assert list(tmp_path.glob("*_final.dst"))


def test_run_matrix(tmp_path):
    rc = main(["run", str(_DAT), "--mode", "matrix",
               "--energy", "800", "--species", "H-",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_matrix.txt"))


def test_run_openpmd_format(tmp_path):
    rc = main(["run", str(_DAT), "--mode", "envelope",
               "--energy", "800", "--species", "H-",
               "--format", "openpmd", "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_results.opmd.h5"))


def test_run_missing_input(tmp_path):
    assert main(["run", str(tmp_path / "nope.dat")]) == 2


def test_run_bad_element_override(tmp_path):
    rc = main(["run", str(_DAT), "--set", "NOSUCH.gradient=1",
               "--out", str(tmp_path), "-q"])
    assert rc == 2


def test_run_fail_under_transmission(tmp_path):
    """A transmission threshold that cannot be met → exit code 1."""
    rc = main(["run", str(_DAT), "--mode", "mp",
               "--energy", "800", "--species", "H-", "--current", "2",
               "--n-particles", "600", "--nx", "12",
               "--fail-under-transmission", "200",
               "--out", str(tmp_path), "-q"])
    assert rc == 1


def _h5(tmp_path):
    files = list(tmp_path.glob("*_results.h5"))
    assert len(files) == 1
    return files[0]


def test_run_mp_action_scan_writes_the_group(tmp_path):
    import h5py
    rc = main(["run", str(_DAT), "--mode", "mp",
               "--energy", "800", "--species", "H-", "--current", "2",
               "--n-particles", "600", "--nx", "12", "--action-scan",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    with h5py.File(_h5(tmp_path), "r") as f:
        g = f["action_scan"]
        n_steps = f["envelope"]["s"].shape[0]
        assert g["count_x"].shape[0] == n_steps > 1
        assert int(g["n_alive"][0]) == 600


def test_run_mp_without_flag_has_no_group(tmp_path):
    import h5py
    rc = main(["run", str(_DAT), "--mode", "mp",
               "--energy", "800", "--species", "H-", "--current", "2",
               "--n-particles", "300", "--nx", "12",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    with h5py.File(_h5(tmp_path), "r") as f:
        assert "action_scan" not in f


def test_action_scan_flag_ignored_with_warning_outside_mp_hdf5(tmp_path,
                                                                capsys):
    import h5py
    rc = main(["run", str(_DAT), "--mode", "envelope",
               "--energy", "800", "--species", "H-", "--action-scan",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert "--action-scan ignored (no particles in envelope mode)" in \
        capsys.readouterr().err
    with h5py.File(_h5(tmp_path), "r") as f:
        assert "action_scan" not in f
    rc = main(["run", str(_DAT), "--mode", "mp", "--format", "openpmd",
               "--energy", "800", "--species", "H-", "--current", "2",
               "--n-particles", "300", "--nx", "12", "--action-scan",
               "--out", str(tmp_path / "o"), "-q"])
    assert rc == 0
    assert "only the hdf5 format carries it, not openpmd" in \
        capsys.readouterr().err
