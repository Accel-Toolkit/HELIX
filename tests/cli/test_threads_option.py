"""``--threads N|all`` on every subcommand: the count reaches the run and
its record; a bad value is a usage error; the matcher takes it too."""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from linac_gen.__main__ import main
from tests.helpers import kept_thread_setting

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "csr_chicane.dat"


@pytest.mark.parametrize("value, expect", [("3", 3), ("all", None)])
def test_run_uses_and_records_the_count(tmp_path, value, expect):
    with kept_thread_setting() as threads:
        threads._setting = threads._requested = None
        rc = main(["run", str(_DAT), "--mode", "mp",
                   "--energy", "800", "--species", "H-", "--current", "2",
                   "--n-particles", "600", "--nx", "12",
                   "--threads", value, "--out", str(tmp_path), "-q"])
        assert rc == 0
        n = expect or threads.available_threads()
        assert threads.get_threads() == n
        assert threads.requested_threads() == expect
    (h5,) = tmp_path.glob("*_results.h5")
    with h5py.File(h5, "r") as f:
        assert f["provenance"].attrs["threads"] == n
        assert f["provenance"].attrs["threads_source"] == "set"


@pytest.mark.parametrize("bad", ["-2", "x", "1.5"])
def test_a_bad_value_is_a_usage_error(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scan", "--threads", bad])
    assert exc.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_the_matcher_takes_it(tmp_path):
    from linac_gen.matching.__main__ import main as match_main
    with kept_thread_setting() as threads:
        # the option is applied before the input is read (here: missing)
        assert match_main([str(tmp_path / "missing.dat"), "--threads", "5"]) == 2
        assert threads.get_threads() == 5


def test_the_surrogate_cli_takes_it():
    from linac_gen.surrogates.cli import main as surrogate_main
    with kept_thread_setting() as threads:
        n = threads.available_threads() + 1          # not what it defaults to
        with pytest.raises(SystemExit):              # no subcommand given
            surrogate_main(["--threads", str(n)])
        assert threads.get_threads() != n            # not applied on errors
        with pytest.raises(SystemExit) as exc:
            surrogate_main(["--threads", "x", "train"])
        assert exc.value.code == 2
