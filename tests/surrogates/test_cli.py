"""M5 — CLI smoke tests: argparse routes correctly; --help works."""
import sys

import pytest

from linac_gen.surrogates import cli


def test_cli_help_runs_without_error(capsys):
    """`python -m linac_gen.surrogates.cli --help` exits cleanly."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "train" in captured.out
    assert "compare" in captured.out
    assert "run-envelope" in captured.out


def test_cli_train_help():
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--help"])
    assert exc.value.code == 0


def test_cli_compare_help():
    with pytest.raises(SystemExit) as exc:
        cli.main(["compare", "--help"])
    assert exc.value.code == 0


def test_cli_run_envelope_help():
    with pytest.raises(SystemExit) as exc:
        cli.main(["run-envelope", "--help"])
    assert exc.value.code == 0


def test_cli_train_invokes_handler(monkeypatch):
    """Train subcommand routes to _cmd_train and forwards args."""
    captured = {}

    def fake_train(args):
        captured["lattice"] = args.lattice
        captured["element"] = args.element
        captured["samples"] = args.samples
        return 0

    monkeypatch.setattr(cli, "_cmd_train", fake_train)
    rc = cli.main([
        "train",
        "--lattice", "test.dat",
        "--element", "FMAP_001",
        "--samples", "42",
    ])
    assert rc == 0
    assert captured["lattice"] == "test.dat"
    assert captured["element"] == "FMAP_001"
    assert captured["samples"] == 42


def test_cli_compare_invokes_handler(monkeypatch):
    """Compare subcommand routes to _cmd_compare with weights list."""
    captured = {}

    def fake_compare(args):
        captured["lattice"] = args.lattice
        captured["weights"] = list(args.weights)
        captured["out"] = args.out
        return 0

    monkeypatch.setattr(cli, "_cmd_compare", fake_compare)
    rc = cli.main([
        "compare",
        "--lattice", "X.dat",
        "--weights", "/path/A", "/path/B",
        "--out", "diff.png",
    ])
    assert rc == 0
    assert captured["lattice"] == "X.dat"
    assert captured["weights"] == ["/path/A", "/path/B"]
    assert captured["out"] == "diff.png"


def test_cli_run_envelope_invokes_handler(monkeypatch):
    """run-envelope routes to _cmd_run_envelope with surrogate list."""
    captured = {}

    def fake_run(args):
        captured["lattice"] = args.lattice
        captured["use_surrogates"] = list(args.use_surrogates)
        return 0

    monkeypatch.setattr(cli, "_cmd_run_envelope", fake_run)
    rc = cli.main([
        "run-envelope",
        "--lattice", "X.dat",
        "--use-surrogates", "/path/A",
    ])
    assert rc == 0
    assert captured["lattice"] == "X.dat"
    assert captured["use_surrogates"] == ["/path/A"]
