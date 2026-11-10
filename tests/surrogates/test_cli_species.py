"""The surrogate CLI runs the species its weights are for.

``compare`` and ``run-envelope`` built an H⁻ beam whatever the weights held, so a map for
protons never engaged (0 NN queries), and ``train`` sampled H⁻ only.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.surrogates import cli, registry
from tests.surrogates.test_frequency_jump import F_OLD, _lattice, _magnet
from tests.surrogates.test_surrogate_provenance import _save


@pytest.fixture(autouse=True)
def _quiet_loads(monkeypatch):
    # pinned maps, not their element's: the CLI's species is under test here, and the
    # accuracy check has its own tests (test_accuracy_check.py)
    from linac_gen.surrogates import training
    real = training.load_surrogate

    def unchecked(*args, **kwargs):
        mlp, meta = real(*args, **kwargs)
        meta.accuracy_due = False
        return mlp, meta

    monkeypatch.setattr(training, "load_surrogate", unchecked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _weights(path, species, name="MAP"):
    el = _magnet(F_OLD)
    el.name = name
    sp = {"proton": PROTON, "H-": H_MINUS}[species]
    return _save(path / name, el, np.eye(6), drop=(), train_frequency_mhz=F_OLD,
                 species_name=sp.name, species_mass_mev=sp.mass)


@pytest.fixture
def line(monkeypatch):
    """The lattice the CLI parses: magnets named MAP and MAP2."""
    from linac_gen.io import tracewin_parser
    a, b = _magnet(F_OLD), _magnet(F_OLD)
    b.name = "MAP2"
    lat = _lattice(a, b)
    monkeypatch.setattr(tracewin_parser, "parse_tracewin", lambda path: (lat, None))
    return lat


def _run(capsys, *args):
    registry.clear()
    rc = cli.main(["run-envelope", "--lattice", "x.dat", "--ref-w-kin", "5.0",
                   "--frequency", str(F_OLD), *args])
    out, err = capsys.readouterr()
    return rc, out, err


def _queries(out):
    return int(out.split("NN full-element queries:")[1].split()[0])


def test_run_envelope_runs_the_species_the_weights_are_for(tmp_path, line, capsys):
    rc, out, err = _run(capsys, "--use-surrogates", str(_weights(tmp_path, "proton")))
    assert rc == 0 and _queries(out) == 1, (out, err)
    assert "will not engage" not in err


def test_h_minus_weights_still_run_an_h_minus_beam(tmp_path, line, capsys):
    rc, out, err = _run(capsys, "--use-surrogates", str(_weights(tmp_path, "H-")))
    assert rc == 0 and _queries(out) == 1


def test_a_beam_asked_for_another_species_says_which_will_not_engage(tmp_path, line, capsys):
    rc, out, err = _run(capsys, "--use-surrogates", str(_weights(tmp_path, "proton")),
                        "--species", "H-")
    assert rc == 0 and _queries(out) == 0
    assert "'MAP'" in err and "is for proton and the beam is H-: it will not engage" in err


def test_weights_for_several_species_need_the_beams(tmp_path, line, capsys):
    a, b = _weights(tmp_path, "proton"), _weights(tmp_path, "H-", name="MAP2")
    with pytest.raises(SystemExit, match=r"several species \(H-, proton\).*--species"):
        _run(capsys, "--use-surrogates", str(a), str(b))
    rc, out, err = _run(capsys, "--use-surrogates", str(a), str(b), "--species", "proton")
    assert rc == 0 and _queries(out) == 1 and "'MAP2'" in err


def test_compare_runs_the_species_the_weights_are_for(tmp_path, line, monkeypatch, capsys):
    from linac_gen.surrogates import compare
    seen = {}

    class _Report:
        def summary_text(self):
            return "report"

    def fake(lat, ref, twiss, current, surrogates):
        seen["species"] = ref.species.name
        return _Report()

    monkeypatch.setattr(compare, "compare_envelope", fake)
    rc = cli.main(["compare", "--lattice", "x.dat", "--weights", str(_weights(tmp_path, "proton")),
                   "--ref-w-kin", "5.0", "--frequency", str(F_OLD)])
    assert rc == 0 and seen["species"] == PROTON.name


def test_train_samples_the_species_asked_for_into_its_folder(tmp_path, monkeypatch):
    from linac_gen.io import tracewin_parser
    from linac_gen.surrogates import training
    from tests.surrogates.test_frequency_jump import _cavity
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return None, type("M", (), {"val_mape": 0.0})()

    lat_file = tmp_path / "line.dat"
    lat_file.write_text("DRIFT 100 10\nEND\n")
    monkeypatch.setattr(tracewin_parser, "parse_tracewin", lambda path: (_lattice(_cavity()), None))
    monkeypatch.setattr(training, "train_surrogate_for_element", fake)
    h16 = registry.hash_lattice_file(lat_file)[:16]
    assert cli.main(["train", "--lattice", str(lat_file), "--element", "MAP", "--species", "proton"]) == 0
    assert seen["ref_template"].species.name == PROTON.name
    assert Path(seen["out_dir"]) == Path("linac_gen/surrogates/weights") / f"{h16}_proton" / "MAP"
    assert cli.main(["train", "--lattice", str(lat_file), "--element", "MAP"]) == 0
    assert seen["ref_template"].species.name == H_MINUS.name
    assert Path(seen["out_dir"]) == Path("linac_gen/surrogates/weights") / h16 / "MAP"


def test_the_tab_and_the_cli_share_the_folder():
    from linac_gen.surrogates.training import weights_folder
    assert weights_folder("0123456789abcdef9999", "H-") == Path("linac_gen/surrogates/weights/0123456789abcdef")
    assert weights_folder("0123456789abcdef9999", "proton").name == "0123456789abcdef_proton"
    assert weights_folder("unknown", None).name == "unknown"


def test_weights_for_a_species_helix_does_not_know_say_so(tmp_path, line, capsys):
    from tests.surrogates.test_frequency_jump import _magnet as magnet
    el = magnet(F_OLD)
    d = _save(tmp_path / "MAP", el, np.eye(6), drop=(), train_frequency_mhz=F_OLD,
              species_name="muon", species_mass_mev=105.66)
    with pytest.raises(SystemExit, match=r"weights are for 'muon'.*--species"):
        _run(capsys, "--use-surrogates", str(d))
    rc, out, err = _run(capsys, "--use-surrogates", str(d), "--species", "H-")
    assert rc == 0 and _queries(out) == 0 and "is for muon" in err
