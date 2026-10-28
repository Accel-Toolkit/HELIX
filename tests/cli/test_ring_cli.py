"""``python -m linac_gen ring optics`` and ``ring run`` end to end."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from linac_gen.__main__ import main
from linac_gen.ring.config import MODEL_ALIASES

#: each model's name in earlier versions
EARLIER = {new: old for old, new in MODEL_ALIASES.items()}

ROOT = Path(__file__).resolve().parents[2]
BOOSTER = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
REF = json.loads((ROOT / "tests" / "ring" / "fixtures" / "refs" / "simple_booster_madx.json").read_text())
FODO = ROOT / "examples" / "fodo_cell.dat"


def test_ring_optics_json_matches_madx(capsys):
    assert main(["ring", "optics", str(BOOSTER), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["qx"] == pytest.approx(REF["q1"], abs=1e-6)
    assert out["qy"] == pytest.approx(REF["q2"], abs=1e-6)
    assert out["dqx"] == pytest.approx(REF["dq1_delta"], abs=0.02)
    assert out["alpha_c"] == pytest.approx(REF["alfa"], rel=1e-6)
    assert out["n_nodes"] == out["n_elements"] == 976 and out["harmonic"] == 84
    assert out["options"]["model"] == "cartesian" and out["options"]["edge_model"] == "exact"


def test_ring_optics_text_six_d_and_twiss_table(tmp_path, capsys):
    table = tmp_path / "twiss.csv"
    assert main(["ring", "optics", str(BOOSTER), "--six-d", "--twiss-out", str(table)]) == 0
    text = capsys.readouterr().out
    assert "Qx 6.70999" in text and "synchrotron tune" in text and "RF bucket" in text
    rows = list(csv.DictReader(table.open()))
    assert len(rows) == 976 and rows[0]["name"] == "sa"
    assert float(rows[-1]["mu_x"]) == pytest.approx(REF["q1"], abs=1e-6)
    js = tmp_path / "twiss.json"
    assert main(["ring", "optics", str(BOOSTER), "--twiss-out", str(js)]) == 0
    capsys.readouterr()
    assert len(json.loads(js.read_text())["betx"]) == 976


def test_ring_optics_curvilinear_model(capsys):
    assert main(["ring", "optics", str(BOOSTER), "--model", "curvilinear", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["options"]["model"] == "curvilinear" and out["options"]["edge_model"] == "linear"
    assert out["dqx"] == pytest.approx(-7.99, abs=0.02)            # curvilinear multipoles


def test_a_deck_without_a_beam_needs_an_energy(capsys):
    assert main(["ring", "optics", str(FODO)]) == 2
    assert "--energy" in capsys.readouterr().err
    assert main(["ring", "optics", str(FODO), "--energy", "100", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kinetic_energy_mev"] == pytest.approx(100.0)
    assert out["gamma_t"] is None                       # no bends: no transition


def test_refusals_are_errors_not_tracebacks(tmp_path, capsys):
    deck = tmp_path / "sol.madx"
    deck.write_text("beam, particle=proton, energy=1.0+pmass;\n"
                    "s: solenoid, l=1, ks=0.1;\nd: drift, l=1;\nr: line=(s, d);\nuse, sequence=r;\n")
    assert main(["ring", "optics", str(deck)]) == 1
    assert "solenoids are not modelled in ring mode" in capsys.readouterr().err
    assert main(["ring", "optics", str(tmp_path / "missing.madx")]) == 2


def test_species_of_a_deck_with_its_own_beam_is_the_decks(capsys):
    # (the MAD-X deck's BEAM is a proton beam: --species does not override it)
    assert main(["ring", "optics", str(BOOSTER), "--species", "H-", "--json"]) == 0
    cap = capsys.readouterr()
    assert "--species is not used" in cap.err
    assert json.loads(cap.out)["qx"] == pytest.approx(REF["q1"], abs=1e-6)


@pytest.mark.parametrize("saved", ["curvilinear", EARLIER["curvilinear"]],
                         ids=["current name", "earlier name"])
def test_a_projects_ring_settings_are_used(tmp_path, capsys, saved):
    # (the element model the GUI saved in the project — also under its name of
    # earlier versions; the command line wins)
    proj = tmp_path / "ring.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {
                                    "model": saved, "edge_model": "linear"}}}))
    assert main(["ring", "optics", str(proj), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["options"]["model"] == "curvilinear" and out["dqx"] == pytest.approx(-7.99, abs=0.02)
    assert main(["ring", "optics", str(proj), "--model", "cartesian", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["options"]["model"] == "cartesian" and out["options"]["edge_model"] == "exact"
    assert out["dqx"] == pytest.approx(REF["dq1_delta"], abs=0.02)


@pytest.mark.parametrize("model", ["cartesian", "curvilinear"])
def test_earlier_model_names_on_the_command_line(capsys, model):
    assert main(["ring", "optics", str(BOOSTER), "--model", EARLIER[model], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["options"]["model"] == model


_BEAM = ["--beam", "emit_nx=1.5", "--beam", "emit_ny=1.5", "--beam", "emit_z=5",
         "--beam", "beta_z=100", "--beam", "frequency=44.7044096"]


def test_ring_run_writes_its_results(tmp_path, capsys):
    from linac_gen.ring.results import load_ring_hdf5
    assert main(["ring", "run", str(BOOSTER), "--turns", "30", "--particles", "200",
                 "--snapshot-turns", "0,30", "--out", str(tmp_path), "--json", *_BEAM]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["turns_done"] == 30 and out["survival"] == 1.0 and out["lost"] == 0
    assert out["device"] == "cpu" and set(out["tunes_naff"]) == {"qx", "qy"}   # (Qs: too few turns)
    res = load_ring_hdf5(Path(out["file"]))
    assert res.turns_done == 30 and sorted(res.snapshots) == [0, 30]
    assert res.meta["config"]["source"] == "matched"
    assert main(["ring", "run", str(BOOSTER), "--turns", "5", "--particles", "50",
                 "--out", str(tmp_path), *_BEAM]) == 0
    text = capsys.readouterr().out
    assert "ring run done: 5 of 5 turns" in text and "survival 100 %" in text


def test_a_projects_run_starts_from_its_beam(tmp_path, capsys):
    # a project's run starts from its beam (the GUI's Beam tab: its particles,
    # emittances and Twiss), whatever source an earlier version saved;
    # --particles sets their number; --source matched the Gaussian matched to
    # the ring; a project without beam settings starts matched
    from linac_gen.ring.results import load_ring_hdf5
    ref = json.loads((ROOT / "tests" / "ring" / "fixtures" / "refs" /
                      "simple_booster_madx.json").read_text())["start"]
    beam = {"species": "proton", "energy": 800.0, "frequency": 44.7044096, "current": 0.0,
            "n_particles": 150, "distribution": "gaussian", "cutoff": 4.0, "emit_nx": 1.5,
            "emit_ny": 1.5, "emit_z": 5.0, "alpha_x": ref["alfx"], "beta_x": ref["betx"],
            "alpha_y": ref["alfy"], "beta_y": ref["bety"], "beta_z": 100.0}
    proj = tmp_path / "ring.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "beam": beam, "ring": {"enabled": True, "config": {
                                    "source": "matched", "n_particles": 999}}}))
    args = ["--turns", "2", "--probes", "0", "--json"]
    assert main(["ring", "run", str(proj), *args, "--out", str(tmp_path / "a")]) == 0
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    res = load_ring_hdf5(Path(out["file"]))
    assert out["particles"] == 150 and res.meta["config"]["source"] == "beam"
    # (its transverse Twiss the ring's — no dispersion given: x a little off —
    # its longitudinal ones not: beta_z 100 deg/MeV for a matched 42.6)
    bm = out["start_mismatch"]
    assert bm[1] == pytest.approx(1.0, abs=1e-3) and bm[2] > 1.1
    assert any("mismatched" in n for n in out["notes"])
    assert main(["ring", "run", str(proj), "--turns", "2", "--probes", "0",
                 "--out", str(tmp_path / "b")]) == 0
    cap = capsys.readouterr()
    assert "beam: the project's, 150 particles (--source matched" in cap.err
    assert "start: BMAG against the ring's matched beam (1 when matched) x 1." in cap.out
    assert main(["ring", "run", str(proj), *args, "--particles", "120",
                 "--out", str(tmp_path / "c")]) == 0
    assert json.loads(capsys.readouterr().out)["particles"] == 120
    assert main(["ring", "run", str(proj), *args, "--source", "matched", "--particles", "130",
                 "--out", str(tmp_path / "d")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["particles"] == 130
    assert load_ring_hdf5(Path(out["file"])).meta["config"]["source"] == "matched"
    assert out["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    # (no beam settings in the project: the matched Gaussian, as for a bare deck)
    bare = tmp_path / "bare.lgproj"
    bare.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {}}}))
    assert main(["ring", "run", str(bare), *args, "--particles", "110", *_BEAM,
                 "--out", str(tmp_path / "e")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["particles"] == 110
    assert load_ring_hdf5(Path(out["file"])).meta["config"]["source"] == "matched"
    assert main(["ring", "run", str(proj), *args, "--particles", "0",
                 "--out", str(tmp_path / "f")]) == 2
    assert "n_particles must be at least 1" in capsys.readouterr().err
    # a beam read from a particle file uses its particles: --particles refused
    import numpy as np

    from linac_gen.io.tracewin_dst import write_dst
    rng = np.random.default_rng(5)
    dst = tmp_path / "beam.dst"
    parts = rng.normal(0.0, [2.0, 0.2, 1.0, 0.3, 5.0, 0.1], (120, 6))
    parts -= parts.mean(axis=0)               # (the file's reference: its mean energy)
    write_dst(str(dst), parts,
              current_mA=0.0, frequency_MHz=44.7044096, mass_MeV=938.27208816,
              w_kin_ref=800.0)
    fproj = tmp_path / "file.lgproj"
    fproj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                 "beam": dict(beam, source="file", distribution_file=str(dst)),
                                 "ring": {"enabled": True, "config": {"source": "beam"}}}))
    assert main(["ring", "run", str(fproj), *args, "--particles", "50",
                 "--out", str(tmp_path / "g")]) == 2
    assert "read from a distribution file" in capsys.readouterr().err
    assert main(["ring", "run", str(fproj), "--turns", "1", "--probes", "0",
                 "--out", str(tmp_path / "g")]) == 0
    cap = capsys.readouterr()
    assert "beam: the project's, its distribution file beam.dst" in cap.err
    assert "1 of 1 turns, 120 particles" in cap.out


def test_a_projects_saved_start_is_the_command_lines(tmp_path, capsys):
    # the Run page's start, as the project saved it: the beam with injection
    # too, or an empty ring (which needs the injection)
    from linac_gen.ring.results import load_ring_hdf5
    deck = tmp_path / "foil.madx"
    text = BOOSTER.read_text()
    line = "cell01 : line = (sa, "
    deck.write_text(text.replace("sa: drift, l=0.176", "foilm: marker;\nsa: drift, l=0.176")
                    .replace(line, "cell01 : line = (foilm, sa, "))
    inj = {"foil": {"at": "foilm", "thickness_ug_cm2": 600.0, "x_min_m": -0.01,
                    "x_max_m": 0.01, "y_min_m": -0.01, "y_max_m": 0.01},
           "source": {"kind": "gaussian", "species": "H-", "kinetic_mev": 800.0,
                      "frequency_mhz": 162.5, "emit_nx_um": 0.3, "emit_ny_um": 0.3,
                      "beta_x_m": 30.0, "beta_y_m": 5.0},
           "turns": 2, "particles_per_turn": 20, "current_ma": 2.0}
    beam = {"species": "proton", "energy": 800.0, "frequency": 44.7044096, "n_particles": 100,
            "emit_nx": 1.5, "emit_ny": 1.5, "emit_z": 5.0, "beta_z": 100.0}
    out = {}
    for src in ("beam", "empty"):
        proj = tmp_path / f"{src}.lgproj"
        proj.write_text(json.dumps({"__kind__": "linac_gen_project",
                                    "lattice_path": str(deck), "beam": beam,
                                    "ring": {"enabled": True, "config": {
                                        "source": src, "injection": inj}}}))
        assert main(["ring", "run", str(proj), "--turns", "2", "--probes", "0", "--json",
                     "--out", str(tmp_path / src)]) == 0, capsys.readouterr().err
        o = json.loads(capsys.readouterr().out)
        out[src] = load_ring_hdf5(Path(o["file"])).meta["config"]["source"]
        # (without the injection an empty start cannot stand: the beam)
        assert main(["ring", "run", str(proj), "--turns", "1", "--probes", "0", "--json",
                     "--no-injection", "--out", str(tmp_path / (src + "_ni"))]) == 0
        o = json.loads(capsys.readouterr().out)
        assert load_ring_hdf5(Path(o["file"])).meta["config"]["source"] == "beam"
    assert out == {"beam": "beam", "empty": "empty"}
    # an injection given on the command line fills an empty ring, whatever start
    # the project saved for its own settings
    f = tmp_path / "inj.json"
    f.write_text(json.dumps(inj))
    plain = tmp_path / "plain.lgproj"
    plain.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(deck),
                                 "beam": beam, "ring": {"enabled": True, "config": {
                                     "source": "beam"}}}))
    assert main(["ring", "run", str(plain), "--injection", str(f), "--turns", "1", "--probes",
                 "0", "--json", "--out", str(tmp_path / "cli_inj")]) == 0
    o = json.loads(capsys.readouterr().out)
    assert load_ring_hdf5(Path(o["file"])).meta["config"]["source"] == "empty"


def test_ring_run_restarts_from_its_checkpoint(tmp_path, capsys):
    from linac_gen.ring.results import load_ring_hdf5
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--turns", "20", "--particles", "100", "--checkpoint-every", "8", *_BEAM]
    assert main(["ring", "run", str(BOOSTER), "--out", str(a), *common]) == 0
    ck = a / "simple_booster_ring_checkpoint.npz"
    assert ck.is_file()
    assert main(["ring", "run", str(BOOSTER), "--out", str(b), "--restart", str(ck),
                 *common]) == 0
    capsys.readouterr()
    ra = load_ring_hdf5(a / "simple_booster_ring.h5")
    rb = load_ring_hdf5(b / "simple_booster_ring.h5")
    import numpy as np
    assert np.array_equal(ra.cov, rb.cov) and np.array_equal(ra.mean, rb.mean)


def test_ring_run_refusals(tmp_path, capsys):
    assert main(["ring", "run", str(FODO), "--out", str(tmp_path)]) == 2       # (no energy)
    assert "--energy" in capsys.readouterr().err
    assert main(["ring", "run", str(BOOSTER), "--beam", "nonsense=1", "--out",
                 str(tmp_path)]) == 2
    assert "unknown beam parameter" in capsys.readouterr().err
    assert main(["ring", "run", str(BOOSTER), "--device", "cuda", "--turns", "1",
                 "--particles", "10", "--out", str(tmp_path), *_BEAM]) in (0, 1)


def test_ring_run_options_after_the_subcommand(tmp_path, capsys):
    # --threads after `run` (and before it), snapshots at 'last', the species
    # printed, the beam settings and the lattice in the results file.  (The
    # threads in use are capped at numba's NUMBA_NUM_THREADS, the cores: 2 on a
    # small CI runner — hence 2, then 1 before the subcommand.)
    import h5py
    import numba

    from linac_gen.ring.results import load_ring_hdf5
    assert main(["ring", "run", str(BOOSTER), "--turns", "6", "--particles", "40",
                 "--snapshot-turns", "0,last", "--threads", "2", "--probes", "0",
                 "--out", str(tmp_path), "--json", *_BEAM]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["threads"] == min(2, numba.config.NUMBA_NUM_THREADS)
    assert out["species"] == "proton"
    res = load_ring_hdf5(Path(out["file"]))
    assert sorted(res.snapshots) == [0, 6] and res.meta["config"]["snapshot_turns"] == [0, "last"]
    with h5py.File(out["file"], "r") as f:
        assert f["beam_config"].attrs["emit_nx"] == pytest.approx(1.5)
        assert f["provenance"].attrs["lattice_path"].endswith("simple_booster.madx")
    assert main(["ring", "--threads", "1", "run", str(BOOSTER), "--turns", "2", "--particles",
                 "20", "--probes", "0", "--out", str(tmp_path), "--json", *_BEAM]) == 0
    assert json.loads(capsys.readouterr().out)["threads"] == 1
    assert main(["ring", "run", str(BOOSTER), "--turns", "2", "--particles", "20", "--probes",
                 "0", "--out", str(tmp_path), *_BEAM]) == 0
    assert "proton 800 MeV" in capsys.readouterr().out


def test_ring_run_restart_and_snapshot_refusals(tmp_path, capsys):
    assert main(["ring", "run", str(BOOSTER), "--restart", str(tmp_path / "none.npz"),
                 "--out", str(tmp_path), *_BEAM]) == 2
    assert "no checkpoint file" in capsys.readouterr().err
    assert main(["ring", "run", str(BOOSTER), "--snapshot-turns", "0,first",
                 "--out", str(tmp_path), *_BEAM]) == 2
    assert "not a whole number" in capsys.readouterr().err


def test_a_restart_keeps_the_checkpoints_probes(tmp_path, capsys):
    # continued without --probes / --stats-every: the checkpoint's settings
    a = tmp_path / "a"
    assert main(["ring", "run", str(BOOSTER), "--out", str(a), "--turns", "6", "--particles",
                 "30", "--probes", "2", "--stats-every", "2", "--checkpoint-every", "3",
                 *_BEAM]) == 0
    capsys.readouterr()
    ck = a / "simple_booster_ring_checkpoint.npz"
    assert main(["ring", "run", str(BOOSTER), "--out", str(tmp_path / "b"), "--turns", "8",
                 "--particles", "30", "--restart", str(ck), "--json", *_BEAM]) == 0
    out = json.loads(capsys.readouterr().out)
    from linac_gen.ring.results import load_ring_hdf5
    res = load_ring_hdf5(Path(out["file"]))
    assert res.turns_done == 8 and res.meta["config"]["tune_probes"] == 2
    assert res.meta["stats_every"] == 2 and list(res.turns) == [0, 2, 4, 6, 8]
    # an explicit other setting is refused with the reason
    assert main(["ring", "run", str(BOOSTER), "--out", str(tmp_path / "c"), "--turns", "8",
                 "--restart", str(ck), "--probes", "1", *_BEAM]) == 1
    assert "continue it with the same settings" in capsys.readouterr().err
    (tmp_path / "junk.npz").write_bytes(b"not a checkpoint")
    assert main(["ring", "run", str(BOOSTER), "--out", str(tmp_path), "--restart",
                 str(tmp_path / "junk.npz"), *_BEAM]) == 2
    assert "is not a ring checkpoint" in capsys.readouterr().err


# -- the ramp (milestone M3) --------------------------------------------------
#: a generic program: a table from 800 to 830 MeV over 0.5 ms (268 turns of the
#: simple Booster) and a constant voltage, given as a table
_SHORT = {"energy": {"kind": "table", "times_s": [0.0, 0.0005], "values": [800.0, 830.0]},
          "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
          "gamma_t": 5.449167323}


def _ramp_file(tmp_path, d=None, name="ramp.json"):
    f = tmp_path / name
    f.write_text(json.dumps(d or _SHORT))
    return f


def test_ring_ramp_prints_the_reference_through_the_program(tmp_path, capsys):
    import numpy as np

    from linac_gen.cli.common import load_lattice
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.programs import Ramp, RampConfig
    table = tmp_path / "turns.csv"
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path)), "--csv",
                 str(table), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    # the same as the reference alone through the API
    rl = compile_from_config(load_lattice(str(BOOSTER)), RingConfig(ramp=_SHORT))
    ramp = Ramp(RampConfig.from_dict(_SHORT), rl)
    o = ramp.orbit(ramp.turns_to_end())
    assert out["turns"] == out["turns_to_end"] == 268 == len(o["t_s"])
    assert out["t_end_s"] == o["t_end_s"] and out["gamma_end"] == o["gamma_end"]
    assert out["kinetic_start_mev"] == pytest.approx(800.0, rel=1e-15)
    assert out["transition_turn"] is None and out["capped_turns"] == 0
    assert out["gamma_t_from"] == "given by the ramp"
    f = o["frequency_hz"]
    assert out["frequency_hz"] == [float(f.min()), float(f.max())]
    assert out["max_gain_over_voltage"] == pytest.approx(float(np.max(o["gain_mev"] / 1.2)),
                                                         rel=1e-14)
    rows = list(csv.DictReader(table.open()))
    assert len(rows) == 268 and float(rows[-1]["t_s"]) == o["t_s"][-1]
    assert float(rows[5]["phi_s_deg"]) == o["phi_deg"][5] + math.degrees(ramp.arg)
    # the text form; a number of turns of its own
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path)),
                 "--turns", "10"]) == 0
    text = capsys.readouterr().out
    assert "turns: 10 from t = 0 ms (268 to the end of the energy program" in text
    assert "not crossed" in text and "turns capped at 90 deg: none" in text


def test_ramp_tables_from_csv_files(tmp_path, capsys):
    # the tables in CSV files next to the ramp file, the header naming the
    # columns; the same turns as the tables inline
    (tmp_path / "energy.csv").write_text("# kinetic energy\nt_s,kinetic_mev\n0.0,800.0\n"
                                         "0.0005,830.0\n")
    (tmp_path / "volts.csv").write_text("t_s,voltage_mv\n0,1.2\n1,1.2\n")
    d = {"energy": {"kind": "table", "file": "energy.csv"},
         "voltage": {"kind": "table", "file": "volts.csv"}, "gamma_t": 5.449167323}
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path, d, "f.json")),
                 "--json"]) == 0
    a = json.loads(capsys.readouterr().out)
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path)),
                 "--json"]) == 0
    b = json.loads(capsys.readouterr().out)
    assert a["t_end_s"] == b["t_end_s"] and a["gamma_end"] == b["gamma_end"]
    # a header that does not say what the column is, a table given twice
    (tmp_path / "bad.csv").write_text("t,energy\n0,800\n")
    d["energy"]["file"] = "bad.csv"
    assert main(["ring", "ramp", str(BOOSTER), "--ramp",
                 str(_ramp_file(tmp_path, d, "g.json"))]) == 2
    assert "the header must be 't_s,<column>'" in capsys.readouterr().err
    d["energy"] = {"kind": "table", "file": "energy.csv", "times_s": [0.0], "values": [800.0]}
    assert main(["ring", "ramp", str(BOOSTER), "--ramp",
                 str(_ramp_file(tmp_path, d, "h.json"))]) == 2
    assert "gives both a file and a table" in capsys.readouterr().err


def test_ring_run_through_a_ramp_to_its_end(tmp_path, capsys):
    from linac_gen.ring.results import load_ring_hdf5
    assert main(["ring", "run", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path)), "--to-end",
                 "--particles", "60", "--probes", "0", "--snapshot-turns", "0,last",
                 "--out", str(tmp_path), "--json", *_BEAM]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["turns_done"] == 268 and out["survival"] == 1.0
    r = out["ramp"]
    assert r["kinetic_start_mev"] == pytest.approx(800.0, rel=1e-15)
    assert r["turns_to_end"] == 268 and r["transition_turn"] is None and r["n_capped"] == 0
    res = load_ring_hdf5(Path(out["file"]))
    assert res.ramp["kinetic_end_mev"] == r["kinetic_end_mev"]
    assert sorted(res.snapshots) == [0, 268] and "reference" in res.snapshots[268]
    # normalized emittances with the reference of each row
    en = res.emit_normalized
    assert out["emit_ny_mm_mrad"] == [float(en[0, 1] * 1e6), float(en[-1, 1] * 1e6)]
    assert res.row_beta0gamma0[-1] > res.row_beta0gamma0[0]


def test_a_projects_ramp_is_used_unless_told_otherwise(tmp_path, capsys):
    proj = tmp_path / "ring.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {"ramp": _SHORT}}}))
    assert main(["ring", "ramp", str(proj), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["turns_to_end"] == 268
    assert main(["ring", "run", str(proj), "--turns", "3", "--particles", "20", "--probes", "0",
                 "--out", str(tmp_path), "--json", *_BEAM]) == 0
    cap = capsys.readouterr()
    assert json.loads(cap.out)["ramp"]["kinetic_start_mev"] == pytest.approx(800.0)
    assert main(["ring", "run", str(proj), "--turns", "3", "--particles", "20", "--probes", "0",
                 "--fixed-energy", "--out", str(tmp_path), "--json", *_BEAM]) == 0
    assert json.loads(capsys.readouterr().out)["ramp"] is None
    # --gamma-t replaces the ramp's (here below the start: above it from the start)
    assert main(["ring", "ramp", str(proj), "--gamma-t", "1.5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gamma_t"] == 1.5 and out["transition_turn"] is None
    assert out["above_transition_at_start"] and out["gamma_t_from"] == "from --gamma-t"
    assert main(["ring", "ramp", str(proj), "--gamma-t", "1.5"]) == 0
    assert "above it from the start" in capsys.readouterr().out
    assert main(["ring", "ramp", str(proj), "--fixed-energy"]) == 2
    assert "--fixed-energy asks for none" in capsys.readouterr().err
    # the project's steerer choice is kept with a ramp from a file, not at fixed energy
    proj2 = tmp_path / "ring2.lgproj"
    proj2.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                 "ring": {"enabled": True, "config": {
                                     "corrector_mode": "fixed_field"}}}))
    from linac_gen.ring.results import load_ring_hdf5
    for extra, want in ((["--ramp", str(_ramp_file(tmp_path))], "fixed_field"), ([], "follow")):
        assert main(["ring", "run", str(proj2), "--turns", "2", "--particles", "20", "--probes",
                     "0", "--out", str(tmp_path / want), "--json", *extra, *_BEAM]) == 0
        out = json.loads(capsys.readouterr().out)
        assert load_ring_hdf5(Path(out["file"])).meta["config"]["corrector_mode"] == want


def test_ramp_files_in_yaml(tmp_path, capsys):
    yaml = pytest.importorskip("yaml")
    d = {"energy": {"kind": "table", "times_s": [0.0, 5e-4], "values": [800.0, 830.0]},
         "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
         "gamma_t": 5.449167323}
    f = tmp_path / "ramp.yaml"
    # (YAML reads 5e-4 and 8.3e2 as text: they are numbers all the same)
    f.write_text(yaml.safe_dump(d).replace("0.0005", "5e-4").replace("830.0", "8.3e2"))
    assert "5e-4" in f.read_text() and "8.3e2" in f.read_text()
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(f), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["turns_to_end"] == 268 and out["kinetic_end_mev"] == pytest.approx(830.0,
                                                                                  rel=1e-6)


def test_ramp_refusals(tmp_path, capsys):
    f = str(_ramp_file(tmp_path))
    cases = [
        (["ring", "ramp", str(BOOSTER)], 2, "no ramp"),
        (["ring", "run", str(BOOSTER), "--to-end"], 2, "--to-end runs to the end of a ramp"),
        (["ring", "run", str(BOOSTER), "--ramp", f, "--to-end", "--turns", "5"], 2,
         "give --turns or --to-end"),
        (["ring", "run", str(BOOSTER), "--gamma-t", "5"], 2, "--gamma-t belongs to a ramp"),
        (["ring", "run", str(BOOSTER), "--ramp", f, "--fixed-energy"], 2, "give one of them"),
        (["ring", "run", str(BOOSTER), "--corrector-mode", "fixed_field"], 2,
         "--corrector-mode applies to a ramp"),
        (["ring", "ramp", str(BOOSTER), "--ramp", str(tmp_path / "none.json")], 2,
         "cannot read"),
        (["ring", "ramp", str(BOOSTER), "--ramp", f, "--turns", "0"], 2, "at least 1"),
    ]
    for argv, code, msg in cases:
        assert main([*argv, "--out", str(tmp_path)] if argv[1] == "run" else argv) == code, argv
        assert msg in capsys.readouterr().err, argv
    flat = _ramp_file(tmp_path, {"energy": {"kind": "flat"}, "gamma_t": 5.4}, "flat.json")
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(flat)]) == 1
    assert "flat, it has no end" in capsys.readouterr().err
    assert main(["ring", "run", str(BOOSTER), "--ramp", str(flat), "--to-end", "--out",
                 str(tmp_path), *_BEAM]) == 1
    assert "flat, it has no end" in capsys.readouterr().err
    # a program whose end is already past at turn 0: no turns, said plainly
    past = dict(_SHORT, t_start_s=0.001)
    pf = str(_ramp_file(tmp_path, past, "past.json"))
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", pf]) == 1
    assert "no turns to run: the clock at turn 0" in capsys.readouterr().err
    assert main(["ring", "run", str(BOOSTER), "--ramp", pf, "--to-end", "--out", str(tmp_path),
                 *_BEAM]) == 1
    assert "no turns to run: the clock at turn 0" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text("{not json")
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(tmp_path / "bad.json")]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    # a checkpoint of one ramp is not continued with another
    a = tmp_path / "a"
    assert main(["ring", "run", str(BOOSTER), "--ramp", f, "--turns", "4", "--particles", "20",
                 "--probes", "0", "--checkpoint-every", "2", "--out", str(a), *_BEAM]) == 0
    capsys.readouterr()
    other = dict(_SHORT, gamma_t=6.0)
    assert main(["ring", "run", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path, other, "o.json")),
                 "--turns", "4", "--particles", "20", "--probes", "0",
                 "--restart", str(a / "simple_booster_ring_checkpoint.npz"),
                 "--out", str(tmp_path / "b"), *_BEAM]) == 1
    assert "followed another ramp" in capsys.readouterr().err
    # a turn refused (no voltage left) stops the run: the turns done are written,
    # the status is 1
    drop = dict(_SHORT, voltage={"kind": "table", "times_s": [0.0, 0.0000091, 0.0000092, 1.0],
                                 "values_mv": [1.2, 1.2, 0.0, 0.0]},
                phase={"kind": "auto", "on_excess": "refuse"})
    assert main(["ring", "run", str(BOOSTER), "--ramp", str(_ramp_file(tmp_path, drop, "d.json")),
                 "--turns", "10", "--particles", "20", "--probes", "0", "--out",
                 str(tmp_path / "d"), "--json", *_BEAM]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["turns_done"] == 5 and out["truncated"]
    assert any("more than the RF gives" in n for n in out["notes"])
    from linac_gen.ring.results import load_ring_hdf5
    assert load_ring_hdf5(Path(out["file"])).turns_done == 5


# ---------------------------------------------------------------------------
# space charge
# ---------------------------------------------------------------------------
_SC = {"intensity": 1e10, "kicks_per_turn": 12, "grid": [16, 16]}


def test_ring_run_with_space_charge(tmp_path, capsys):
    f = tmp_path / "sc.json"
    f.write_text(json.dumps(_SC))
    assert main(["ring", "run", str(BOOSTER), "--space-charge", str(f), "--turns", "2",
                 "--particles", "300", "--probes", "0", "--out", str(tmp_path), "--json",
                 *_BEAM]) == 0
    out = json.loads(capsys.readouterr().out)
    sc = out["space_charge"]
    assert sc["model"] == "pic" and sc["kicks"] == 12 and sc["intensity"] == 1e10
    assert sc["bunched"] and sc["laslett_gaussian"][0] < 0.0
    # (the line density's bins: "auto" by default, the bins it used from turn 0)
    assert sc["z_bins"] == "auto" and sc["z_bins_used"][0][0] == 0
    from linac_gen.ring.results import load_ring_hdf5
    assert load_ring_hdf5(Path(out["file"])).meta["space_charge"]["grid"] == [16, 16]
    # the text summary says what it had
    assert main(["ring", "run", str(BOOSTER), "--space-charge", str(f), "--turns", "1",
                 "--particles", "100", "--probes", "0", "--out", str(tmp_path), *_BEAM]) == 0
    text = capsys.readouterr().out
    assert "space charge: 2.5-D particle-in-cell, 1e+10 particles in the bunch, 12 kicks" in text
    assert "analytic tune shift at the centre of a Gaussian bunch: dQx -" in text
    assert "line density: bins following the bunch (auto) — " in text
    assert " from turn 0" in text
    # --intensity alone: the default settings with it
    assert main(["ring", "run", str(BOOSTER), "--intensity", "2e9", "--turns", "1",
                 "--particles", "100", "--probes", "0", "--out", str(tmp_path), "--json",
                 *_BEAM]) == 0
    sc = json.loads(capsys.readouterr().out)["space_charge"]
    assert sc["intensity"] == 2e9 and sc["grid"] == [64, 64]


def test_a_coasting_beams_space_charge_from_the_current(tmp_path, capsys):
    deck = tmp_path / "coasting.madx"
    text = BOOSTER.read_text()
    assert text.count("total_voltage = 0.2;") == 1
    deck.write_text(text.replace("total_voltage = 0.2;", "total_voltage = 0.0;"))
    f = tmp_path / "sc.json"
    f.write_text(json.dumps({"kicks_per_turn": 12, "grid": [16, 16]}))
    args = ["ring", "run", str(deck), "--space-charge", str(f), "--turns", "1", "--particles",
            "200", "--probes", "0", "--out", str(tmp_path), *_BEAM, "--beam", "current=5"]
    assert main([*args, "--json"]) == 0
    sc = json.loads(capsys.readouterr().out)["space_charge"]
    assert not sc["bunched"] and not sc["longitudinal_kick"] and sc["intensity"] is None
    t_rev = 474.202752 / (0.841810706 * 299792458.0)
    assert sc["intensity_used"] == pytest.approx(5e-3 * t_rev / 1.602176634e-19, rel=1e-6)
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "particles in the ring (the beam's current times the revolution time)" in out
    assert "no longitudinal kick (a coasting beam)" in out
    assert "analytic tune shift of a coasting Gaussian beam: dQx -" in out


def test_a_projects_space_charge_is_used_unless_told_otherwise(tmp_path, capsys):
    proj = tmp_path / "ring.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {"space_charge": _SC}}}))
    args = ["--turns", "1", "--particles", "100", "--probes", "0", "--out", str(tmp_path),
            "--json", *_BEAM]
    assert main(["ring", "run", str(proj), *args]) == 0
    assert json.loads(capsys.readouterr().out)["space_charge"]["kicks"] == 12
    assert main(["ring", "run", str(proj), *args[:-1 - len(_BEAM)], *_BEAM]) == 0
    assert "space charge: the project's" in capsys.readouterr().err
    assert main(["ring", "run", str(proj), "--no-space-charge", *args]) == 0
    assert json.loads(capsys.readouterr().out)["space_charge"] is None
    # --intensity replaces the project's intensity, keeping its other settings
    assert main(["ring", "run", str(proj), "--intensity", "3e9", *args]) == 0
    sc = json.loads(capsys.readouterr().out)["space_charge"]
    assert sc["intensity"] == 3e9 and sc["kicks"] == 12


def test_space_charge_refusals(tmp_path, capsys):
    f = tmp_path / "sc.json"
    f.write_text(json.dumps(dict(_SC, pipe="round")))
    assert main(["ring", "run", str(BOOSTER), "--space-charge", str(f), "--turns", "1",
                 "--out", str(tmp_path), *_BEAM]) == 2
    assert "pipe_a_m" in capsys.readouterr().err
    assert main(["ring", "run", str(BOOSTER), "--no-space-charge", "--intensity", "1e9",
                 "--turns", "1", "--out", str(tmp_path), *_BEAM]) == 2
    assert "give one of them" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text("{not json")
    assert main(["ring", "run", str(BOOSTER), "--space-charge", str(tmp_path / "bad.json"),
                 "--turns", "1", "--out", str(tmp_path), *_BEAM]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    # no intensity at all (no current in the beam): refused before the run
    f2 = tmp_path / "noint.json"
    f2.write_text(json.dumps({"kicks_per_turn": 4}))
    assert main(["ring", "run", str(BOOSTER), "--space-charge", str(f2), "--turns", "1",
                 "--particles", "50", "--out", str(tmp_path), *_BEAM]) == 1
    assert "intensity" in capsys.readouterr().err


def test_space_charge_with_the_beams_current_as_intensity(tmp_path, capsys):
    # (intensity left out: the beam's current over its frequency; the summary
    # reports the intensity used, in text and in JSON)
    f = tmp_path / "sc.json"
    f.write_text(json.dumps({"kicks_per_turn": 6, "grid": [16, 16]}))
    args = ["ring", "run", str(BOOSTER), "--space-charge", str(f), "--turns", "1",
            "--particles", "50", "--probes", "0", "--out", str(tmp_path), *_BEAM,
            "--beam", "current=5"]
    assert main(args) == 0
    text = capsys.readouterr().out
    want = 5e-3 / 44.7044096e6 / 1.602176634e-19
    assert f"{want:.6g} particles in the bunch (the beam's current over its frequency)" in text
    assert main([*args, "--json"]) == 0
    sc = json.loads(capsys.readouterr().out)["space_charge"]
    assert sc["intensity"] is None and sc["intensity_used"] == pytest.approx(want, rel=1e-12)


def test_an_injection_from_a_file_fills_the_ring(tmp_path, capsys):
    deck = tmp_path / "foil.madx"
    text = BOOSTER.read_text()
    line = "cell01 : line = (sa, "
    assert text.count(line) == 1
    deck.write_text(text.replace("sa: drift, l=0.176", "foilm: marker;\nsa: drift, l=0.176")
                    .replace(line, "cell01 : line = (foilm, sa, "))
    f = tmp_path / "inj.json"
    f.write_text(json.dumps({
        "foil": {"at": "foilm", "thickness_ug_cm2": 600.0, "x_min_m": -0.01, "x_max_m": 0.01,
                 "y_min_m": -0.01, "y_max_m": 0.01},
        "source": {"kind": "gaussian", "species": "H-", "kinetic_mev": 800.0,
                   "frequency_mhz": 162.5, "emit_nx_um": 0.3, "emit_ny_um": 0.3,
                   "beta_x_m": 30.0, "beta_y_m": 5.0},
        "turns": 3, "particles_per_turn": 40, "current_ma": 2.0}))
    base = ["ring", "run", str(deck), "--turns", "4", "--probes", "0", "--out", str(tmp_path)]
    args = [*base, "--injection", str(f)]
    assert main([*args, "--json"]) == 0

    def no_nan(name):
        raise AssertionError(f"not strict JSON: {name}")

    # (strict JSON: the start's emittances of a ring that starts empty are null)
    out = json.loads(capsys.readouterr().out, parse_constant=no_nan)
    assert out["emit_nx_mm_mrad"][0] is None and out["emit_nx_mm_mrad"][1] > 0.0
    inj = out["injection"]
    assert inj["injected"] == 120 and inj["circulating"] + sum(
        inj["waste"][k] for k in ("missed", "unstripped")) == 120
    # (the real particles: the ring's, 84 buckets each as the one tracked)
    assert inj["buckets"] == 84
    assert inj["injected_particles"] == pytest.approx(
        3 * 2e-3 * 474.202752 / (0.841810706 * 299792458.0) / 1.602176634e-19, rel=1e-6)
    assert inj["electron_mev"] > 100.0 * inj["deposit_mev"] > 0.0
    assert out["particles"] == inj["circulating"]
    assert inj["foil"]["name"] == "foilm" and inj["foil"]["theta0_rad"] > 0.0
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "foil foilm: theta0" in text and "injected 120 macroparticles" in text
    assert "particles into the ring (84 filled buckets" in text
    assert "ionization energy left in the foil" in text and "electrons carry" in text
    assert "normalized rms emittance at the end" in text
    # (no injection: a matched beam, the marker a marker)
    assert main([*base, "--no-injection", "--particles", "50", *_BEAM, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["injection"] is None
    assert main([*args, "--no-injection"]) == 2
    assert "give one of them" in capsys.readouterr().err


def test_the_manuals_command_line_page_lists_every_option():
    """docs/manual/17_ring/10_cli.md documents every option of ring optics,
    ring ramp and ring run (an option added to the parser fails this until
    the page has it)."""
    import argparse

    from linac_gen.cli import ring as ring_cmd
    p = argparse.ArgumentParser()
    ring_cmd.add_arguments(p)
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {"optics", "ramp", "run"}
    page = (ROOT / "docs" / "manual" / "17_ring" / "10_cli.md").read_text(encoding="utf-8")
    missing = sorted({opt for name, sp in sub.choices.items() for a in sp._actions
                      for opt in a.option_strings
                      if opt.startswith("--") and opt != "--help" and f"`{opt}" not in page})
    assert not missing, missing


def test_a_bare_deck_says_its_beam_settings_are_the_defaults(tmp_path, capsys):
    """(the matched beam's emittances and bunch length come from the beam
    settings: a bare lattice file runs the defaults, and says so)"""
    args = ["ring", "run", str(BOOSTER), "--turns", "1", "--particles", "100", "--probes", "0",
            "--out", str(tmp_path)]
    assert main(args) == 0
    err = capsys.readouterr().err
    assert "note: the beam settings are the defaults (normalized rms emittances 0.25" in err
    assert main([*args, *_BEAM]) == 0
    assert "the beam settings are the defaults" not in capsys.readouterr().err
    assert main([*args, "--json"]) == 0                  # (JSON: no note on stderr)
    assert "the beam settings are the defaults" not in capsys.readouterr().err


def test_every_ring_command_prints_strict_json(tmp_path, capsys):
    """(a number that is not finite is null: the 4-D optics have no
    synchrotron tune)"""
    def strict(text):
        return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert main(["ring", "optics", str(BOOSTER), "--json"]) == 0
    assert strict(capsys.readouterr().out)["qs"] is None
    ramp = tmp_path / "ramp.json"
    ramp.write_text(json.dumps(_SHORT))
    assert main(["ring", "ramp", str(BOOSTER), "--ramp", str(ramp), "--json"]) == 0
    strict(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# restarts take the checkpoint's seed; clear errors; a project's probes
def test_a_restart_takes_the_checkpoints_seed_and_refuses_another(tmp_path, capsys):
    from linac_gen.ring.results import load_ring_hdf5
    args = ["ring", "run", str(BOOSTER), "--particles", "60", "--probes", "0", "--json",
            "--source", "matched", *_BEAM]
    assert main([*args, "--turns", "4", "--seed", "7", "--checkpoint-every", "2",
                 "--out", str(tmp_path / "full")]) == 0
    capsys.readouterr()
    ck = tmp_path / "full" / "simple_booster_ring_checkpoint.npz"
    assert main([*args, "--turns", "4", "--restart", str(ck),
                 "--out", str(tmp_path / "again")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert load_ring_hdf5(Path(out["file"])).meta["config"]["seed"] == 7
    assert main([*args, "--turns", "4", "--restart", str(ck), "--seed", "3",
                 "--out", str(tmp_path / "other")]) == 1      # (refused by the run, as probes)
    err = capsys.readouterr().err
    assert "seed=7" in err and "same settings" in err


def test_bad_inputs_give_an_error_line_not_a_traceback(tmp_path, capsys):
    run = ["ring", "run", str(BOOSTER), "--turns", "2", "--particles", "20", "--probes", "0",
           *_BEAM]
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"PK\x03\x04 not a zip file at all")
    assert main([*run, "--restart", str(bad), "--out", str(tmp_path)]) == 2
    assert "not a ring checkpoint" in capsys.readouterr().err
    a_file = tmp_path / "taken"
    a_file.write_text("x")
    assert main([*run, "--out", str(a_file)]) == 2
    assert "--out" in capsys.readouterr().err
    lost = tmp_path / "moved.lgproj"
    lost.write_text(json.dumps({"__kind__": "linac_gen_project",
                                "lattice_path": str(tmp_path / "no_such_deck.madx")}))
    assert main(["ring", "optics", str(lost)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_particles_with_an_empty_start_are_refused(tmp_path, capsys):
    inj = tmp_path / "inj.json"
    inj.write_text(json.dumps({"foil": {"at": "sa", "thickness_ug_cm2": 600.0},
                               "source": {"kind": "gaussian", "species": "H-",
                                          "kinetic_mev": 800.0, "frequency_mhz": 162.5,
                                          "emit_nx_um": 0.3, "emit_ny_um": 0.3,
                                          "beta_x_m": 30.0, "beta_y_m": 5.0},
                               "turns": 2, "particles_per_turn": 10, "current_ma": 2.0}))
    assert main(["ring", "run", str(BOOSTER), "--turns", "2", "--injection", str(inj),
                 "--source", "empty", "--particles", "30", "--out", str(tmp_path),
                 *_BEAM]) == 2
    assert "--particles" in capsys.readouterr().err


def test_a_projects_saved_probes_are_used(tmp_path, capsys):
    from linac_gen.ring.results import load_ring_hdf5
    proj = tmp_path / "ring.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {
                                    "source": "matched", "tune_probes": 2}}}))
    assert main(["ring", "run", str(proj), "--turns", "3", "--particles", "40", "--json",
                 "--source", "matched", "--out", str(tmp_path), *_BEAM]) == 0
    res = load_ring_hdf5(Path(json.loads(capsys.readouterr().out)["file"]))
    assert res.meta["config"]["tune_probes"] == 2
    assert main(["ring", "run", str(proj), "--turns", "3", "--particles", "40", "--json",
                 "--source", "matched", "--probes", "1", "--out", str(tmp_path), *_BEAM]) == 0
    res = load_ring_hdf5(Path(json.loads(capsys.readouterr().out)["file"]))
    assert res.meta["config"]["tune_probes"] == 1


def test_the_ring_commands_threads_help_is_ring_modes(capsys):
    with pytest.raises(SystemExit):
        main(["ring", "--help"])
    out = capsys.readouterr().out
    assert "ring kernels" in out and "field maps" not in out


def test_a_deck_whose_beam_gives_no_energy_needs_one(tmp_path, capsys):
    # (MAD-X: a BEAM with only the particle — the magnets were scaled with the
    # importer's default energy, which says nothing about the ring's)
    deck = tmp_path / "ring.madx"
    deck.write_text(BOOSTER.read_text().replace("beam, particle=proton, energy=0.8+pmass;",
                                                "beam, particle=proton;"))
    assert main(["ring", "optics", str(deck)]) == 2
    assert "--energy" in capsys.readouterr().err
    assert main(["ring", "optics", str(deck), "--energy", "800", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kinetic_energy_mev"] == pytest.approx(800.0)
    # (the magnets keep the deck's normalized strengths: the same optics)
    assert out["qx"] == pytest.approx(REF["q1"], abs=1e-6)
    assert out["qy"] == pytest.approx(REF["q2"], abs=1e-6)
