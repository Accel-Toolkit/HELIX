"""The ring-mode example projects (``examples/ring/*.lgproj``): each opens —
its lattice found, its ring settings valid ring settings — and runs from the
command line with the project's own beam, ramp and space charge (two turns of
a few particles); its Beam tab holds the beam matched to the ring at the start
of its run (BMAG 1); the ramp's clock puts transition where the example
says."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from linac_gen.__main__ import main
from linac_gen.io.project import load_project
from linac_gen.ring.config import RingConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "ring"
PROJECTS = ("simple_booster_storage_sc", "simple_booster_transition_sc")


def test_every_ring_example_is_checked():
    assert sorted(p.stem for p in EXAMPLES.glob("*.lgproj")) == sorted(PROJECTS)


@pytest.mark.parametrize("name", PROJECTS)
def test_a_ring_example_opens_and_runs(name, tmp_path, capsys):
    proj = EXAMPLES / f"{name}.lgproj"
    data = json.loads(proj.read_text())
    assert data["ring"]["enabled"] is True
    rc = RingConfig.from_dict(dict(data["ring"]["config"]))
    # (the run takes the Beam tab's beam: 5000 particles)
    assert rc.source == "beam" and "n_particles" not in data["ring"]["config"]
    assert data["beam"]["n_particles"] == 5000 and rc.space_charge["z_bins"] == "auto"
    assert Path(load_project(proj).lattice_path).is_file()
    assert main(["ring", "run", str(proj), "--turns", "2", "--particles", "200",
                 "--out", str(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["turns_done"] == 2 and out["survival"] == 1.0 and out["particles"] == 200
    assert out["species"] == "proton" and out["space_charge"]["intensity"] == 1.5e10
    assert (out["ramp"] is not None) == (rc.ramp is not None)
    # the Beam tab's beam is the one matched to the ring at the start of the
    # run, at the run's starting energy (no note about another energy)
    assert out["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=2e-3)
    assert data["beam"]["energy"] == pytest.approx(out["kinetic_mev"], rel=1e-9)
    assert not any("generated at the ring's reference" in n for n in out["notes"])


def test_the_transition_example_crosses_at_turn_300(capsys):
    proj = EXAMPLES / "simple_booster_transition_sc.lgproj"
    assert main(["ring", "ramp", str(proj), "--turns", "600", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["transition_turn"] == 300
    assert out["kinetic_start_mev"] == pytest.approx(4059.111, abs=1e-3)
