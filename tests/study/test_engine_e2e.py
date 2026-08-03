"""End-to-end StudyManager: layout, resume, both modes, failure rows.

Dual-regime rule: the engine is exercised in BOTH envelope and mp
(current > 0, SC active) branches — every branch untested is a branch
wrong.
"""
import json

import numpy as np
import pytest

from linac_gen.study.engine import StudyManager
from linac_gen.study.spec import ObservableSpec, ParamSpec, StudySpec

FODO = """DRIFT 100 20 0 0 0
QUAD 80 8.0 20 0 0 0 0 0 0
DRIFT 200 20 0 0 0
QUAD 80 -8.0 20 0 0 0 0 0 0
DRIFT 100 20 0 0 0
END
"""


@pytest.fixture()
def deck(tmp_path):
    p = tmp_path / "fodo.dat"
    p.write_text(FODO)
    return p


def _spec(deck, mode="envelope", **kw):
    base = dict(
        name="e2e", input=str(deck), mode=mode, strategy="grid",
        parameters=[ParamSpec(selector="@2.gradient", start=7.0,
                              stop=9.0, n=3)],
        observables=[ObservableSpec(name="sx_mid", quantity="sigma_x",
                                    at={"s_m": 0.28})],
        beam={"energy": 2.1, "current": 0.0, "n_particles": 400},
        numerics={"nx": 32},
    )
    base.update(kw)
    return StudySpec(**base)


def _run(tmp_path, deck, mode="envelope", **kw):
    mgr = StudyManager.create(tmp_path / "study", _spec(deck, mode, **kw))
    mgr.run(serial=True)
    return mgr


class TestEnvelopeE2E:
    def test_layout_and_summary(self, tmp_path, deck):
        mgr = _run(tmp_path, deck)
        sd = tmp_path / "study"
        assert (sd / "study.json").exists()
        assert (sd / "summary" / "runs_manifest.json").exists()
        run_dirs = sorted((sd / "runs").iterdir())
        assert len(run_dirs) == 3
        for rd in run_dirs:
            assert (rd / "results.h5").exists()
            st = json.loads((rd / "status.json").read_text())
            assert st["status"] == "ok"
            assert st["metrics"]["sigma_x"] is not None
        rows = (sd / "summary" / "summary.csv").read_text().splitlines()
        assert len(rows) == 4                       # header + 3
        assert "sx_mid" in rows[0]
        # physics: stronger focusing quad -> smaller sigma_x at exit
        import csv as _csv
        recs = list(_csv.DictReader(rows))
        sx = [float(r["sigma_x"]) for r in recs]
        assert sx[0] > sx[-1]
        # observable evaluated (s-units mm/m handling pinned here)
        assert all(0.1 < float(r["sx_mid"]) < 50.0 for r in recs)

    def test_resume_runs_only_missing(self, tmp_path, deck):
        mgr = _run(tmp_path, deck)
        victim = sorted((tmp_path / "study" / "runs").iterdir())[1]
        (victim / "status.json").unlink()
        mgr2 = StudyManager.load(tmp_path / "study")
        assert len(mgr2.pending()) == 1
        mgr2.run(serial=True)
        assert len(mgr2.pending()) == 0

    def test_orphan_part_swept(self, tmp_path, deck):
        mgr = _run(tmp_path, deck)
        victim = sorted((tmp_path / "study" / "runs").iterdir())[0]
        (victim / "status.json").unlink()
        (victim / "results.h5.part").write_bytes(b"torn")
        mgr2 = StudyManager.load(tmp_path / "study")
        mgr2.run(serial=True)
        assert not (victim / "results.h5.part").exists()
        assert json.loads(
            (victim / "status.json").read_text())["status"] == "ok"

    def test_lattice_drift_refused(self, tmp_path, deck):
        _run(tmp_path, deck)
        deck.write_text(FODO.replace("8.0", "8.5"))
        with pytest.raises(RuntimeError, match="changed since"):
            StudyManager.load(tmp_path / "study")

    def test_spec_edit_refused(self, tmp_path, deck):
        _run(tmp_path, deck)
        sp = tmp_path / "study" / "study.json"
        doc = json.loads(sp.read_text())
        doc["parameters"][0]["n"] = 5
        sp.write_text(json.dumps(doc))
        with pytest.raises(RuntimeError, match="run plan"):
            StudyManager.load(tmp_path / "study")

    def test_provenance_pins_lattice_sha(self, tmp_path, deck):
        from linac_gen.study.observables import read_provenance
        mgr = _run(tmp_path, deck)
        rd = sorted((tmp_path / "study" / "runs").iterdir())[0]
        prov = read_provenance(str(rd / "results.h5"))
        assert prov.get("lattice_sha256") == mgr.spec.lattice_sha256


class TestMpE2E:
    def test_mp_with_sc_and_repeats(self, tmp_path, deck):
        mgr = StudyManager.create(
            tmp_path / "study",
            _spec(deck, mode="mp", repeats=2,
                  beam={"energy": 2.1, "current": 5.0,
                        "n_particles": 300}))
        assert len(mgr.plan()) == 6
        mgr.run(serial=True)
        assert len(mgr.pending()) == 0
        st = json.loads(
            (sorted((tmp_path / "study" / "runs").iterdir())[0]
             / "status.json").read_text())
        assert st["status"] == "ok"
        assert st["metrics"]["transmission"] is not None
        seeds = {json.loads((rd / "status.json").read_text())["seed"]
                 for rd in (tmp_path / "study" / "runs").iterdir()}
        assert seeds == {42, 43}


class TestValidation:
    def test_bad_selector_refused_before_any_run(self, tmp_path, deck):
        spec = _spec(deck)
        spec.parameters[0].selector = "NOSUCH.gradient"
        with pytest.raises(ValueError):
            StudyManager.create(tmp_path / "study", spec)
        assert not (tmp_path / "study" / "runs").exists()

    def test_int_attr_truncation_guard(self, tmp_path, deck):
        spec = _spec(deck)
        spec.parameters = [ParamSpec(selector="@2.n_steps",
                                     start=10.0, stop=11.5, n=2)]
        with pytest.raises(ValueError, match="[Ii]nteger"):
            StudyManager.create(tmp_path / "study", spec)

    def test_failed_run_is_a_row_not_a_crash(self, tmp_path, deck):
        spec = _spec(deck)
        # sweep into a nonsense negative n that BeamConfig rejects at
        # run time (validation can't see beam blowups, only shapes)
        spec.beam = {"energy": 2.1, "current": 0.0,
                     "n_particles": 400, "cutoff": -4.0}
        mgr = StudyManager.create(tmp_path / "study", spec)
        mgr.run(serial=True)
        stats = [json.loads((rd / "status.json").read_text())["status"]
                 for rd in sorted((tmp_path / "study" / "runs").iterdir())]
        assert stats == ["failed"] * 3
        # --retry-failed re-queues them
        assert len(mgr.pending()) == 0
        assert len(mgr.pending(retry_failed=True)) == 3

    def test_element_observable_resolved(self, tmp_path, deck):
        spec = _spec(deck)
        spec.observables = [ObservableSpec(
            name="sx_q2", quantity="sigma_x",
            at={"element": "QUAD_002"})]
        mgr = StudyManager.create(tmp_path / "study", spec)
        assert mgr.spec.observables[0].s_m == pytest.approx(0.46, abs=1e-6)
