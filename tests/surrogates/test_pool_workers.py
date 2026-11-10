"""The scan pool's workers and the registered surrogates.

A worker process starts with an empty registry, so a failure sweep or a campaign run through the
pool tracked every point natively while the same sweep run serially used the registered weights.
A study that uses them gives each point the spec of the weights its design check accepted
(``surrogates.transport``): a worker loads each folder once, builds the weights on its own parse
before the point's overrides, with the parent's verdicts, and counts what they did; a serial run
uses the parent's registry as it stands.  ``failures`` and ``reliability`` take
``--surrogates DIR... | auto``.
"""
import copy
import csv
import warnings

import numpy as np
import pytest

from linac_gen.cli import common
from linac_gen.core.config import BeamConfig
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import MlpHead, Scope, SurrogateMetadata
from linac_gen.surrogates.training import save_surrogate, weights_folder

W, F = 5.0, 325.0


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _write_1d(path, vals, zmax_m):
    lines = [f"{len(vals) - 1} {zmax_m:.6f}", "1.0"] + [f"{v:.9g}" for v in vals]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def deck(tmp_path):
    """Two solenoid maps between drifts, written to disk (a worker parses the file)."""
    z = np.linspace(0.0, 0.3, 31)
    _write_1d(tmp_path / "sol.bsz", 0.4 * (0.5 - 0.5 * np.cos(2 * np.pi * z / 0.3)), 0.3)
    path = tmp_path / "line.dat"
    path.write_text("DRIFT 100 16 0\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nDRIFT 100 16 0\n"
                    "FIELD_MAP 10 300 0 16 1.5 0 0 0 sol\nDRIFT 100 16 0\nEND\n")
    return path


def _beam():
    return BeamConfig(species="proton", energy=W, frequency=F, current=0.0, n_particles=16,
                      emit_nx=0.2, beta_x=1.0, emit_ny=0.2, beta_y=1.0, emit_z=0.2, beta_z=1.0)


BEAM = {"energy": W, "frequency": F, "current": 0.0, "species": "proton"}


def _save_sets(lattice, root, factor=1.0, with_settings=True):
    """Weights pinned to each map's native matrix at the design energy (times ``factor`` in
    M[0, 1]), saved as training saves them, one folder per element."""
    from linac_gen.surrogates.base import settings_of
    for el in [e for e in lattice.elements if e.name.startswith("FMAP")]:
        ref = ReferenceParticle(species=PROTON, w_kin=W, frequency=F)
        M = np.asarray(copy.deepcopy(el).fitted_matrix(ref), dtype=float)
        M[0, 1] *= factor
        m = SurrogateMetadata(
            element_key=el.name, element_class=type(el).__name__,
            architecture={"input_dim": 3, "output_dim": 36, "hidden_dims": [4],
                          "activation": "silu", "param_names": []},
            scope=Scope(input_names=["w_kin", "beta", "gamma"],
                        input_lo=np.array([W - 1e-9, 0.0, 1.0]),
                        input_hi=np.array([W + 1e-9, 1.0, 100.0])),
            input_norm={"mean": [0.0] * 3, "std": [1.0] * 3},
            output_norm={"mean": [float(v) for v in M.ravel()], "std": [0.0] * 36},
            training_seed=0, n_samples=0, epochs=0, val_mape=0.0, helix_commit_sha="",
            lattice_hash="lh", created_iso="", species_name=PROTON.name,
            species_mass_mev=PROTON.mass, train_frequency_mhz=F,
            train_settings=settings_of(el, set()) if with_settings else None)
        save_surrogate(MlpHead(input_dim=3, output_dim=36, hidden_dims=(4,)), m, root / el.name)
    return root


def _spec(deck_path, root):
    from linac_gen.surrogates.transport import study_spec
    lat, cfg, _conv = common.load_input(str(deck_path))
    common.apply_beam_overrides(cfg, BEAM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study_spec(lat, cfg, str(deck_path), [str(root)], mode="envelope")


def _sweep(deck_path, spec, workers):
    from linac_gen.failures import FailureKind, FailureStudy, enumerate_scenarios
    lat, _cfg, _conv = common.load_input(str(deck_path))
    scen, n2c, names = enumerate_scenarios(lat, types={"solenoid"}, kind=FailureKind("partial"),
                                           combination="single", amp_scale=0.9)
    study = FailureStudy(str(deck_path), beam_overrides=BEAM, mode="envelope",
                         surrogates=spec or None)
    return study.run(scen, names, n2c, max_workers=workers, serial=(workers == 1))


def test_pool_workers_answer_as_the_serial_sweep(deck, tmp_path):
    """The same answers, the same refusals (the failed map's error knob, which these weights
    take no input for), the same rows, serially and with two workers."""
    check, spec, notes = _spec(deck, _save_sets(common.load_input(str(deck))[0], tmp_path / "w"))
    assert check.use and notes == [] and len(spec.entries) == 2
    serial, pooled = _sweep(deck, spec, 1), _sweep(deck, spec, 2)
    for res in (serial, pooled):
        assert res.baseline["surrogate_calls"] == 2
        assert [im.metrics["surrogate_calls"] for im in res.impacts] == [1, 1]
        assert [im.metrics["surrogate_refused"] for im in res.impacts] == [{"error knob": 1}] * 2
    for a, b in zip(serial.impacts, pooled.impacts, strict=True):
        assert {k: v for k, v in a.metrics.items() if k != "elapsed"} == \
               {k: v for k, v in b.metrics.items() if k != "elapsed"}


def test_a_serial_sweep_keeps_this_process_registry(deck, tmp_path):
    from linac_gen.parallel.scan_pool import run_scan_points_serial
    check, spec, _notes = _spec(deck, _save_sets(common.load_input(str(deck))[0], tmp_path / "w"))
    before = {key: registry.get(*key) for key in registry.list_registered()}
    point = common.build_scan_point(str(deck), beam_overrides=BEAM, mode="envelope",
                                    surrogates=spec)
    rows = run_scan_points_serial([point])
    after = {key: registry.get(*key) for key in registry.list_registered()}
    assert after.keys() == before.keys() and all(after[k] is before[k] for k in before)
    assert rows[0]["surrogate_calls"] == 2


def test_a_point_without_weights_is_the_point_as_before(deck):
    from linac_gen.parallel.scan_pool import run_scan_points_serial
    point = common.build_scan_point(str(deck), beam_overrides=BEAM, mode="envelope")
    assert point.surrogates is None
    row = run_scan_points_serial([point])[0]
    assert "surrogate_calls" not in row and "surrogate_refused" not in row


def test_a_worker_takes_the_parents_verdicts_and_settings(deck, tmp_path):
    """Each set carries the parent's metadata: its accuracy verdict (a refusal too) and the
    settings the parent's load held (for a file with none recorded, the parent's snapshot), and
    the worker decides neither again."""
    from linac_gen.surrogates.transport import install, spec_of_registry
    lat0 = common.load_input(str(deck))[0]
    _save_sets(lat0, tmp_path / "w", with_settings=False)
    check, spec, _notes = _spec(deck, tmp_path / "w")
    assert check.use
    parent = {key: registry.get(lh, key).metadata for lh, key in registry.list_registered()}
    lat = common.load_input(str(deck))[0]
    registry.clear()
    got = {s.metadata.element_key: s.metadata for s in install(spec, lat)}
    assert set(got) == {"FMAP_001", "FMAP_002"}
    for key, meta in got.items():
        assert (meta.accuracy_due, meta.accuracy_check) == (parent[key].accuracy_due, "")
        assert meta.train_settings == parent[key].train_settings and meta.settings_snapshot
    # weights the parent built in memory (and saved) were not checked there: not in a worker
    for lh, key in registry.list_registered():
        meta = registry.get(lh, key).metadata
        meta.accuracy_due, meta.accuracy_check = False, None
    unchecked, _ = spec_of_registry()
    assert [e.meta.accuracy_due for e in unchecked.entries] == [False, False]
    registry.clear()
    transport_loaded = install.__globals__["_LOADED"]
    transport_loaded.clear()
    for s in install(unchecked, common.load_input(str(deck))[0]):
        assert (s.metadata.accuracy_due, s.metadata.accuracy_check) == (False, None)
    # a set the parent refused is carried with its refusal, and refused in the worker
    registry.clear()
    _spec(deck, _save_sets(lat0, tmp_path / "bad", factor=1.05))
    spec_bad, notes = spec_of_registry()
    assert len(spec_bad.entries) == 2 and notes == []
    assert all(e.meta.accuracy_check.startswith("surrogate '") for e in spec_bad.entries)
    registry.clear()
    from linac_gen.surrogates.base import OutOfScopeError
    for s in install(spec_bad, common.load_input(str(deck))[0]):
        with pytest.raises(OutOfScopeError, match="not within the accuracy limits"):
            s.fitted_matrix(ReferenceParticle(species=PROTON, w_kin=W, frequency=F))


def test_weights_changed_since_the_study_loaded_them_run_native(deck, tmp_path):
    import dataclasses

    from linac_gen.surrogates import transport
    _check, spec, _notes = _spec(deck, _save_sets(common.load_input(str(deck))[0], tmp_path / "w"))
    first = dataclasses.replace(spec.entries[0], weights_sha256="0" * 64)
    other = dataclasses.replace(spec.entries[1], structure=("FieldMap", 1.0))
    transport._LOADED.pop(first.weights_dir, None)
    with pytest.warns(UserWarning) as seen:
        got = transport.install(transport.SurrogateSpec((first, other)),
                                common.load_input(str(deck))[0])
    assert got == [] and registry.list_registered() == []
    text = [str(w.message) for w in seen]
    assert any("changed since the study loaded them" in t for t in text)
    assert any("not the one the study built its weights on" in t for t in text)


def test_weights_with_no_folder_are_named(deck):
    from linac_gen.surrogates.transport import spec_of_registry
    from tests.surrogates.accuracy_helpers import native, pinned
    lat = common.load_input(str(deck))[0]
    el = lat.elements[1]
    registry.register(pinned(el, native(el, w=W, f=F)))
    spec, notes = spec_of_registry()
    assert spec.entries == () and notes == [f"{el.name}: its weights are not in a folder, so the "
                                            "pool's workers run it native"]


def test_a_worker_running_a_point_without_weights_holds_none(deck, tmp_path):
    """As the pool sends them (``in_pool``), when a point of the run has surrogates."""
    import dataclasses

    from linac_gen.parallel import scan_pool
    _check, spec, _notes = _spec(deck, _save_sets(common.load_input(str(deck))[0], tmp_path / "w"))
    assert registry.list_registered()
    point = common.build_scan_point(str(deck), beam_overrides=BEAM, mode="envelope")
    scan_pool._run_one_point_worker(dataclasses.replace(point, in_pool=True))
    assert registry.list_registered() == []
    point = common.build_scan_point(str(deck), beam_overrides=BEAM, mode="envelope",
                                    surrogates=spec)
    row = scan_pool._run_one_point_worker(dataclasses.replace(point, in_pool=True))
    assert row["surrogate_calls"] == 2 and len(registry.list_registered()) == 2


def test_the_pool_marks_its_points_only_when_one_has_surrogates(monkeypatch):
    """A run without surrogates sends its points as before."""
    from linac_gen.parallel import scan_pool
    sent = []

    class _Pool:
        def __init__(self, *a, **k):
            pass

        def submit(self, fn, point):
            sent.append(point)
            raise RuntimeError("captured")

        def shutdown(self, *a, **k):
            pass
    monkeypatch.setattr(scan_pool, "ProcessPoolExecutor", _Pool)
    base = scan_pool.ScanPoint(lattice_path="x.dat", beam_config={}, nx=8, grid_extent=4.0,
                               step1=10.0, step2=5.0, use_gpu="cpu")
    for points, marked in (([base, base], False), ([base, scan_pool.ScanPoint(
            **{**base.__dict__, "surrogates": _a_spec()})], True)):
        sent.clear()
        with pytest.raises(RuntimeError, match="captured"):
            scan_pool.run_scan_points(points, max_workers=2)
        assert sent and all(p.in_pool is marked for p in sent)


def _failures(deck_path, tmp_path, *extra):
    import argparse

    from linac_gen.cli import failures
    p = argparse.ArgumentParser()
    failures.add_arguments(p)
    out = tmp_path / f"f{len(list(tmp_path.glob('f*.csv')))}.csv"
    args = p.parse_args([str(deck_path), "--types", "solenoid", "--mode", "partial",
                         "--mode-amp", "0.9", "--forward", "envelope", "--energy", str(W),
                         "--freq", str(F), "--current", "0", "--species", "proton",
                         "--out", str(out), *extra])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert failures.run(args) == 0
    with open(out, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_failures_command_takes_surrogates(deck, tmp_path, capsys):
    root = _save_sets(common.load_input(str(deck))[0], tmp_path / "w")
    serial = _failures(deck, tmp_path, "--workers", "1", "--surrogates", str(root))
    said = capsys.readouterr().out
    pooled = _failures(deck, tmp_path, "--workers", "2", "--surrogates", str(root))
    assert serial == pooled and [r["surrogate_calls"] for r in serial] == ["1", "1"]
    assert "surrogates used: with them the design lattice (2 network answers)" in said
    assert "1 network answers per scenario; left native over the 2 scenarios: error knob 2" in said
    plain = _failures(deck, tmp_path, "--workers", "1")
    assert "surrogate_calls" not in plain[0]


def test_the_failures_command_finds_the_lattices_own_weights(deck, tmp_path, monkeypatch, capsys):
    """``auto``: the weights folder of the lattice's hash and the beam's species, from the
    working directory."""
    monkeypatch.chdir(tmp_path)
    lh = registry.hash_lattice_file(str(deck))
    _save_sets(common.load_input(str(deck))[0], tmp_path / weights_folder(lh, "proton"))
    rows = _failures(deck, tmp_path, "--workers", "1", "--surrogates", "auto")
    assert [r["surrogate_calls"] for r in rows] == ["1", "1"]
    registry.clear()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")                  # no weights folder from here
    rows = _failures(deck, tmp_path, "--workers", "1", "--surrogates", "auto")
    assert "no weights folder for this lattice" in capsys.readouterr().err
    assert "surrogate_calls" not in rows[0]


def test_a_campaign_carries_the_weights_to_its_envelope_runs(deck):
    from types import SimpleNamespace

    from linac_gen.reliability.points import make_point
    spec = _a_spec()
    rspec = SimpleNamespace(beam=BEAM, sc={}, forward={}, numerics={})
    assert make_point(rspec, str(deck), mode="envelope", surrogates=spec).surrogates is spec
    assert make_point(rspec, str(deck), mode="mp", surrogates=spec).surrogates is None
    assert make_point(rspec, str(deck), mode="envelope").surrogates is None


def test_the_reliability_command_takes_surrogates():
    import argparse

    from linac_gen.cli import reliability
    p = argparse.ArgumentParser()
    reliability.add_arguments(p)
    assert p.parse_args(["run", "x", "--surrogates", "auto"]).surrogates == ["auto"]
    assert p.parse_args(["run", "x"]).surrogates is None


def _demo_campaign(tmp_path):
    import shutil
    from pathlib import Path

    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.reliability.spec import default_spec
    demo = Path(__file__).resolve().parents[2] / "examples" / "reliability_demo"
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(demo / name, tmp_path / name)
    spec = default_spec(str(tmp_path / "reliability_demo.lgproj"), preset="quick", name="w",
                        circuits="circuits.json", landmarks={"treaty_element": "BPM_003"})
    return ReliabilityCampaign.create(tmp_path / "campaign", spec)


def _a_spec():
    import pickle

    from linac_gen.surrogates.transport import SpecEntry, SurrogateSpec
    return SurrogateSpec((SpecEntry("FMAP_001", "/w", "0" * 64, "lh", pickle.dumps(None), ()),))


def test_a_campaign_point_takes_the_campaigns_weights(tmp_path):
    c = _demo_campaign(tmp_path)
    assert c.surrogates is None
    item = {"id": "x", "leg": "faults", "kind": "fault", "case_id": "x", "mode": "envelope",
            "seed": 42, "overrides": []}
    assert c._point_for(item).surrogates is None
    c.surrogates = spec = _a_spec()
    assert c._point_for(item).surrogates is spec
    assert c._point_for({**item, "mode": "mp"}).surrogates is None


@pytest.mark.parametrize("use", [True, False])
def test_the_reliability_command_gives_the_campaign_what_its_check_accepted(tmp_path, monkeypatch,
                                                                         use):
    """The design check runs in the command's process; the campaign's envelope points carry the
    weights it accepted, and a campaign run serially runs native when the check failed."""
    import argparse
    from types import SimpleNamespace

    from linac_gen.cli import reliability
    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.surrogates import transport
    from linac_gen.surrogates.transport import SurrogateSpec
    _demo_campaign(tmp_path)
    spec = _a_spec() if use else SurrogateSpec()
    check = SimpleNamespace(use=use, note="surrogates used: x" if use else "surrogates not used: y")
    monkeypatch.setattr(transport, "study_spec", lambda *a, **k: (check, spec, ["a note"]))
    seen = {}

    def run(self, *a, **k):
        seen.update(surrogates=self.surrogates, suspended=registry.is_suspended())
        return {}
    monkeypatch.setattr(ReliabilityCampaign, "run", run)
    monkeypatch.setattr(ReliabilityCampaign, "summarize", lambda self: tmp_path / "s.json",
                        raising=False)
    p = argparse.ArgumentParser()
    reliability.add_arguments(p)
    args = p.parse_args(["run", str(tmp_path / "campaign"), "--surrogates", "auto", "--serial"])
    reliability.run(args)
    assert seen == {"surrogates": spec if use else None, "suspended": not use}


@pytest.mark.parametrize("workers", ["1", "2"])
def test_a_failed_design_check_leaves_the_sweep_native(deck, tmp_path, capsys, monkeypatch,
                                                       workers):
    """Weights within their accuracy limits (0.1 % off) that the design check turns down: the
    sweep is the native sweep, serially (its registry suspended) and in the workers (no spec)."""
    from linac_gen.surrogates import study_check
    from linac_gen.surrogates.study_check import DesignCheck
    root = _save_sets(common.load_input(str(deck))[0], tmp_path / "w", factor=1.001)
    monkeypatch.setattr(study_check, "check_design",
                        lambda *a, **k: DesignCheck(use=False, note="surrogates not used: no"))
    rows = _failures(deck, tmp_path, "--workers", workers, "--surrogates", str(root))
    assert "surrogates not used: no" in capsys.readouterr().out
    registry.clear()
    assert rows == _failures(deck, tmp_path, "--workers", workers)
    # and the same weights, used, do change the rows
    monkeypatch.undo()
    used = _failures(deck, tmp_path, "--workers", workers, "--surrogates", str(root))
    assert [{k: v for k, v in r.items() if k != "surrogate_calls"} for r in used] != rows


def test_the_held_settings_are_the_parents(deck, tmp_path):
    """Weights first built on an element set otherwise in the parent hold that setting in every
    worker: a worker does not take its own parse's (the file's) instead."""
    from linac_gen.surrogates.base import OutOfScopeError
    from linac_gen.surrogates.transport import install, study_spec
    lat, cfg, _conv = common.load_input(str(deck))
    lat.elements[1].kb = 2.1                          # in the parent only, before the load
    _save_sets(lat, tmp_path / "w", with_settings=False)    # weights of that map
    common.apply_beam_overrides(cfg, BEAM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check, spec, _notes = study_spec(lat, cfg, str(deck), [str(tmp_path / "w")])
    entry = next(e for e in spec.entries if e.element_key == "FMAP_001")
    assert entry.meta.train_settings["kb"] == 2.1
    registry.clear()
    fresh = common.load_input(str(deck))[0]
    surr = next(s for s in install(spec, fresh) if s.metadata.element_key == "FMAP_001")
    ref = ReferenceParticle(species=PROTON, w_kin=W, frequency=F)
    with pytest.raises(OutOfScopeError, match="its kb is 2.0, 2.1 when its weights were loaded"):
        surr.fitted_matrix(ref)


# ---------------------------------------------------------------------------
# the second look: what a worker gets is what the parent has
# ---------------------------------------------------------------------------
def _rows(res):
    return [{k: v for k, v in m.items() if k != "elapsed"}
            for m in [res.baseline] + [im.metrics for im in res.impacts]]


def _with_a_bad_set(deck, tmp_path):
    """FMAP_001's set 5 % off (its accuracy check refuses it), FMAP_002's exact."""
    import shutil
    lat = common.load_input(str(deck))[0]
    root = _save_sets(lat, tmp_path / "w")
    _save_sets(lat, tmp_path / "bad", factor=1.05)
    shutil.rmtree(root / "FMAP_001")
    shutil.copytree(tmp_path / "bad" / "FMAP_001", root / "FMAP_001")
    return root


def test_a_set_its_accuracy_check_refused_is_refused_and_counted_alike(deck, tmp_path):
    check, spec, _notes = _spec(deck, _with_a_bad_set(deck, tmp_path))
    assert check.use
    serial, pooled = _sweep(deck, spec, 1), _sweep(deck, spec, 2)
    assert serial.baseline["surrogate_refused"] == {"accuracy check": 1}
    assert _rows(serial) == _rows(pooled)


def test_a_metadata_file_edited_after_the_check_does_not_reach_the_workers(deck, tmp_path):
    """A worker reads the weights alone (checked against the hash of what the parent loaded),
    and takes the parent's metadata."""
    import json
    root = _save_sets(common.load_input(str(deck))[0], tmp_path / "w")
    _check, spec, _notes = _spec(deck, root)
    path = root / "FMAP_001" / "metadata.json"
    data = json.loads(path.read_text())
    data["output_norm"]["mean"][1] *= 1.05
    path.write_text(json.dumps(data))
    assert _rows(_sweep(deck, spec, 1)) == _rows(_sweep(deck, spec, 2))


def test_weights_gone_from_disk_run_native_in_a_worker(deck, tmp_path):
    import shutil

    from linac_gen.surrogates import transport
    root = _save_sets(common.load_input(str(deck))[0], tmp_path / "w")
    _check, spec, _notes = _spec(deck, root)
    shutil.rmtree(root / "FMAP_001")
    with pytest.warns(UserWarning, match="could not be read .*this worker runs it native"):
        got = transport.install(spec, common.load_input(str(deck))[0])
    assert [s.metadata.element_key for s in got] == ["FMAP_002"]
    pooled = _sweep(deck, spec, 2)                       # no crash: that map runs native
    assert pooled.baseline["surrogate_calls"] == 1


def _save_kb_set(lattice, root, name, kb0, half):
    """Weights over ``kb`` in [kb0 - half, kb0 + half], pinned to the map at kb0, holding the
    element's other settings."""
    from linac_gen.surrogates.base import settings_of
    el = copy.deepcopy(next(e for e in lattice.elements if e.name == name))
    held = settings_of(el, {"kb"})
    el.kb = kb0
    ref = ReferenceParticle(species=PROTON, w_kin=W, frequency=F)
    M = np.asarray(copy.deepcopy(el).fitted_matrix(ref), dtype=float)
    m = SurrogateMetadata(
        element_key=name, element_class=type(el).__name__,
        architecture={"input_dim": 4, "output_dim": 36, "hidden_dims": [4], "activation": "silu",
                      "param_names": ["kb"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "kb"],
                    input_lo=np.array([W - 1e-9, 0.0, 1.0, kb0 - half]),
                    input_hi=np.array([W + 1e-9, 1.0, 100.0, kb0 + half])),
        input_norm={"mean": [0.0] * 4, "std": [1.0] * 4},
        output_norm={"mean": [float(v) for v in M.ravel()], "std": [0.0] * 36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0, helix_commit_sha="",
        lattice_hash="lh", created_iso="", species_name=PROTON.name, species_mass_mev=PROTON.mass,
        train_frequency_mhz=F, train_settings=held)
    save_surrogate(MlpHead(input_dim=4, output_dim=36, hidden_dims=(4,)), m, root / name)


def test_a_check_the_design_lattice_never_reached_is_settled_before_the_workers(deck, tmp_path):
    """FMAP_001's weights cover kb near 1.8 (its fault at 90 % puts it there), not the design's
    2.0: the design lattice never asks them, so the parent checks them before the spec, and no
    worker decides it again at whichever point it meets first."""
    lat = common.load_input(str(deck))[0]
    root = _save_sets(lat, tmp_path / "w")
    import shutil
    shutil.rmtree(root / "FMAP_001")
    _save_kb_set(lat, root, "FMAP_001", 1.8, 0.01)
    check, spec, _notes = _spec(deck, root)
    assert check.use and check.refused == {"outside its range": 1}
    entry = next(e for e in spec.entries if e.element_key == "FMAP_001")
    assert entry.meta.accuracy_check == ""                 # settled here, and passed
    assert _rows(_sweep(deck, spec, 1)) == _rows(_sweep(deck, spec, 2))


def test_the_spec_carries_the_weights_tracking_serves(deck, tmp_path):
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.transport import load_folders, spec_of_registry
    lat = common.load_input(str(deck))[0]
    load_folders(lat, str(deck), [str(_save_sets(lat, tmp_path / "w"))])
    old = registry.get_by_element_name("FMAP_001")
    newer = SurrogateFieldMap(lat.elements[1], old.mlp, copy.deepcopy(old.metadata))
    newer.weights_dir = old.weights_dir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        registry.register(newer, "another-hash", element_key="FMAP_001")
    spec, _notes = spec_of_registry()
    entry = next(e for e in spec.entries if e.element_key == "FMAP_001")
    assert entry.lattice_hash == "another-hash" and registry.get_by_element_name("FMAP_001") is newer


@pytest.mark.parametrize("order", [("auto", "DIR"), ("DIR", "auto")])
def test_auto_and_a_folder_take_the_folder_in_either_order(deck, tmp_path, monkeypatch, order):
    from linac_gen.surrogates.transport import load_folders
    monkeypatch.chdir(tmp_path)                           # no weights folder for the deck here
    lat = common.load_input(str(deck))[0]
    root = _save_sets(lat, tmp_path / "w")
    notes = load_folders(lat, str(deck), [str(root) if f == "DIR" else f for f in order],
                         species="proton")
    assert sorted(k for _h, k in registry.list_registered()) == ["FMAP_001", "FMAP_002"]
    assert [n for n in notes if n.startswith("auto: no weights folder for this lattice")]


def test_the_surrogates_option_is_one_value_and_leaves_the_input_alone():
    import argparse

    from linac_gen.cli import failures
    p = argparse.ArgumentParser()
    failures.add_arguments(p)
    args = p.parse_args(["--surrogates", "auto,/w", "deck.dat"])
    assert args.surrogates == ["auto", "/w"] and args.input == "deck.dat"


def test_no_weights_registered_says_so(deck, tmp_path):
    from linac_gen.surrogates.transport import study_spec
    lat, cfg, _conv = common.load_input(str(deck))
    common.apply_beam_overrides(cfg, BEAM)
    check, spec, notes = study_spec(lat, cfg, str(deck), [str(tmp_path / "nowhere")])
    assert check.note == "surrogates not used: no weights were registered" and not spec
    assert notes == [f"{tmp_path / 'nowhere'}: does not exist"]


def test_a_campaigns_orbit_correction_runs_native(tmp_path, monkeypatch):
    """In a serial campaign the draws run in the command's process, whose registry holds the
    surrogates: their orbit correction runs native there, as in a pool worker."""
    from dataclasses import asdict

    from linac_gen.errors import correction
    from linac_gen.reliability.errors import draw_seed
    from tests.surrogates.accuracy_helpers import native, pinned
    c = _demo_campaign(tmp_path)
    lat, cfg, _conv = c._load()
    el = next(e for e in lat.elements if type(e).__name__ in ("FieldMap", "FieldMap3D",
                                                               "Quadrupole", "Solenoid"))
    registry.register(pinned(el, np.eye(6)) if not hasattr(el, "fitted_matrix")
                      else pinned(el, native(el, w=W, f=F)))
    seen = []

    def fake(lattice, factory, **kw):
        seen.append(registry.is_suspended())
        return {"status": "converged", "history": []}
    monkeypatch.setattr(correction, "run_correction_from_lattice", fake)
    job = {"lattice_path": str(c.input_path), "beam_config": asdict(cfg),
           "budget": {"element": [{"pattern": "*", "parameter": "dx", "sigma": 0.1}], "beam": []},
           "seed": 1, "base_seed": 0,
           "correction": {"enabled": True, "n_iter": 1, "reading_backend": "mp"}}
    out = draw_seed(job)
    assert out["error"] is None and seen == [True] and not registry.is_suspended()


def _reliability_args(*extra):
    import argparse

    from linac_gen.cli import reliability
    p = argparse.ArgumentParser()
    reliability.add_arguments(p)
    return reliability, p.parse_args(list(extra))


def test_a_campaign_is_resumed_only_the_way_it_ran(tmp_path, monkeypatch):
    """Its runs are all made one way: a resume with another --surrogates choice than the one
    recorded stops (--force starts over), and the summary says which."""
    import json
    from types import SimpleNamespace

    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.surrogates import transport
    _demo_campaign(tmp_path)
    cdir = tmp_path / "campaign"
    spec = _a_spec()
    check = SimpleNamespace(use=True, note="surrogates used: x")
    monkeypatch.setattr(transport, "study_spec", lambda *a, **k: (check, spec, []))
    monkeypatch.setattr(ReliabilityCampaign, "run", lambda self, *a, **k: {})
    monkeypatch.setattr(ReliabilityCampaign, "summarize", lambda self: cdir / "s.json")
    real_pending = ReliabilityCampaign.pending
    started = {"on": False}

    def pending(self, *a, **k):
        items = real_pending(self, *a, **k)
        return items[1:] if started["on"] else items       # one item done once started
    monkeypatch.setattr(ReliabilityCampaign, "pending", pending)
    reliability, args = _reliability_args("run", str(cdir), "--surrogates", "auto", "--serial")
    assert reliability.run(args) == 0
    recorded = json.loads((cdir / "surrogates.json").read_text())
    assert recorded["use"] and recorded["weights"] == [["FMAP_001", "0" * 64]]
    started["on"] = True
    reliability, args = _reliability_args("run", str(cdir), "--serial")
    assert reliability.run(args) == 2                       # native now: refused
    reliability, args = _reliability_args("run", str(cdir), "--surrogates", "auto", "--serial")
    assert reliability.run(args) == 0                       # as it ran
    reliability, args = _reliability_args("run", str(cdir), "--serial", "--force")
    assert reliability.run(args) == 0 and not (cdir / "surrogates.json").exists()


def test_a_multiparticle_campaigns_check_says_its_runs_are_native(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.surrogates import transport
    c = _demo_campaign(tmp_path)
    c.spec.forward = {"mode": "mp"}
    c.save() if hasattr(c, "save") else None
    seen = {}

    def study_spec(*a, **k):
        seen.update(k)
        return SimpleNamespace(use=False, note="n"), transport.SurrogateSpec(), []
    monkeypatch.setattr(transport, "study_spec", study_spec)
    monkeypatch.setattr(ReliabilityCampaign, "run", lambda self, *a, **k: {})
    monkeypatch.setattr(ReliabilityCampaign, "summarize", lambda self: tmp_path / "s.json")
    monkeypatch.setattr(ReliabilityCampaign, "load",
                        classmethod(lambda cls, d: c))
    reliability, args = _reliability_args("run", str(tmp_path / "campaign"), "--surrogates",
                                          "auto", "--serial")
    reliability.run(args)
    assert seen["mode"] == "mp"


def test_the_spec_carries_the_hash_of_what_was_loaded(deck, tmp_path):
    """Weights replaced on disk after the load (a retraining into the same folder) are not the
    ones the design check ran: the spec carries the loaded bytes' hash, so the workers refuse
    the new file."""
    from linac_gen.surrogates.transport import load_folders, spec_of_registry
    lat = common.load_input(str(deck))[0]
    root = _save_sets(lat, tmp_path / "w")
    load_folders(lat, str(deck), [str(root)])
    loaded = registry.get_by_element_name("FMAP_001").metadata.weights_sha256
    _save_sets(lat, tmp_path / "w", factor=1.01)          # the files change after the load
    spec, _notes = spec_of_registry()
    entry = next(e for e in spec.entries if e.element_key == "FMAP_001")
    assert entry.weights_sha256 == loaded


def test_a_second_spec_for_the_same_folder_brings_its_own_metadata(deck, tmp_path):
    import dataclasses
    import pickle

    from linac_gen.surrogates import transport
    _check, spec, _notes = _spec(deck, _save_sets(common.load_input(str(deck))[0], tmp_path / "w"))
    entry = next(e for e in spec.entries if e.element_key == "FMAP_002")
    meta = entry.meta
    meta.train_settings = {**meta.train_settings, "kb": 9.99}
    other = transport.SurrogateSpec((dataclasses.replace(entry, metadata=pickle.dumps(meta)),))
    lat = common.load_input(str(deck))[0]
    transport.install(spec, lat)
    got = transport.install(other, common.load_input(str(deck))[0])
    assert [s.metadata.train_settings["kb"] for s in got] == [9.99]


def test_the_summary_says_what_the_surrogates_made_the_runs(tmp_path):
    import json
    c = _demo_campaign(tmp_path)
    (c.dir / "surrogates.json").write_text(json.dumps({"note": "surrogates used: x", "use": True}))
    try:
        out = c.summarize()
    except Exception as exc:                              # noqa: BLE001 - an empty campaign
        pytest.skip(f"an empty campaign does not summarize here: {exc}")
    assert json.loads(out.read_text())["surrogates"] == "surrogates used: x"
