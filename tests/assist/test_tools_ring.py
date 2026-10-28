"""The assistant's ring tools (linac_gen.assist.tools_ring): the ring verdict,
the optics, the ramp and its preview, ring mode, ring runs — on the simple
Fermilab Booster of the ring tests, headless (the GUI's adapter:
tests/gui/test_assistant_ring.py)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numba")

from linac_gen.assist.agent import LONG_RUNNING  # noqa: E402
from linac_gen.assist.jobs import JobManager  # noqa: E402
from linac_gen.assist.tools import TOOLS, WorkContext  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BOOSTER = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
#: a short ramp: 800 to 801 MeV in half a 20 kHz cycle (about 13 turns) with
#: the cavities' own voltage
SHORT_RAMP = {"energy": {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 801.0,
                         "frequency_hz": 20000.0},
              "voltage": {"kind": "constant"}, "phase": {"kind": "auto"}}


def _call(name, ctx, **kw):
    return TOOLS[name].fn(ctx, **kw)


@pytest.fixture()
def ctx(tmp_path):
    c = WorkContext(calc_dir=str(tmp_path))
    r = _call("load_lattice", c, path=str(BOOSTER))
    assert r["status"] == "ok", r
    return c


def test_the_tools_are_registered_with_their_tiers():
    tiers = {n: TOOLS[n].tier for n in ("ring_status", "ring_optics", "ring_programs",
                                         "set_ring_program", "set_ring_mode", "run_ring")}
    assert tiers == {"ring_status": "read", "ring_optics": "compute", "ring_programs": "read",
                     "set_ring_program": "mutate", "set_ring_mode": "mutate",
                     "run_ring": "compute"}
    assert "run_ring" in LONG_RUNNING and "ring_optics" not in LONG_RUNNING
    assert "ring" in TOOLS["run_in_gui"].schema["properties"]["kind"]["enum"]
    from linac_gen.assist.mcp_server import tool_specs
    names = {s["name"] for s in tool_specs()}
    assert {"ring_status", "ring_optics", "run_ring", "set_ring_program"} <= names


def test_a_linac_session_never_loads_ring_mode():
    code = ("import sys, linac_gen.assist.tools, linac_gen.assist.agent; "
            "print(sorted(m for m in sys.modules if m.startswith('linac_gen.ring')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=120, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_status_gives_the_verdict_and_the_settings(ctx):
    assert _call("ring_status", WorkContext())["status"] == "error"      # no lattice
    r = _call("ring_status", ctx)
    assert r["status"] == "ok", r
    d = r["data"]
    assert d["verdict"]["is_ring"] is True and d["verdict"]["harmonic"] == 84
    assert abs(d["verdict"]["total_angle_deg"] - 360.0) < 1e-5
    assert d["ring_mode"] is False and "last_run" not in d
    assert d["settings"]["ramp"].startswith("none") and d["settings"]["space_charge"] == "off"


def test_optics_are_the_ring_tabs(ctx):
    r = _call("ring_optics", ctx)
    assert r["status"] == "ok", r
    d = r["data"]
    assert abs(d["qx"] - 6.71) < 1e-4 and abs(d["qy"] - 6.81) < 1e-4
    assert abs(d["dqx"] + 8.65) < 0.01 and abs(d["dqy"] + 8.09) < 0.01
    assert abs(d["gamma_t"] - 5.449) < 1e-3 and d["harmonic"] == 84
    # (no beam settings in the session: the deck's BEAM and default emittances, said)
    assert any("default emittances" in w for w in r["warnings"])
    r6 = _call("ring_optics", ctx, six_d=True)
    assert r6["status"] == "ok" and r6["data"]["bucket"]["stable"] is True
    assert 0.02 < r6["data"]["qs"] < 0.03


def test_optics_follow_the_session_settings(ctx):
    base = _call("ring_optics", ctx)["data"]
    ctx.set_ring_config({"model": "curvilinear"})
    curv = _call("ring_optics", ctx)["data"]
    assert curv["options"]["model"] == "curvilinear" != base["options"]["model"]
    assert abs(curv["dqx"] - base["dqx"]) > 0.1       # (the bend multipoles' frame)


def test_a_lattice_without_its_beam_needs_the_sessions(ctx):
    ctx.lattice.ring_reference_beam = None
    r = _call("ring_optics", ctx)
    assert r["status"] == "refused" and "set_beam_config" in r["data"]["message"]
    from linac_gen.core.config import BeamConfig
    ctx.beam_config = BeamConfig(species="proton", energy=800.0)
    r = _call("ring_optics", ctx)
    assert r["status"] == "ok" and not any("default" in w for w in r["warnings"])
    assert abs(r["data"]["qx"] - 6.71) < 1e-4


def test_the_ramp_is_checked_then_kept_and_cleared(ctx):
    from linac_gen.ring.programs import RampConfig
    assert _call("ring_programs", ctx)["data"]["ramp"] == "none"
    bad = _call("set_ring_program", ctx, ramp={"energy": {"kind": "sine", "bogus": 1}})
    assert bad["status"] == "refused" and "bogus" in bad["data"]["message"]
    assert _call("set_ring_program", ctx, ramp="sine")["status"] == "refused"
    assert _call("set_ring_program", ctx)["status"] == "refused"
    assert "ramp" not in ctx.ring_config
    r = _call("set_ring_program", ctx, ramp=SHORT_RAMP)
    assert r["status"] == "ok", r
    assert ctx.ring_config["ramp"] == RampConfig.from_dict(SHORT_RAMP).to_dict()
    p = _call("ring_programs", ctx)
    assert p["status"] == "ok", p
    pv = p["data"]["preview"]
    assert 5 <= pv["turns_to_end"] <= 30 and pv["turns"] == pv["turns_to_end"]
    # (the reference ends within a turn's gain of the program's top: the turns
    # are counted on the program's clock)
    assert 800.98 < pv["kinetic_end_mev"] <= 801.0 and pv["capped_turns"] == 0
    assert pv["gamma_t_from"] == "from the ring's optics" and pv["transition_turn"] is None
    assert _call("set_ring_program", ctx, clear=True)["status"] == "ok"
    assert "ramp" not in ctx.ring_config


def test_ring_mode_is_the_guis(ctx):
    r = _call("set_ring_mode", ctx, on=True)
    assert r["status"] == "refused" and "desktop GUI" in r["data"]["message"]


def test_a_run_writes_its_file_and_keeps_its_results(ctx, tmp_path):
    from linac_gen.ring.results import load_ring_hdf5
    ticks = []
    r = _call("run_ring", ctx, turns=6, particles=200,
              progress_callback=lambda s, i, n: ticks.append((i, n)))
    assert r["status"] == "ok", r
    d = r["data"]
    assert (d["turns_done"], d["particles"], d["survival"], d["truncated"]) == (6, 200, 1.0,
                                                                              False)
    assert ticks[-1] == (6, 6) and all(n == 6 for _i, n in ticks)
    path = Path(d["file"])
    assert path.parent == tmp_path and path.name.startswith("simple_booster_")
    assert path.name.endswith("_ring.h5") and path.is_file()
    assert load_ring_hdf5(path).turns_done == 6
    assert ctx.ring_results is not None and ctx.ring_results_path == str(path)
    assert r["provenance"]["ring_results_path"] == str(path)
    last = _call("ring_status", ctx)["data"]["last_run"]
    assert last["turns_done"] == 6 and last["file"] == str(path)
    # (a second run in the same second gets its own file)
    r2 = _call("run_ring", ctx, turns=1, particles=50)
    assert r2["status"] == "ok" and Path(r2["data"]["file"]) != path


def test_ring_status_says_the_start_a_run_would_take(ctx):
    # without beam settings in the session: the Gaussian matched to the ring
    # and its 10000 particles; with an empty start: none
    st = _call("ring_status", ctx)["data"]["settings"]
    assert (st["source"], st["particles"]) == ("matched", 10000)
    ctx.set_ring_config({"source": "empty"})
    st = _call("ring_status", ctx)["data"]["settings"]
    assert (st["source"], st["particles"]) == ("empty", 0)


def test_a_run_starts_from_the_sessions_beam(ctx, tmp_path):
    # with beam settings in the session a run starts from that beam, as the
    # Ring tab's does (without, from the Gaussian matched to the ring: above)
    import json

    from linac_gen.core.config import BeamConfig
    from linac_gen.ring.results import load_ring_hdf5
    ctx.beam_config = BeamConfig(species="proton", energy=800.0, frequency=44.7044096,
                                 n_particles=120, emit_nx=1.5, emit_ny=1.5, emit_z=5.0,
                                 beta_z=100.0, cutoff=4.0)
    st = _call("ring_status", ctx)["data"]["settings"]
    assert st["particles"] == 120 and st["source"] == "beam"
    r = _call("run_ring", ctx, turns=2)
    assert r["status"] == "ok", r
    d = r["data"]
    assert d["particles"] == 120
    assert load_ring_hdf5(Path(d["file"])).meta["config"]["source"] == "beam"
    # (the beam's Twiss are the settings' defaults, a linac's: mismatched, and said)
    assert d["start_mismatch"][0] > 1.01 and any("mismatched" in n for n in d["notes"])
    # particles: this run's number, the session's beam unchanged
    r = _call("run_ring", ctx, turns=1, particles=90)
    assert r["data"]["particles"] == 90 and ctx.beam_config.n_particles == 120
    r = _call("run_ring", ctx, turns=1, particles=80, source="matched")
    assert r["data"]["particles"] == 80
    assert load_ring_hdf5(Path(r["data"]["file"])).meta["config"]["source"] == "matched"
    bad = _call("run_ring", ctx, turns=1, particles=0)
    assert bad["status"] == "refused" and "at least 1" in bad["data"]["message"]
    # (a beam read from a particle file uses its particles)
    import dataclasses
    ctx.beam_config = dataclasses.replace(ctx.beam_config, source="file",
                                          distribution_file=str(tmp_path / "none.dst"))
    bad = _call("run_ring", ctx, turns=1, particles=10)
    assert bad["status"] == "refused" and "distribution file" in bad["data"]["message"]
    ctx.beam_config = dataclasses.replace(ctx.beam_config, source="generate",
                                          distribution_file=None)
    # a project of an earlier version whose run started matched: the beam's now
    proj = tmp_path / "old.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {
                                    "source": "matched", "n_particles": 5000, "n_turns": 4}}}))
    c = WorkContext(calc_dir=str(tmp_path))
    assert _call("load_lattice", c, path=str(proj))["status"] == "ok"
    assert c.ring_config == {"source": "beam", "n_turns": 4}


def test_a_cancelled_run_keeps_the_turns_done(ctx):
    asked = [0]

    def stop():
        asked[0] += 1
        return asked[0] > 2

    r = _call("run_ring", ctx, turns=40, particles=100, should_abort=stop)
    assert r["status"] == "ok", r
    assert r["data"]["truncated"] is True and r["data"]["turns_done"] < 40


def test_a_run_to_the_end_of_the_ramp(ctx):
    assert _call("run_ring", ctx, to_end=True)["status"] == "refused"        # no ramp
    _call("set_ring_program", ctx, ramp=SHORT_RAMP)
    pv = _call("ring_programs", ctx)["data"]["preview"]
    assert _call("run_ring", ctx, to_end=True, turns=3)["status"] == "refused"
    r = _call("run_ring", ctx, to_end=True, particles=100)
    assert r["status"] == "ok", r
    assert r["data"]["turns_done"] == pv["turns_to_end"]
    assert r["data"]["ramp"]["kinetic_end_mev"] == pv["kinetic_end_mev"]   # (the same reference)


def test_run_refusals(ctx):
    assert _call("run_ring", WorkContext())["status"] == "error"          # no lattice
    assert _call("run_ring", ctx, turns=0)["status"] == "refused"
    assert _call("run_ring", ctx, turns="many")["status"] == "refused"
    bad = _call("run_ring", ctx, turns=2, particles=50, source="nowhere")
    assert bad["status"] == "refused"


def test_a_ring_job_reports_turns():
    jm = JobManager(max_concurrent=1)
    try:
        def runner(progress, _abort):
            progress(7.0, 7, 12)
            return {"status": "ok", "data": {}, "provenance": {}, "warnings": []}
        jid = jm.submit("run_ring", {}, runner)
        import time
        for _ in range(200):
            snap = jm.status(jid)["data"]
            if snap["state"] == "done":
                break
            time.sleep(0.01)
        assert snap["state"] == "done"
        assert (snap["progress_turn"], snap["turns"]) == (7, 12)
        # (the key every reader knows stays: the terminal assistant's /jobs reads it)
        assert snap["progress_s_mm"] == 7.0
        jid2 = jm.submit("run_mp", {}, runner)
        for _ in range(200):
            snap2 = jm.status(jid2)["data"]
            if snap2["state"] == "done":
                break
            time.sleep(0.01)
        assert snap2["progress_s_mm"] == 7.0 and "progress_turn" not in snap2
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# the review's cases (2026-10-25)
# ---------------------------------------------------------------------------
def test_two_runs_at_once_write_two_files(ctx):
    """(the job pool runs two jobs at a time: each run's file its own)"""
    import threading
    out = {}

    def go(k):
        out[k] = _call("run_ring", ctx, turns=1, particles=50, seed=k)

    ts = [threading.Thread(target=go, args=(k,)) for k in (1, 2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(300)
    assert all(out[k]["status"] == "ok" for k in (1, 2)), out
    files = {out[k]["data"]["file"] for k in (1, 2)}
    assert len(files) == 2 and all(Path(f).is_file() for f in files)
    from linac_gen.assist import tools_ring as TR
    a = TR._reserve_name(Path(ctx.calc_dir), "x")
    b = TR._reserve_name(Path(ctx.calc_dir), "x")
    assert a != b and b.name.endswith("_2.h5")


def test_a_lattice_changed_during_a_run_keeps_its_results_out(ctx):
    other = object()
    seen = []

    def swap(s, i, n):
        if not seen:
            seen.append(i)
            ctx.lattice = other                      # (a new lattice loaded meanwhile)

    r = _call("run_ring", ctx, turns=3, particles=50, progress_callback=swap)
    assert r["status"] == "ok", r
    assert ctx.ring_results is None
    assert any("lattice changed during the run" in n for n in r["data"]["notes"])
    assert Path(r["data"]["file"]).is_file()


def test_a_run_of_the_ring_tab_in_progress_holds_run_ring_back(ctx):
    class Busy(WorkContext):
        def ring_running(self):
            return True
    busy = Busy(calc_dir=ctx.calc_dir, lattice=ctx.lattice, lattice_path=ctx.lattice_path)
    r = _call("run_ring", busy, turns=1, particles=50)
    assert r["status"] == "refused" and "Ring tab" in r["data"]["message"]
    assert _call("ring_status", busy)["data"]["run_in_progress"] is True
    assert _call("ring_status", ctx)["data"]["run_in_progress"] is False


def test_a_run_that_cannot_fit_in_memory_is_refused(ctx, monkeypatch):
    from linac_gen.ring import memory
    monkeypatch.setattr(memory, "physical_memory", lambda: 2 ** 20)
    r = _call("run_ring", ctx, turns=10, particles=5000)
    assert r["status"] == "refused" and "GB" in r["data"]["message"]
    assert not list(Path(ctx.calc_dir).glob("*_ring.h5"))


def test_a_run_stopped_before_its_first_turn_keeps_the_last_results(ctx):
    first = _call("run_ring", ctx, turns=2, particles=50)
    assert first["status"] == "ok"
    kept = ctx.ring_results
    r = _call("run_ring", ctx, turns=5, particles=50, should_abort=lambda: True)
    assert r["status"] == "refused" and "before its first turn" in r["data"]["message"]
    assert ctx.ring_results is kept
    assert len(list(Path(ctx.calc_dir).glob("*_ring.h5"))) == 1


def test_a_projects_ring_settings_become_the_sessions(tmp_path):
    import json
    proj = tmp_path / "ring.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                                "ring": {"enabled": True, "config": {
                                    "model": "curvilinear", "ramp": SHORT_RAMP,
                                    "n_turns": 7}}}))
    c = WorkContext(calc_dir=str(tmp_path))
    r = _call("load_lattice", c, path=str(proj))
    assert r["status"] == "ok" and r["data"]["ring_settings"] == "the project's"
    assert c.ring_config["model"] == "curvilinear" and c.ring_config["n_turns"] == 7
    st = _call("ring_status", c)["data"]["settings"]
    assert st["model"] == "curvilinear" and st["ramp"] == "set" and st["turns"] == 7
    # (the same numbers as `python -m linac_gen ring optics` on the project)
    assert _call("ring_optics", c)["data"]["dqx"] == pytest.approx(-7.99, abs=0.02)
    # a bare lattice file: no ring settings key
    c2 = WorkContext(calc_dir=str(tmp_path))
    assert "ring_settings" not in _call("load_lattice", c2, path=str(BOOSTER))["data"]


def test_a_long_ramp_preview_is_refused_at_once(ctx, monkeypatch):
    from linac_gen.assist import tools_ring as TR
    monkeypatch.setattr(TR, "PREVIEW_TURNS_CAP", 5)
    _call("set_ring_program", ctx, ramp=SHORT_RAMP)
    r = _call("ring_programs", ctx)
    assert r["status"] == "refused" and "ring ramp" in r["data"]["message"]


def test_the_ring_tools_results_are_strict_json(ctx):
    import json
    for name, kw in (("ring_status", {}), ("ring_optics", {}), ("ring_optics", {"six_d": True}),
                     ("run_ring", {"turns": 2, "particles": 50})):
        r = _call(name, ctx, **kw)
        assert r["status"] == "ok", (name, r)
        json.dumps(r, allow_nan=False)


def test_a_project_without_ring_settings_leaves_none_of_the_last_ones(tmp_path):
    # (as File > Open Project: a project without a ring section has the
    # default ring settings, not the previous project's)
    import json
    a = tmp_path / "a.lgproj"
    a.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER),
                             "ring": {"enabled": True, "config": {
                                 "ramp": SHORT_RAMP, "model": "curvilinear"}}}))
    b = tmp_path / "b.lgproj"
    b.write_text(json.dumps({"__kind__": "linac_gen_project", "lattice_path": str(BOOSTER)}))
    c = WorkContext(calc_dir=str(tmp_path))
    assert _call("load_lattice", c, path=str(a))["status"] == "ok"
    assert c.ring_config["ramp"]
    r = _call("load_lattice", c, path=str(b))
    assert r["status"] == "ok" and r["data"]["ring_settings"].startswith("none")
    assert c.ring_config == {}
    st = _call("ring_status", c)["data"]["settings"]
    assert st["ramp"].startswith("none") and st["model"] == "cartesian"


def test_another_lattice_drops_the_last_ring_run(ctx):
    r = _call("run_ring", ctx, turns=1, particles=20)
    assert r["status"] == "ok" and ctx.ring_results is not None
    assert _call("load_lattice", ctx, path=str(BOOSTER))["status"] == "ok"
    assert ctx.ring_results is None and ctx.ring_results_path == ""
    assert "last_run" not in _call("ring_status", ctx)["data"]


def test_load_results_reads_a_ring_file_into_the_ring_results(ctx, tmp_path):
    r = _call("run_ring", ctx, turns=2, particles=20)
    path = r["data"]["file"]
    c = WorkContext(calc_dir=str(tmp_path))
    assert _call("load_lattice", c, path=str(BOOSTER))["status"] == "ok"
    got = _call("load_results", c, path=path)
    assert got["status"] == "ok", got
    assert got["data"]["kind"] == "ring" and got["data"]["turns_done"] == 2
    assert c.ring_results is not None and c.ring_results_path == path
    assert c.results is None                       # (not a linac run)
    assert _call("ring_status", c)["data"]["last_run"]["turns_done"] == 2


def test_the_ring_tools_take_the_beam_a_ring_run_takes(ctx):
    # (the hook the GUI points at the Beam tab's form; headless the session's)
    from linac_gen.core.config import BeamConfig

    class Ctx(WorkContext):
        def ring_beam_config(self):
            return BeamConfig(species="proton", energy=800.0, n_particles=33)

    c = Ctx(calc_dir=ctx.calc_dir)
    c.set_lattice(ctx.lattice, ctx.lattice_path)
    c.beam_config = BeamConfig(species="proton", energy=800.0, n_particles=77)
    st = _call("ring_status", c)["data"]["settings"]
    assert (st["source"], st["particles"]) == ("beam", 33)


def test_a_deck_whose_beam_is_not_declared_needs_the_sessions(ctx):
    rb = dict(ctx.lattice.ring_reference_beam, declared=False)
    ctx.lattice.ring_reference_beam = rb
    r = _call("ring_optics", ctx)
    assert r["status"] == "refused" and "set_beam_config" in r["data"]["message"]
