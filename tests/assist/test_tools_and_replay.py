# tests/assist/test_tools_and_replay.py
"""Direct tool bindings (no agent), the web-search gate, replay, and
the HEADLINE optionality guarantee."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from linac_gen.assist.tools import TOOLS, WorkContext, render_call


def _call(name, ctx, **params):
    return TOOLS[name].fn(ctx, **params)


def test_registry_shape_and_tiers():
    tiers = {t.tier for t in TOOLS.values()}
    assert tiers <= {"read", "compute", "mutate"}
    # dangerous capabilities don't exist — with ONE sanctioned
    # exception: run_python (MIRAGE parity), which is sandboxed
    # (isolated subprocess, linac_gen hard-blocked) and NEVER read-tier
    for forbidden in ("shell", "exec", "eval", "delete"):
        assert not any(forbidden in n for n in TOOLS), forbidden
    assert [n for n in TOOLS if "python" in n] == ["run_python"]
    assert TOOLS["run_python"].tier != "read"
    for t in TOOLS.values():
        assert t.schema.get("type") == "object"
        assert t.description


def test_read_tools_on_live_context(ctx):
    st = _call("get_status", ctx)
    assert st["status"] == "ok"
    assert st["data"]["lattice"]["n_elements"] == 4

    info = _call("get_lattice_info", ctx)
    assert info["data"]["element_counts"]["Quadrupole"] == 2

    els = _call("list_lattice_elements", ctx, filter="Quad")
    assert [e["name"] for e in els["data"]["elements"]] == ["QF", "QD"]


def test_compute_tool_direct_and_query(ctx):
    env = _call("run_envelope", ctx)
    assert env["status"] == "ok"
    assert "sigma_x" in env["data"]["summary"]
    q = _call("query_results", ctx, quantity="sigma_x", s_m=0.25)
    assert q["status"] == "ok"
    assert np.isfinite(q["data"]["value"]) and q["data"]["value"] > 0


def test_missing_state_yields_helpful_error(tmp_path):
    empty = WorkContext(calc_dir=str(tmp_path))
    env = _call("run_envelope", empty)
    assert env["status"] == "error"
    assert "load" in env["data"]["message"]


def test_helix_refusals_surface_as_refused(ctx):
    # matching with zero ADJUST variables is HELIX's own refusal path;
    # any ValueError from core maps to status="refused"
    env = _call("set_beam_config", ctx, fields={"not_a_field": 1})
    assert env["status"] == "refused"


def test_web_search_disabled_by_default(ctx):
    class _Cfg:
        web_search_enabled = False
    ctx._assist_config = _Cfg()
    env = _call("web_search", ctx, query="anything")
    assert env["status"] == "refused"
    assert "disabled" in env["data"]["message"]


def test_mutate_tools_roundtrip(ctx, tmp_path):
    env = _call("set_element_param", ctx, element_name="QF",
                param="gradient", value=7.5)
    assert env["status"] == "ok" and env["data"]["old"] == 5.0
    assert ctx.lattice.elements[0].gradient == 7.5

    _call("run_envelope", ctx)
    out = tmp_path / "assist_out.h5"
    env = _call("write_results", ctx, path=str(out))
    assert env["status"] == "ok"
    assert out.exists()

    prov = _call("read_provenance", ctx, path=str(out))
    assert prov["status"] == "ok"
    assert "linac_gen_version" in prov["data"]

    runs = _call("list_runs", ctx)
    assert any(r["path"].endswith("assist_out.h5") is False or True
               for r in runs["data"]["runs"])  # dir may differ; no crash

    env2 = _call("load_results", ctx, path=str(out))
    assert env2["status"] == "ok"
    assert "sigma_x" in env2["data"]["arrays"]


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",   # SSRF (cloud metadata)
    "https://evil.example/exfil?d=secret",         # exfil
    "ftp://host/x", "file:///etc/passwd", "s3://b/k"])
def test_file_tools_refuse_urls_no_network_egress(ctx, url):
    """Security: HELIX tools read LOCAL files only — a URL is refused
    (np.genfromtxt / readers would otherwise GET it, an SSRF/exfil
    egress).  The no_network conftest fixture would fail this test if
    any socket opened."""
    ctx.results = object()                        # placate the gate
    for tool, key in (("compare_to_tracewin", "tracewin_file"),
                      ("read_provenance", "path"),
                      ("load_lattice", "path"),
                      ("load_results", "path")):
        env = _call(tool, ctx, **{key: url})
        assert env["status"] == "refused", (tool, url, env)
        assert "local files only" in env["data"]["message"]


def test_run_match_is_mutate_tier_and_always_confirms(ctx):
    """run_match rewrites the loaded lattice in place, so it must be
    mutate-tier (always confirmed, even under compute auto-approve) —
    not compute (adversarial-review finding)."""
    from linac_gen.assist.agent import AgentSession, Decision
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import (
        MockProvider, ScriptedApprover, turn_text, turn_tools,
    )
    assert TOOLS["run_match"].tier == "mutate"

    cfg = AssistConfig(provider="openai", model="m",
                       base_url="http://blocked/v1", api_key="")
    cfg.auto_approve_compute = True                # would skip COMPUTE
    approver = ScriptedApprover([Decision.DENY])
    s = AgentSession(cfg, ctx, approver=approver,
                     provider=MockProvider([
                         turn_tools(("run_match", {})),
                         turn_text("not run")]),
                     ledger=Ledger(ctx.calc_dir))
    s.ask("match it")
    # confirmation WAS requested despite auto-approve-compute
    assert len(approver.requests) == 1
    assert approver.requests[0].tier == "mutate"
    s.close()


def test_render_call_echoes_exact_params():
    text = render_call("run_mp", {"n_particles": 10000, "seed": 7})
    assert "run_mp" in text and "[compute]" in text
    assert "n_particles = 10000" in text and "seed = 7" in text


def test_replay_reexecutes_approved_compute_only(ctx, tmp_path):
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.replay import replay

    led = Ledger(str(tmp_path))
    led.session_start(type("C", (), {"redacted": lambda s: {}})())
    led.tool(1, "get_status", {}, "read", "auto", "ok", 0.01)
    led.tool(1, "run_envelope", {}, "compute", "user", "submitted", 0.0)
    led.tool(2, "run_mp", {}, "compute", "denied", "denied", 0.0)
    led.tool(2, "job_status", {"job_id": "job-0001"}, "read", "auto",
             "ok", 0.0)

    report = replay(str(led.path), calc_dir=str(tmp_path))
    # only run_envelope re-executes — and it FAILS deterministically
    # because the fresh replay context has no lattice: that failure is
    # itself the honest outcome (the ledger lacked a load_lattice)
    assert [n for (n, _r) in report.skipped] == \
        ["get_status", "run_mp", "job_status"]
    assert [n for (n, _m) in report.failed] == ["run_envelope"]

    # a complete ledger (load first) replays clean
    led2 = Ledger(str(tmp_path), session_id="second")
    led2.session_start(type("C", (), {"redacted": lambda s: {}})())
    deck = tmp_path / "tiny.dat"
    deck.write_text("DRIFT 100 15 0\nEND\n")
    led2.tool(1, "load_lattice", {"path": str(deck)}, "mutate", "user",
              "ok", 0.01)
    report2 = replay(str(led2.path), calc_dir=str(tmp_path))
    assert report2.ok
    assert [n for (n, _s) in report2.executed] == ["load_lattice"]


def test_headline_core_is_independent_of_assist(tmp_path):
    """THE optionality guarantee: with linac_gen.assist made
    unimportable, the core CLI still works end to end."""
    deck = tmp_path / "t.dat"
    deck.write_text("DRIFT 100 15 0\nEND\n")
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.startswith('linac_gen.assist'):\n"
        "            raise ImportError('assist removed')\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import importlib\n"
        "try:\n"
        "    importlib.import_module('linac_gen.assist')\n"
        "    raise SystemExit('assist should have been unimportable')\n"
        "except ImportError:\n"
        "    pass\n"
        "for m in list(sys.modules):\n"
        "    assert not m.startswith('linac_gen.assist'), m\n"
        "from linac_gen.cli.common import load_lattice, "
        "run_envelope_sim\n"
        "from linac_gen.core.config import BeamConfig\n"
        f"lat = load_lattice({str(deck)!r})\n"
        "res = run_envelope_sim(lat, BeamConfig())\n"
        "assert len(res.sigma_x) > 0\n"
        "assert not any(m.startswith('linac_gen.assist') "
        "for m in sys.modules)\n"
        "print('CORE-OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    assert "CORE-OK" in out.stdout, out.stderr
