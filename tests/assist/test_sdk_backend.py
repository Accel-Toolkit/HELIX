"""Keyless Claude Agent SDK backend.

The SDK transport (loop/client/network) is never touched here — these tests
cover the parts we own: config acceptance, the availability probe, the
tier-gate → approver bridge, and the tool wrapper (run + ledger + refusal),
all with NO network and NO real SDK, matching the assist test discipline."""
from __future__ import annotations

import asyncio

import pytest

from linac_gen.assist import sdk_backend
from linac_gen.assist.agent import Decision
from linac_gen.assist.config import AssistConfig, resolve_config
from linac_gen.assist.ledger import Ledger
from linac_gen.assist.messages import Usage
from linac_gen.assist.tools import TOOLS, WorkContext


# ---- config -----------------------------------------------------------
def test_config_accepts_claude_sdk_without_key(monkeypatch):
    for v in ("HELIX_ASSIST_BASE_URL", "HELIX_ASSIST_API_KEY",
              "ANTHROPIC_API_KEY", "HELIX_ASSIST_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HELIX_ASSIST_PROVIDER", "claude_sdk")
    cfg = resolve_config()
    assert cfg is not None and cfg.provider == "claude_sdk"
    assert cfg.api_key == "" and cfg.base_url == ""   # keyless, no url


# ---- availability probe ----------------------------------------------
def test_sdk_available_reports_missing_sdk(monkeypatch):
    monkeypatch.setattr(sdk_backend.importlib.util, "find_spec",
                        lambda name: None)
    ok, why = sdk_backend.sdk_available()
    assert ok is False and "not installed" in why


def test_sdk_available_reports_missing_login(monkeypatch):
    monkeypatch.setattr(sdk_backend.importlib.util, "find_spec",
                        lambda name: object())        # SDK "present"
    monkeypatch.setattr(sdk_backend, "_ensure_claude_cli", lambda: None)
    ok, why = sdk_backend.sdk_available()
    assert ok is False and "claude` CLI" in why


# ---- a minimal fake session (no real SDK) ----------------------------
class _FakeSession:
    def __init__(self, tmp_path, approver):
        self.tools = TOOLS
        self.context = WorkContext(calc_dir=str(tmp_path))
        self.config = AssistConfig(provider="claude_sdk", model="")
        self.approver = approver
        self.ledger = Ledger(str(tmp_path))
        self.ledger.session_start(self.config)
        self.auto_approve_compute = False
        self.turn = 1
        self.usage_total = Usage()
        self.events = []
        self.stopped = False

    def _emit(self, **e):
        self.events.append(e)

    def request_stop(self):
        self.stopped = True


def _backend(tmp_path, approver):
    return sdk_backend.ClaudeSdkBackend(_FakeSession(tmp_path, approver))


# ---- tier gate -> approver bridge ------------------------------------
def test_gate_read_tier_is_auto(tmp_path):
    be = _backend(tmp_path, lambda r: Decision.DENY)   # never consulted
    got = asyncio.run(be._gate(TOOLS["get_status"], {}))
    assert got == "auto"


def test_gate_compute_auto_session(tmp_path):
    be = _backend(tmp_path, lambda r: Decision.DENY)
    be.session.auto_approve_compute = True
    got = asyncio.run(be._gate(TOOLS["run_envelope"], {}))
    assert got == "auto_session"


def test_gate_mutate_denied_returns_none(tmp_path):
    seen = []
    be = _backend(tmp_path, lambda r: seen.append(r) or Decision.DENY)
    got = asyncio.run(be._gate(TOOLS["load_lattice"], {"path": "x.dat"}))
    assert got is None                    # denied -> caller must refuse
    assert seen and seen[0].tier == "mutate"


def test_gate_mutate_approved_returns_user(tmp_path):
    be = _backend(tmp_path, lambda r: Decision.APPROVE)
    got = asyncio.run(be._gate(TOOLS["load_lattice"], {"path": "x.dat"}))
    assert got == "user"


# ---- tool wrapper: run + ledger, and denied -> refused ----------------
def test_run_tool_read_executes_and_ledgers(tmp_path):
    be = _backend(tmp_path, lambda r: Decision.APPROVE)
    out = asyncio.run(be._run_tool(TOOLS["get_status"], {}))
    text = out["content"][0]["text"]
    assert '"status": "ok"' in text
    # ledger recorded exactly this tool call
    lines = be.session.ledger.path.read_text().splitlines()
    assert any('"tool": "get_status"' in ln for ln in lines)


def test_run_tool_denied_mutate_does_not_run(tmp_path, monkeypatch):
    be = _backend(tmp_path, lambda r: Decision.DENY)
    called = {"n": 0}
    orig = TOOLS["load_lattice"].fn

    def _spy(ctx, **kw):
        called["n"] += 1
        return orig(ctx, **kw)

    monkeypatch.setattr(TOOLS["load_lattice"], "fn", _spy)
    out = asyncio.run(be._run_tool(TOOLS["load_lattice"],
                                   {"path": "examples/pipii/btl/btl.lgproj"}))
    assert '"status": "refused"' in out["content"][0]["text"]
    assert called["n"] == 0               # the mutate never executed


# ---- AgentSession delegates to the backend ---------------------------
def test_agent_session_uses_sdk_backend(tmp_path, monkeypatch):
    from linac_gen.assist import agent as agent_mod

    class _StubBackend:
        def __init__(self, session): self.session = session
        def ask(self, text): return f"stub-reply:{text}"
        def close(self): ...

    monkeypatch.setattr(agent_mod, "ClaudeSdkBackend", _StubBackend,
                        raising=False)
    # ClaudeSdkBackend is imported lazily inside __init__; patch the source
    monkeypatch.setattr(sdk_backend, "ClaudeSdkBackend", _StubBackend)
    cfg = AssistConfig(provider="claude_sdk", model="")
    sess = agent_mod.AgentSession(
        cfg, WorkContext(calc_dir=str(tmp_path)),
        approver=lambda r: Decision.APPROVE)
    assert sess._sdk is not None
    assert sess.ask("hello") == "stub-reply:hello"
    sess.close()


# ---- streaming: text deltas are emitted live -------------------------
def test_render_emits_live_text_deltas(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    be = _backend(tmp_path, lambda r: Decision.APPROVE)

    class _FakeStreamEvent:
        def __init__(self, text):
            self.event = {"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": text}}

    monkeypatch.setattr(be, "_stream_event_cls", lambda: _FakeStreamEvent)
    be._render(_FakeStreamEvent("Load"))
    be._render(_FakeStreamEvent("ed."))
    deltas = [e["text"] for e in be.session.events
              if e.get("type") == "assistant_delta"]
    assert deltas == ["Load", "ed."]
    assert be._streamed is True          # a streamed segment is open


def test_render_non_text_stream_event_ignored(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    be = _backend(tmp_path, lambda r: Decision.APPROVE)

    class _FakeStreamEvent:
        def __init__(self): self.event = {"type": "message_start"}

    monkeypatch.setattr(be, "_stream_event_cls", lambda: _FakeStreamEvent)
    be._render(_FakeStreamEvent())
    assert not [e for e in be.session.events
                if e.get("type") == "assistant_delta"]
    assert be._streamed is False


# ---- lifecycle: close() must cancel an in-flight turn (no hang) --------
def test_close_cancels_inflight_turn_no_hang(tmp_path, monkeypatch):
    """Regression (adversarial review #1): closing the panel mid-turn used
    to orphan ask()'s future and hang the worker QThread forever.  close()
    must cancel the in-flight turn so ask() returns promptly."""
    pytest.importorskip("claude_agent_sdk")
    import threading
    import time

    import claude_agent_sdk
    be = _backend(tmp_path, lambda r: Decision.APPROVE)

    class _FakeClient:                     # blocks in receive_response()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def query(self, text): return None
        async def receive_response(self):
            await asyncio.Event().wait()   # never set -> blocks until cancel
            if False:                      # make it an async generator
                yield None

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient",
                        lambda *a, **k: _FakeClient())
    out = {}

    def run():
        out["r"] = be.ask("hello")

    t = threading.Thread(target=run, daemon=True)   # daemon: never hang pytest
    t.start()
    time.sleep(1.0)                         # let the turn block
    be.close()                             # must cancel the turn
    t.join(timeout=6)
    assert not t.is_alive()                # ask() unblocked — no hang
    assert out.get("r") == ""              # cancelled turn -> empty reply


def test_turn_breaks_on_abort(tmp_path):
    """#4: _turn stops consuming messages once the session is aborted."""
    be = _backend(tmp_path, lambda r: Decision.APPROVE)
    seen = []

    class _Client:
        async def query(self, text): return None
        async def receive_response(self):
            for i in range(5):
                yield i

    be._client = _Client()
    be._render = lambda m: seen.append(m) or ""
    be.session._abort = __import__("threading").Event()
    be.session._abort.set()                # already aborted
    asyncio.run(be._turn("x"))
    assert seen == []                      # broke before rendering anything
