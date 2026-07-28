"""Keyless Claude backend — reasons via the **Claude Agent SDK**, reusing
the user's **Claude Code login** (no API key).

This is the third assistant transport, beside the stdlib ``urllib`` cloud
path (``providers.AnthropicProvider``, needs a key) and the local
OpenAI-compatible path (``providers.OpenAIProvider``, offline).  Unlike
those, the Agent SDK owns its own agent loop, so this module is a drop-in
**runtime** that ``AgentSession.ask()`` delegates to — it reuses the same
tool registry, tier gate, approver, ledger and event stream; only the
transport changes.

Optionality contract (identical to the rest of ``assist``): nothing here
is imported unless ``provider="claude_sdk"`` is chosen; ``claude_agent_sdk``
is imported lazily inside methods; with the SDK absent or the ``claude`` CLI
not logged in, :func:`sdk_available` returns ``(False, reason)`` and the
surfaces show setup guidance — the app never breaks.

Threading: a private asyncio loop runs a persistent ``ClaudeSDKClient`` on a
daemon thread (so multi-turn context survives across ``ask()`` calls);
``ask(text) -> str`` bridges to it synchronously via
``run_coroutine_threadsafe``.  HELIX tools are blocking, so they run on
executor threads — which also means their PIC FFTs take the safe
single-threaded path (see ``pic/gpu_backend._workers``).

SAFETY: the tier gate + human confirmation live **inside** the tool wrapper
(``_run_tool``), not in the SDK permission callback — so a compute/mutate
call always pauses for the approver regardless of the SDK's permission mode.
The SDK is sandboxed to the HELIX tools only (``allowed_tools`` +
``disallowed_tools`` block shell/file/web).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import json
import os
import shutil
import threading
import time

_MAX_TOOL_RESULT_CHARS = 20_000

# builtin Claude Code tools the model must never touch through HELIX
_DISALLOWED = ["Bash", "Read", "Write", "Edit", "MultiEdit", "NotebookEdit",
               "Glob", "Grep", "WebFetch", "WebSearch", "Task", "TodoWrite",
               "SlashCommand", "KillShell", "BashOutput"]


# ---------------------------------------------------------------------------
# availability probe (I/O-light: import check + PATH probe, no network)
# ---------------------------------------------------------------------------
def _ensure_claude_cli() -> str | None:
    """Locate the ``claude`` CLI whose login carries the subscription token.
    GUI launches inherit launchd's minimal PATH, so restore common bins."""
    found = shutil.which("claude")
    if found:
        return found
    for cand in (os.path.expanduser("~/.local/bin/claude"),
                 "/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        if os.path.exists(cand):
            os.environ["PATH"] = (os.path.dirname(cand) + os.pathsep
                                  + os.environ.get("PATH", ""))
            return cand
    return None


def sdk_available() -> tuple[bool, str]:
    """``(ok, reason)`` — ok when the SDK is importable AND a ``claude`` CLI
    login is reachable.  Used by the surfaces to gate the backend."""
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return (False, "the Claude Agent SDK is not installed — "
                       "pip install linac_gen[assist-sdk]")
    if _ensure_claude_cli() is None:
        return (False, "the `claude` CLI was not found — install Claude Code "
                       "and log in once (https://claude.com/claude-code), "
                       "then relaunch")
    return (True, "ok")


# ---------------------------------------------------------------------------
# the backend runtime
# ---------------------------------------------------------------------------
class ClaudeSdkBackend:
    """Owns a persistent ``ClaudeSDKClient`` on a private loop thread and
    exposes a synchronous :meth:`ask`.  Constructed with the owning
    :class:`AgentSession` so it reuses its tools/approver/ledger/events."""

    def __init__(self, session):
        self.session = session
        self._loop = None
        self._client = None
        self._thread = None
        self._ready = threading.Event()
        self._start_error: str | None = None
        self._stop: asyncio.Event | None = None
        self._turn_lock = threading.Lock()     # serialise turns (defensive)
        self._streamed = False                 # a delta was shown this segment
        self._se_cls = False                   # cached StreamEvent (False=unresolved)

    # -- lifecycle -------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._loop is not None:
            if self._start_error:
                raise RuntimeError(self._start_error)
            return
        _ensure_claude_cli()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="assist-claude-sdk")
        self._thread.start()
        if not self._ready.wait(timeout=90):
            raise RuntimeError("Claude SDK session did not start in time")
        if self._start_error:
            raise RuntimeError(self._start_error)

    def _run_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())
        except Exception as exc:                            # noqa: BLE001
            self._start_error = f"{type(exc).__name__}: {exc}"
            self._ready.set()

    async def _serve(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient, create_sdk_mcp_server
        self._stop = asyncio.Event()
        try:
            server = create_sdk_mcp_server("helix", "1.0.0",
                                           tools=self._sdk_tools())
            async with ClaudeSDKClient(self._options(server)) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()             # keep the session alive
                # cancel any in-flight turn AND await it so its future
                # actually resolves before the loop stops — otherwise
                # close() orphans the future and the worker QThread hangs
                # forever (crash on app quit).
                pending = [t for t in asyncio.all_tasks(self._loop)
                           if t is not asyncio.current_task()]
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:                            # noqa: BLE001
            self._start_error = f"{type(exc).__name__}: {exc}"
            self._ready.set()

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except Exception:                               # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    def interrupt(self) -> None:
        """Stop the CURRENT turn's server-side generation (best effort).

        Without this the abort flag only stops our *iteration* over
        ``receive_response`` — the model keeps generating in the CLI
        subprocess and the tokens land in the next receive."""
        client, loop = self._client, self._loop
        if client is None or loop is None:
            return

        async def _do():
            try:
                await client.interrupt()
            except Exception:                               # noqa: BLE001
                pass
        try:
            asyncio.run_coroutine_threadsafe(_do(), loop)
        except Exception:                                   # noqa: BLE001
            pass

    # -- the synchronous bridge -----------------------------------------
    def ask(self, user_text: str) -> str:
        try:
            self._ensure_started()
        except Exception as exc:                            # noqa: BLE001
            msg = f"[claude-sdk] cannot start: {exc}"
            self.session._emit(type="error", message=msg)
            return msg
        with self._turn_lock:
            fut = asyncio.run_coroutine_threadsafe(
                self._turn(user_text), self._loop)
            try:
                return fut.result()
            except (asyncio.CancelledError,
                    concurrent.futures.CancelledError):
                return ""                 # turn cancelled (panel closing)
            except Exception as exc:                        # noqa: BLE001
                msg = f"[claude-sdk] turn failed: {exc}"
                self.session._emit(type="error", message=msg)
                return msg

    async def _turn(self, text: str) -> str:
        await self._client.query(text)
        final = ""
        async for msg in self._client.receive_response():
            if self.session._abort.is_set():   # user denied/aborted or closing
                break
            piece = self._render(msg)
            if piece:
                final = piece
        return final

    def _stream_event_cls(self):
        """StreamEvent class if this SDK build streams, else None (cached)."""
        if self._se_cls is False:
            try:
                from claude_agent_sdk import StreamEvent
            except ImportError:
                try:
                    from claude_agent_sdk.types import StreamEvent
                except ImportError:
                    StreamEvent = None
            self._se_cls = StreamEvent
        return self._se_cls

    # -- rendering (feeds our existing event stream + ledger) -----------
    def _render(self, msg) -> str:
        from claude_agent_sdk import (
            AssistantMessage, ResultMessage, TextBlock, ToolUseBlock,
        )
        sess = self.session
        # 1. live streaming: emit each text delta as it is generated
        se = self._stream_event_cls()
        if se is not None and isinstance(msg, se):
            ev = getattr(msg, "event", {}) or {}
            if ev.get("type") == "content_block_delta":
                delta = ev.get("delta", {}) or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    self._streamed = True
                    sess._emit(type="assistant_delta", text=delta["text"])
            return ""
        if isinstance(msg, AssistantMessage):
            text_out = ""
            for b in getattr(msg, "content", []) or []:
                if isinstance(b, TextBlock) and b.text.strip():
                    text_out = b.text
                    if self._streamed:        # already shown live via deltas
                        sess._emit(type="assistant_delta_done")
                        self._streamed = False
                    else:
                        sess._emit(type="assistant_text", text=b.text)
                elif isinstance(b, ToolUseBlock):
                    if self._streamed:
                        sess._emit(type="assistant_delta_done")
                        self._streamed = False
                    name = b.name.replace("mcp__helix__", "")
                    sess._emit(type="tool_start", tool=name,
                               params=dict(b.input or {}))
            return text_out
        if isinstance(msg, ResultMessage):
            if self._streamed:
                sess._emit(type="assistant_delta_done")
                self._streamed = False
            from linac_gen.assist.messages import Usage
            u = _usage_from(getattr(msg, "usage", None))
            sess.usage_total.add(u)
            sess._emit(type="usage",
                       input_tokens=sess.usage_total.input_tokens,
                       output_tokens=sess.usage_total.output_tokens)
            sess.ledger.assistant(sess.turn, getattr(msg, "result", "") or "",
                                  u)
            if getattr(msg, "is_error", False):
                sess._emit(type="error",
                           message=f"turn error: {getattr(msg,'result','')}")
        return ""

    # -- SDK plumbing ----------------------------------------------------
    def _options(self, server):
        import dataclasses

        from claude_agent_sdk import ClaudeAgentOptions
        from linac_gen.assist.prompts import build_system_prompt
        names = list(self.session.tools.keys())
        extra = ""
        try:
            extra = self.session._prompt_extra()
        except Exception:                                   # noqa: BLE001
            pass
        want = dict(
            system_prompt=build_system_prompt(self.session.context,
                                              extra=extra),
            mcp_servers={"helix": server},
            allowed_tools=[f"mcp__helix__{n}" for n in names],
            disallowed_tools=list(_DISALLOWED),
            # our in-tool gate is the real control; don't let the SDK prompt
            permission_mode="bypassPermissions",
            max_turns=max(1, self.session.config.max_tool_iterations) * 3,
            cwd=os.getcwd(),
            setting_sources=[],
            # stream tokens live for a responsive, "typing" reply
            include_partial_messages=self._stream_event_cls() is not None,
        )
        model = (getattr(self.session.config, "model", "") or "").strip()
        # the anthropic-provider default id is not an SDK model name; skip it
        if model and model != "claude-sonnet-5":
            want["model"] = model
        valid = {f.name for f in dataclasses.fields(ClaudeAgentOptions)}
        return ClaudeAgentOptions(**{k: v for k, v in want.items()
                                     if k in valid})

    def _sdk_tools(self) -> list:
        from claude_agent_sdk import tool
        out = []
        for spec in self.session.tools.values():
            @tool(spec.name, spec.description, spec.schema)
            async def _t(args, _spec=spec):
                return await self._run_tool(_spec, dict(args or {}))
            out.append(_t)
        return out

    async def _run_tool(self, spec, params: dict) -> dict:
        """Tier-gate + confirm (via our approver) then run the HELIX tool on
        an executor thread; log to the ledger; return SDK tool content."""
        loop = asyncio.get_running_loop()
        approved_by = await self._gate(spec, params)
        if approved_by is None:                     # denied / aborted
            return _content({"status": "refused",
                             "data": {"message": "the user denied this call"},
                             "provenance": {}, "warnings": []})
        # long-running tools run inline here (no job manager in SDK mode) —
        # pass a cooperative abort so a close/deny can stop them, and a
        # throttled progress callback so the GUI progress label stays live
        call = dict(params)
        from linac_gen.assist.agent import LONG_RUNNING
        if spec.name in LONG_RUNNING:
            call["should_abort"] = lambda: self.session._abort.is_set()
            last = [0.0]

            def _progress(s_mm, i, n, _name=spec.name):
                now = time.time()
                if now - last[0] >= 0.25:
                    last[0] = now
                    self.session._emit(
                        type="progress", tool=_name,
                        text=f"{_name}: s = {float(s_mm) / 1000.0:.1f} m")
            call["progress_callback"] = _progress
        t0 = time.time()
        try:
            env = await loop.run_in_executor(
                None, lambda: spec.fn(self.session.context, **call))
        except TypeError as exc:
            env = {"status": "refused",
                   "data": {"message": f"bad parameters: {exc}"},
                   "provenance": {}, "warnings": []}
        except Exception as exc:                            # noqa: BLE001
            env = {"status": "error",
                   "data": {"message": f"{type(exc).__name__}: {exc}"},
                   "provenance": {}, "warnings": []}
        # provenance stamp for assistant-written HDF5 (parity with agent.py)
        if (spec.name == "write_results" and env.get("status") == "ok"
                and str(params.get("format", "hdf5")) == "hdf5"):
            try:
                from linac_gen.assist.ledger import stamp_assist_provenance
                stamp_assist_provenance(
                    env["data"]["written"],
                    session_id=self.session.ledger.session_id,
                    turn=self.session.turn, tool=spec.name,
                    approved_by=approved_by,
                    ledger_path=str(self.session.ledger.path))
            except Exception:                               # noqa: BLE001
                pass
        self.session.ledger.tool(self.session.turn, spec.name, params,
                                 spec.tier, approved_by,
                                 env.get("status", "?"), time.time() - t0,
                                 provenance=env.get("provenance"))
        self.session._emit(type="tool_done", tool=spec.name,
                           status=env.get("status"))
        data = env.get("data")
        if (isinstance(data, dict) and data.get("saved_to")
                and str(data.get("mime", "")).startswith("image/")):
            self.session._emit(type="capture",
                               path=str(data["saved_to"]),
                               tool=spec.name)
        return _content(env)

    async def _gate(self, spec, params: dict):
        """Return an ``approved_by`` string, or ``None`` if denied.
        Mirrors ``agent.AgentSession._execute`` tier logic exactly."""
        from linac_gen.assist.agent import ConfirmationRequest, Decision
        from linac_gen.assist.tools import render_call
        sess = self.session
        if spec.tier not in ("compute", "mutate"):
            return "auto"
        if spec.tier == "compute" and sess.auto_approve_compute:
            return "auto_session"
        req = ConfirmationRequest(
            tool=spec.name, tier=spec.tier, params=params,
            pretty=render_call(spec.name, params),
            allow_session_auto=(spec.tier == "compute"))
        sess._emit(type="confirm_pending", request=req)
        loop = asyncio.get_running_loop()
        decision = await loop.run_in_executor(None, sess.approver, req)
        sess._emit(type="confirm_resolved", tool=spec.name,
                   decision=decision.value)
        if decision is Decision.ABORT:
            sess.request_stop()
            decision = Decision.DENY
        if decision is Decision.DENY:
            sess.ledger.tool(sess.turn, spec.name, params, spec.tier,
                             "denied", "denied", 0.0)
            return None
        if decision is Decision.APPROVE_SESSION:
            sess.auto_approve_compute = True
        return decision.value


def _usage_from(usage) -> "object":
    from linac_gen.assist.messages import Usage
    if usage is None:
        return Usage()
    if isinstance(usage, dict):
        return Usage(input_tokens=int(usage.get("input_tokens", 0) or 0),
                     output_tokens=int(usage.get("output_tokens", 0) or 0))
    return Usage(input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                 output_tokens=int(getattr(usage, "output_tokens", 0) or 0))


def _content(env: dict) -> dict:
    # Sight: a capture payload becomes a real MCP image block so the model
    # SEES the figure; the JSON text block keeps the metadata (minus b64).
    data = env.get("data")
    if isinstance(data, dict) and data.get("img_b64"):
        b64 = data["img_b64"]
        mime = data.get("mime", "image/png")
        meta = {k: v for k, v in data.items() if k != "img_b64"}
        text = json.dumps({"status": env.get("status"), "data": meta,
                           "warnings": env.get("warnings", [])},
                          default=str)
        return {"content": [
            {"type": "text", "text": text},
            {"type": "image", "data": b64, "mimeType": mime},
        ]}
    text = json.dumps(env, default=str)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        text = json.dumps(
            {"status": env.get("status"),
             "warnings": env.get("warnings", []),
             "data": {"truncated": True,
                      "preview": text[:_MAX_TOOL_RESULT_CHARS]}},
            default=str)
    return {"content": [{"type": "text", "text": text}]}
