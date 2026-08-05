"""The agent loop: provider ↔ tools, with tier gating and audit.

Flow per user turn::

    drain finished jobs → system notes
    send transcript → provider
    while the model requests tools:
        for each call: gate by tier → confirm (echo the resolved call)
                     → execute inline or submit as a background job
        feed ALL outcomes back in one message; repeat (bounded)

Safety properties:
* ``read`` auto-runs; ``compute`` requires confirmation (session
  auto-approve possible); ``mutate`` ALWAYS requires confirmation.
* The confirmation shows the EXACT resolved call (``tools.render_call``).
* A denied call returns a refused outcome to the model (it re-plans);
  the ledger records who approved what.
* Unknown tool names are refused — there is no fallthrough to any
  dynamic execution.
"""
from __future__ import annotations

import enum
import json
import queue
import threading
import time
from dataclasses import dataclass

from linac_gen.assist.messages import (
    AssistantMsg, StopReason, SystemNote, ToolOutcome, ToolResultsMsg,
    Usage, UserMsg,
)

#: compute tools executed through the job manager (return a job_id
#: immediately); other approved tools run inline.
LONG_RUNNING = {"run_envelope", "run_mp", "run_match", "parameter_scan",
                "run_campaign", "tuning_plan", "hofmann_stability",
                "lebt_scc"}

_MAX_TOOL_RESULT_CHARS = 20_000


class Decision(enum.Enum):
    APPROVE = "user"
    APPROVE_SESSION = "auto_session"     # compute only
    DENY = "denied"
    ABORT = "abort"


@dataclass(frozen=True)
class ConfirmationRequest:
    tool: str
    tier: str
    params: dict
    pretty: str
    allow_session_auto: bool


class AgentSession:
    """One conversation.  ``approver(req) -> Decision`` is supplied by
    the surface (CLI prompt / GUI buttons); ``on_event(dict)`` receives
    render events and must never raise."""

    def __init__(self, config, context, *, approver, provider=None,
                 ledger=None, jobs=None, on_event=None):
        from linac_gen.assist.jobs import JobManager
        from linac_gen.assist.ledger import Ledger
        from linac_gen.assist.tools import TOOLS
        self.config = config
        self.context = context
        context._assist_config = config          # web_search gate
        self.approver = approver
        self.on_event = on_event or (lambda e: None)
        self.tools = TOOLS
        self.jobs = jobs or JobManager(config.max_concurrent_jobs)
        self.ledger = ledger or Ledger(getattr(context, "calc_dir", "."))
        self.ledger.session_start(config)
        # Three transports: the Claude Agent SDK backend (keyless — reuses
        # the Claude Code login, owns its own loop) OR a request/response
        # Provider (anthropic key / local OpenAI-compatible).  An explicitly
        # injected provider (tests' MockProvider) always wins.
        self._sdk = None
        if provider is None and getattr(config, "provider", "") == "claude_sdk":
            from linac_gen.assist.sdk_backend import ClaudeSdkBackend
            self._sdk = ClaudeSdkBackend(self)
        elif provider is None:
            from linac_gen.assist.providers import make_provider
            provider = make_provider(config)
        self.provider = provider
        self.transcript: list = []
        self.usage_total = Usage()
        self.turn = 0
        # cross-session memory: recent lab-notebook entries -> system prompt
        try:
            self._digest = ""
            from linac_gen.assist.notebook import load_tail
            self.notebook_tail = load_tail(
                getattr(context, "calc_dir", ".") or ".")
        except Exception:                                   # noqa: BLE001
            self.notebook_tail = ""
        self.auto_approve_compute = bool(config.auto_approve_compute)
        self._abort = threading.Event()
        self._closed = False
        # Machine-event channel: thread-safe inbox for watcher alerts /
        # job completions; rendered immediately, injected into the model
        # at the next turn boundary (see submit_event / ask).
        self._events: queue.Queue[str] = queue.Queue()

    # ------------------------------------------------------------------
    def request_stop(self) -> None:
        """Stop the CURRENT turn.  Reusable: ``ask()`` re-arms."""
        self._abort.set()
        if self._sdk is not None:
            try:            # actually stop the server-side generation too
                self._sdk.interrupt()
            except Exception:                               # noqa: BLE001
                pass

    def close(self) -> None:
        self._closed = True
        self._abort.set()
        try:                       # an abandoned drill restores the lattice
            from linac_gen.assist.instructor import restore_if_active
            restore_if_active(self.context)
        except Exception:                               # noqa: BLE001
            pass
        if self._sdk is not None:
            try:
                self._sdk.close()
            except Exception:                               # noqa: BLE001
                pass
        self.jobs.shutdown()
        try:                       # lab-notebook memory (never raises)
            from linac_gen.assist.notebook import append_session_summary
            append_session_summary(self)
        except Exception:                                   # noqa: BLE001
            pass
        self.ledger.session_end(self.usage_total)

    def _prompt_extra(self) -> str:
        parts = []
        if self._digest:
            parts.append("LIVE STATE (auto-refreshed at the start of this "
                         "turn — orientation only, use tools for anything "
                         "deeper):\n" + self._digest)
        if self.notebook_tail:
            parts.append("PAST SESSIONS (your lab notebook — what was done "
                         "and concluded before):\n" + self.notebook_tail)
        return "\n\n".join(parts)

    def _refresh_digest(self) -> None:
        try:
            from linac_gen.assist.prompts import state_digest
            self._digest = state_digest(self.context, self.jobs)
        except Exception:                                   # noqa: BLE001
            self._digest = ""

    def run_fast(self, name: str, params: dict):
        """Direct READ-tier execution for the GUI's local intent
        fast-path ('instant commands').  No LLM turn: the tool runs,
        the call is ledgered like any other, and a quiet note is queued
        so the MODEL learns of it at its next turn (it must never be
        surprised by state the user changed around it).  Returns the
        ToolResult, or None when not eligible (unknown tool, not
        read-tier, or session closed)."""
        tool = self.tools.get(name)
        if tool is None or tool.tier != "read" or self._closed:
            return None
        t0 = time.time()
        try:
            res = tool.fn(self.context, **dict(params or {}))
        except Exception as exc:                            # noqa: BLE001
            res = {"status": "error",
                   "data": {"message": f"{type(exc).__name__}: {exc}"},
                   "provenance": {}, "warnings": []}
        self.ledger.tool(self.turn, name, dict(params or {}), tool.tier,
                         approved_by="fast_path",
                         status=str(res.get("status", "?")),
                         duration_s=time.time() - t0)
        try:
            self._events.put(f"user ran {name} via an instant command "
                             f"(status {res.get('status')})")
        except Exception:                                   # noqa: BLE001
            pass
        return res

    def _emit(self, **event) -> None:
        try:
            self.on_event(event)
        except Exception:                                   # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def _tool_specs(self) -> list[dict]:
        from linac_gen.assist.tools import provider_tool_specs
        specs = provider_tool_specs()
        specs.extend([
            {"name": "job_status", "description":
                "Status of a background job started by a compute tool.",
             "input_schema": {"type": "object",
                              "properties": {"job_id": {"type": "string"}},
                              "required": ["job_id"]}},
            {"name": "job_list", "description":
                "List all background jobs of this session.",
             "input_schema": {"type": "object", "properties": {},
                              "required": []}},
            {"name": "job_cancel", "description":
                "Request cancellation of a running background job.",
             "input_schema": {"type": "object",
                              "properties": {"job_id": {"type": "string"}},
                              "required": ["job_id"]}},
        ])
        return specs

    # ------------------------------------------------------------------
    # Headless run watching (the GUI panel wires RunWatch to the app's
    # results_changed signal instead; this hook serves REPL/MCP-less
    # surfaces): inspect after compute completions.
    def attach_watch(self, watch) -> None:
        self._watch = watch

    def _inspect_results(self) -> None:
        w = getattr(self, "_watch", None)
        if w is None:
            return
        try:
            from linac_gen.assist.tools_analysis import _identity
            alerts = w.inspect(getattr(self.context, "results", None),
                               identity=_identity(self.context))
        except Exception:                                   # noqa: BLE001
            return
        for a in alerts:
            self.submit_event(a)

    # ------------------------------------------------------------------
    # Machine-event channel (watcher alerts, job completions, anything a
    # surface wants narrated).  Thread-safe; render is immediate, model
    # injection happens at turn boundaries so events never interleave a
    # streaming reply.
    def submit_event(self, text: str) -> None:
        text = str(text).strip()
        if not text:
            return
        self._emit(type="event", text=text)
        self._events.put(text)

    def _drain_events_into_notes(self) -> None:
        while True:
            try:
                text = self._events.get_nowait()
            except queue.Empty:
                break
            self.transcript.append(SystemNote("[machine event] " + text))

    def _pop_events(self) -> list[str]:
        out: list[str] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def _drain_jobs_into_notes(self) -> None:
        drained = False
        for job in self.jobs.drain_completions():
            drained = True
            snap = job.snapshot()
            note = (f"job {job.id} ({job.tool}) finished with state "
                    f"{job.state}: "
                    + json.dumps(snap.get("result"), default=str)[:2000])
            self.transcript.append(SystemNote(note))
            self._emit(type="event",
                       text=f"job {job.id} ({job.tool}) finished: "
                            f"{job.state}", job=snap)
        if drained:
            self._inspect_results()

    # ------------------------------------------------------------------
    def _execute(self, call) -> ToolOutcome:
        """Gate, confirm, run (inline or as a job) one tool call."""
        from linac_gen.assist.tools import render_call

        def outcome(env, is_error=None):
            # this transport is text-only: replace a capture's b64 payload
            # with a pointer to the saved file (the SDK backend shows the
            # actual image)
            data = env.get("data")
            if isinstance(data, dict) and data.get("img_b64"):
                env = dict(env)
                env["data"] = {k: v for k, v in data.items()
                               if k != "img_b64"}
                env["data"]["image_note"] = (
                    "image captured (not viewable over this backend — "
                    "saved to " + str(data.get("saved_to", "disk"))
                    + "; the claude_sdk backend can see images)")
            content = json.dumps(env, default=str)
            if len(content) > _MAX_TOOL_RESULT_CHARS:
                # keep the envelope VALID JSON: replace the payload
                # with a preview + marker (full record is in the ledger)
                content = json.dumps(
                    {"status": env.get("status"),
                     "warnings": env.get("warnings", []),
                     "data": {"truncated": True,
                              "preview":
                              content[:_MAX_TOOL_RESULT_CHARS]}},
                    default=str)
            return ToolOutcome(
                call_id=call.id, name=call.name, content=content,
                is_error=(env.get("status") != "ok"
                          if is_error is None else is_error))

        if "__parse_error__" in (call.args or {}):
            return outcome({"status": "refused",
                            "data": {"message":
                                     "tool arguments were not valid "
                                     "JSON — resend the call"},
                            "provenance": {}, "warnings": []})

        # runtime job-control tools
        if call.name == "job_status":
            return outcome(self.jobs.status(str(call.args.get("job_id"))))
        if call.name == "job_list":
            return outcome(self.jobs.list_jobs())
        if call.name == "job_cancel":
            return outcome(self.jobs.cancel(str(call.args.get("job_id"))))

        tool = self.tools.get(call.name)
        if tool is None:
            return outcome({"status": "refused",
                            "data": {"message":
                                     f"unknown tool {call.name!r}"},
                            "provenance": {}, "warnings": []})

        params = dict(call.args or {})
        approved_by = "auto"
        if tool.tier in ("compute", "mutate"):
            if tool.tier == "compute" and self.auto_approve_compute:
                approved_by = "auto_session"
            else:
                req = ConfirmationRequest(
                    tool=call.name, tier=tool.tier, params=params,
                    pretty=render_call(call.name, params),
                    allow_session_auto=(tool.tier == "compute"))
                self._emit(type="confirm_pending", request=req)
                decision = self.approver(req)
                self._emit(type="confirm_resolved", tool=call.name,
                           decision=decision.value)
                if decision is Decision.ABORT:
                    self._abort.set()
                    decision = Decision.DENY
                if decision is Decision.DENY:
                    self.ledger.tool(self.turn, call.name, params,
                                     tool.tier, "denied", "denied", 0.0)
                    return outcome({"status": "refused",
                                    "data": {"message":
                                             "the user denied this call"},
                                    "provenance": {}, "warnings": []})
                if decision is Decision.APPROVE_SESSION:
                    self.auto_approve_compute = True
                approved_by = decision.value

        t0 = time.time()
        self._emit(type="tool_start", tool=call.name, params=params)

        if call.name in LONG_RUNNING:
            def runner(progress_cb, should_abort,
                       _tool=tool, _params=params, _name=call.name):
                last = [0.0]      # agent-side throttle: queued GUI signals
                                  # must not be flooded by per-step calls

                def cb(s_mm, i, n):
                    progress_cb(s_mm, i, n)
                    now = time.time()
                    if now - last[0] >= 0.25:
                        last[0] = now
                        self._emit(type="progress", tool=_name,
                                   text=f"{_name}: s = "
                                        f"{float(s_mm) / 1000.0:.1f} m")
                return _tool.fn(self.context,
                                progress_callback=cb,
                                should_abort=should_abort, **_params)
            job_id = self.jobs.submit(call.name, params, runner)
            env = {"status": "ok",
                   "data": {"job_id": job_id, "state": "running",
                            "note": "long-running job started; use "
                                    "job_status to poll"},
                   "provenance": {}, "warnings": []}
            self.ledger.tool(self.turn, call.name, params, tool.tier,
                             approved_by, "submitted",
                             time.time() - t0, job_id=job_id)
            self._emit(type="job_submitted", tool=call.name,
                       job_id=job_id)
            return outcome(env)

        try:
            env = tool.fn(self.context, **params)
        except TypeError as exc:
            env = {"status": "refused",
                   "data": {"message": f"bad parameters: {exc}"},
                   "provenance": {}, "warnings": []}
        except Exception as exc:                            # noqa: BLE001
            env = {"status": "error",
                   "data": {"message": f"{type(exc).__name__}: {exc}"},
                   "provenance": {}, "warnings": []}
        # provenance stamp for assistant-written HDF5 results
        if (call.name == "write_results" and env.get("status") == "ok"
                and str(params.get("format", "hdf5")) == "hdf5"):
            from linac_gen.assist.ledger import stamp_assist_provenance
            stamp_assist_provenance(
                env["data"]["written"],
                session_id=self.ledger.session_id, turn=self.turn,
                tool=call.name, approved_by=approved_by,
                ledger_path=str(self.ledger.path))
        self.ledger.tool(self.turn, call.name, params, tool.tier,
                         approved_by, env.get("status", "?"),
                         time.time() - t0,
                         provenance=env.get("provenance"))
        self._emit(type="tool_done", tool=call.name,
                   status=env.get("status"))
        data = env.get("data")
        if (isinstance(data, dict) and data.get("saved_to")
                and str(data.get("mime", "")).startswith("image/")):
            # let the surface show the user what the assistant saw
            self._emit(type="capture", path=str(data["saved_to"]),
                       tool=call.name)
        if tool.tier == "compute" and env.get("status") == "ok":
            self._inspect_results()          # headless run watching
        return outcome(env)

    # ------------------------------------------------------------------
    def ask_event(self) -> str:
        """Give the model a turn to react to PENDING machine events
        (already queued via ``submit_event``).  A normal ``ask()`` whose
        input is system-generated, not typed — surfaces suppress the
        'you' transcript line for these turns."""
        return self.ask("[machine event] New machine event(s) arrived — "
                        "acknowledge briefly and flag anything that "
                        "needs attention; do not start actions unless "
                        "previously asked to.")

    def ask(self, user_text: str) -> str:
        """One full user turn; returns the final assistant text."""
        if not self._closed:
            # A previous Stop must not kill this turn (the abort event
            # used to be one-way — first Stop silently broke the session).
            self._abort.clear()
        self.turn += 1
        self.ledger.user(self.turn, user_text)
        self._refresh_digest()
        if self._sdk is not None:
            # the Agent SDK owns its own loop, transcript, tool dispatch and
            # confirmation (via our approver, inside the tool wrapper).
            # It also owns the transcript, so pending machine events are
            # injected by prefixing this query.
            events = self._pop_events()
            if events:
                user_text = ("SYSTEM EVENTS (machine notifications, not "
                             "from the user):\n"
                             + "\n".join(f"- {e}" for e in events)
                             + "\n\n" + user_text)
            text = self._sdk.ask(user_text)
            self._emit(type="turn_done")
            return text
        from linac_gen.assist.prompts import build_system_prompt
        self._drain_jobs_into_notes()
        self._drain_events_into_notes()
        self.transcript.append(UserMsg(user_text))
        final_text = ""
        for _ in range(max(1, self.config.max_tool_iterations)):
            if self._abort.is_set():
                break
            system = build_system_prompt(self.context,
                                         extra=self._prompt_extra())
            turn = self.provider.send(system=system,
                                      transcript=self.transcript,
                                      tools=self._tool_specs(),
                                      max_tokens=4096)
            self.usage_total.add(turn.usage)
            self._emit(type="usage",
                       input_tokens=self.usage_total.input_tokens,
                       output_tokens=self.usage_total.output_tokens)
            self.transcript.append(AssistantMsg(turn))
            if turn.text:
                final_text = turn.text
                self._emit(type="assistant_text", text=turn.text)
            self.ledger.assistant(self.turn, turn.text, turn.usage)
            # Defensive: some local models emit tool calls but report
            # stop_reason=end.  Any tool call MUST be answered, or the
            # next request has a dangling tool_use id that both APIs
            # reject — so drive off tool_calls, not the stop reason.
            if not turn.tool_calls:
                break
            outcomes = tuple(self._execute(c) for c in turn.tool_calls)
            self.transcript.append(ToolResultsMsg(outcomes))
            self._drain_jobs_into_notes()
            self._drain_events_into_notes()
        else:
            note = ("tool-call budget for this turn exhausted "
                    f"({self.config.max_tool_iterations} iterations)")
            self.transcript.append(SystemNote(note))
            self._emit(type="error", message=note)
        self._emit(type="turn_done")
        return final_text
