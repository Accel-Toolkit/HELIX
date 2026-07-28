"""Network-free test doubles, shipped so GUI tests can import them.

``MockProvider`` replays a script of assistant turns; nothing here
opens a socket, and the assist test-suite additionally blocks all
network at the socket layer.
"""
from __future__ import annotations

from linac_gen.assist.messages import (
    AssistantTurn, StopReason, ToolCall, Usage,
)


def turn_text(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, stop_reason=StopReason.END,
                         usage=Usage(10, 5))


def turn_tools(*calls: tuple, text: str = "") -> AssistantTurn:
    """``turn_tools(("run_envelope", {}), ("get_status", {}))``"""
    tcs = tuple(ToolCall(id=f"tc-{i}", name=name, args=dict(args))
                for i, (name, args) in enumerate(calls))
    return AssistantTurn(text=text, tool_calls=tcs,
                         stop_reason=StopReason.TOOL_USE,
                         usage=Usage(10, 5))


class MockProvider:
    """Scripted provider: each ``send`` pops the next turn.  Entries
    may be callables receiving the transcript (for assertions)."""

    name = "mock"

    def __init__(self, turns):
        self._turns = list(turns)
        self.requests: list[dict] = []

    def send(self, *, system, transcript, tools, max_tokens):
        self.requests.append({"system": system,
                              "transcript": list(transcript),
                              "tools": [t["name"] for t in tools],
                              "max_tokens": max_tokens})
        if not self._turns:
            raise AssertionError("MockProvider script exhausted")
        t = self._turns.pop(0)
        return t(transcript) if callable(t) else t


class ScriptedApprover:
    """Feeds a fixed list of Decisions; records every request."""

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.requests = []

    def __call__(self, req):
        self.requests.append(req)
        if not self._decisions:
            raise AssertionError("ScriptedApprover script exhausted")
        return self._decisions.pop(0)
