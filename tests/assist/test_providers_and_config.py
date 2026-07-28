# tests/assist/test_providers_and_config.py
"""Provider adapters (pure build/parse, canned payloads — no network)
and config resolution precedence."""
from __future__ import annotations

import json

import pytest

from linac_gen.assist.config import AssistConfig, resolve_config
from linac_gen.assist.messages import (
    AssistantMsg, StopReason, ToolOutcome, ToolResultsMsg, UserMsg,
)
from linac_gen.assist.providers import (
    anthropic_build_body, anthropic_parse_response,
    openai_build_body, openai_parse_response,
)
from linac_gen.assist.testing import turn_tools


def test_anthropic_round_trip_build_and_parse():
    transcript = [UserMsg("run it")]
    body = anthropic_build_body("m", "SYS", transcript,
                                [{"name": "t", "description": "d",
                                  "input_schema": {"type": "object"}}],
                                1024)
    assert body["system"] == "SYS" and body["max_tokens"] == 1024
    assert body["messages"][0] == {"role": "user", "content": "run it"}

    payload = {"content": [{"type": "text", "text": "on it — "},
                           {"type": "tool_use", "id": "toolu_1",
                            "name": "run_envelope", "input": {}}],
               "stop_reason": "tool_use",
               "usage": {"input_tokens": 100, "output_tokens": 20}}
    turn = anthropic_parse_response(payload)
    assert turn.stop_reason is StopReason.TOOL_USE
    assert turn.tool_calls[0].name == "run_envelope"
    assert turn.usage.input_tokens == 100

    # tool results render as ONE user message with tool_result blocks
    transcript = [UserMsg("go"), AssistantMsg(turn),
                  ToolResultsMsg((ToolOutcome("toolu_1", "run_envelope",
                                              '{"status":"ok"}'),))]
    body2 = anthropic_build_body("m", "SYS", transcript, [], 1024)
    last = body2["messages"][-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "toolu_1"


def test_openai_round_trip_and_malformed_arguments_guard():
    turn = turn_tools(("run_mp", {"seed": 7}))
    transcript = [UserMsg("go"), AssistantMsg(turn),
                  ToolResultsMsg((ToolOutcome("tc-0", "run_mp",
                                              '{"status":"ok"}'),))]
    body = openai_build_body("m", "SYS", transcript,
                             [{"name": "run_mp", "description": "d",
                               "input_schema": {"type": "object"}}],
                             512)
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    am = body["messages"][2]
    assert json.loads(am["tool_calls"][0]["function"]["arguments"]) == \
        {"seed": 7}
    assert body["messages"][3]["role"] == "tool"

    payload = {"choices": [{"finish_reason": "tool_calls",
                            "message": {"content": None, "tool_calls": [
                                {"id": "call_1", "type": "function",
                                 "function": {"name": "run_mp",
                                              "arguments":
                                              "{seed: NOT JSON"}}]}}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    parsed = openai_parse_response(payload)
    assert parsed.tool_calls[0].args.get("__parse_error__")


def test_config_resolution_precedence(monkeypatch):
    for var in ("HELIX_ASSIST_PROVIDER", "HELIX_ASSIST_BASE_URL",
                "HELIX_ASSIST_API_KEY", "HELIX_ASSIST_MODEL",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_config() is None              # nothing configured

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    cfg = resolve_config()
    assert cfg.provider == "anthropic" and cfg.api_key == "sk-x"

    monkeypatch.setenv("HELIX_ASSIST_BASE_URL",
                       "http://localhost:11434/v1")
    monkeypatch.setenv("HELIX_ASSIST_PROVIDER", "openai")
    cfg = resolve_config()
    assert cfg.provider == "openai"
    assert cfg.base_url.endswith("/v1")

    explicit = AssistConfig(provider="anthropic", model="m", api_key="k")
    assert resolve_config(explicit) is explicit  # explicit wins


def test_config_redacted_never_contains_key():
    cfg = AssistConfig(provider="anthropic", model="m",
                       api_key="sk-SECRET")
    red = cfg.redacted()
    assert "api_key" not in red
    assert red["api_key_set"] is True
    assert "sk-SECRET" not in json.dumps(red)


def test_available_probe_no_network(monkeypatch):
    for var in ("HELIX_ASSIST_PROVIDER", "HELIX_ASSIST_BASE_URL",
                "HELIX_ASSIST_API_KEY", "HELIX_ASSIST_MODEL",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    import linac_gen.assist as assist
    ok, reason = assist.available()
    assert ok is False and "provider" in reason.lower()
    monkeypatch.setenv("HELIX_ASSIST_BASE_URL", "http://localhost:1/v1")
    ok, reason = assist.available()
    assert ok is True                            # config only, no I/O
