"""LLM provider adapters — stdlib only, no SDK dependency.

Two wire formats behind one interface:
* :class:`AnthropicProvider` — the Anthropic Messages API.
* :class:`OpenAIProvider`   — any OpenAI-compatible
  ``/v1/chat/completions`` endpoint, which includes LOCAL servers
  (ollama, llama.cpp, vLLM) so the assistant can run fully offline
  with no API key.

Adapters are split into pure ``build_body`` / ``parse_response``
functions (unit-tested on canned payloads, no network) plus one thin
``_post_json`` HTTP call.  API keys never appear in exceptions, logs
or the ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from linac_gen.assist.messages import (
    AssistantMsg, AssistantTurn, StopReason, SystemNote, ToolCall,
    ToolResultsMsg, Usage, UserMsg,
)


class ProviderError(RuntimeError):
    """Network/HTTP/protocol failure with an actionable message."""


def _post_json(url: str, headers: dict, body: dict,
               timeout_s: float) -> dict:
    import socket
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        snippet = ""
        try:
            snippet = exc.read().decode("utf-8", "replace")[:400]
        except Exception:                                   # noqa: BLE001
            pass
        raise ProviderError(
            f"provider HTTP {exc.code} from {url}: {snippet}") from None
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        raise ProviderError(
            f"cannot reach {url} ({exc}) — for a local model, is the "
            f"server running (e.g. `ollama serve`)?") from None


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------
_ANTHROPIC_STOP = {"end_turn": StopReason.END,
                   "tool_use": StopReason.TOOL_USE,
                   "max_tokens": StopReason.LENGTH,
                   "refusal": StopReason.REFUSED}


def anthropic_build_body(model: str, system: str, transcript: list,
                         tools: list[dict], max_tokens: int) -> dict:
    messages = []
    for entry in transcript:
        if isinstance(entry, UserMsg):
            messages.append({"role": "user", "content": entry.text})
        elif isinstance(entry, SystemNote):
            messages.append({"role": "user",
                             "content": f"[system-note] {entry.text}"})
        elif isinstance(entry, AssistantMsg):
            blocks = []
            if entry.turn.text:
                blocks.append({"type": "text", "text": entry.turn.text})
            for tc in entry.turn.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id,
                               "name": tc.name, "input": tc.args})
            messages.append({"role": "assistant",
                             "content": blocks or
                             [{"type": "text", "text": "(no content)"}]})
        elif isinstance(entry, ToolResultsMsg):
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": o.call_id,
                 "content": o.content, "is_error": bool(o.is_error)}
                for o in entry.outcomes]})
    return {"model": model, "max_tokens": int(max_tokens),
            "system": system, "messages": messages,
            "tools": tools}


def anthropic_parse_response(payload: dict) -> AssistantTurn:
    text_parts, calls = [], []
    for i, block in enumerate(payload.get("content") or []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(ToolCall(id=block.get("id") or f"tc-{i}",
                                  name=block.get("name", ""),
                                  args=dict(block.get("input") or {})))
    usage = payload.get("usage") or {}
    return AssistantTurn(
        text="".join(text_parts),
        tool_calls=tuple(calls),
        stop_reason=_ANTHROPIC_STOP.get(payload.get("stop_reason"),
                                        StopReason.END),
        usage=Usage(input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0))))


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, config):
        self._cfg = config

    def send(self, *, system: str, transcript: list, tools: list[dict],
             max_tokens: int) -> AssistantTurn:
        cfg = self._cfg
        base = cfg.base_url or "https://api.anthropic.com"
        body = anthropic_build_body(cfg.model, system, transcript,
                                    tools, max_tokens)
        payload = _post_json(
            base.rstrip("/") + "/v1/messages",
            {"x-api-key": cfg.api_key,
             "anthropic-version": "2023-06-01",
             **cfg.extra_headers},
            body, cfg.timeout_s)
        return anthropic_parse_response(payload)


# ---------------------------------------------------------------------------
# OpenAI-compatible chat/completions (incl. local servers)
# ---------------------------------------------------------------------------
_OPENAI_STOP = {"stop": StopReason.END,
                "tool_calls": StopReason.TOOL_USE,
                "length": StopReason.LENGTH}


def openai_build_body(model: str, system: str, transcript: list,
                      tools: list[dict], max_tokens: int) -> dict:
    messages = [{"role": "system", "content": system}]
    for entry in transcript:
        if isinstance(entry, UserMsg):
            messages.append({"role": "user", "content": entry.text})
        elif isinstance(entry, SystemNote):
            messages.append({"role": "user",
                             "content": f"[system-note] {entry.text}"})
        elif isinstance(entry, AssistantMsg):
            msg = {"role": "assistant",
                   "content": entry.turn.text or None}
            if entry.turn.tool_calls:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.args)}}
                    for tc in entry.turn.tool_calls]
            messages.append(msg)
        elif isinstance(entry, ToolResultsMsg):
            for o in entry.outcomes:
                messages.append({"role": "tool",
                                 "tool_call_id": o.call_id,
                                 "content": o.content})
    return {"model": model, "messages": messages,
            "tools": [{"type": "function",
                       "function": {"name": t["name"],
                                    "description": t["description"],
                                    "parameters": t["input_schema"]}}
                      for t in tools],
            "tool_choice": "auto",
            "max_tokens": int(max_tokens)}


def openai_parse_response(payload: dict) -> AssistantTurn:
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(args, dict):
                args = {"__parse_error__": str(raw)}
        except (TypeError, ValueError):
            # Small local models emit malformed JSON — surface it to
            # the model as an error outcome instead of crashing.
            args = {"__parse_error__": str(raw)}
        calls.append(ToolCall(id=tc.get("id") or f"tc-{i}",
                              name=fn.get("name", ""), args=args))
    usage = payload.get("usage") or {}
    return AssistantTurn(
        text=msg.get("content") or "",
        tool_calls=tuple(calls),
        stop_reason=_OPENAI_STOP.get(choice.get("finish_reason"),
                                     StopReason.END),
        usage=Usage(input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0))))


class OpenAIProvider:
    name = "openai"

    def __init__(self, config):
        self._cfg = config

    def send(self, *, system: str, transcript: list, tools: list[dict],
             max_tokens: int) -> AssistantTurn:
        cfg = self._cfg
        headers = dict(cfg.extra_headers)
        if cfg.api_key:                      # local servers need no key
            headers["authorization"] = f"Bearer {cfg.api_key}"
        body = openai_build_body(cfg.model, system, transcript,
                                 tools, max_tokens)
        payload = _post_json(cfg.base_url.rstrip("/") +
                             "/chat/completions"
                             if cfg.base_url.rstrip("/").endswith("/v1")
                             else cfg.base_url.rstrip("/") +
                             "/v1/chat/completions",
                             headers, body, cfg.timeout_s)
        return openai_parse_response(payload)


def make_provider(config):
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    if config.provider == "openai":
        return OpenAIProvider(config)
    raise ProviderError(f"unknown provider {config.provider!r}")
