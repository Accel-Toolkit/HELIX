"""Assistant configuration — resolution order and storage.

Resolution order (first hit wins):
1. Explicit ``AssistConfig`` passed by the caller (GUI settings panel).
2. Environment: ``HELIX_ASSIST_PROVIDER`` / ``HELIX_ASSIST_MODEL`` /
   ``HELIX_ASSIST_BASE_URL`` / ``HELIX_ASSIST_API_KEY``, with
   ``ANTHROPIC_API_KEY`` accepted as the key for the anthropic provider.
3. Nothing configured → ``resolve_config()`` returns ``None`` and every
   surface degrades to setup guidance (the app never breaks).

API keys are NEVER written to disk by this module and NEVER appear in
the session ledger; the GUI stores its settings via QSettings in its
own layer and passes an explicit config here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "llama3.1",          # sensible local-server default
    "claude_sdk": "",              # empty -> the SDK/login default model
}


@dataclass
class AssistConfig:
    provider: str                   # "anthropic" | "openai" (compatible)
    model: str
    api_key: str = ""               # may legitimately be empty (local)
    base_url: str = ""              # required for provider="openai"
    max_tool_iterations: int = 12   # per user turn
    max_concurrent_jobs: int = 2
    auto_approve_compute: bool = False
    web_search_enabled: bool = False
    timeout_s: float = 120.0
    #: unanswered confirmation prompts auto-DENY after this many seconds
    #: (0 = wait forever).  Keeps a forgotten prompt from blocking the
    #: worker thread indefinitely.
    confirm_timeout_s: float = 120.0
    extra_headers: dict = field(default_factory=dict)

    def redacted(self) -> dict:
        """Loggable form — the key never leaves the process."""
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("api_key", "extra_headers")}
        d["api_key_set"] = bool(self.api_key)
        return d


def resolve_config(explicit: AssistConfig | None = None
                   ) -> AssistConfig | None:
    if explicit is not None:
        return explicit
    provider = os.environ.get("HELIX_ASSIST_PROVIDER", "").strip().lower()
    base_url = os.environ.get("HELIX_ASSIST_BASE_URL", "").strip()
    key = (os.environ.get("HELIX_ASSIST_API_KEY", "").strip()
           or os.environ.get("ANTHROPIC_API_KEY", "").strip())
    model = os.environ.get("HELIX_ASSIST_MODEL", "").strip()

    if not provider:
        # Infer: a base_url implies an OpenAI-compatible (local) server;
        # a bare Anthropic key implies anthropic; neither -> unconfigured.
        # (claude_sdk is never inferred — it must be chosen explicitly,
        #  since it needs no key/url and would otherwise mask "unconfigured".)
        if base_url:
            provider = "openai"
        elif key:
            provider = "anthropic"
        else:
            return None
    if provider not in ("anthropic", "openai", "claude_sdk"):
        return None
    if provider == "anthropic" and not key:
        return None
    if provider == "openai" and not base_url:
        return None
    # claude_sdk needs neither a key nor a base_url — auth is the Claude
    # Code login, validated at connect time by sdk_backend.sdk_available().
    return AssistConfig(
        provider=provider,
        model=model or _DEFAULT_MODELS[provider],
        api_key=key,
        base_url=base_url,
    )
