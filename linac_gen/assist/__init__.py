"""HELIX AI Assistant — OPTIONAL natural-language agent over HELIX.

STRICT OPTIONALITY CONTRACT
---------------------------
* This subpackage is imported by NOTHING in the core package or the
  GUI at startup: the ``assist`` CLI subcommand and the GUI Tools-menu
  entry lazy-import it on use.  Deleting this directory must leave the
  rest of HELIX fully functional (guarded by a test).
* Importing it requires only the standard library plus HELIX's existing
  hard dependencies — the LLM client is stdlib ``urllib``; there is no
  SDK dependency and no new entry in ``pyproject`` ``dependencies``.
* It makes NO network connection until the user explicitly sends a
  message to a configured provider.  With no provider configured the
  feature reports itself unavailable and every surface shows setup
  guidance instead of an error.

ARCHITECTURE (see docs/manual "AI Assistant" chapter)
-----------------------------------------------------
tools.py      declarative registry of read/compute/mutate tools, each a
              thin wrapper over an EXISTING audited HELIX API — the
              agent orchestrates, it never computes, and no tool can
              reach a shell or ``eval``.
providers.py  Anthropic Messages API + any OpenAI-compatible endpoint
              (ollama / llama.cpp / vLLM — fully offline, no API key),
              plus the network-free MockProvider used by the tests.
agent.py      the tool-use loop with tier gating: ``read`` auto-runs,
              ``compute`` requires confirmation, ``mutate`` ALWAYS
              requires confirmation, and every confirmation echoes the
              exact resolved call with units.
jobs.py       background execution of long compute tools with
              progress + cancel (wired to should_abort).
ledger.py     JSONL session ledger of every tool call (never secrets)
              + LLM-free deterministic replay.
"""
from __future__ import annotations


def available() -> tuple[bool, str]:
    """Cheap probe: is any provider configured?  Never touches the
    network — it only inspects configuration (env vars / stored
    settings).  Returns ``(ok, human_reason)``."""
    from linac_gen.assist.config import resolve_config
    cfg = resolve_config()
    if cfg is None:
        return (False,
                "No assistant provider configured.  Set "
                "HELIX_ASSIST_API_KEY (or ANTHROPIC_API_KEY) for a "
                "cloud model, or HELIX_ASSIST_BASE_URL to a local "
                "OpenAI-compatible server (e.g. ollama at "
                "http://localhost:11434/v1) — see the manual's "
                "'AI Assistant' chapter.")
    return True, f"provider: {cfg.provider} ({cfg.model})"
