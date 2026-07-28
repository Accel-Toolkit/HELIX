"""Session ledger — the deterministic audit trail of every tool call.

One JSONL file per session under ``<calc_dir>/assist_sessions/``.  The
words exchanged with the model are advisory; the LEDGER is the record:
every tool call with its exact parameters, tier, who approved it, its
status and duration.  ``replay.py`` re-executes a ledger without any
LLM.  API keys are excluded structurally (config is serialized through
``AssistConfig.redacted()``) AND scrubbed defensively.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from pathlib import Path

_SECRET_KEYS = {"api_key", "authorization", "x-api-key", "apikey"}


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()
                if str(k).lower() not in _SECRET_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


class Ledger:
    def __init__(self, calc_dir: str, session_id: str | None = None):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"{stamp}_{os.getpid()}"
        d = Path(calc_dir) / "assist_sessions"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{self.session_id}.jsonl"
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("ts",
                          datetime.datetime.now().astimezone().isoformat())
        line = json.dumps(_scrub(record), default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    # -- convenience writers --------------------------------------------
    def session_start(self, config) -> None:
        self.append({"event": "session_start",
                     "session_id": self.session_id,
                     "config": config.redacted()})

    def user(self, turn: int, text: str) -> None:
        self.append({"event": "user", "turn": turn, "text": text})

    def assistant(self, turn: int, text: str, usage) -> None:
        self.append({"event": "assistant", "turn": turn, "text": text,
                     "usage": {"input_tokens": usage.input_tokens,
                               "output_tokens": usage.output_tokens}})

    def tool(self, turn: int, tool: str, params: dict, tier: str,
             approved_by: str, status: str, duration_s: float,
             provenance: dict | None = None,
             job_id: str | None = None) -> None:
        self.append({"event": "tool", "turn": turn, "tool": tool,
                     "params": params, "tier": tier,
                     "approved_by": approved_by, "status": status,
                     "duration_s": round(float(duration_s), 3),
                     "provenance": provenance or {},
                     "job_id": job_id})

    def session_end(self, usage) -> None:
        self.append({"event": "session_end",
                     "usage_total": {"input_tokens": usage.input_tokens,
                                     "output_tokens":
                                     usage.output_tokens}})


def stamp_assist_provenance(h5_path: str, *, session_id: str, turn: int,
                            tool: str, approved_by: str,
                            ledger_path: str) -> None:
    """Post-write stamp: mark an assistant-initiated results file so it
    is traceable to the exact ledger entry.  Never raises — a failed
    stamp must not fail the run that produced the file."""
    try:
        import h5py
        with h5py.File(h5_path, "r+") as f:
            prov = f.require_group("provenance")
            prov.attrs["assist_session"] = str(session_id)
            prov.attrs["assist_turn"] = int(turn)
            prov.attrs["assist_tool"] = str(tool)
            prov.attrs["assist_approved_by"] = str(approved_by)
            prov.attrs["assist_ledger"] = str(ledger_path)
    except Exception:                                       # noqa: BLE001
        pass
