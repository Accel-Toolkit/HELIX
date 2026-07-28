"""LLM-free deterministic replay of a session ledger.

Re-executes the ledger's approved ``compute``/``mutate`` tool calls in
order against a fresh :class:`~linac_gen.assist.tools.WorkContext` —
no provider is ever constructed, no network is touched.  ``read``
calls, denied/refused calls and the runtime ``job_*`` tools are
skipped; a job's underlying tool runs synchronously (determinism over
concurrency).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ReplayReport:
    executed: list = field(default_factory=list)   # (tool, status)
    skipped: list = field(default_factory=list)    # (tool, reason)
    failed: list = field(default_factory=list)     # (tool, message)

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        return (f"replay: {len(self.executed)} executed, "
                f"{len(self.skipped)} skipped, "
                f"{len(self.failed)} failed")


def replay(ledger_path: str, *, calc_dir: str = ".") -> ReplayReport:
    from linac_gen.assist.tools import TOOLS, WorkContext
    ctx = WorkContext(calc_dir=calc_dir)
    report = ReplayReport()
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") != "tool":
                continue
            name = rec.get("tool", "")
            tier = rec.get("tier", "")
            status = rec.get("status", "")
            if name.startswith("job_"):
                report.skipped.append((name, "runtime job tool"))
                continue
            if tier == "read":
                report.skipped.append((name, "read tier"))
                continue
            if status in ("denied", "refused", "cancelled", "error"):
                report.skipped.append((name, f"original status={status}"))
                continue
            tool = TOOLS.get(name)
            if tool is None:
                report.failed.append((name, "tool no longer exists"))
                continue
            try:
                env = tool.fn(ctx, **(rec.get("params") or {}))
            except Exception as exc:                    # noqa: BLE001
                report.failed.append((name,
                                      f"{type(exc).__name__}: {exc}"))
                continue
            st = env.get("status")
            if st == "ok":
                report.executed.append((name, st))
            else:
                report.failed.append(
                    (name, str(env.get("data", {}).get("message", st))))
    return report
