"""Background execution of long compute tools.

Long-running tools (multiparticle runs, matches) must not block the
agent loop: they are submitted here, the model immediately receives a
``job_id``, and completion is injected into the conversation as a
system note at the next opportunity.  Cancellation wires the solvers'
own cooperative ``should_abort`` hook (``core/simulation.py``,
``tracking/envelope.py``) — the run stops at the next element boundary.

Threads, not processes: the solvers release the GIL inside numpy /
C++ kernels and already support cooperative cancel.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    tool: str
    params: dict
    state: str = "queued"        # queued|running|done|error|cancelled
    progress: tuple = (0.0, 0)   # (s_mm, n_steps_reported)
    submitted: float = field(default_factory=time.time)
    ended: float = 0.0
    result: dict | None = None
    abort: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict:
        return {"job_id": self.id, "tool": self.tool,
                "state": self.state,
                "progress_s_mm": float(self.progress[0]),
                "elapsed_s": round((self.ended or time.time())
                                   - self.submitted, 1),
                "result": (self.result if self.state in
                           ("done", "error", "cancelled") else None)}


class JobManager:
    def __init__(self, max_concurrent: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_concurrent),
                                        thread_name_prefix="assist-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._unreported: list[str] = []
        self._n = 0

    # -- lifecycle -------------------------------------------------------
    def submit(self, tool_name: str, params: dict, runner) -> str:
        """``runner(progress_cb, should_abort) -> ToolResult dict``."""
        with self._lock:
            self._n += 1
            job = Job(id=f"job-{self._n:04d}", tool=tool_name,
                      params=dict(params))
            self._jobs[job.id] = job

        def _progress(s_mm, i, n):
            job.progress = (float(s_mm), int(n))

        def _run():
            job.state = "running"
            try:
                res = runner(_progress, job.abort.is_set)
                job.result = res
                job.state = ("cancelled" if job.abort.is_set()
                             else ("error" if res.get("status") == "error"
                                   else "done"))
            except Exception as exc:            # noqa: BLE001
                job.result = {"status": "error",
                              "data": {"message":
                                       f"{type(exc).__name__}: {exc}"},
                              "provenance": {}, "warnings": []}
                job.state = "error"
            finally:
                job.ended = time.time()
                with self._lock:
                    self._unreported.append(job.id)

        self._pool.submit(_run)
        return job.id

    # -- queries ---------------------------------------------------------
    def status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"status": "error",
                    "data": {"message": f"unknown job {job_id!r}"},
                    "provenance": {}, "warnings": []}
        return {"status": "ok", "data": job.snapshot(),
                "provenance": {}, "warnings": []}

    def list_jobs(self) -> dict:
        return {"status": "ok",
                "data": {"jobs": [j.snapshot() for j in
                                  self._jobs.values()]},
                "provenance": {}, "warnings": []}

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"status": "error",
                    "data": {"message": f"unknown job {job_id!r}"},
                    "provenance": {}, "warnings": []}
        job.abort.set()
        return {"status": "ok",
                "data": {"job_id": job_id, "cancel_requested": True},
                "provenance": {}, "warnings": []}

    def drain_completions(self) -> list[Job]:
        """Jobs finished since the last drain (agent-thread only)."""
        with self._lock:
            ids, self._unreported = self._unreported, []
        return [self._jobs[i] for i in ids]

    def shutdown(self) -> None:
        for job in self._jobs.values():
            job.abort.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
