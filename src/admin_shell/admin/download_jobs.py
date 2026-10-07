"""In-memory download job ledger: queueing, progress updates, history limits.

This is one implementation of the JobLedger port in model_catalog.py. Services
with their own persistent job store (SQLite, cancel support) implement the port
with an adapter instead. The runner (HTTP fetching) is a port too; the ledger owns
the job dict shape, the lock, eviction, and the concurrency limit.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .table_spec import ACTIVE_JOB_STATUSES, MODEL_STATUSES

ProgressReport = Callable[[dict[str, Any]], None]
Spawn = Callable[[Callable[[], None]], None]

# Keys a runner may update while a job is running. Status transitions are the
# ledger's own responsibility, so they are deliberately absent here.
PROGRESS_KEYS = frozenset(
    {
        "total_bytes",
        "downloaded_bytes",
        "speed_mbps",
        "progress_percent",
        "elapsed_seconds",
        "eta_seconds",
    }
)
TERMINAL_STATUSES = frozenset({"completed", "failed"})


class DownloadFailed(Exception):
    """Raised by a runner with a user-facing message (hints included)."""


class TooManyDownloads(Exception):
    pass


class JobNotFound(Exception):
    pass


class DownloadRunner(Protocol):
    def run(self, repo_id: str, target_dir: Path, report: ProgressReport) -> None:
        """Fetch repo_id into target_dir. Raise DownloadFailed with a message on error."""
        ...


def _thread_spawn(target: Callable[[], None]) -> None:
    threading.Thread(target=target, daemon=True).start()


class DownloadJobRegistry:
    def __init__(
        self,
        runner: DownloadRunner,
        *,
        max_concurrent: int = 3,
        max_history: int = 100,
        ttl_seconds: float = 86400.0,
        clock: Callable[[], float] = time.time,
        spawn: Spawn = _thread_spawn,
    ) -> None:
        self._runner = runner
        self._max_concurrent = max_concurrent
        self._max_history = max_history
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._spawn = spawn
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ----- queries -----

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values()]
        jobs.sort(key=lambda job: job["created_at"], reverse=True)
        return jobs

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(f"download job not found: {job_id}")
            return dict(job)

    def active_names(self) -> set[str]:
        with self._lock:
            return {
                job["local_name"]
                for job in self._jobs.values()
                if job["status"] in {"queued", "downloading"}
            }

    # ----- commands -----

    def start(
        self,
        repo_id: str,
        local_name: str,
        target_dir: Path,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = self._clock()
        job_id = str(uuid.uuid4())
        job: dict[str, Any] = {
            "id": job_id,
            "repo_id": repo_id,
            "local_name": local_name,
            "target_dir": str(target_dir),
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": 0.0,
            "progress_percent": None,
            "total_bytes": None,
            "downloaded_bytes": 0,
            "speed_mbps": 0.0,
            "eta_seconds": None,
            "error": None,
            "extra": dict(extra or {}),
        }
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j["status"] in ACTIVE_JOB_STATUSES)
            if active >= self._max_concurrent:
                raise TooManyDownloads(
                    f"too many concurrent downloads (max {self._max_concurrent}); "
                    "wait for a job to finish"
                )
            self._evict_locked(now)
            self._jobs[job_id] = job
        self._spawn(lambda: self._execute(job_id, repo_id, target_dir))
        return dict(job)

    # ----- internals -----

    def _update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(updates)
            job["updated_at"] = self._clock()

    def _report_for(self, job_id: str) -> ProgressReport:
        def report(updates: dict[str, Any]) -> None:
            unknown = set(updates) - PROGRESS_KEYS
            if unknown:
                raise ValueError(f"runner reported non-progress keys: {sorted(unknown)}")
            self._update(job_id, **updates)

        return report

    def _execute(self, job_id: str, repo_id: str, target_dir: Path) -> None:
        started = self._clock()
        self._update(job_id, status="downloading", started_at=started)
        try:
            self._runner.run(repo_id, target_dir, self._report_for(job_id))
        except DownloadFailed as exc:
            self._finish(job_id, started, status="failed", error=str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a runner bug must not leave a job running
            self._finish(job_id, started, status="failed", error=f"{type(exc).__name__}: {exc}")
            return
        self._finish(job_id, started, status="completed", error=None)

    def _finish(self, job_id: str, started: float, *, status: str, error: str | None) -> None:
        assert status in MODEL_STATUSES and status in TERMINAL_STATUSES
        finished = self._clock()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            progress = (
                100.0
                if status == "completed" and job.get("total_bytes")
                else job.get("progress_percent")
            )
            job.update(
                {
                    "status": status,
                    "error": error,
                    "finished_at": finished,
                    "updated_at": finished,
                    "elapsed_seconds": round(finished - started, 2),
                    "speed_mbps": 0.0,
                    "progress_percent": progress,
                    "eta_seconds": 0.0
                    if status == "completed" and job.get("total_bytes")
                    else None,
                }
            )

    def _evict_locked(self, now: float) -> None:
        cutoff = now - self._ttl_seconds
        for job_id in [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in TERMINAL_STATUSES and (job["finished_at"] or 0) < cutoff
        ]:
            del self._jobs[job_id]
        # Leave room for the job about to be inserted so history never exceeds the limit.
        excess = len(self._jobs) - (self._max_history - 1)
        if excess > 0:
            oldest = sorted(self._jobs, key=lambda job_id: self._jobs[job_id]["created_at"])
            for job_id in oldest[:excess]:
                del self._jobs[job_id]
