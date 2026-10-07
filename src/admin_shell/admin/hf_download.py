"""DownloadRunner adapter over huggingface_hub.snapshot_download.

Progress is estimated from the target directory size, like the original
tts-gateway / embedding-api workers. huggingface_hub is imported lazily so the
admin package stays importable without it; a missing package fails the job.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .download_jobs import DownloadFailed, ProgressReport


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def failure_message(exc: BaseException) -> str:
    message = str(exc) or "unknown download error"
    if "401" in message or "403" in message or "gated" in message.lower():
        return f"{message} (set HF_TOKEN for private/gated models)"
    return message


class HfSnapshotDownloader:
    def __init__(
        self,
        token: str | None,
        *,
        poll_interval: float = 1.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._poll_interval = max(0.2, poll_interval)
        self._clock = clock
        self._sleep = sleep

    def estimate_size(self, repo_id: str) -> int | None:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=self._token).model_info(repo_id, files_metadata=True)
        except Exception:  # noqa: BLE001 - estimate is best-effort
            return None
        sizes: list[int] = []
        for sibling in info.siblings or []:
            size = getattr(sibling, "size", None)
            if isinstance(size, int) and size > 0:
                sizes.append(size)
        return sum(sizes) if sizes else None

    def run(self, repo_id: str, target_dir: Path, report: ProgressReport) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # noqa: BLE001
            raise DownloadFailed(f"missing huggingface_hub: {exc}") from exc

        total_bytes = self.estimate_size(repo_id)
        report({"total_bytes": total_bytes})
        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                snapshot_download(repo_id=repo_id, local_dir=str(target_dir), token=self._token)
                result["ok"] = True
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        start_wall = self._clock()
        start_size = directory_size(target_dir)
        last_t, last_size = start_wall, start_size
        thread.start()
        while thread.is_alive():
            self._sleep(self._poll_interval)
            now = self._clock()
            current = directory_size(target_dir)
            downloaded = max(0, current - start_size)
            speed_bps = max(0.0, (current - last_size) / max(1e-6, now - last_t))
            progress = eta = None
            if total_bytes:
                progress = min(100.0, downloaded / total_bytes * 100.0)
                if speed_bps > 1:
                    eta = max(0, total_bytes - downloaded) / speed_bps
            report(
                {
                    "downloaded_bytes": downloaded,
                    "speed_mbps": round(speed_bps / (1024 * 1024), 2),
                    "elapsed_seconds": round(now - start_wall, 2),
                    "progress_percent": round(progress, 2) if progress is not None else None,
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                }
            )
            last_t, last_size = now, current
        thread.join()
        report({"downloaded_bytes": max(0, directory_size(target_dir) - start_size)})
        if "error" in result:
            raise DownloadFailed(failure_message(result["error"])) from result["error"]
