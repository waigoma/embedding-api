"""SSE snapshot broadcaster. Providers are injected; the hub knows no routes.

Each provider is an async callable returning JSON-serializable data. The hub
polls them, fingerprints the combined snapshot, and pushes only changes to
subscribers. A failing provider logs and retries next tick.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

SnapshotProvider = Callable[[], Awaitable[Any]]


class AdminEventHub:
    def __init__(
        self, providers: dict[str, SnapshotProvider], *, poll_interval: float = 0.5
    ) -> None:
        if not providers:
            raise ValueError("AdminEventHub needs at least one provider")
        self._providers = dict(providers)
        self.poll_interval = poll_interval
        self._subscribers: set[asyncio.Queue] = set()
        self._current: dict[str, Any] | None = None
        self._fingerprint: str | None = None
        self._task: asyncio.Task | None = None

    @property
    def keys(self) -> list[str]:
        return list(self._providers)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="admin-event-hub")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        if self._current is not None:
            self._put_latest(queue, self._current)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def build_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key, provider in self._providers.items():
            snapshot[key] = await provider()
        return snapshot

    async def tick(self) -> bool:
        """Poll once. Returns True when a changed snapshot was broadcast."""
        snapshot = await self.build_snapshot()
        fingerprint = _fingerprint(snapshot)
        if fingerprint == self._fingerprint:
            return False
        self._current, self._fingerprint = snapshot, fingerprint
        for queue in list(self._subscribers):
            self._put_latest(queue, snapshot)
        return True

    @staticmethod
    def _put_latest(queue: asyncio.Queue, snapshot: dict[str, Any]) -> None:
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(snapshot)

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad poll must not stop the stream
                log.exception("admin_event_hub_poll_error")
            await asyncio.sleep(self.poll_interval)


def _fingerprint(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha1(encoded.encode()).hexdigest()


def format_sse(data: str, event: str | None = None) -> str:
    parts = [f"event: {event}"] if event else []
    parts.append(f"data: {data}")
    return "\n".join(parts) + "\n\n"
