"""Server-Sent Events stream of AdminEventHub snapshots (event: admin_update)."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..admin.event_hub import AdminEventHub, format_sse
from ._auth import AuthDependency, router_dependencies


def create_events_router(
    hub: AdminEventHub, *, auth: AuthDependency, prefix: str = "/admin"
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["admin"], dependencies=router_dependencies(auth))

    @router.get("/events")
    async def admin_events(request: Request) -> StreamingResponse:
        queue = await hub.subscribe()

        async def generate():
            yield "retry: 3000\n\n"
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        snapshot = await asyncio.wait_for(queue.get(), timeout=20.0)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield format_sse(
                        json.dumps(snapshot, ensure_ascii=False, default=str), event="admin_update"
                    )
            finally:
                await hub.unsubscribe(queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    return router
