"""Recent activity listing with server-declared columns."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..admin.interactions import InteractionsFeed
from ._auth import AuthDependency, router_dependencies


def create_interactions_router(
    feed: InteractionsFeed, *, auth: AuthDependency, prefix: str = "/admin"
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["admin"], dependencies=router_dependencies(auth))

    @router.get("/interactions")
    async def list_interactions(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        return await feed.recent(limit)

    return router
