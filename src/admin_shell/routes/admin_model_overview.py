"""HTTP adapter for ModelOverviewService: GET /admin/models/overview.

Actions in the snapshot point at the service's existing routes (download jobs,
load/unload, cancel); this router only serves the merged read model.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..admin.model_overview import ModelOverviewService
from ._auth import AuthDependency, router_dependencies


def create_model_overview_router(
    overview: ModelOverviewService, *, auth: AuthDependency, prefix: str = "/admin"
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["admin"], dependencies=router_dependencies(auth))

    @router.get("/models/overview")
    async def model_overview() -> dict[str, Any]:
        return await overview.snapshot()

    return router
