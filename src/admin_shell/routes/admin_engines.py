"""HTTP adapter for engine status and engine settings forms."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from ..admin.engine_status import (
    EngineNotFound,
    EngineSettingsProvider,
    EngineStatusProvider,
    InvalidSetting,
)
from ._auth import AuthDependency, router_dependencies


def create_engines_router(
    engines: EngineStatusProvider,
    settings: EngineSettingsProvider | None,
    *,
    auth: AuthDependency,
    prefix: str = "/admin",
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["admin"], dependencies=router_dependencies(auth))

    @router.get("/engines")
    async def list_engines() -> dict[str, Any]:
        return {"items": [item.to_dict() for item in await engines.statuses()]}

    @router.post("/engines/{name}/unload")
    async def unload_engine(name: str) -> dict[str, Any]:
        try:
            return await engines.unload(name)
        except EngineNotFound as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    if settings is not None:

        @router.get("/engine-settings")
        async def describe_settings() -> dict[str, Any]:
            return {"engines": [form.to_dict() for form in await settings.describe()]}

        @router.post("/engine-settings/{name}")
        async def apply_settings(name: str, body: dict[str, Any]) -> dict[str, Any]:
            try:
                return await settings.apply(name, body)
            except EngineNotFound as error:
                raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
            except InvalidSetting as error:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    return router
