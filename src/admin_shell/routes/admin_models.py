"""HTTP adapter for ModelCatalogService. Routes exist only for declared capabilities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..admin.download_jobs import JobNotFound, TooManyDownloads
from ..admin.model_catalog import (
    CapabilityUnsupported,
    CatalogError,
    InvalidDownloadField,
    InvalidLocalName,
    InvalidRepoId,
    JobNotCancellable,
    ModelCatalogService,
    ModelNotLoadable,
    TargetExists,
)
from ._auth import AuthDependency, router_dependencies

_ERROR_STATUS: dict[type[Exception], int] = {
    InvalidLocalName: status.HTTP_400_BAD_REQUEST,
    InvalidRepoId: status.HTTP_400_BAD_REQUEST,
    InvalidDownloadField: status.HTTP_400_BAD_REQUEST,
    TargetExists: status.HTTP_409_CONFLICT,
    TooManyDownloads: status.HTTP_429_TOO_MANY_REQUESTS,
    CapabilityUnsupported: status.HTTP_404_NOT_FOUND,
    JobNotFound: status.HTTP_404_NOT_FOUND,
    ModelNotLoadable: status.HTTP_409_CONFLICT,
    JobNotCancellable: status.HTTP_409_CONFLICT,
}


@contextmanager
def translate_errors() -> Iterator[None]:
    try:
        yield
    except (CatalogError, TooManyDownloads, JobNotFound) as error:
        raise HTTPException(_ERROR_STATUS.get(type(error), 500), str(error)) from error


class DownloadBody(BaseModel):
    repo_id: str
    local_name: str | None = None
    force: bool = False
    extra: dict[str, Any] = {}


def create_models_router(
    catalog: ModelCatalogService,
    *,
    auth: AuthDependency,
    prefix: str = "/admin",
    exclude: frozenset[str] = frozenset(),
) -> APIRouter:
    """Routes follow the catalog capabilities. `exclude` names routes the service
    already serves on the same path with its own (compatible) body, e.g.
    {"download", "downloads"} when a legacy POST /admin/models/download must stay."""
    router = APIRouter(prefix=prefix, tags=["admin"], dependencies=router_dependencies(auth))
    capabilities = catalog.capabilities

    @router.get("/models")
    def list_models() -> dict[str, Any]:
        return catalog.listing().to_dict()

    if capabilities.download and "download" not in exclude:

        @router.post("/models/download")
        def start_download(body: DownloadBody) -> dict[str, Any]:
            with translate_errors():
                return catalog.start_download(body.repo_id, body.local_name, body.force, body.extra)

    if capabilities.download and "downloads" not in exclude:

        @router.get("/models/downloads")
        def list_downloads() -> dict[str, Any]:
            return {"items": catalog.list_downloads()}

        @router.get("/models/downloads/{job_id}")
        def get_download(job_id: str) -> dict[str, Any]:
            with translate_errors():
                return catalog.get_download(job_id)

    if capabilities.cancel:

        @router.post("/models/downloads/{job_id}/cancel")
        def cancel_download(job_id: str) -> dict[str, Any]:
            with translate_errors():
                return catalog.cancel_download(job_id)

    if capabilities.load:

        @router.post("/models/{local_name:path}/load")
        async def load_model(local_name: str) -> dict[str, str]:
            with translate_errors():
                await catalog.load(local_name)
            return {"status": "loaded", "local_name": local_name}

    if capabilities.unload:

        @router.post("/models/{local_name:path}/unload")
        async def unload_model(local_name: str) -> dict[str, str]:
            with translate_errors():
                await catalog.unload(local_name)
            return {"status": "unloaded", "local_name": local_name}

    return router
