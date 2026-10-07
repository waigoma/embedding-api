"""Protected UI delivery for static/admin/. One canonical source, no public mount.

create_admin_ui_router(static_dir, auth=require_basic_auth) serves admin.html at
{prefix}/ui and every shipped .html/.css/.js below static_dir at
{prefix}/assets/{path}. Subdirectories (core/, screens/) are allowed; traversal,
unknown extensions and missing files are 404.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from ._auth import AuthDependency, router_dependencies

MEDIA_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; font-src 'self'; img-src 'self'; media-src blob:; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
}


def resolve_asset(static_dir: Path, requested: str) -> Path | None:
    """Return the file for a relative asset path, or None when it must be 404."""
    posix = PurePosixPath(requested)
    if posix.is_absolute() or any(
        part in {".", ".."} or part.startswith(".") for part in posix.parts
    ):
        return None
    if posix.suffix not in MEDIA_TYPES:
        return None
    root = static_dir.resolve()
    candidate = (root / Path(*posix.parts)).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def create_admin_ui_router(
    static_dir: Path, *, auth: AuthDependency, prefix: str = "/admin"
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["admin"], dependencies=router_dependencies(auth))
    index = static_dir / "admin.html"

    @router.get("/ui", response_class=FileResponse)
    async def admin_ui() -> FileResponse:
        if not index.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "admin.html not found")
        return FileResponse(index, media_type="text/html", headers=SECURITY_HEADERS)

    @router.get("/assets/{path:path}", response_class=FileResponse)
    async def admin_asset(path: str) -> FileResponse:
        file = resolve_asset(static_dir, path)
        if file is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Asset not found")
        return FileResponse(file, media_type=MEDIA_TYPES[file.suffix], headers=SECURITY_HEADERS)

    return router
