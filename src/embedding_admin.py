"""embedding-api adapters and composition for the shared admin shell.

Ports implemented here: a MODEL_DIR scanner (HF / PEFT roots with a guessed
model type), load state over the in-process registry, and an inference-log
reader. server.py injects callables instead of its module globals so tests can
replace the registry, loaders and logs without rebuilding the routers.

The admin routes are mounted WITHOUT authentication on purpose: this service
already exposes /v1/* unauthenticated on a trusted network, and the admin UI
keeps that same trust boundary.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from admin_shell.admin.event_hub import AdminEventHub
from admin_shell.admin.interactions import InteractionsFeed
from admin_shell.admin.model_catalog import (
    JobLedger,
    LocalModel,
    ModelCatalogCapabilities,
    ModelCatalogService,
    ModelNotLoadable,
)
from admin_shell.admin.table_spec import ColumnSpec
from admin_shell.routes.admin_events import create_events_router
from admin_shell.routes.admin_interactions import create_interactions_router
from admin_shell.routes.admin_models import create_models_router, translate_errors
from admin_shell.routes.admin_ui import create_admin_ui_router

STATIC_DIR = Path(__file__).resolve().parent / "static" / "admin"
CAPABILITIES = ModelCatalogCapabilities(download=True, load=True, unload=True, cancel=False)
MODEL_COLUMNS = [ColumnSpec("type", "type", "text")]
INTERACTION_COLUMNS = [
    ColumnSpec("timestamp", "Time", "time"),
    ColumnSpec("event", "Event", "mono"),
    ColumnSpec("model", "Model", "mono"),
    ColumnSpec("status", "Status", "status"),
    ColumnSpec("duration_ms", "Duration", "duration"),
]
SKIP_WALK_SUBDIRS = frozenset({".cache", "__pycache__", ".git"})
MODEL_CONFIG_FILES = frozenset({"config.json", "adapter_config.json"})
# Architecture suffixes that mean "cross-encoder / reranker"; everything else embeds.
RERANKER_ARCHITECTURE_SUFFIXES = ("ForSequenceClassification",)
RERANKER_NAME_HINTS = ("rerank",)
# A model whose latest download job is in one of these states may have partial files.
LOAD_BLOCKING_JOB_STATUSES = {
    "queued": "download in progress; wait for it to finish before loading",
    "downloading": "download in progress; wait for it to finish before loading",
    "failed": "last download failed and files may be partial; retry the download first",
}

ModelLoader = Callable[[str, str], Any]  # (model_id, model_type) -> registers the model
ModelUnloader = Callable[[str], bool]
LogSource = Callable[[int], list[dict[str, Any]]]
HealthProvider = Callable[[], Awaitable[dict[str, Any]]]


def find_model_roots(root: Path) -> list[Path]:
    """Model roots: directories with config.json / adapter_config.json, outermost wins.

    Cache / VCS / hidden directories are skipped, so a nested config inside a
    model (or a huggingface .cache folder) never shows up as its own model.
    """
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_WALK_SUBDIRS and not d.startswith(".")]
        if Path(current) == root:
            continue
        if MODEL_CONFIG_FILES & set(files):
            candidates.append(Path(current))
    candidates.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    roots: list[Path] = []
    for candidate in candidates:
        if any(candidate.is_relative_to(parent) for parent in roots):
            continue
        roots.append(candidate)
    return sorted(roots)


def guess_model_type(directory: Path) -> str:
    """Decide embedding vs reranker from config.json architectures, then the directory name."""
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        architectures = config.get("architectures") or []
    except (OSError, ValueError, AttributeError):
        architectures = []
    if any(str(arch).endswith(RERANKER_ARCHITECTURE_SUFFIXES) for arch in architectures):
        return "reranker"
    if any(hint in directory.name.lower() for hint in RERANKER_NAME_HINTS):
        return "reranker"
    return "embedding"


def _directory_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


class EmbeddingModelScanner:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir

    def scan(self) -> list[LocalModel]:
        root = self._model_dir
        models: list[LocalModel] = []
        for directory in find_model_roots(root):
            models.append(
                LocalModel(
                    local_name=directory.relative_to(root).as_posix(),
                    size_bytes=_directory_size(directory),
                    modified_at=directory.stat().st_mtime,
                    extra={"type": guess_model_type(directory)},
                )
            )
        return models

    def type_of(self, local_name: str) -> str | None:
        """Guessed type of an on-disk model root, or None when local_name is not one."""
        root = self._model_dir
        for directory in find_model_roots(root):
            if directory.relative_to(root).as_posix() == local_name:
                return guess_model_type(directory)
        return None


class RegistryLoadState:
    """LoadStateProvider over server.py's registry, reached through injected callables."""

    def __init__(
        self,
        *,
        loaded_names: Callable[[], set[str]],
        loader: ModelLoader,
        unloader: ModelUnloader,
        scanner: EmbeddingModelScanner,
        jobs: JobLedger,
    ) -> None:
        self._loaded_names = loaded_names
        self._loader = loader
        self._unloader = unloader
        self._scanner = scanner
        self._jobs = jobs

    def loaded_names(self) -> set[str]:
        return self._loaded_names()

    def _blocking_reason(self, local_name: str) -> str | None:
        jobs = [job for job in self._jobs.all() if job.get("local_name") == local_name]
        if not jobs:
            return None
        latest = max(jobs, key=lambda job: job.get("created_at") or 0)
        return LOAD_BLOCKING_JOB_STATUSES.get(latest.get("status"))

    async def load(self, local_name: str) -> None:
        if local_name in self._loaded_names():
            return
        reason = self._blocking_reason(local_name)
        if reason is not None:
            raise ModelNotLoadable(f"'{local_name}': {reason}")
        # Only on-disk model roots: an unknown name would make sentence-transformers
        # fetch it from the Hub instead of failing.
        model_type = await asyncio.to_thread(self._scanner.type_of, local_name)
        if model_type is None:
            raise ModelNotLoadable(f"'{local_name}' is not a model directory under MODEL_DIR")
        try:
            await asyncio.to_thread(self._loader, local_name, model_type)
        except HTTPException as exc:
            raise ModelNotLoadable(str(exc.detail)) from exc
        except Exception as exc:  # noqa: BLE001 - model libraries raise arbitrary errors
            raise ModelNotLoadable(f"{type(exc).__name__}: {exc}") from exc

    async def unload(self, local_name: str) -> None:
        if not await asyncio.to_thread(self._unloader, local_name):
            raise ModelNotLoadable(f"'{local_name}' is not loaded")


class InferenceLogReader:
    """InteractionReader over the in-memory inference log (oldest first in the source)."""

    def __init__(self, source: LogSource) -> None:
        self._source = source

    async def recent(self, limit: int) -> list[dict[str, Any]]:
        rows = list(self._source(limit))
        rows.reverse()
        return [
            {
                "timestamp": row.get("timestamp"),
                "event": row.get("event", "inference"),
                "model": row.get("model"),
                "status": (row.get("details") or {}).get("status") or row.get("level") or "info",
                "duration_ms": row.get("duration_ms"),
            }
            for row in rows
        ]


def build_catalog(
    model_dir: Path,
    *,
    jobs: JobLedger,
    loaded_names: Callable[[], set[str]],
    loader: ModelLoader,
    unloader: ModelUnloader,
) -> ModelCatalogService:
    scanner = EmbeddingModelScanner(model_dir)
    return ModelCatalogService(
        scanner=scanner,
        models_dir=model_dir,
        capabilities=CAPABILITIES,
        columns=MODEL_COLUMNS,
        jobs=jobs,
        load_state=RegistryLoadState(
            loaded_names=loaded_names, loader=loader, unloader=unloader, scanner=scanner, jobs=jobs
        ),
    )


@contextmanager
def v1_download_errors() -> Iterator[None]:
    """Map catalog errors for the legacy /v1/models/download* routes (same codes as /admin)."""
    try:
        with translate_errors():
            yield
    except OSError as exc:
        raise HTTPException(
            500,
            f"failed to prepare model directory: {exc}. "
            "Check if MODEL_DIR is mounted writable (remove ':ro' from volume mount).",
        ) from exc


def mount_admin(
    app: FastAPI,
    *,
    catalog: ModelCatalogService,
    logs: LogSource,
    health: HealthProvider,
    poll_interval: float = 1.0,
) -> AdminEventHub:
    """Mount /admin/* (UI, models, events, interactions, health). Start/stop the hub in lifespan."""
    feed = InteractionsFeed(InferenceLogReader(logs), INTERACTION_COLUMNS)
    hub = AdminEventHub(
        {"models": catalog.snapshot, "interactions": feed.snapshot, "health": health},
        poll_interval=poll_interval,
    )
    app.state.model_catalog = catalog
    app.state.admin_event_hub = hub

    auth = None  # explicit: same trust boundary as the unauthenticated /v1/* API
    app.include_router(create_admin_ui_router(STATIC_DIR, auth=auth))
    app.include_router(create_models_router(catalog, auth=auth))
    app.include_router(create_events_router(hub, auth=auth))
    app.include_router(create_interactions_router(feed, auth=auth))

    extra = APIRouter(prefix="/admin", tags=["admin"])

    @extra.get("/health")
    async def admin_health() -> dict[str, Any]:
        return await health()

    app.include_router(extra)
    return hub
