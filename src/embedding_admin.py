"""embedding-api composition root for the shared admin shell.

Ports implemented here: a MODEL_DIR scanner (HF / PEFT roots with a guessed
model type), load state over the in-process registry, and an inference-log
reader. The admin routes are mounted WITHOUT authentication on purpose: this
service already exposes /v1/* unauthenticated on a trusted network.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from admin_shell.admin.download_jobs import DownloadJobRegistry
from admin_shell.admin.event_hub import AdminEventHub
from admin_shell.admin.hf_download import HfSnapshotDownloader
from admin_shell.admin.interactions import InteractionsFeed
from admin_shell.admin.model_catalog import (
    LocalModel,
    ModelCatalogCapabilities,
    ModelCatalogService,
    ModelNotLoadable,
)
from admin_shell.admin.table_spec import ColumnSpec
from admin_shell.routes.admin_events import create_events_router
from admin_shell.routes.admin_interactions import create_interactions_router
from admin_shell.routes.admin_models import create_models_router
from admin_shell.routes.admin_ui import create_admin_ui_router

STATIC_DIR = Path(__file__).resolve().parent / "static" / "admin"
CAPABILITIES = ModelCatalogCapabilities(download=True, load=True, unload=True)
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

ModelLoader = Callable[[str, str], Any]      # (model_id, model_type) -> entry
ModelUnloader = Callable[[str], bool]


def guess_model_type(directory: Path) -> str:
    """Decide embedding vs reranker from config.json architectures, then the directory name."""
    config = directory / "config.json"
    try:
        architectures = json.loads(config.read_text(encoding="utf-8")).get("architectures") or []
    except (OSError, ValueError, AttributeError):
        architectures = []
    if any(str(arch).endswith(RERANKER_ARCHITECTURE_SUFFIXES) for arch in architectures):
        return "reranker"
    if any(hint in directory.name.lower() for hint in RERANKER_NAME_HINTS):
        return "reranker"
    return "embedding"


class EmbeddingModelScanner:
    """Model roots: directories with config.json / adapter_config.json, outermost wins."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._types: dict[str, str] = {}

    def scan(self) -> list[LocalModel]:
        root = self._model_dir
        if not root.is_dir():
            return []
        candidates: list[Path] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_WALK_SUBDIRS and not d.startswith(".")]
            if Path(current) == root:
                continue
            if MODEL_CONFIG_FILES & set(files):
                candidates.append(Path(current))
        candidates.sort(key=lambda p: (len(p.relative_to(root).parts), str(p)))
        roots: list[Path] = []
        for candidate in candidates:
            if any(candidate.is_relative_to(parent) for parent in roots):
                continue
            roots.append(candidate)
        models: list[LocalModel] = []
        types: dict[str, str] = {}
        for directory in sorted(roots):
            local_name = directory.relative_to(root).as_posix()
            model_type = guess_model_type(directory)
            types[local_name] = model_type
            models.append(
                LocalModel(
                    local_name=local_name,
                    size_bytes=sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()),
                    modified_at=directory.stat().st_mtime,
                    extra={"type": model_type},
                )
            )
        self._types = types
        return models

    def type_of(self, local_name: str) -> str:
        if local_name not in self._types:
            self.scan()
        return self._types.get(local_name, "embedding")


class RegistryLoadState:
    def __init__(
        self,
        registry: dict[str, Any],
        lock: threading.Lock,
        loader: ModelLoader,
        unloader: ModelUnloader,
        scanner: EmbeddingModelScanner,
    ) -> None:
        self._registry = registry
        self._lock = lock
        self._loader = loader
        self._unloader = unloader
        self._scanner = scanner

    def loaded_names(self) -> set[str]:
        with self._lock:
            return set(self._registry)

    async def load(self, local_name: str) -> None:
        with self._lock:
            if local_name in self._registry:
                return
        model_type = self._scanner.type_of(local_name)
        try:
            entry = await asyncio.to_thread(self._loader, local_name, model_type)
        except HTTPException as exc:
            raise ModelNotLoadable(str(exc.detail)) from exc
        except Exception as exc:  # noqa: BLE001 - model libraries raise arbitrary errors
            raise ModelNotLoadable(f"{type(exc).__name__}: {exc}") from exc
        with self._lock:
            self._registry[local_name] = entry

    async def unload(self, local_name: str) -> None:
        if not await asyncio.to_thread(self._unloader, local_name):
            raise ModelNotLoadable(f"'{local_name}' is not loaded")


class InferenceLogReader:
    def __init__(self, logs: list[dict[str, Any]], lock: threading.Lock) -> None:
        self._logs = logs
        self._lock = lock

    async def recent(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._logs[-limit:])
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


def build_admin(
    app: FastAPI,
    *,
    model_dir: str,
    registry: dict[str, Any],
    registry_lock: threading.Lock,
    loader: ModelLoader,
    unloader: ModelUnloader,
    inference_logs: list[dict[str, Any]],
    inference_lock: threading.Lock,
    health: Callable[[], Any],
    hf_token: str | None,
    download_poll_interval: float,
) -> ModelCatalogService:
    scanner = EmbeddingModelScanner(Path(model_dir))
    jobs = DownloadJobRegistry(HfSnapshotDownloader(token=hf_token, poll_interval=download_poll_interval))
    catalog = ModelCatalogService(
        scanner=scanner,
        models_dir=Path(model_dir),
        capabilities=CAPABILITIES,
        columns=MODEL_COLUMNS,
        jobs=jobs,
        load_state=RegistryLoadState(registry, registry_lock, loader, unloader, scanner),
    )
    feed = InteractionsFeed(InferenceLogReader(inference_logs, inference_lock), INTERACTION_COLUMNS)

    async def health_snapshot() -> Any:
        return await health()

    hub = AdminEventHub({"models": catalog.snapshot, "interactions": feed.snapshot, "health": health_snapshot})
    app.state.model_catalog = catalog
    app.state.admin_event_hub = hub

    auth = None  # explicit: same trust boundary as the unauthenticated /v1/* API
    app.include_router(create_admin_ui_router(STATIC_DIR, auth=auth))
    app.include_router(create_models_router(catalog, auth=auth))
    app.include_router(create_events_router(hub, auth=auth))
    app.include_router(create_interactions_router(feed, auth=auth))
    extra = APIRouter(prefix="/admin", tags=["admin"])

    @extra.get("/health")
    async def admin_health() -> Any:
        return await health()

    app.include_router(extra)
    return catalog
