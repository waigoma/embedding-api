"""Model overview adapter for embedding-api (shared "モデル管理" screen).

Turns MODEL_CATALOG (repo_id / type / family entries) and the MODEL_DIR scan into
overview items. Catalog entries are saved under "<first root>/<repo name>" so they
land inside the storage roots the overview lists; /v1/models/download keeps its own
naming and is untouched. Load / unload / download actions point at the existing
/admin/models routes, and the load gate is RegistryLoadState's (an unfinished or
failed download blocks loading).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from admin_shell.admin.model_catalog import JobLedger, LocalModel
from admin_shell.admin.table_spec import ACTIVE_JOB_STATUSES
from admin_shell.admin.model_overview import LOAD_STEPS, FetchForm, ModelOverviewService, family_by_rules
from embedding_admin import (
    LOAD_BLOCKING_REASONS_JA,
    EmbeddingModelScanner,
    RegistryLoadState,
    find_model_roots,
)

DEFAULT_ROOTS = "embedding"
TYPE_LABELS = {"embedding": "埋め込み", "reranker": "リランカー"}
TYPE_PURPOSES = {"embedding": "テキスト埋め込み", "reranker": "検索結果の再順位付け"}
CATALOG_FAMILY_FALLBACK = "その他"
FAMILY_RULES = (
    ("ruri", "Ruri"),
    ("qwen3-embedding", "Qwen3-Embedding"),
    ("qwen3-reranker", "Qwen3-Reranker"),
)

CatalogEntries = Callable[[], list[dict[str, Any]]]
LoadedNames = Callable[[], set[str]]


def parse_roots(raw: str | None, model_dir: Path) -> tuple[Path, ...]:
    """ADMIN_MODEL_ROOTS (comma-separated, relative to model_dir or absolute) -> paths under model_dir."""
    base = model_dir.resolve()
    roots: list[Path] = []
    for part in (raw if raw and raw.strip() else DEFAULT_ROOTS).split(","):
        part = part.strip()
        if not part:
            continue
        root = (base / part).resolve()
        if root == base or base not in root.parents:
            raise ValueError(f"ADMIN_MODEL_ROOTS entry must be a directory under MODEL_DIR: {part!r}")
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("ADMIN_MODEL_ROOTS must name at least one directory")
    return tuple(roots)


def admin_path(*segments: str) -> str:
    """Request path relative to /admin with each segment of a nested model name URL-encoded."""
    parts: list[str] = []
    for segment in segments:
        parts.extend(quote(piece, safe="") for piece in segment.split("/"))
    return "/".join(parts)


def _action(
    action_id: str, label: str, *, enabled: bool, reason: str, primary: bool, request: dict[str, Any], tone: str | None = None
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "id": action_id,
        "label": label,
        "enabled": enabled,
        "reason": None if enabled else reason,
        "primary": primary,
        "request": request,
    }
    if tone:
        action["tone"] = tone
    return action


class EmbeddingOverviewActions:
    """Load / unload (and download) actions from lifecycle facts, shared by catalog and local rows."""

    def __init__(self, load_state: RegistryLoadState) -> None:
        self._load_state = load_state

    def load_actions(self, local_name: str, *, downloaded: bool, loaded: bool) -> list[dict[str, Any]]:
        blocked = LOAD_BLOCKING_REASONS_JA.get(self._load_state.blocking_status(local_name) or "")
        reason = "すでにロードされています。" if loaded else blocked or ""
        load = _action(
            "load",
            "ロード",
            enabled=not loaded and not blocked,
            reason=reason,
            primary=downloaded and not loaded,
            request={"method": "POST", "path": admin_path("models", local_name, "load")},
        )
        # The overview enables it once the merge settles "downloaded" (unverified files count).
        load["requires"] = ["downloaded"]
        load["unmet_reason"] = "先にモデルを取得してください。"
        unload = _action(
            "unload",
            "解放",
            enabled=loaded,
            reason="ロードされていません。",
            primary=loaded,
            tone="danger",
            request={"method": "POST", "path": admin_path("models", local_name, "unload")},
        )
        return [load, unload]

    def for_local(self, model: LocalModel, loaded: bool | None) -> list[dict[str, Any]]:
        return self.load_actions(model.local_name, downloaded=True, loaded=bool(loaded))


class EmbeddingCatalogSource:
    """CatalogSource over MODEL_CATALOG: repo_id / type / family entries -> overview items."""

    def __init__(
        self,
        *,
        entries: CatalogEntries,
        models_root: Path,
        first_root: Path,
        loaded_names: LoadedNames,
        jobs: JobLedger,
        actions: EmbeddingOverviewActions,
    ) -> None:
        self._entries = entries
        self._models_root = models_root.resolve()
        self._first_root = first_root.resolve()
        self._loaded_names = loaded_names
        self._jobs = jobs
        self._actions = actions
        self._family_of = family_by_rules(FAMILY_RULES, default=CATALOG_FAMILY_FALLBACK)

    def local_name_of(self, entry: dict[str, Any]) -> str:
        """Save name: an explicit entry local_name, else "<first root>/<repo name>"."""
        explicit = str(entry.get("local_name") or "").strip().strip("/")
        if explicit:
            return explicit
        prefix = self._first_root.relative_to(self._models_root).as_posix()
        return f"{prefix}/{str(entry['repo_id']).rsplit('/', 1)[-1]}"

    async def entries(self) -> list[dict[str, Any]]:
        # Same detection as the scanner (outermost config wins), limited to the first root.
        present = {path.relative_to(self._models_root).as_posix() for path in find_model_roots(self._first_root)}
        loaded = self._loaded_names()
        jobs = self._jobs.all()
        return [self._item(entry, present, loaded, jobs) for entry in self._entries() if entry.get("repo_id")]

    def _item(self, entry: dict[str, Any], present: set[str], loaded: set[str], jobs: list[dict[str, Any]]) -> dict[str, Any]:
        repo_id = str(entry["repo_id"])
        local_name = self.local_name_of(entry)
        model_type = entry.get("type")
        downloaded = local_name in present
        is_loaded = local_name in loaded
        mine = [job for job in jobs if job.get("local_name") == local_name]
        latest = max(mine, key=lambda job: job.get("created_at") or 0) if mine else None
        active = latest is not None and latest.get("status") in ACTIVE_JOB_STATUSES
        retry = latest is not None and latest.get("status") == "failed"
        target = str(self._models_root / local_name)
        if downloaded and not retry:
            download_reason = "すでに取得済みです。"
        elif active:
            download_reason = "取得中です。"
        else:
            download_reason = ""
        download = _action(
            "download",
            "取得",
            enabled=not download_reason,
            reason=download_reason,
            primary=not downloaded,
            request={
                "method": "POST",
                "path": "models/download",
                "body": {"repo_id": repo_id, "local_name": local_name, "force": retry},
            },
        )
        return {
            "id": local_name,
            "title": repo_id.rsplit("/", 1)[-1],
            "family": str(entry.get("family") or self._family_of(repo_id)),
            "version": TYPE_LABELS.get(model_type, "種別未判定"),
            "purpose": TYPE_PURPOSES.get(model_type, ""),
            "support": "adapter",
            "source": f"https://huggingface.co/{repo_id}",
            "license": None,
            "local_name": local_name,
            "api_name": local_name,
            "paths": [target],
            "claims": [target],
            "lifecycle": {"downloaded": downloaded, "loaded": is_loaded},
            "actions": [download, *self._actions.load_actions(local_name, downloaded=downloaded, loaded=is_loaded)],
        }


def build_overview(
    model_dir: Path,
    *,
    jobs: JobLedger,
    loaded_names: LoadedNames,
    loader: Callable[[str, str], Any],
    unloader: Callable[[str], bool],
    catalog_entries: CatalogEntries,
    roots: tuple[Path, ...],
) -> ModelOverviewService:
    scanner = EmbeddingModelScanner(model_dir)
    load_state = RegistryLoadState(
        loaded_names=loaded_names, loader=loader, unloader=unloader, scanner=scanner, jobs=jobs
    )
    actions = EmbeddingOverviewActions(load_state)
    first_root = roots[0]
    source = EmbeddingCatalogSource(
        entries=catalog_entries,
        models_root=model_dir,
        first_root=first_root,
        loaded_names=loaded_names,
        jobs=jobs,
        actions=actions,
    )
    prefix = first_root.relative_to(model_dir.resolve()).as_posix() + "/"
    return ModelOverviewService(
        catalog=source,
        scanner=scanner,
        models_root=model_dir,
        roots=roots,
        steps=LOAD_STEPS,
        fetch=FetchForm(name_prefix=prefix, api_name=True, path="models/download"),
        family_of_local=family_by_rules(FAMILY_RULES),
        local_actions=actions.for_local,
        load_state=load_state,
        jobs=jobs,
        api_name_for_local=True,
        version_label="種別",
    )

