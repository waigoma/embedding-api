"""Model overview use case: one family-grouped list of every model a service knows.

It merges three sources into one snapshot for the shared "モデル管理" screen:

- the service's catalog (CatalogSource): models it knows how to fetch and run,
  already shaped as overview items with lifecycle facts and actions;
- the storage scan (LocalModelScanner): directories found under the declared roots;
- the generic download ledger (JobLedger, optional): free-form "repo から取得" jobs.

Rules owned here (and nowhere else):
- a storage directory is "claimed" by a catalog item when it equals, contains, or
  sits inside one of the item's paths / claims; unclaimed directories become
  "カタログ外" (support="local") items;
- a catalog item that reports not-downloaded while every claimed path exists on
  disk is shown as present but unverified (files without a receipt);
- free-form jobs attach to the item whose local_name matches, else become their
  own pending rows.

Item keys (all optional except id/title/family/lifecycle): version, purpose,
support ("adapter" | "candidate" | "local"), recommended, license, source, revision,
integrity, paths, size_bytes, local_name, api_name, note, group/group_rank (sub-heading
inside a long family), specs [{key, label, value, kind, conditions, source}], job, and
actions [{id, label, enabled, reason, primary, tone, request{method,path,body},
short_label (row button), navigate{tab,item,hook}, payload,
confirm{title,text,accept_label,body_patch},
requires [lifecycle keys], unmet_reason}]. Use `requires` instead of computing
`enabled` from lifecycle in the service: the merge may change the facts.
Claims must name the model's own directories, never a shared parent.

Items, steps and actions are plain data. The UI renders them without knowing the
service; services describe requests (method/path/body) and navigation hooks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .model_catalog import JobLedger, LoadStateProvider, LocalModel, LocalModelScanner
from .table_spec import ACTIVE_JOB_STATUSES

Item = dict[str, Any]
Action = dict[str, Any]

SUPPORT_KINDS = ("adapter", "candidate", "local")
LOCAL_NOTE = (
    "保存領域で見つかったカタログ外のモデルです。固定版と checksum が無いため検証はしません。"
)
UNVERIFIED_INTEGRITY = "受領証なし・未検証"
UNVERIFIED_REASON = "保存領域にファイルがあります。取得の受領証が無いため検証はしていません。"


@dataclass(frozen=True)
class LifecycleStep:
    """One fact in a model's lifecycle. Order is progression order."""

    key: str
    label: str
    hint: str
    done_label: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "label": self.label,
            "hint": self.hint,
            "done_label": self.done_label,
        }


STANDARD_STEPS: tuple[LifecycleStep, ...] = (
    LifecycleStep("downloaded", "取得", "ファイルが保存領域にある", "取得済み"),
    LifecycleStep(
        "files_verified", "ファイル検証", "固定版・サイズ・checksum を確認済み", "検証済み"
    ),
    LifecycleStep(
        "runtime_compatible",
        "runtime 互換",
        "依存とエンジンで読み込めることを確認済み",
        "動作確認済み",
    ),
    LifecycleStep("loaded", "ロード", "この worker のメモリに載っている", "ロード中"),
    LifecycleStep("inference_confirmed", "推論確認", "試験入力で出力を確認済み", "推論確認済み"),
)

LOAD_STEPS: tuple[LifecycleStep, ...] = (
    STANDARD_STEPS[0],
    LifecycleStep("loaded", "ロード", "メモリに載っていて API から使える", "ロード中"),
)


@dataclass(frozen=True)
class FetchForm:
    """The free-form "リポジトリから取得" form. None on the service means no form.

    name_prefix pre-fills the save name (e.g. "embedding/"); api_name says the
    save name is also the API model name; fields are extra inputs sent in body.extra.
    """

    name_prefix: str
    api_name: bool = False
    fields: tuple[dict[str, Any], ...] = ()
    path: str = "models/download"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_prefix": self.name_prefix,
            "api_name": self.api_name,
            "fields": [dict(spec) for spec in self.fields],
            "request": {"method": "POST", "path": self.path},
        }


class CatalogSource(Protocol):
    """Service catalog as overview items (see module docstring for the item shape).

    Items may carry "claims": absolute paths the model would occupy once fetched.
    The key is consumed by the overview and never reaches the client.
    """

    async def entries(self) -> list[Item]: ...


class PathProbe(Protocol):
    def exists_nonempty(self, path: Path) -> bool: ...


class FilesystemProbe:
    def exists_nonempty(self, path: Path) -> bool:
        if not path.exists():
            return False
        if not path.is_dir():
            return True
        return any(path.iterdir())


LocalActions = Callable[[LocalModel, "bool | None"], list[Action]]
FamilyOf = Callable[[str], str]


def family_by_rules(rules: tuple[tuple[str, str], ...], default: str = "カタログ外") -> FamilyOf:
    """Family of a storage directory from (substring, family) rules, first match wins."""

    def family_of(local_name: str) -> str:
        lowered = local_name.lower()
        for needle, family in rules:
            if needle.lower() in lowered:
                return family
        return default

    return family_of


@dataclass
class ModelOverviewService:
    catalog: CatalogSource
    scanner: LocalModelScanner
    models_root: Path
    roots: tuple[Path, ...]
    steps: tuple[LifecycleStep, ...] = STANDARD_STEPS
    fetch: FetchForm | None = None
    family_of_local: FamilyOf = field(default=family_by_rules(()))
    local_actions: LocalActions | None = None
    load_state: LoadStateProvider | None = None
    jobs: JobLedger | None = None
    api_name_for_local: bool = False
    version_label: str = "版"
    job_cancel_path: str = "models/downloads/{id}/cancel"
    probe: PathProbe = field(default_factory=FilesystemProbe)

    def __post_init__(self) -> None:
        self.models_root = self.models_root.resolve()
        self.roots = tuple(root.resolve() for root in self.roots)
        if not self.roots:
            raise ValueError("ModelOverviewService needs at least one storage root")
        keys = [step.key for step in self.steps]
        if not keys or keys[0] != "downloaded" or len(set(keys)) != len(keys):
            raise ValueError("steps must start with 'downloaded' and have unique keys")

    async def snapshot(self) -> dict[str, Any]:
        entries = await self.catalog.entries()
        # Disk scans block; keep them off the event loop (the SSE hub polls often).
        scanned = await asyncio.to_thread(self.scanner.scan)
        local = [model for model in scanned if self._in_roots(model)]
        loaded = self.load_state.loaded_names() if self.load_state is not None else None
        jobs = self.jobs.all() if self.jobs is not None else []
        items = merge_overview(
            entries,
            local,
            jobs,
            models_root=self.models_root,
            step_keys=[step.key for step in self.steps],
            family_of_local=self.family_of_local,
            local_actions=self.local_actions,
            loaded=loaded,
            api_name_for_local=self.api_name_for_local,
            probe=self.probe,
            cancel_path=self.job_cancel_path
            if self.jobs is not None and hasattr(self.jobs, "cancel")
            else None,
        )
        return {
            "schema_version": 1,
            "steps": [step.to_dict() for step in self.steps],
            "roots": [str(root) for root in self.roots],
            "models_root": str(self.models_root) + "/",
            "version_label": self.version_label,
            "fetch": self.fetch.to_dict() if self.fetch is not None else None,
            "items": items,
        }

    def _in_roots(self, model: LocalModel) -> bool:
        path = (self.models_root / model.local_name).resolve()
        return any(path == root or root in path.parents for root in self.roots)


def _related(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def merge_overview(
    entries: list[Item],
    local: list[LocalModel],
    jobs: list[dict[str, Any]],
    *,
    models_root: Path,
    step_keys: list[str],
    family_of_local: FamilyOf,
    local_actions: LocalActions | None,
    loaded: set[str] | None,
    api_name_for_local: bool,
    probe: PathProbe,
    cancel_path: str | None,
) -> list[Item]:
    local_paths = {model.local_name: (models_root / model.local_name).resolve() for model in local}
    claimed: set[str] = set()
    items: list[Item] = []

    for raw in entries:
        item = {key: value for key, value in raw.items() if key != "claims"}
        claims = [Path(p).resolve() for p in (raw.get("claims") or []) + (raw.get("paths") or [])]
        mine = {
            name
            for name, path in local_paths.items()
            if any(_related(path, claim) for claim in claims)
        }
        if raw.get("local_name") in local_paths:
            mine.add(raw["local_name"])
        claimed |= mine
        if item.get("size_bytes") is None and mine:
            item["size_bytes"] = sum(
                model.size_bytes for model in local if model.local_name in mine
            )
        lifecycle = dict(item.get("lifecycle") or {})
        own_claims = [Path(p).resolve() for p in raw.get("claims") or []]
        if (
            not lifecycle.get("downloaded")
            and own_claims
            and all(probe.exists_nonempty(path) for path in own_claims)
        ):
            item = _present_unverified(item, lifecycle, own_claims)
        items.append(_normalize(item, step_keys))

    by_local_name = {item.get("local_name"): item for item in items if item.get("local_name")}
    for model in sorted(local, key=lambda m: m.local_name):
        if model.local_name in claimed:
            continue
        is_loaded = None if loaded is None else model.local_name in loaded
        item = _local_item(model, local_paths[model.local_name], family_of_local, is_loaded)
        if api_name_for_local:
            item["api_name"] = model.local_name
        if local_actions is not None:
            item["actions"] = local_actions(model, is_loaded)
        normalized = _normalize(item, step_keys)
        items.append(normalized)
        by_local_name[model.local_name] = normalized

    for job in _latest_jobs(jobs):
        view = _job_view(job, cancel_path)
        target = by_local_name.get(job.get("local_name"))
        if target is not None:
            if job["status"] in ACTIVE_JOB_STATUSES or job["status"] == "failed":
                target["job"] = view
            continue
        if job["status"] not in ACTIVE_JOB_STATUSES and job["status"] != "failed":
            continue
        name = str(job.get("local_name") or job.get("repo_id") or job.get("id"))
        items.append(
            _normalize(
                {
                    "id": "job:" + str(job.get("id")),
                    "title": name.rsplit("/", 1)[-1],
                    "family": family_of_local(name),
                    "version": job.get("repo_id") or "",
                    "support": "local",
                    "local_name": name,
                    "api_name": name if api_name_for_local else None,
                    "paths": [str((models_root / name).resolve())],
                    "lifecycle": {},
                    "actions": [],
                    "job": view,
                },
                step_keys,
            )
        )
    return items


def _present_unverified(item: Item, lifecycle: dict[str, Any], claims: list[Path]) -> Item:
    lifecycle["downloaded"] = True
    if "files_verified" in lifecycle:
        lifecycle["files_verified"] = False
    actions = []
    for action in item.get("actions") or []:
        if action.get("id") == "download":
            action = {**action, "enabled": False, "reason": UNVERIFIED_REASON}
        actions.append(action)
    return {
        **item,
        "lifecycle": lifecycle,
        "integrity": UNVERIFIED_INTEGRITY,
        "paths": item.get("paths") or [str(path) for path in claims],
        "unverified": True,
        "actions": actions,
    }


def _local_item(
    model: LocalModel, path: Path, family_of_local: FamilyOf, is_loaded: bool | None
) -> Item:
    return {
        "id": "local:" + model.local_name,
        "title": model.local_name.rsplit("/", 1)[-1],
        "family": family_of_local(model.local_name),
        "version": "カタログ外",
        "support": "local",
        "local_name": model.local_name,
        "paths": [str(path)],
        "size_bytes": model.size_bytes,
        "lifecycle": {"downloaded": True, "loaded": bool(is_loaded)},
        "note": LOCAL_NOTE,
        "license": None,
        "actions": [],
        "extra": dict(model.extra),
    }


def _normalize(item: Item, step_keys: list[str]) -> Item:
    lifecycle = item.get("lifecycle") or {}
    facts = {key: bool(lifecycle.get(key)) for key in step_keys}
    support = item.get("support") or "adapter"
    if support not in SUPPORT_KINDS:
        raise ValueError(f"unknown support kind {support!r} on {item.get('id')}")
    return {
        "version": "",
        "purpose": "",
        "recommended": False,
        "license": None,
        "source": None,
        "revision": None,
        "integrity": None,
        "paths": [],
        "size_bytes": None,
        "local_name": None,
        "api_name": None,
        "note": "",
        "specs": [],
        "group": None,
        "group_rank": 0,
        "job": None,
        **item,
        "support": support,
        "lifecycle": facts,
        "actions": [_gate(action, facts) for action in item.get("actions") or []],
    }


def _gate(action: Action, facts: dict[str, bool]) -> Action:
    """Apply an action's lifecycle requirements after the merge has settled the facts.

    {"requires": ["downloaded"], "unmet_reason": "..."} keeps the action disabled until
    those facts hold, so a model found on disk without a receipt still unlocks it.
    """
    requires = action.get("requires") or []
    if not requires:
        return dict(action)
    met = all(facts.get(key) for key in requires)
    gated = {**action, "enabled": bool(action.get("enabled", True)) and met}
    if not met:
        gated["reason"] = action.get("unmet_reason") or action.get("reason") or ""
    return gated


def _latest_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for job in sorted(jobs, key=lambda j: j.get("created_at") or 0):
        latest[str(job.get("local_name") or job.get("id"))] = job
    return list(latest.values())


def _job_view(job: dict[str, Any], cancel_path: str | None) -> dict[str, Any]:
    active = job["status"] in ACTIVE_JOB_STATUSES
    cancel = (
        {
            "id": "cancel",
            "label": "取消",
            "enabled": True,
            "tone": "danger",
            "request": {"method": "POST", "path": cancel_path.format(id=job["id"])},
        }
        if active and cancel_path and job.get("id")
        else None
    )
    return {
        "id": job.get("id"),
        "status": job["status"],
        "progress_percent": job.get("progress_percent"),
        "speed_mbps": job.get("speed_mbps"),
        "error": job.get("error"),
        "cancel": cancel,
    }
