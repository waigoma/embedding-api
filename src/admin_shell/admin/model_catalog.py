"""Model catalog use case: one listing that merges disk, download jobs, and load state.

Ports: LocalModelScanner (disk), DownloadJobRegistry (ledger), LoadStateProvider
(runtime), TargetDirectory (filesystem checks for a download target).
Capabilities declare what a service can do; routes and the UI read them instead
of branching on the service name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .table_spec import ACTIVE_JOB_STATUSES, ColumnSpec

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class LocalModel:
    local_name: str
    size_bytes: int
    modified_at: float
    extra: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCatalogCapabilities:
    download: bool = False
    load: bool = False
    unload: bool = False
    cancel: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "download": self.download,
            "load": self.load,
            "unload": self.unload,
            "cancel": self.cancel,
        }


class JobLedger(Protocol):
    """Download job history as flat dicts (see DownloadJobRegistry for the shape).

    Required: all(), get(), active_names(). start() is needed for the download
    capability, cancel() for the cancel capability. Status values must be in
    MODEL_STATUSES minus "installed".
    """

    def all(self) -> list[dict[str, Any]]: ...
    def get(self, job_id: str) -> dict[str, Any]: ...
    def active_names(self) -> set[str]: ...


class LocalModelScanner(Protocol):
    def scan(self) -> list[LocalModel]: ...


class LoadStateProvider(Protocol):
    def loaded_names(self) -> set[str]: ...
    async def load(self, local_name: str) -> None: ...
    async def unload(self, local_name: str) -> None: ...


class TargetDirectory(Protocol):
    def exists_nonempty(self, path: Path) -> bool: ...
    def prepare(self, path: Path) -> None: ...


class FilesystemTargets:
    def exists_nonempty(self, path: Path) -> bool:
        if not path.exists():
            return False
        if not path.is_dir():
            return True
        return any(path.iterdir())

    def prepare(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)


class CatalogError(Exception):
    pass


class InvalidLocalName(CatalogError):
    pass


class InvalidRepoId(CatalogError):
    pass


class TargetExists(CatalogError):
    pass


class CapabilityUnsupported(CatalogError):
    pass


class InvalidDownloadField(CatalogError):
    pass


class ModelNotLoadable(CatalogError):
    """Raised by a LoadStateProvider when the model cannot be (un)loaded."""


class JobNotCancellable(CatalogError):
    """Raised by a JobLedger.cancel() when the job is finished or owned elsewhere."""


@dataclass(frozen=True)
class CatalogListing:
    models_dir: str
    capabilities: ModelCatalogCapabilities
    columns: list[ColumnSpec]
    items: list[dict[str, JsonValue]]
    download_fields: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "models_dir": self.models_dir,
            "capabilities": self.capabilities.to_dict(),
            "columns": [column.to_dict() for column in self.columns],
            "download_fields": list(self.download_fields),
            "items": self.items,
        }


_ROW_KEYS = (
    "repo_id",
    "progress_percent",
    "speed_mbps",
    "downloaded_bytes",
    "total_bytes",
    "error",
)


def validate_local_name(local_name: str) -> str:
    cleaned = local_name.strip()
    if not cleaned or cleaned != local_name:
        raise InvalidLocalName("local_name must be nonempty without surrounding whitespace")
    if "\\" in cleaned or cleaned.startswith("/"):
        raise InvalidLocalName("local_name must be a relative POSIX path")
    parts = PurePosixPath(cleaned).parts
    if any(part in {".", ".."} or part.startswith(".") for part in parts):
        raise InvalidLocalName("local_name must not contain '.', '..' or hidden segments")
    return cleaned


def validate_repo_id(repo_id: str) -> str:
    cleaned = repo_id.strip()
    parts = cleaned.split("/")
    if len(parts) != 2 or not all(parts):
        raise InvalidRepoId("repo_id must be in 'owner/repo' format")
    return cleaned


class ModelCatalogService:
    def __init__(
        self,
        *,
        scanner: LocalModelScanner,
        models_dir: Path,
        capabilities: ModelCatalogCapabilities,
        columns: list[ColumnSpec],
        jobs: JobLedger | None = None,
        load_state: LoadStateProvider | None = None,
        targets: TargetDirectory | None = None,
        download_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        if capabilities.download and (jobs is None or not hasattr(jobs, "start")):
            raise ValueError("download capability requires a JobLedger with start()")
        if capabilities.cancel and (jobs is None or not hasattr(jobs, "cancel")):
            raise ValueError("cancel capability requires a JobLedger with cancel()")
        if (capabilities.load or capabilities.unload) and load_state is None:
            raise ValueError("load/unload capability requires a LoadStateProvider")
        self._scanner = scanner
        self._models_dir = models_dir.resolve()
        self._capabilities = capabilities
        self._columns = list(columns)
        self._jobs = jobs
        self._load_state = load_state
        self._targets = targets or FilesystemTargets()
        # Extra form fields the UI shows next to repo_id/local_name; values are passed
        # to the ledger's start() as keyword `extra` (e.g. a pinned revision).
        self._download_fields = list(download_fields or [])

    @property
    def capabilities(self) -> ModelCatalogCapabilities:
        return self._capabilities

    # ----- listing -----

    def listing(self) -> CatalogListing:
        jobs = self._jobs.all() if self._jobs is not None else []
        installed = self._scanner.scan()
        loaded = self._load_state.loaded_names() if self._load_state is not None else None
        return CatalogListing(
            models_dir=str(self._models_dir),
            capabilities=self._capabilities,
            columns=self._columns,
            items=merge_models_and_jobs(installed, jobs, loaded),
            download_fields=self._download_fields,
        )

    async def snapshot(self) -> dict[str, JsonValue]:
        return self.listing().to_dict()

    # ----- downloads -----

    def resolve_target(self, local_name: str) -> Path:
        cleaned = validate_local_name(local_name)
        target = (self._models_dir / cleaned).resolve()
        if target == self._models_dir or self._models_dir not in target.parents:
            raise InvalidLocalName("local_name must resolve under models_dir")
        return target

    def start_download(
        self,
        repo_id: str,
        local_name: str | None,
        force: bool,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._capabilities.download or self._jobs is None:
            raise CapabilityUnsupported("this service does not download models")
        repo = validate_repo_id(repo_id)
        name = local_name or repo.rsplit("/", 1)[-1]
        target = self.resolve_target(name)
        if self._targets.exists_nonempty(target) and not force:
            raise TargetExists(
                f"model directory already exists: {name} (set force=true to overwrite)"
            )
        self._targets.prepare(target)
        allowed = {str(spec["key"]) for spec in self._download_fields}
        unknown = set(extra or {}) - allowed
        if unknown:
            raise InvalidDownloadField(f"unknown download fields: {sorted(unknown)}")
        start = getattr(self._jobs, "start", None)
        if start is None:
            raise CapabilityUnsupported("the job ledger cannot start downloads")
        return start(repo, validate_local_name(name), target, extra=dict(extra or {}))

    def get_download(self, job_id: str) -> dict[str, Any]:
        if not self._capabilities.download or self._jobs is None:
            raise CapabilityUnsupported("this service does not download models")
        return self._jobs.get(job_id)

    def list_downloads(self) -> list[dict[str, Any]]:
        return self._jobs.all() if self._jobs is not None else []

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        if not self._capabilities.cancel or self._jobs is None:
            raise CapabilityUnsupported("this service does not cancel downloads")
        cancel = getattr(self._jobs, "cancel", None)
        if cancel is None:
            raise CapabilityUnsupported("the job ledger cannot cancel downloads")
        return cancel(job_id)

    # ----- load state -----

    async def load(self, local_name: str) -> None:
        if not self._capabilities.load or self._load_state is None:
            raise CapabilityUnsupported("this service does not load models on demand")
        await self._load_state.load(validate_local_name(local_name))

    async def unload(self, local_name: str) -> None:
        if not self._capabilities.unload or self._load_state is None:
            raise CapabilityUnsupported("this service does not unload models on demand")
        await self._load_state.unload(validate_local_name(local_name))


def merge_models_and_jobs(
    installed: list[LocalModel],
    jobs: list[dict[str, Any]],
    loaded: set[str] | None,
) -> list[dict[str, JsonValue]]:
    """Port of the tts admin UI merge rules; see spec section 3.3."""
    active = {job["local_name"] for job in jobs if job["status"] in ACTIVE_JOB_STATUSES}
    installed_names = {model.local_name for model in installed}

    rows: list[dict[str, JsonValue]] = []
    for model in sorted(installed, key=lambda item: item.local_name):
        if model.local_name in active:
            continue
        rows.append(
            {
                "local_name": model.local_name,
                "status": "installed",
                "size_bytes": model.size_bytes,
                "loaded": None if loaded is None else model.local_name in loaded,
                **{key: None for key in _ROW_KEYS},
                "id": None,
                "extra": dict(model.extra),
            }
        )

    shown_jobs = [
        job
        for job in jobs
        if job["status"] in {"queued", "downloading", "failed"}
        or (job["status"] == "completed" and job["local_name"] not in installed_names)
    ]
    for job in sorted(shown_jobs, key=lambda item: item["created_at"], reverse=True):
        rows.append(
            {
                "local_name": job["local_name"],
                "status": job["status"],
                "size_bytes": None,
                "loaded": None,
                **{key: job.get(key) for key in _ROW_KEYS},
                "id": job.get("id"),
                "extra": dict(job.get("extra") or {}),
            }
        )
    return rows


SnapshotProvider = Callable[[], Awaitable[JsonValue]]
