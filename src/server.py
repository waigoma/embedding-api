"""
OpenAI-compatible Embedding & Reranking API Server
- On-demand model loading / unloading
- Auto-load on first request (AUTO_LOAD=1)
- Idle TTL auto-unload (IDLE_TTL=300 etc.)
- Preload specific models on startup (PRELOAD_EMBEDDING / PRELOAD_RERANKER)
"""

import os
import gc
import time
import threading
import logging
import json
import uuid
import traceback
import asyncio
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
WEBUI_DIR = os.path.join(os.path.dirname(__file__), "webui")
DEVICE_MODE = os.environ.get("DEVICE_MODE", "auto").strip().lower()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
DOWNLOAD_PROGRESS_INTERVAL_SEC = float(os.environ.get("DOWNLOAD_PROGRESS_INTERVAL_SEC", "1.0"))
DOWNLOAD_MAX_LOGS = int(os.environ.get("DOWNLOAD_MAX_LOGS", "200"))
INFERENCE_MAX_LOGS = int(os.environ.get("INFERENCE_MAX_LOGS", "300"))
LLM_PROXY_BASE_URL = os.environ.get("LLM_PROXY_BASE_URL", "").strip().rstrip("/")
LLM_PROXY_API_KEY = os.environ.get("LLM_PROXY_API_KEY", "").strip()
LLM_PROXY_TIMEOUT_SEC = float(os.environ.get("LLM_PROXY_TIMEOUT_SEC", "120"))
IDLE_TTL = int(os.environ.get("IDLE_TTL", "0"))  # seconds, 0 = disabled
AUTO_LOAD = os.environ.get("AUTO_LOAD", "1") == "1"
PRELOAD_EMBEDDING = os.environ.get("PRELOAD_EMBEDDING", "").split(",")
PRELOAD_RERANKER = os.environ.get("PRELOAD_RERANKER", "").split(",")
SENTENCE_TRANSFORMER_KWARGS = os.environ.get("SENTENCE_TRANSFORMER_KWARGS", "").strip()
CROSS_ENCODER_KWARGS = os.environ.get("CROSS_ENCODER_KWARGS", "").strip()
MODEL_CATALOG_JSON = os.environ.get("MODEL_CATALOG_JSON", "").strip()


def _parse_json_env(raw_value: str, env_name: str) -> dict:
    if not raw_value:
        return {}
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{env_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{env_name} must be a JSON object")
    return value


def _parse_json_list_env(raw_value: str, env_name: str) -> list[dict[str, Any]]:
    if not raw_value:
        return []
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{env_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"{env_name} must be a JSON array")
    out: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            raise RuntimeError(f"{env_name} must contain JSON objects only")
        out.append(row)
    return out


def _detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if torch.version.hip:
        return "rocm"
    if torch.version.cuda:
        return "cuda"
    return "cuda-unknown"


def _detect_device(mode: str) -> tuple[str, str]:
    if mode == "cpu":
        return "cpu", "cpu"
    if mode in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE_MODE requests GPU but no CUDA/HIP device is available")
        return "cuda", _detect_backend()
    if mode == "auto":
        if torch.cuda.is_available():
            return "cuda", _detect_backend()
        return "cpu", "cpu"
    raise RuntimeError("DEVICE_MODE must be one of: auto, cpu, gpu, cuda")


DEVICE, ACCELERATOR_BACKEND = _detect_device(DEVICE_MODE)
EMBEDDING_KWARGS = _parse_json_env(SENTENCE_TRANSFORMER_KWARGS, "SENTENCE_TRANSFORMER_KWARGS")
RERANKER_KWARGS = _parse_json_env(CROSS_ENCODER_KWARGS, "CROSS_ENCODER_KWARGS")
MODEL_CATALOG = _parse_json_list_env(MODEL_CATALOG_JSON, "MODEL_CATALOG_JSON")

if not MODEL_CATALOG:
    MODEL_CATALOG = [
        {"repo_id": "Qwen/Qwen3-Embedding-0.6B", "type": "embedding"},
        {"repo_id": "Qwen/Qwen3-Embedding-4B", "type": "embedding"},
        {"repo_id": "Qwen/Qwen3-Reranker-0.6B", "type": "reranker"},
        {"repo_id": "Qwen/Qwen3-Reranker-4B", "type": "reranker"},
        {"repo_id": "cl-nagoya/ruri-v3-310m", "type": "embedding"},
        {"repo_id": "cl-nagoya/ruri-v3-reranker-310m", "type": "reranker"},
    ]


class ModelEntry:
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self.last_used = time.time()

    def touch(self):
        self.last_used = time.time()


# --- Registry ---
registry: dict[str, ModelEntry] = {}
registry_lock = threading.Lock()
download_jobs: dict[str, dict[str, Any]] = {}
download_lock = threading.Lock()
inference_logs: list[dict[str, Any]] = []
inference_lock = threading.Lock()


def _model_local_path(model_id: str) -> str:
    """Full path under MODEL_DIR for a relative model id (supports nested dirs). Rejects path traversal."""
    stripped = model_id.strip()
    parts = [p for p in stripped.replace("\\", "/").split("/") if p and p != "."]
    if not parts:
        raise HTTPException(400, "model_id is empty or invalid")
    if ".." in parts:
        raise HTTPException(400, "invalid model_id")
    return os.path.join(MODEL_DIR, *parts)


_SKIP_WALK_SUBDIRS = frozenset({".cache", "__pycache__", ".git"})


def _dir_has_model_config(files: list[str]) -> bool:
    """Typical Hugging Face / PEFT root: config at this directory level (not every subfolder with files)."""
    names = frozenset(files)
    return "config.json" in names or "adapter_config.json" in names


def _iter_local_model_relative_ids() -> list[str]:
    """List model roots: directories with config.json (or adapter_config.json), excluding cache/submodules."""
    if not os.path.isdir(MODEL_DIR):
        return []
    candidates: list[str] = []
    for root, dirs, files in os.walk(MODEL_DIR):
        dirs[:] = [d for d in dirs if d not in _SKIP_WALK_SUBDIRS and not d.startswith(".")]
        if root == MODEL_DIR:
            continue
        if not _dir_has_model_config(files):
            continue
        rel = os.path.relpath(root, MODEL_DIR)
        if rel in (".", ""):
            continue
        candidates.append(rel.replace(os.sep, "/"))

    candidates.sort(key=lambda r: (len(r.split("/")), r))
    roots: list[str] = []
    for rel in candidates:
        if any(rel.startswith(p + "/") for p in roots):
            continue
        roots.append(rel)
    return sorted(roots)


def _resolve_path(model_id: str) -> str:
    stripped = model_id.strip()
    if not stripped:
        raise HTTPException(400, "model_id is empty")
    local = _model_local_path(stripped)
    if os.path.isdir(local):
        return local
    if os.path.isdir(stripped):
        return stripped
    return stripped


def _default_local_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def _sanitize_local_name(local_name: str) -> str:
    """Relative path under MODEL_DIR, e.g. embedding/Qwen3-Embedding-0.6B or reranker/ruri-v3-reranker-310m."""
    cleaned = local_name.strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise HTTPException(400, "local_name is empty")
    parts = [p for p in cleaned.split("/") if p]
    for p in parts:
        if p in (".", ".."):
            raise HTTPException(400, "local_name must not contain '.' or '..' segments")
    rel = "/".join(parts)
    base = os.path.abspath(MODEL_DIR)
    candidate = os.path.abspath(os.path.join(MODEL_DIR, *parts))
    if candidate != base and not candidate.startswith(base + os.sep):
        raise HTTPException(400, "local_name must stay under MODEL_DIR")
    return rel


def _set_download_status(job_id: str, **updates: Any) -> None:
    with download_lock:
        job = download_jobs.get(job_id)
        if not job:
            return
        job.update(updates)


def _append_download_log(job_id: str, level: str, message: str) -> None:
    event = {
        "timestamp": time.time(),
        "level": level,
        "message": message,
    }
    with download_lock:
        job = download_jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        logs.append(event)
        if len(logs) > DOWNLOAD_MAX_LOGS:
            del logs[: len(logs) - DOWNLOAD_MAX_LOGS]
        job["updated_at"] = event["timestamp"]


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _estimate_repo_size(repo_id: str) -> Optional[int]:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=HF_TOKEN).model_info(repo_id, files_metadata=True)
        total = 0
        found = False
        for sibling in info.siblings or []:
            size = getattr(sibling, "size", None)
            if isinstance(size, int) and size > 0:
                total += size
                found = True
        return total if found else None
    except Exception:
        return None


def _serialize_download_job(job: dict[str, Any], include_logs: bool = False) -> dict[str, Any]:
    payload = {k: v for k, v in job.items() if k != "logs"}
    payload["logs_count"] = len(job.get("logs", []))
    payload["last_log"] = job.get("logs", [])[-1] if job.get("logs") else None
    if include_logs:
        payload["logs"] = list(job.get("logs", []))
    return payload


def _append_inference_log(
    event: str,
    model_id: str,
    duration_ms: float,
    details: dict[str, Any],
    level: str = "info",
) -> None:
    record = {
        "timestamp": time.time(),
        "level": level,
        "event": event,
        "model": model_id,
        "duration_ms": round(duration_ms, 2),
        "details": details,
    }
    with inference_lock:
        inference_logs.append(record)
        if len(inference_logs) > INFERENCE_MAX_LOGS:
            del inference_logs[: len(inference_logs) - INFERENCE_MAX_LOGS]


def _normalize_response_input(input_value: Any) -> list[str]:
    # Accept a subset of OpenAI Responses API input shapes:
    # - "text"
    # - ["text1", "text2"]
    # - [{"role":"user","content":"text"}]
    # - [{"role":"user","content":[{"type":"input_text","text":"text"}]}]
    if isinstance(input_value, str):
        return [input_value]
    if isinstance(input_value, list):
        texts: list[str] = []
        for item in input_value:
            if isinstance(item, str):
                texts.append(item)
                continue
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    texts.append(content)
                    continue
                if isinstance(content, list):
                    for chunk in content:
                        if not isinstance(chunk, dict):
                            continue
                        if chunk.get("type") in {"input_text", "text"} and isinstance(chunk.get("text"), str):
                            texts.append(chunk["text"])
                    continue
                if isinstance(item.get("text"), str):
                    texts.append(item["text"])
                    continue
        return [t for t in texts if t.strip()]
    return []


def _encode_embeddings(model_id: str, inputs: list[str]) -> tuple[list[list[float]], int]:
    entry = _get_model(model_id, "embedding")
    encode_kwargs: dict[str, Any] = {"convert_to_numpy": True, "normalize_embeddings": True}
    if "query" in getattr(entry.model, "prompts", {}):
        encode_kwargs["prompt_name"] = "query"
    vectors = entry.model.encode(inputs, **encode_kwargs)
    embeddings = [vec.tolist() for vec in vectors]
    tokens = sum(len(s) // 4 for s in inputs)
    return embeddings, tokens


def _proxy_chat_completions(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if not LLM_PROXY_BASE_URL:
        raise HTTPException(
            501,
            "chat completions backend is not configured. Set LLM_PROXY_BASE_URL "
            "(e.g. http://localhost:11434/v1 or llama.cpp OpenAI endpoint).",
        )
    url = f"{LLM_PROXY_BASE_URL}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if LLM_PROXY_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_PROXY_API_KEY}"
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=LLM_PROXY_TIMEOUT_SEC) as res:
            raw = res.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return int(res.status), parsed
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            msg = parsed.get("error", {}).get("message") or parsed.get("detail") or detail
        except Exception:
            msg = detail or str(exc)
        raise HTTPException(exc.code, f"upstream chat backend error: {msg}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(502, f"failed to reach chat backend: {exc}") from exc


def _run_download_job(job_id: str, repo_id: str, target_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        _set_download_status(
            job_id,
            status="failed",
            error=f"missing huggingface_hub: {exc}",
            finished_at=time.time(),
        )
        return

    total_bytes = _estimate_repo_size(repo_id)
    _set_download_status(job_id, total_bytes=total_bytes)
    if total_bytes:
        _append_download_log(job_id, "info", f"estimated total size: {total_bytes} bytes")
    else:
        _append_download_log(job_id, "info", "total size estimate unavailable")

    result: dict[str, Any] = {}
    start_wall = time.time()
    start_size = _dir_size_bytes(target_dir)

    def _download_worker():
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=HF_TOKEN,
            )
            result["ok"] = True
        except Exception as exc:
            result["error"] = exc
            result["traceback"] = traceback.format_exc()

    worker = threading.Thread(target=_download_worker, daemon=True)
    _set_download_status(job_id, status="downloading", started_at=time.time())
    _append_download_log(job_id, "info", f"download started: {repo_id} -> {target_dir}")
    worker.start()

    last_t = start_wall
    last_size = start_size
    interval = max(0.2, DOWNLOAD_PROGRESS_INTERVAL_SEC)
    while worker.is_alive():
        time.sleep(interval)
        now = time.time()
        current_size = _dir_size_bytes(target_dir)
        downloaded = max(0, current_size - start_size)
        elapsed = max(1e-6, now - start_wall)
        speed_bps = max(0.0, (current_size - last_size) / max(1e-6, now - last_t))
        speed_mbps = speed_bps / (1024 * 1024)
        progress_percent = None
        eta_seconds = None
        if total_bytes and total_bytes > 0:
            progress_percent = min(100.0, (downloaded / total_bytes) * 100.0)
            if speed_bps > 1:
                remain = max(0, total_bytes - downloaded)
                eta_seconds = remain / speed_bps
        _set_download_status(
            job_id,
            updated_at=now,
            downloaded_bytes=downloaded,
            speed_mbps=round(speed_mbps, 2),
            elapsed_seconds=round(elapsed, 2),
            progress_percent=round(progress_percent, 2) if progress_percent is not None else None,
            eta_seconds=round(eta_seconds, 1) if eta_seconds is not None else None,
        )
        last_t = now
        last_size = current_size

    worker.join()
    finished = time.time()
    final_size = _dir_size_bytes(target_dir)
    downloaded_final = max(0, final_size - start_size)

    if result.get("ok"):
        _set_download_status(
            job_id,
            status="completed",
            finished_at=finished,
            updated_at=finished,
            downloaded_bytes=downloaded_final,
            speed_mbps=0.0,
            progress_percent=100.0 if total_bytes else None,
            eta_seconds=0.0 if total_bytes else None,
        )
        _append_download_log(job_id, "info", f"download completed: {repo_id}")
        return

    exc = result.get("error")
    msg = str(exc) if exc else "unknown download error"
    hint = ""
    lower = msg.lower()
    if "401" in msg or "403" in msg or "gated" in lower:
        hint = " (set HF_TOKEN for private/gated models)"
    if "timeout" in lower:
        _append_download_log(job_id, "error", "timeout detected during model download")
    _append_download_log(job_id, "error", msg)
    if result.get("traceback"):
        _append_download_log(job_id, "error", result["traceback"])
    _set_download_status(
        job_id,
        status="failed",
        error=f"{msg}{hint}",
        finished_at=finished,
        updated_at=finished,
        downloaded_bytes=downloaded_final,
        speed_mbps=0.0,
    )


def _load_embedding(model_id: str) -> ModelEntry:
    from sentence_transformers import SentenceTransformer
    path = _resolve_path(model_id)
    logger.info(f"Loading embedding: {model_id} from {path}")
    kwargs = dict(EMBEDDING_KWARGS)
    kwargs.setdefault("device", DEVICE)
    model = SentenceTransformer(path, **kwargs)
    logger.info(f"Loaded embedding: {model_id} (dim={model.get_sentence_embedding_dimension()})")
    return ModelEntry(model, "embedding")


def _load_reranker(model_id: str) -> ModelEntry:
    from sentence_transformers import CrossEncoder
    path = _resolve_path(model_id)
    logger.info(f"Loading reranker: {model_id} from {path}")
    kwargs = dict(RERANKER_KWARGS)
    kwargs.setdefault("device", DEVICE)
    model = CrossEncoder(path, **kwargs)
    logger.info(f"Loaded reranker: {model_id}")
    return ModelEntry(model, "reranker")


def _unload(model_id: str) -> bool:
    with registry_lock:
        entry = registry.pop(model_id, None)
    if entry is None:
        return False
    del entry.model
    del entry
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    logger.info(f"Unloaded: {model_id}")
    return True


def _get_model(model_id: str, expected_type: str) -> ModelEntry:
    with registry_lock:
        entry = registry.get(model_id)
    if entry is not None:
        if entry.model_type != expected_type:
            raise HTTPException(400, f"'{model_id}' is {entry.model_type}, not {expected_type}")
        entry.touch()
        return entry

    if AUTO_LOAD and os.path.isdir(_model_local_path(model_id)):
        loader = _load_embedding if expected_type == "embedding" else _load_reranker
        entry = loader(model_id)
        with registry_lock:
            registry[model_id] = entry
        return entry

    available = [k for k, v in registry.items() if v.model_type == expected_type]
    raise HTTPException(404, f"'{model_id}' not loaded. Available: {available}")


# --- Idle TTL checker ---
def _idle_checker():
    while True:
        time.sleep(30)
        if IDLE_TTL <= 0:
            continue
        now = time.time()
        to_unload = []
        with registry_lock:
            for mid, entry in registry.items():
                idle = now - entry.last_used
                if idle > IDLE_TTL:
                    to_unload.append(mid)
        for mid in to_unload:
            logger.info(f"Auto-unloading idle model: {mid} (idle {IDLE_TTL}s)")
            _unload(mid)


def _startup_once():
    logger.info(f"Device mode: {DEVICE_MODE}")
    logger.info(f"Device: {DEVICE}, backend: {ACCELERATOR_BACKEND}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {mem:.1f} GB")
    logger.info(f"Model dir: {MODEL_DIR}")
    logger.info(f"Auto-load: {AUTO_LOAD}, Idle TTL: {IDLE_TTL}s")

    for mid in PRELOAD_EMBEDDING:
        mid = mid.strip()
        if mid:
            registry[mid] = _load_embedding(mid)
    for mid in PRELOAD_RERANKER:
        mid = mid.strip()
        if mid:
            registry[mid] = _load_reranker(mid)

    threading.Thread(target=_idle_checker, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_once()
    yield


app = FastAPI(title="Embedding Server", version="2.0.0", lifespan=lifespan)


# --- Schemas ---
class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: Optional[str] = "float"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


class ResponsesEmbeddingRequest(BaseModel):
    model: str
    input: Any


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None

class RerankResponse(BaseModel):
    object: str = "list"
    results: list[RerankResult]
    model: str
    usage: dict

class LoadRequest(BaseModel):
    model_id: str
    model_type: str = "embedding"

class UnloadRequest(BaseModel):
    model_id: str

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"
    type: str
    loaded: bool = True
    last_used: Optional[float] = None

class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class DownloadRequest(BaseModel):
    repo_id: str
    local_name: Optional[str] = None
    force: bool = False


class DownloadStatusResponse(BaseModel):
    id: str
    repo_id: str
    local_name: str
    target_dir: str
    status: str
    created_at: float
    updated_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    progress_percent: Optional[float] = None
    total_bytes: Optional[int] = None
    downloaded_bytes: Optional[int] = None
    speed_mbps: Optional[float] = None
    eta_seconds: Optional[float] = None
    logs_count: Optional[int] = None
    last_log: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# --- Management ---
@app.get("/v1/models", response_model=ModelListResponse)
@app.get("/models", response_model=ModelListResponse)
async def list_models():
    models = []
    with registry_lock:
        for mid, entry in registry.items():
            models.append(ModelInfo(
                id=mid, type=entry.model_type,
                loaded=True, last_used=entry.last_used,
            ))
    loaded_ids = {m.id for m in models}
    if os.path.isdir(MODEL_DIR):
        for rel in _iter_local_model_relative_ids():
            if rel not in loaded_ids:
                models.append(ModelInfo(id=rel, type="unknown", loaded=False))
    return ModelListResponse(data=models)


@app.get("/ui")
@app.get("/webui")
async def webui():
    index_html = os.path.join(WEBUI_DIR, "index.html")
    if not os.path.isfile(index_html):
        raise HTTPException(404, "webui not found")
    return FileResponse(index_html, media_type="text/html")


@app.get("/v1/models/catalog")
@app.get("/models/catalog")
async def model_catalog():
    return {"object": "list", "data": MODEL_CATALOG}


@app.post("/v1/models/download", response_model=DownloadStatusResponse)
@app.post("/models/download", response_model=DownloadStatusResponse)
async def download_model(request: DownloadRequest):
    repo_id = request.repo_id.strip()
    if not repo_id:
        raise HTTPException(400, "repo_id is empty")
    if repo_id.count("/") != 1:
        raise HTTPException(
            400,
            "repo_id must be in 'owner/repo' format (e.g. Qwen/Qwen3-Embedding-0.6B)",
        )

    local_name = _sanitize_local_name(request.local_name or _default_local_name(repo_id))
    target_dir = os.path.join(MODEL_DIR, local_name)
    if os.path.exists(target_dir):
        if not os.path.isdir(target_dir):
            raise HTTPException(409, f"target exists but is not a directory: {local_name}")
        if os.listdir(target_dir) and not request.force:
            raise HTTPException(409, f"target already exists: {local_name} (set force=true)")

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            500,
            f"failed to prepare model directory: {exc}. "
            "Check if MODEL_DIR is mounted writable (remove ':ro' from volume mount).",
        ) from exc

    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "id": job_id,
        "repo_id": repo_id,
        "local_name": local_name,
        "target_dir": target_dir,
        "status": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "progress_percent": 0.0,
        "total_bytes": None,
        "downloaded_bytes": 0,
        "speed_mbps": 0.0,
        "eta_seconds": None,
        "logs": [],
        "error": None,
    }
    with download_lock:
        download_jobs[job_id] = job

    threading.Thread(
        target=_run_download_job,
        args=(job_id, repo_id, target_dir),
        daemon=True,
    ).start()
    return DownloadStatusResponse(**_serialize_download_job(job))


@app.get("/v1/models/downloads")
@app.get("/models/downloads")
async def list_download_jobs():
    with download_lock:
        jobs = [_serialize_download_job(job) for job in download_jobs.values()]
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"object": "list", "data": jobs}


@app.get("/v1/models/downloads/{job_id}", response_model=DownloadStatusResponse)
@app.get("/models/downloads/{job_id}", response_model=DownloadStatusResponse)
async def get_download_job(job_id: str):
    with download_lock:
        job = download_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"download job not found: {job_id}")
    return DownloadStatusResponse(**_serialize_download_job(job))


@app.get("/v1/logs/inference")
@app.get("/logs/inference")
async def get_inference_logs(limit: int = 50):
    capped = max(1, min(limit, 500))
    with inference_lock:
        data = list(inference_logs[-capped:])
    return {"object": "list", "data": data}


@app.post("/v1/models/load")
@app.post("/models/load")
async def load_model(request: LoadRequest):
    mid = request.model_id
    with registry_lock:
        if mid in registry:
            return {"status": "already_loaded", "model_id": mid}
    if request.model_type == "embedding":
        entry = _load_embedding(mid)
    elif request.model_type == "reranker":
        entry = _load_reranker(mid)
    else:
        raise HTTPException(400, f"Unknown type: {request.model_type}")
    with registry_lock:
        registry[mid] = entry
    return {"status": "loaded", "model_id": mid, "type": request.model_type}


@app.post("/v1/models/unload")
@app.post("/models/unload")
async def unload_model(request: UnloadRequest):
    if _unload(request.model_id):
        return {"status": "unloaded", "model_id": request.model_id}
    raise HTTPException(404, f"'{request.model_id}' not loaded")


# --- Inference ---
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    t0 = time.time()
    inputs = request.input if isinstance(request.input, list) else [request.input]
    try:
        embeddings, tokens = _encode_embeddings(request.model, inputs)
        data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
        _append_inference_log(
            event="embedding",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"inputs_count": len(inputs), "prompt_tokens": tokens, "status": "ok"},
        )
        return EmbeddingResponse(
            data=data, model=request.model,
            usage={"prompt_tokens": tokens, "total_tokens": tokens},
        )
    except Exception as exc:
        tokens = sum(len(s) // 4 for s in inputs)
        _append_inference_log(
            event="embedding",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"inputs_count": len(inputs), "prompt_tokens": tokens, "status": "error", "error": str(exc)},
            level="error",
        )
        raise


@app.post("/v1/responses")
@app.post("/responses")
async def create_responses_embedding(request: ResponsesEmbeddingRequest):
    t0 = time.time()
    inputs = _normalize_response_input(request.input)
    if not inputs:
        raise HTTPException(
            400,
            "input must include text. Supported formats: string, list[string], or Responses API message input.",
        )
    try:
        embeddings, tokens = _encode_embeddings(request.model, inputs)
        _append_inference_log(
            event="responses_embedding",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"inputs_count": len(inputs), "prompt_tokens": tokens, "status": "ok"},
        )
        return {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": request.model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Embedding generated successfully."}
                    ],
                }
            ],
            "output_text": "Embedding generated successfully.",
            # Custom extension for embedding use-cases.
            "data": [
                {"object": "embedding", "embedding": emb, "index": i}
                for i, emb in enumerate(embeddings)
            ],
            "usage": {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
        }
    except Exception as exc:
        _append_inference_log(
            event="responses_embedding",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"inputs_count": len(inputs), "status": "error", "error": str(exc)},
            level="error",
        )
        raise


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionsRequest):
    t0 = time.time()
    payload = request.model_dump(exclude_none=True)
    try:
        status, data = await asyncio.to_thread(_proxy_chat_completions, payload)
        _append_inference_log(
            event="chat_completions",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"messages_count": len(request.messages), "status": "ok", "upstream_status": status},
        )
        return data
    except HTTPException as exc:
        _append_inference_log(
            event="chat_completions",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"messages_count": len(request.messages), "status": "error", "error": str(exc.detail)},
            level="error",
        )
        raise


@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    t0 = time.time()
    tokens = sum(len(d) // 4 for d in request.documents)
    try:
        entry = _get_model(request.model, "reranker")
        pairs = [[request.query, doc] for doc in request.documents]
        scores = entry.model.predict(pairs).tolist()
        results = [
            RerankResult(
                index=i, relevance_score=float(s),
                document=request.documents[i] if request.return_documents else None,
            )
            for i, s in enumerate(scores)
        ]
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if request.top_n:
            results = results[: request.top_n]
        _append_inference_log(
            event="rerank",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"documents_count": len(request.documents), "prompt_tokens": tokens, "status": "ok"},
        )
        return RerankResponse(
            results=results, model=request.model,
            usage={"prompt_tokens": tokens, "total_tokens": tokens},
        )
    except Exception as exc:
        _append_inference_log(
            event="rerank",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={"documents_count": len(request.documents), "prompt_tokens": tokens, "status": "error", "error": str(exc)},
            level="error",
        )
        raise


@app.get("/health")
async def health():
    gpu = None
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_mb": round(total / 1024**2),
            "vram_free_mb": round(free / 1024**2),
            "vram_used_mb": round((total - free) / 1024**2),
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "accelerator_backend": ACCELERATOR_BACKEND,
        "device_mode": DEVICE_MODE,
        "gpu": gpu,
        "idle_ttl": IDLE_TTL,
        "auto_load": AUTO_LOAD,
        "loaded": {k: v.model_type for k, v in registry.items()},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7997)
