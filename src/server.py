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
import asyncio
import base64
import struct
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from admin_shell.admin.download_jobs import DownloadJobRegistry
from admin_shell.admin.hf_download import HfSnapshotDownloader
from embedding_admin import build_catalog, find_model_roots, mount_admin, v1_download_errors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
DEVICE_MODE = os.environ.get("DEVICE_MODE", "auto").strip().lower()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
DOWNLOAD_PROGRESS_INTERVAL_SEC = float(os.environ.get("DOWNLOAD_PROGRESS_INTERVAL_SEC", "1.0"))
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
        self.last_inference_at: Optional[float] = None

    def mark_inference(self):
        """成功した推論の時刻を現ロードに記録する。アンロード後は引き継がない。"""
        self.last_inference_at = time.time()

    def touch(self):
        self.last_used = time.time()


# --- Registry ---
registry: dict[str, ModelEntry] = {}
registry_lock = threading.Lock()
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


def _iter_local_model_relative_ids() -> list[str]:
    """List model roots (config.json / adapter_config.json), excluding cache/hidden dirs."""
    root = Path(MODEL_DIR)
    return [path.relative_to(root).as_posix() for path in find_model_roots(root)]


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


def _encode_embeddings(
    model_id: str,
    inputs: list[str],
    dimensions: Optional[int] = None,
) -> tuple[list[list[float]], int]:
    entry = _get_model(model_id, "embedding")

    if dimensions is not None:
        max_dim = entry.model.get_sentence_embedding_dimension()
        if not (1 <= dimensions <= max_dim):
            raise HTTPException(
                400,
                f"dimensions must be between 1 and {max_dim} (model embedding dimension), got {dimensions}",
            )

    encode_kwargs: dict[str, Any] = {"convert_to_numpy": True, "normalize_embeddings": True}
    if "query" in getattr(entry.model, "prompts", {}):
        encode_kwargs["prompt_name"] = "query"

    if dimensions is not None:
        try:
            vectors = entry.model.encode(inputs, truncate_dim=dimensions, **encode_kwargs)
            embeddings = [vec.tolist() for vec in vectors]
        except TypeError:
            # truncate_dim not supported by this version; slice after encoding
            vectors = entry.model.encode(inputs, **encode_kwargs)
            embeddings = [vec[:dimensions].tolist() for vec in vectors]
    else:
        vectors = entry.model.encode(inputs, **encode_kwargs)
        embeddings = [vec.tolist() for vec in vectors]

    if embeddings:
        entry.mark_inference()
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


def _load_into_registry(model_id: str, model_type: str) -> str:
    """Load and register a model. Returns "loaded" or "already_loaded"."""
    with registry_lock:
        if model_id in registry:
            return "already_loaded"
    if model_type == "embedding":
        entry = _load_embedding(model_id)
    elif model_type == "reranker":
        entry = _load_reranker(model_id)
    else:
        raise HTTPException(400, f"Unknown type: {model_type}")
    with registry_lock:
        registry[model_id] = entry
    return "loaded"


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
    await admin_hub.start()
    try:
        yield
    finally:
        await admin_hub.stop()


app = FastAPI(title="Embedding Server", version="2.0.0", lifespan=lifespan)


# --- Admin shell (/admin/*) ---
# Callables read module globals at call time, so tests may replace registry,
# loaders and logs. The download ledger is in-memory, like the old job dict.
def _loaded_names() -> set[str]:
    with registry_lock:
        return set(registry)


def _recent_inference_logs(limit: int) -> list[dict[str, Any]]:
    with inference_lock:
        return list(inference_logs[-limit:])


def _admin_config() -> dict[str, Any]:
    """Non-secret startup configuration (read-only; changed via env + restart)."""
    return {
        "model_dir": MODEL_DIR,
        "device_mode": DEVICE_MODE,
        "auto_load": AUTO_LOAD,
        "idle_ttl": IDLE_TTL,
        "preload_embedding": [m.strip() for m in PRELOAD_EMBEDDING if m.strip()],
        "preload_reranker": [m.strip() for m in PRELOAD_RERANKER if m.strip()],
        "chat_proxy_configured": bool(LLM_PROXY_BASE_URL),
    }


async def admin_health_snapshot() -> dict[str, Any]:
    """/health plus the config block and per-load inference evidence (formerly /ui/status)."""
    payload = dict(await health())
    with registry_lock:
        payload["loaded_models"] = [
            {"id": mid, "type": entry.model_type, "last_inference_at": entry.last_inference_at}
            for mid, entry in registry.items()
        ]
    payload["config"] = _admin_config()
    return payload


def new_download_jobs() -> DownloadJobRegistry:
    return DownloadJobRegistry(
        HfSnapshotDownloader(token=HF_TOKEN, poll_interval=DOWNLOAD_PROGRESS_INTERVAL_SEC)
    )


def new_admin_catalog(model_dir: str, jobs: DownloadJobRegistry):
    return build_catalog(
        Path(model_dir),
        jobs=jobs,
        loaded_names=lambda: _loaded_names(),
        loader=lambda model_id, model_type: _load_into_registry(model_id, model_type),
        unloader=lambda model_id: _unload(model_id),
    )


# /v1/models/download* read this global at call time (tests swap it for a temp MODEL_DIR).
admin_catalog = new_admin_catalog(MODEL_DIR, new_download_jobs())
admin_hub = mount_admin(
    app, catalog=admin_catalog, logs=lambda limit: _recent_inference_logs(limit),
    health=lambda: admin_health_snapshot(),
)


# --- Schemas ---
class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None


def _format_embedding_output(vec: list[float], fmt: Literal["float", "base64"]) -> list[float] | str:
    if fmt == "base64":
        return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")
    return vec


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float] | str
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
async def legacy_ui_redirect():
    """The management UI moved to /admin/ui. Relative, so a reverse-proxy path prefix survives."""
    return RedirectResponse("admin/ui", status_code=307)


@app.get("/v1/models/catalog")
@app.get("/models/catalog")
async def model_catalog():
    return {"object": "list", "data": MODEL_CATALOG}


def _v1_download_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Shared job ledger row -> legacy DownloadStatusResponse fields. Logs are no longer kept."""
    payload = {key: value for key, value in job.items() if key != "extra"}
    payload["logs_count"] = 0
    payload["last_log"] = None
    return payload


def _legacy_local_name(local_name: Optional[str]) -> Optional[str]:
    """Keep the old leniency for /v1 clients: trim, '\\' -> '/', strip outer slashes."""
    if local_name is None:
        return None
    cleaned = local_name.strip().replace("\\", "/").strip("/")
    return cleaned or None


@app.post("/v1/models/download", response_model=DownloadStatusResponse)
@app.post("/models/download", response_model=DownloadStatusResponse)
async def download_model(request: DownloadRequest):
    with v1_download_errors():
        job = admin_catalog.start_download(
            request.repo_id, _legacy_local_name(request.local_name), request.force
        )
    return DownloadStatusResponse(**_v1_download_payload(job))


@app.get("/v1/models/downloads")
@app.get("/models/downloads")
async def list_download_jobs():
    jobs = [
        DownloadStatusResponse(**_v1_download_payload(job)).model_dump()
        for job in admin_catalog.list_downloads()
    ]
    return {"object": "list", "data": jobs}


@app.get("/v1/models/downloads/{job_id}", response_model=DownloadStatusResponse)
@app.get("/models/downloads/{job_id}", response_model=DownloadStatusResponse)
async def get_download_job(job_id: str):
    with v1_download_errors():
        job = admin_catalog.get_download(job_id)
    return DownloadStatusResponse(**_v1_download_payload(job))


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
    status = _load_into_registry(mid, request.model_type)
    if status == "already_loaded":
        return {"status": status, "model_id": mid}
    return {"status": status, "model_id": mid, "type": request.model_type}


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
        embeddings, tokens = _encode_embeddings(request.model, inputs, request.dimensions)
        fmt = request.encoding_format
        data = [
            EmbeddingData(embedding=_format_embedding_output(emb, fmt), index=i)
            for i, emb in enumerate(embeddings)
        ]
        dimensions_actual = len(embeddings[0]) if embeddings else 0
        _append_inference_log(
            event="embedding",
            model_id=request.model,
            duration_ms=(time.time() - t0) * 1000.0,
            details={
                "inputs_count": len(inputs),
                "prompt_tokens": tokens,
                "status": "ok",
                "dimensions": request.dimensions,
                "dimensions_actual": dimensions_actual,
            },
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
            details={
                "inputs_count": len(inputs),
                "prompt_tokens": tokens,
                "status": "error",
                "error": str(exc),
                "dimensions": request.dimensions,
            },
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
        if scores:
            entry.mark_inference()
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
