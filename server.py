"""
OpenAI-compatible Embedding & Reranking API Server
- On-demand model loading / unloading
- Auto-load on first request (AUTO_LOAD=1)
- Idle TTL auto-unload (IDLE_TTL=300 etc.)
- Preload specific models on startup (PRELOAD_EMBEDDING / PRELOAD_RERANKER)
"""

import os
import time
import threading
import logging
import json
import uuid
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

app = FastAPI(title="Embedding Server", version="2.0.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
WEBUI_DIR = os.path.join(os.path.dirname(__file__), "webui")
DEVICE_MODE = os.environ.get("DEVICE_MODE", "auto").strip().lower()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
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


def _resolve_path(model_id: str) -> str:
    local = os.path.join(MODEL_DIR, model_id)
    return local if os.path.isdir(local) else model_id


def _default_local_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def _sanitize_local_name(local_name: str) -> str:
    cleaned = local_name.strip().replace("\\", "/")
    if not cleaned:
        raise HTTPException(400, "local_name is empty")
    if "/" in cleaned or cleaned in {".", ".."}:
        raise HTTPException(400, "local_name must be a single directory name")
    return cleaned


def _set_download_status(job_id: str, **updates: Any) -> None:
    with download_lock:
        job = download_jobs.get(job_id)
        if not job:
            return
        job.update(updates)


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

    _set_download_status(job_id, status="downloading", started_at=time.time())
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN,
        )
        _set_download_status(job_id, status="completed", finished_at=time.time())
    except Exception as exc:
        hint = ""
        if "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
            hint = " (set HF_TOKEN for private/gated models)"
        _set_download_status(
            job_id,
            status="failed",
            error=f"{exc}{hint}",
            finished_at=time.time(),
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

    if AUTO_LOAD and os.path.isdir(os.path.join(MODEL_DIR, model_id)):
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


@app.on_event("startup")
async def startup():
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
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
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
        for name in sorted(os.listdir(MODEL_DIR)):
            if os.path.isdir(os.path.join(MODEL_DIR, name)) and name not in loaded_ids:
                models.append(ModelInfo(id=name, type="unknown", loaded=False))
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
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    with download_lock:
        download_jobs[job_id] = job

    threading.Thread(
        target=_run_download_job,
        args=(job_id, repo_id, target_dir),
        daemon=True,
    ).start()
    return DownloadStatusResponse(**job)


@app.get("/v1/models/downloads")
@app.get("/models/downloads")
async def list_download_jobs():
    with download_lock:
        jobs = list(download_jobs.values())
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"object": "list", "data": jobs}


@app.get("/v1/models/downloads/{job_id}", response_model=DownloadStatusResponse)
@app.get("/models/downloads/{job_id}", response_model=DownloadStatusResponse)
async def get_download_job(job_id: str):
    with download_lock:
        job = download_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"download job not found: {job_id}")
    return DownloadStatusResponse(**job)


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
    entry = _get_model(request.model, "embedding")
    inputs = request.input if isinstance(request.input, list) else [request.input]
    embeddings = entry.model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True)
    data = [EmbeddingData(embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]
    tokens = sum(len(s) // 4 for s in inputs)
    return EmbeddingResponse(
        data=data, model=request.model,
        usage={"prompt_tokens": tokens, "total_tokens": tokens},
    )


@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
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
    tokens = sum(len(d) // 4 for d in request.documents)
    return RerankResponse(
        results=results, model=request.model,
        usage={"prompt_tokens": tokens, "total_tokens": tokens},
    )


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
