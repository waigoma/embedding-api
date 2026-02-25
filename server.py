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
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-server")

app = FastAPI(title="Embedding Server", version="2.0.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_TTL = int(os.environ.get("IDLE_TTL", "0"))  # seconds, 0 = disabled
AUTO_LOAD = os.environ.get("AUTO_LOAD", "1") == "1"
PRELOAD_EMBEDDING = os.environ.get("PRELOAD_EMBEDDING", "").split(",")
PRELOAD_RERANKER = os.environ.get("PRELOAD_RERANKER", "").split(",")


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


def _resolve_path(model_id: str) -> str:
    local = os.path.join(MODEL_DIR, model_id)
    return local if os.path.isdir(local) else model_id


def _load_embedding(model_id: str) -> ModelEntry:
    from sentence_transformers import SentenceTransformer
    path = _resolve_path(model_id)
    logger.info(f"Loading embedding: {model_id} from {path}")
    model = SentenceTransformer(path, device=DEVICE)
    logger.info(f"Loaded embedding: {model_id} (dim={model.get_sentence_embedding_dimension()})")
    return ModelEntry(model, "embedding")


def _load_reranker(model_id: str) -> ModelEntry:
    from sentence_transformers import CrossEncoder
    path = _resolve_path(model_id)
    logger.info(f"Loading reranker: {model_id} from {path}")
    model = CrossEncoder(path, device=DEVICE)
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
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
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
        "gpu": gpu,
        "idle_ttl": IDLE_TTL,
        "auto_load": AUTO_LOAD,
        "loaded": {k: v.model_type for k, v in registry.items()},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7997)
