"""Loopback-only browser fixture. No weights, external downloads, or production calls."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
import uvicorn

model_dir = tempfile.TemporaryDirectory(prefix="embedding-ui-fixture-")
os.environ["MODEL_DIR"] = model_dir.name
os.environ["DEVICE_MODE"] = "cpu"
os.environ["PRELOAD_EMBEDDING"] = ""
os.environ["PRELOAD_RERANKER"] = ""
os.environ["IDLE_TTL"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = MagicMock()
torch.cuda.is_available.return_value = False
sys.modules["torch"] = torch
import server
local_model_ids = server._iter_local_model_relative_ids
catalog = list(server.MODEL_CATALOG)


def fake_entry(model_type="embedding"):
    model = MagicMock()
    model.prompts = {}
    model.get_sentence_embedding_dimension.return_value = 8
    model.encode.side_effect = lambda texts, **kw: np.array([
        [0.8, 0.6, 0.4, 0.2, 0.1, -0.1, 0.05, 0.02] if i == 0 else
        [0.7, 0.6, 0.5, 0.3, 0.1, -0.2, 0.04, 0.01]
        for i, _ in enumerate(texts)
    ], dtype=np.float32)[:, :kw.get("truncate_dim", 8)]
    return server.ModelEntry(model, model_type)


def setup():
    server._iter_local_model_relative_ids = local_model_ids
    server.MODEL_CATALOG = list(catalog)
    server.registry.clear()
    server.download_jobs.clear()
    server.inference_logs.clear()
    for name in ["embedding/ruri-v3-310m", "Qwen3-Embedding-0.6B", "reranker/synthetic", "synthetic/<img onerror=alert(1)>"]:
        path = Path(model_dir.name) / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text('{}')
    server.registry["embedding/ruri-v3-310m"] = fake_entry()
    now = time.time()
    for i, (repo, local, status) in enumerate([
        ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding-4B", "downloading"),
        ("Qwen/Qwen3-Reranker-0.6B", "reranker/synthetic", "failed"),
        ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B", "completed"),
    ]):
        server.download_jobs[str(i)] = {
            "id": str(i), "repo_id": repo, "local_name": local,
            "target_dir": str(Path(model_dir.name) / local), "status": status,
            "created_at": now - i, "progress_percent": 37.5 if status == "downloading" else None,
            "downloaded_bytes": 384 * 1024**2, "total_bytes": 1024**3 if status != "failed" else None,
            "speed_mbps": 18.4, "error": "synthetic network timeout · retry available" if status == "failed" else None,
            "logs": [{"message": "synthetic fixture: no files downloaded"}],
        }


server._startup_once = lambda: None
server._load_embedding = lambda _: fake_entry()
server._load_reranker = lambda _: fake_entry("reranker")
server._run_download_job = lambda job_id, *_: server._set_download_status(job_id, status="completed")
server.DEVICE = "cuda"
server.ACCELERATOR_BACKEND = "cuda"
torch.cuda.mem_get_info.return_value = (2377 * 1024**2, 15850 * 1024**2)
torch.cuda.get_device_name.return_value = "Synthetic GPU · UI fixture"
setup()


@server.app.post("/fixture/reset")
async def reset():
    setup()
    return {"ok": True}


@server.app.post("/fixture/empty")
async def empty():
    server.registry.clear()
    server.download_jobs.clear()
    server.inference_logs.clear()
    server._iter_local_model_relative_ids = lambda: []
    server.MODEL_CATALOG = []
    return {"ok": True}


outer = FastAPI()
outer.mount("/prefix", server.app)
outer.mount("/", server.app)
if __name__ == "__main__":
    uvicorn.run(outer, host="127.0.0.1", port=19975)
