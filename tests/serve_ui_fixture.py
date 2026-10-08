"""Loopback-only browser fixture for the admin UI. No weights, external downloads, or production calls.

Runs the real server.app (admin routers, SSE hub, /v1 API) with stubbed model
loaders and a fake download runner. Never use it as a production entrypoint.
"""
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
os.environ["DOWNLOAD_PROGRESS_INTERVAL_SEC"] = "0.2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = MagicMock()
torch.cuda.is_available.return_value = False
sys.modules["torch"] = torch

from admin_shell.admin.download_jobs import DownloadFailed  # noqa: E402
from admin_shell.admin.hf_download import HfSnapshotDownloader  # noqa: E402


def fake_run(self, repo_id, target_dir, report):
    """Synthetic downloader: 'fail' repos fail, others write a config after a short progress."""
    report({"total_bytes": 1024**3})
    time.sleep(0.3)
    report({"downloaded_bytes": 384 * 1024**2, "progress_percent": 37.5, "speed_mbps": 18.4})
    time.sleep(0.3)
    if "fail" in repo_id:
        raise DownloadFailed("synthetic network timeout · retry available")
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    (Path(target_dir) / "config.json").write_text('{"architectures": ["BertModel"]}')


HfSnapshotDownloader.run = fake_run  # fixture process only

import server  # noqa: E402


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
    for name, arch in [
        ("embedding/ruri-v3-310m", "ModernBertModel"),
        ("embedding/Qwen3-Embedding-0.6B", "Qwen3Model"),
        ("embedding/synthetic-reranker", "XLMRobertaForSequenceClassification"),
        ("embedding/<img onerror=alert(1)>", "BertModel"),
        # Outside ADMIN_MODEL_ROOTS (default "embedding"): the overview must not list it.
        ("stt/onnx-whisper", "WhisperModel"),
    ]:
        path = Path(model_dir.name) / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text('{"architectures": ["%s"]}' % arch)
    server.registry["embedding/ruri-v3-310m"] = fake_entry()
    # A failed job over an existing directory: the admin load must refuse it.
    server.admin_catalog.start_download("Qwen/fail-reranker", "embedding/synthetic-reranker", True)


server._startup_once = lambda: None
server._load_embedding = lambda _: fake_entry()
server._load_reranker = lambda _: fake_entry("reranker")
server.DEVICE = "cuda"
server.ACCELERATOR_BACKEND = "cuda"
torch.cuda.mem_get_info.return_value = (2377 * 1024**2, 15850 * 1024**2)
torch.cuda.get_device_name.return_value = "Synthetic GPU · UI fixture"
setup()

outer = FastAPI(lifespan=server.lifespan)
outer.mount("/prefix", server.app)
outer.mount("/", server.app)
if __name__ == "__main__":
    uvicorn.run(outer, host="127.0.0.1", port=19975)
