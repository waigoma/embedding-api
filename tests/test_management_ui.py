"""UI contracts: reads do not load models, evidence resets, clients keep their payloads."""
import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(server, "registry", {})
    monkeypatch.setattr(server, "download_jobs", {})
    monkeypatch.setattr(server, "inference_logs", [])
    # No lifespan: these contract tests must not run configured startup preloads.
    return TestClient(server.app, raise_server_exceptions=False)


def entry(model_type="embedding"):
    model = MagicMock()
    model.prompts = {}
    model.get_sentence_embedding_dimension.return_value = 4
    model.encode.side_effect = lambda texts, **kw: np.ones((len(texts), kw.get("truncate_dim", 4)), dtype=np.float32)
    model.predict.side_effect = lambda pairs: np.arange(len(pairs), dtype=np.float32)
    return server.ModelEntry(model, model_type)


@pytest.mark.parametrize("path", ["/ui", "/webui"])
def test_ui_aliases_and_external_modules(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert 'lang="ja"' in response.text
    assert 'type="module" src="ui/assets/admin.js"' in response.text
    assert '<script>' not in response.text
    assert 'onclick=' not in response.text


@pytest.mark.parametrize("asset", sorted(server._WEBUI_ASSETS))
def test_assets_ship_and_have_correct_mime(client, asset):
    response = client.get(f"/ui/assets/{asset}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css" if asset.endswith(".css") else "text/javascript")
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("asset", ["server.py", "index.html", ".env", "admin.js.map", "%2e%2e%2fserver.py"])
def test_asset_allowlist_denies_other_files(client, asset):
    assert client.get(f"/ui/assets/{asset}").status_code == 404


def test_reads_preserve_unloaded_files_and_do_not_call_loaders(client, tmp_path, monkeypatch):
    local = tmp_path / "embedding" / "synthetic"
    local.mkdir(parents=True)
    (local / "config.json").write_text('{}')
    loader = MagicMock(side_effect=AssertionError("read loaded a model"))
    monkeypatch.setattr(server, "_load_embedding", loader)
    monkeypatch.setattr(server, "_load_reranker", loader)
    for path in ["/health", "/ui/status", "/v1/models/catalog", "/models/catalog", "/v1/models/downloads", "/models/downloads"]:
        assert client.get(path).status_code == 200
    data = client.get("/v1/models").json()["data"]
    assert data[0]["id"] == "embedding/synthetic"
    assert data[0]["loaded"] is False
    assert data[0]["type"] == "unknown"
    assert client.get("/models").json() == client.get("/v1/models").json()
    loader.assert_not_called()
    assert server.registry == {}


def test_status_reports_safe_config_and_current_load_only(client, monkeypatch):
    server.registry["synthetic"] = entry()
    monkeypatch.setattr(server, "HF_TOKEN", "synthetic-secret-token")
    monkeypatch.setattr(server, "LLM_PROXY_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(server, "LLM_PROXY_BASE_URL", "https://private.example.test")
    status = client.get("/ui/status").json()
    assert status["loaded"] == [{"id": "synthetic", "type": "embedding", "last_inference_at": None}]
    assert status["config"]["chat_proxy_configured"] is True
    assert "synthetic-secret" not in json.dumps(status)
    assert "private.example" not in json.dumps(status)
    assert "sentence_transformer_kwargs" not in status["config"]


@pytest.mark.parametrize("path", ["/v1/embeddings", "/embeddings", "/v1/responses", "/responses"])
def test_successful_embedding_marks_current_load_without_changing_response(client, path):
    loaded = entry()
    server.registry["synthetic"] = loaded
    response = client.post(path, json={"model": "synthetic", "input": ["synthetic A", "synthetic B"]})
    assert response.status_code == 200
    assert loaded.last_inference_at is not None
    if "responses" not in path:
        body = response.json()
        assert set(body) == {"object", "data", "model", "usage"}
        assert body["object"] == "list"
        assert [d["index"] for d in body["data"]] == [0, 1]
        assert len(body["data"][0]["embedding"]) == 4


def test_invalid_dimensions_does_not_mark_inference(client):
    loaded = entry()
    server.registry["synthetic"] = loaded
    response = client.post("/v1/embeddings", json={"model": "synthetic", "input": "synthetic", "dimensions": 5})
    assert response.status_code == 400
    assert loaded.last_inference_at is None


def test_empty_input_does_not_mark_inference(client):
    loaded = entry()
    server.registry["synthetic"] = loaded
    assert client.post("/v1/embeddings", json={"model": "synthetic", "input": []}).status_code == 200
    assert loaded.last_inference_at is None


def test_dimension_and_base64_contract(client):
    import base64
    server.registry["synthetic"] = entry()
    response = client.post("/v1/embeddings", json={"model": "synthetic", "input": "synthetic", "dimensions": 2, "encoding_format": "base64"})
    assert response.status_code == 200
    assert len(base64.b64decode(response.json()["data"][0]["embedding"])) == 8


def test_model_reload_resets_inference_evidence(client, monkeypatch):
    monkeypatch.setattr(server, "_load_embedding", lambda _: entry())
    assert client.post("/models/load", json={"model_id": "synthetic"}).json()["status"] == "loaded"
    client.post("/v1/embeddings", json={"model": "synthetic", "input": "synthetic"})
    assert client.get("/ui/status").json()["loaded"][0]["last_inference_at"] is not None
    assert client.post("/models/unload", json={"model_id": "synthetic"}).status_code == 200
    assert client.get("/ui/status").json()["loaded"] == []
    assert client.post("/v1/models/load", json={"model_id": "synthetic"}).status_code == 200
    assert client.get("/ui/status").json()["loaded"][0]["last_inference_at"] is None


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank"])
def test_rerank_contract_and_evidence(client, path):
    loaded = entry("reranker")
    server.registry["synthetic"] = loaded
    response = client.post(path, json={"model": "synthetic", "query": "synthetic", "documents": ["A", "B"]})
    assert response.status_code == 200
    assert response.json()["results"][0]["index"] == 1
    assert loaded.last_inference_at is not None


@pytest.mark.parametrize("path", ["/v1/models/download", "/models/download"])
def test_download_validation_and_existing_job_payload(client, tmp_path, monkeypatch, path):
    started = threading.Event()
    worker = MagicMock(side_effect=lambda *_: started.set())
    monkeypatch.setattr(server, "_run_download_job", worker)
    assert client.post(path, json={"repo_id": "invalid"}).status_code == 400
    assert client.post(path, json={"repo_id": "synthetic/model", "local_name": "../outside"}).status_code == 400
    assert server.download_jobs == {}
    response = client.post(path, json={"repo_id": "synthetic/model", "local_name": "embedding/synthetic"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["local_name"] == "embedding/synthetic"
    assert body["repo_id"] == "synthetic/model"
    assert (tmp_path / "embedding" / "synthetic").is_dir()
    assert len(server.download_jobs) == 1
    assert started.wait(1)
    worker.assert_called_once()
    assert client.get(f'/v1/models/downloads/{body["id"]}').json()["status"] == "queued"


def test_inference_error_does_not_mark_success(client):
    loaded = entry()
    loaded.model.encode.side_effect = RuntimeError("synthetic inference failure")
    server.registry["synthetic"] = loaded
    assert client.post("/v1/embeddings", json={"model": "synthetic", "input": "synthetic"}).status_code == 500
    assert loaded.last_inference_at is None
    assert client.get("/v1/logs/inference").json()["data"][0]["details"]["status"] == "error"


def test_repeated_load_is_idempotent_and_unload_is_explicit(client, monkeypatch):
    loader = MagicMock(side_effect=lambda _: entry())
    monkeypatch.setattr(server, "_load_embedding", loader)
    payload = {"model_id": "synthetic", "model_type": "embedding"}
    assert client.post("/v1/models/load", json=payload).json()["status"] == "loaded"
    assert client.post("/models/load", json=payload).json()["status"] == "already_loaded"
    loader.assert_called_once_with("synthetic")
    assert client.post("/v1/models/unload", json=payload).json()["status"] == "unloaded"
    assert client.post("/models/unload", json=payload).status_code == 404
    assert client.get("/ui/status").json()["loaded"] == []


def test_load_failure_and_invalid_type_preserve_registry(client, monkeypatch):
    loader = MagicMock(side_effect=RuntimeError("synthetic load failure"))
    monkeypatch.setattr(server, "_load_embedding", loader)
    assert client.post("/models/load", json={"model_id": "synthetic", "model_type": "unsupported"}).status_code == 400
    loader.assert_not_called()
    assert client.post("/models/load", json={"model_id": "synthetic"}).status_code == 500
    assert server.registry == {}
    assert client.get("/health").json()["loaded"] == {}


def test_existing_download_conflict_creates_no_job_or_worker(client, tmp_path, monkeypatch):
    local = tmp_path / "synthetic"
    local.mkdir()
    (local / "config.json").write_text('{}')
    worker = MagicMock()
    monkeypatch.setattr(server, "_run_download_job", worker)
    response = client.post("/v1/models/download", json={"repo_id": "synthetic/model", "local_name": "synthetic"})
    assert response.status_code == 409
    assert "force=true" in response.json()["detail"]
    assert server.download_jobs == {}
    worker.assert_not_called()


def test_failed_job_reads_preserve_unknown_progress_and_error(client):
    server.download_jobs["synthetic"] = {
        "id": "synthetic", "repo_id": "synthetic/model", "local_name": "synthetic",
        "target_dir": "/synthetic", "status": "failed", "created_at": 1,
        "progress_percent": None, "total_bytes": None, "error": "synthetic timeout",
        "logs": [{"message": "retry required"}],
    }
    job = client.get("/models/downloads/synthetic").json()
    assert job["status"] == "failed"
    assert job["progress_percent"] is None
    assert job["total_bytes"] is None
    assert job["error"] == "synthetic timeout"
    assert job["last_log"]["message"] == "retry required"
    assert "logs" not in job
    assert client.get("/v1/models/downloads/missing").status_code == 404
