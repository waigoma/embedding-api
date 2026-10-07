"""API contracts behind the admin UI: reads do not load models, evidence resets, clients keep payloads."""
import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
import server
from admin_shell.admin.download_jobs import DownloadFailed, DownloadJobRegistry


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(server, "registry", {})
    monkeypatch.setattr(server, "inference_logs", [])
    # /v1/models/download* use the shared catalog; swap in one over tmp_path whose
    # ledger never starts a real download (tests opt in through `spawned`).
    spawned.clear()
    jobs = DownloadJobRegistry(MagicMock(), spawn=spawned.append)
    monkeypatch.setattr(server, "admin_catalog", server.new_admin_catalog(str(tmp_path), jobs))
    # No lifespan: these contract tests must not run configured startup preloads.
    return TestClient(server.app, raise_server_exceptions=False)


spawned = []


def entry(model_type="embedding"):
    model = MagicMock()
    model.prompts = {}
    model.get_sentence_embedding_dimension.return_value = 4
    model.encode.side_effect = lambda texts, **kw: np.ones((len(texts), kw.get("truncate_dim", 4)), dtype=np.float32)
    model.predict.side_effect = lambda pairs: np.arange(len(pairs), dtype=np.float32)
    return server.ModelEntry(model, model_type)


def test_reads_preserve_unloaded_files_and_do_not_call_loaders(client, tmp_path, monkeypatch):
    local = tmp_path / "embedding" / "synthetic"
    local.mkdir(parents=True)
    (local / "config.json").write_text('{}')
    loader = MagicMock(side_effect=AssertionError("read loaded a model"))
    monkeypatch.setattr(server, "_load_embedding", loader)
    monkeypatch.setattr(server, "_load_reranker", loader)
    for path in ["/health", "/admin/health", "/admin/interactions", "/v1/models/catalog", "/models/catalog", "/v1/models/downloads", "/models/downloads"]:
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
    status = client.get("/admin/health").json()
    assert status["loaded_models"] == [{"id": "synthetic", "type": "embedding", "last_inference_at": None}]
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
    assert client.get("/admin/health").json()["loaded_models"][0]["last_inference_at"] is not None
    assert client.post("/models/unload", json={"model_id": "synthetic"}).status_code == 200
    assert client.get("/admin/health").json()["loaded_models"] == []
    assert client.post("/v1/models/load", json={"model_id": "synthetic"}).status_code == 200
    assert client.get("/admin/health").json()["loaded_models"][0]["last_inference_at"] is None


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank"])
def test_rerank_contract_and_evidence(client, path):
    loaded = entry("reranker")
    server.registry["synthetic"] = loaded
    response = client.post(path, json={"model": "synthetic", "query": "synthetic", "documents": ["A", "B"]})
    assert response.status_code == 200
    assert response.json()["results"][0]["index"] == 1
    assert loaded.last_inference_at is not None


@pytest.mark.parametrize("path", ["/v1/models/download", "/models/download"])
def test_download_validation_and_existing_job_payload(client, tmp_path, path):
    assert client.post(path, json={"repo_id": "invalid"}).status_code == 400
    assert client.post(path, json={"repo_id": "synthetic/model", "local_name": "../outside"}).status_code == 400
    assert server.admin_catalog.list_downloads() == []
    # Legacy leniency: surrounding slashes are trimmed before validation.
    response = client.post(path, json={"repo_id": "synthetic/model", "local_name": "/embedding/synthetic/"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["local_name"] == "embedding/synthetic"
    assert body["repo_id"] == "synthetic/model"
    assert body["logs_count"] == 0 and body["last_log"] is None
    assert set(body) == set(server.DownloadStatusResponse.model_fields)
    assert (tmp_path / "embedding" / "synthetic").is_dir()
    assert len(server.admin_catalog.list_downloads()) == 1
    assert len(spawned) == 1
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
    assert client.get("/admin/health").json()["loaded_models"] == []


def test_load_failure_and_invalid_type_preserve_registry(client, monkeypatch):
    loader = MagicMock(side_effect=RuntimeError("synthetic load failure"))
    monkeypatch.setattr(server, "_load_embedding", loader)
    assert client.post("/models/load", json={"model_id": "synthetic", "model_type": "unsupported"}).status_code == 400
    loader.assert_not_called()
    assert client.post("/models/load", json={"model_id": "synthetic"}).status_code == 500
    assert server.registry == {}
    assert client.get("/health").json()["loaded"] == {}


def test_existing_download_conflict_creates_no_job_or_worker(client, tmp_path):
    local = tmp_path / "synthetic"
    local.mkdir()
    (local / "config.json").write_text('{}')
    response = client.post("/v1/models/download", json={"repo_id": "synthetic/model", "local_name": "synthetic"})
    assert response.status_code == 409
    assert "force=true" in response.json()["detail"]
    assert server.admin_catalog.list_downloads() == []
    assert spawned == []


def test_failed_job_reads_preserve_unknown_progress_and_error(client, tmp_path, monkeypatch):
    runner = MagicMock()
    runner.run.side_effect = DownloadFailed("synthetic timeout")
    jobs = DownloadJobRegistry(runner, spawn=lambda run: run())
    monkeypatch.setattr(server, "admin_catalog", server.new_admin_catalog(str(tmp_path), jobs))
    job_id = client.post("/v1/models/download", json={"repo_id": "synthetic/model"}).json()["id"]
    job = client.get(f"/models/downloads/{job_id}").json()
    assert job["status"] == "failed"
    assert job["progress_percent"] is None
    assert job["total_bytes"] is None
    assert job["error"] == "synthetic timeout"
    assert "logs" not in job and "extra" not in job
    assert client.get("/v1/models/downloads").json()["data"][0]["id"] == job_id
    assert client.get("/v1/models/downloads/missing").status_code == 404
