"""Admin shell + model catalog wiring in server.py (torch mocked, no models loaded).

Run inside the container (or any env with fastapi + httpx):
    python -m pytest tests/test_admin.py -v
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False
_torch_mock.version.hip = None
_torch_mock.version.cuda = None
sys.modules.setdefault("torch", _torch_mock)

_MODEL_DIR = tempfile.mkdtemp(prefix="embedding-admin-test-")
os.environ["MODEL_DIR"] = _MODEL_DIR
os.makedirs(os.path.join(_MODEL_DIR, "embedding", "demo"), exist_ok=True)
with open(os.path.join(_MODEL_DIR, "embedding", "demo", "config.json"), "w", encoding="utf-8") as fh:
    fh.write('{"architectures": ["BertModel"]}')
os.makedirs(os.path.join(_MODEL_DIR, "reranker", "demo-rr"), exist_ok=True)
with open(os.path.join(_MODEL_DIR, "reranker", "demo-rr", "config.json"), "w", encoding="utf-8") as fh:
    fh.write('{"architectures": ["XLMRobertaForSequenceClassification"]}')
os.makedirs(os.path.join(_MODEL_DIR, ".cache", "x"), exist_ok=True)
with open(os.path.join(_MODEL_DIR, ".cache", "x", "config.json"), "w", encoding="utf-8") as fh:
    fh.write("{}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


class TestAdminWiring(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_ui_is_unauthenticated_and_legacy_urls_redirect(self):
        page = self.client.get("/admin/ui")
        self.assertEqual(page.status_code, 200)
        self.assertIn('data-service="embedding-api"', page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        for legacy in ("/ui", "/webui"):
            response = self.client.get(legacy, follow_redirects=False)
            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "/admin/ui")
        self.assertEqual(self.client.get("/admin/assets/core/table.js").status_code, 200)
        self.assertEqual(self.client.get("/admin/assets/../server.py").status_code, 404)

    def test_catalog_lists_models_with_guessed_type(self):
        listing = self.client.get("/admin/models").json()
        self.assertEqual(listing["capabilities"], {"download": True, "load": True, "unload": True})
        self.assertEqual(listing["columns"], [{"key": "type", "label": "type", "kind": "text"}])
        rows = {row["local_name"]: row for row in listing["items"]}
        self.assertEqual(set(rows), {"embedding/demo", "reranker/demo-rr"})
        self.assertEqual(rows["embedding/demo"]["extra"]["type"], "embedding")
        self.assertEqual(rows["reranker/demo-rr"]["extra"]["type"], "reranker")
        self.assertFalse(rows["embedding/demo"]["loaded"])

    def test_v1_models_still_lists_local_models(self):
        ids = {m["id"] for m in self.client.get("/v1/models").json()["data"]}
        self.assertIn("embedding/demo", ids)

    def test_download_validation_goes_through_catalog(self):
        self.assertEqual(self.client.post("/v1/models/download", json={"repo_id": "nope"}).status_code, 400)
        self.assertEqual(self.client.post("/v1/models/download", json={"repo_id": "a/b", "local_name": "../x"}).status_code, 400)
        self.assertEqual(self.client.post("/v1/models/download", json={"repo_id": "a/demo", "local_name": "embedding/demo"}).status_code, 409)
        self.assertEqual(self.client.get("/v1/models/downloads").json(), {"object": "list", "data": []})
        self.assertEqual(self.client.get("/v1/models/downloads/zzz").status_code, 404)

    def test_unload_of_not_loaded_model_is_409_and_health_is_served(self):
        self.assertEqual(self.client.post("/admin/models/embedding%2Fdemo/unload").status_code, 409)
        self.assertEqual(self.client.post("/admin/models/embedding/demo/unload").status_code, 409)
        health = self.client.get("/admin/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["loaded"], {})
        logs = self.client.get("/admin/interactions?limit=5").json()
        self.assertEqual([c["key"] for c in logs["columns"]][:2], ["timestamp", "event"])


if __name__ == "__main__":
    unittest.main()
