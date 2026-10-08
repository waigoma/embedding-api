"""Admin shell wiring for embedding-api (torch mocked, no model loaded, no network).

Run with any env that has fastapi + httpx:
    python -m pytest tests/test_admin.py -v
    python -m unittest tests.test_admin -v   (from the repository root)
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urljoin

_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False
_torch_mock.version.hip = None
_torch_mock.version.cuda = None
sys.modules.setdefault("torch", _torch_mock)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import httpx  # noqa: F401  (TestClient needs it)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - dependency-light environments
    TestClient = None

import server  # noqa: E402
import embedding_admin  # noqa: E402
import embedding_overview  # noqa: E402
from admin_shell.admin.download_jobs import DownloadFailed, DownloadJobRegistry  # noqa: E402
from admin_shell.admin.model_catalog import ModelNotLoadable  # noqa: E402


def write_model(root: Path, relative: str, architectures=None, filename="config.json") -> Path:
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"architectures": architectures} if architectures is not None else {}
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")
    return directory


class _TempModelDir(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="embedding-admin-test-")
        self.addCleanup(temp.cleanup)
        self.model_dir = Path(temp.name)
        self.spawned = []

    def ledger(self, runner=None, spawn=None):
        return DownloadJobRegistry(runner or MagicMock(), spawn=spawn or self.spawned.append)


@unittest.skipIf(TestClient is None, "fastapi/httpx not installed")
class TestServerAdminRoutes(_TempModelDir):
    """Routes on server.app: UI delivery, redirects, legacy /v1 download contract, health."""

    def setUp(self):
        super().setUp()
        catalog = server.new_admin_catalog(str(self.model_dir), self.ledger())
        for target, value in (
            ("admin_catalog", catalog),
            ("MODEL_DIR", str(self.model_dir)),
            ("registry", {}),
            ("inference_logs", []),
        ):
            patcher = patch.object(server, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # No lifespan: preloads and the SSE hub stay off.
        self.client = TestClient(server.app, raise_server_exceptions=False)

    def test_admin_ui_is_served_without_authentication(self):
        page = self.client.get("/admin/ui")
        self.assertEqual(page.status_code, 200)
        self.assertIn('data-service="embedding-api"', page.text)
        self.assertIn("<title>embedding-api Admin</title>", page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])

    def test_legacy_ui_urls_redirect_relative_to_the_admin_ui(self):
        for legacy in ("/ui", "/webui"):
            response = self.client.get(legacy, follow_redirects=False)
            self.assertEqual(response.status_code, 307)
            self.assertEqual(urljoin("http://testserver" + legacy, response.headers["location"]),
                             "http://testserver/admin/ui")
            # A reverse-proxy prefix is preserved because the Location is relative.
            self.assertEqual(urljoin("https://host/prefix" + legacy, response.headers["location"]),
                             "https://host/prefix/admin/ui")
        self.assertEqual(self.client.get("/ui").status_code, 200)

    def test_assets_ship_and_traversal_is_404(self):
        for asset, media in (
            ("shell.js", "text/javascript"),
            ("core/table.js", "text/javascript"),
            ("screens/model_overview.js", "text/javascript"),
            ("screens/playground.js", "text/javascript"),
            ("screens/health.js", "text/javascript"),
            ("service/v1.js", "text/javascript"),
            ("admin.css", "text/css"),
        ):
            response = self.client.get(f"/admin/assets/{asset}")
            self.assertEqual(response.status_code, 200, asset)
            self.assertTrue(response.headers["content-type"].startswith(media), asset)
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        for bad in ("../server.py", "%2e%2e/embedding_admin.py", "..%2fserver.py", ".hidden.js",
                    "missing.js", "core/../../server.py", "admin.css.map"):
            self.assertEqual(self.client.get(f"/admin/assets/{bad}").status_code, 404, bad)
        # The previous UI's routes are gone.
        self.assertEqual(self.client.get("/ui/assets/admin.js").status_code, 404)
        self.assertEqual(self.client.get("/ui/status").status_code, 404)

    def test_v1_download_validation_goes_through_the_catalog(self):
        write_model(self.model_dir, "embedding/demo", ["BertModel"])
        cases = [
            ({"repo_id": "nope"}, 400),
            ({"repo_id": "a/b/c"}, 400),
            ({"repo_id": "a/b", "local_name": "../x"}, 400),
            ({"repo_id": "a/b", "local_name": "x/.hidden"}, 400),
            ({"repo_id": "a/demo", "local_name": "embedding/demo"}, 409),
        ]
        for body, expected in cases:
            response = self.client.post("/v1/models/download", json=body)
            self.assertEqual(response.status_code, expected, body)
        self.assertEqual(self.spawned, [])
        ok = self.client.post("/models/download", json={"repo_id": "a/demo", "local_name": "embedding/demo", "force": True})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["status"], "queued")
        self.assertEqual(len(self.spawned), 1)

    def test_v1_download_concurrency_limit_is_429(self):
        for index in range(3):
            self.assertEqual(self.client.post("/v1/models/download", json={"repo_id": f"o/m{index}"}).status_code, 200)
        self.assertEqual(self.client.post("/v1/models/download", json={"repo_id": "o/m9"}).status_code, 429)

    def test_v1_downloads_list_shape_when_empty_and_unknown_job(self):
        self.assertEqual(self.client.get("/v1/models/downloads").json(), {"object": "list", "data": []})
        self.assertEqual(self.client.get("/models/downloads").json(), {"object": "list", "data": []})
        self.assertEqual(self.client.get("/v1/models/downloads/zzz").status_code, 404)

    def test_v1_models_lists_local_roots_and_skips_caches(self):
        write_model(self.model_dir, "embedding/demo", ["BertModel"])
        write_model(self.model_dir, "embedding/demo/nested", ["BertModel"])
        write_model(self.model_dir, ".cache/x")
        data = self.client.get("/v1/models").json()["data"]
        self.assertEqual([(m["id"], m["type"], m["loaded"]) for m in data], [("embedding/demo", "unknown", False)])

    def test_admin_health_merges_health_config_and_inference_evidence(self):
        entry = server.ModelEntry(MagicMock(), "embedding")
        server.registry["embedding/demo"] = entry
        health = self.client.get("/admin/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["loaded"], {"embedding/demo": "embedding"})
        self.assertEqual(health["loaded_models"], [{"id": "embedding/demo", "type": "embedding", "last_inference_at": None}])
        self.assertEqual(
            set(health["config"]),
            {"model_dir", "device_mode", "auto_load", "idle_ttl", "preload_embedding", "preload_reranker", "chat_proxy_configured"},
        )
        plain = self.client.get("/health").json()
        self.assertNotIn("config", plain)  # the public /health payload is unchanged

    def test_hub_providers_cover_the_ui_feed_keys(self):
        self.assertEqual(sorted(server.admin_hub.keys), ["health", "interactions", "models", "overview"])
        snapshot = asyncio.run(server.admin_hub.build_snapshot())
        self.assertEqual(snapshot["health"]["status"], "ok")
        self.assertEqual([c["key"] for c in snapshot["interactions"]["columns"]],
                         ["timestamp", "event", "model", "status", "duration_ms"])


@unittest.skipIf(TestClient is None, "fastapi/httpx not installed")
class TestEmbeddingAdminComposition(_TempModelDir):
    """mount_admin + build_catalog on a fresh app over a temporary MODEL_DIR."""

    def setUp(self):
        super().setUp()
        self.loaded = {}
        self.load_calls = []
        self.logs = []
        self.jobs = self.ledger()
        self.catalog_entries = [
            {"repo_id": "Qwen/Qwen3-Embedding-0.6B", "family": "Qwen3-Embedding", "type": "embedding"},
            {"repo_id": "Qwen/Qwen3-Reranker-0.6B", "family": "Qwen3-Reranker", "type": "reranker"},
            {"repo_id": "cl-nagoya/ruri-v3-310m", "family": "Ruri", "type": "embedding"},
        ]
        catalog = embedding_admin.build_catalog(
            self.model_dir,
            jobs=self.jobs,
            loaded_names=lambda: set(self.loaded),
            loader=self._load,
            unloader=lambda name: self.loaded.pop(name, None) is not None,
        )
        overview = self.build_overview(self.jobs)

        async def health():
            return {"status": "ok", "loaded": dict(self.loaded)}

        app = FastAPI()
        self.hub = embedding_admin.mount_admin(app, catalog=catalog, overview=overview, logs=lambda limit: self.logs[-limit:], health=health)
        self.client = TestClient(app, raise_server_exceptions=False)

    def build_overview(self, jobs, roots="embedding"):
        return embedding_overview.build_overview(
            self.model_dir,
            jobs=jobs,
            loaded_names=lambda: set(self.loaded),
            loader=self._load,
            unloader=lambda name: self.loaded.pop(name, None) is not None,
            catalog_entries=lambda: self.catalog_entries,
            roots=embedding_overview.parse_roots(roots, self.model_dir),
        )

    def _load(self, name, model_type):
        self.load_calls.append((name, model_type))
        self.loaded[name] = model_type

    def test_catalog_lists_model_roots_with_guessed_type(self):
        write_model(self.model_dir, "embedding/demo", ["BertModel"])
        write_model(self.model_dir, "reranker/demo-rr", ["XLMRobertaForSequenceClassification"])
        write_model(self.model_dir, "Qwen3-Reranker-0.6B", ["Qwen3ForCausalLM"])
        write_model(self.model_dir, "adapters/lora", filename="adapter_config.json")
        write_model(self.model_dir, ".cache/x")
        write_model(self.model_dir, "embedding/demo/1_Pooling")
        listing = self.client.get("/admin/models").json()
        self.assertEqual(listing["capabilities"], {"download": True, "load": True, "unload": True, "cancel": False})
        self.assertEqual(listing["columns"], [{"key": "type", "label": "type", "kind": "text"}])
        rows = {row["local_name"]: row for row in listing["items"]}
        self.assertEqual(set(rows), {"embedding/demo", "reranker/demo-rr", "Qwen3-Reranker-0.6B", "adapters/lora"})
        self.assertEqual(rows["embedding/demo"]["extra"]["type"], "embedding")
        self.assertEqual(rows["reranker/demo-rr"]["extra"]["type"], "reranker")
        self.assertEqual(rows["Qwen3-Reranker-0.6B"]["extra"]["type"], "reranker")
        self.assertEqual(rows["adapters/lora"]["extra"]["type"], "embedding")
        self.assertTrue(all(row["status"] == "installed" and row["loaded"] is False for row in rows.values()))
        self.assertGreater(rows["embedding/demo"]["size_bytes"], 0)
        # cancel capability is off, so the cancel route is not registered.
        self.assertEqual(self.client.post("/admin/models/downloads/x/cancel").status_code, 404)

    def test_load_uses_the_guessed_type_and_unload_of_not_loaded_is_409(self):
        write_model(self.model_dir, "reranker/demo-rr", ["XLMRobertaForSequenceClassification"])
        self.assertEqual(self.client.post("/admin/models/reranker/demo-rr/unload").status_code, 409)
        response = self.client.post("/admin/models/reranker/demo-rr/load")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.load_calls, [("reranker/demo-rr", "reranker")])
        self.assertTrue(self.client.get("/admin/models").json()["items"][0]["loaded"])
        self.assertEqual(self.client.post("/admin/models/reranker/demo-rr/load").status_code, 200)
        self.assertEqual(len(self.load_calls), 1)  # already loaded: no second load
        self.assertEqual(self.client.post("/admin/models/reranker/demo-rr/unload").status_code, 200)
        self.assertEqual(self.loaded, {})

    def test_load_refuses_unknown_names_and_partial_downloads(self):
        self.assertEqual(self.client.post("/admin/models/not/on-disk/load").status_code, 409)
        write_model(self.model_dir, "partial", ["BertModel"])
        runner = MagicMock()
        runner.run.side_effect = DownloadFailed("synthetic network error")
        failing = DownloadJobRegistry(runner, spawn=lambda run: run())
        state = embedding_admin.RegistryLoadState(
            loaded_names=lambda: set(), loader=self._load, unloader=lambda name: False,
            scanner=embedding_admin.EmbeddingModelScanner(self.model_dir), jobs=failing,
        )
        failing.start("o/partial", "partial", self.model_dir / "partial")
        with self.assertRaisesRegex(ModelNotLoadable, "files may be partial"):
            asyncio.run(state.load("partial"))
        self.assertEqual(self.load_calls, [])

    def test_loader_failures_are_409_with_the_message(self):
        write_model(self.model_dir, "embedding/demo", ["BertModel"])

        def broken(name, model_type):
            raise RuntimeError("synthetic VRAM unavailable")

        async def health():
            return {}

        catalog = embedding_admin.build_catalog(
            self.model_dir, jobs=self.jobs, loaded_names=set, loader=broken, unloader=lambda name: False,
        )
        app = FastAPI()
        embedding_admin.mount_admin(
            app, catalog=catalog, overview=self.build_overview(self.jobs), logs=lambda limit: [], health=health
        )
        response = TestClient(app).post("/admin/models/embedding/demo/load")
        self.assertEqual(response.status_code, 409)
        self.assertIn("synthetic VRAM unavailable", response.json()["detail"])

    def test_admin_download_and_downloads_routes(self):
        response = self.client.post("/admin/models/download", json={"repo_id": "o/new-model", "local_name": "embedding/new"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["local_name"], "embedding/new")
        items = self.client.get("/admin/models/downloads").json()["items"]
        self.assertEqual([job["status"] for job in items], ["queued"])
        row = next(item for item in self.client.get("/admin/models").json()["items"] if item["local_name"] == "embedding/new")
        self.assertEqual(row["status"], "queued")

    # ----- model overview -----

    def overview(self):
        response = self.client.get("/admin/models/overview")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        return data, {item["id"]: item for item in data["items"]}

    @staticmethod
    def actions(item):
        return {action["id"]: action for action in item["actions"]}

    def test_overview_envelope_and_catalog_names_under_the_first_root(self):
        data, items = self.overview()
        self.assertEqual(data["version_label"], "種別")
        self.assertEqual([step["key"] for step in data["steps"]], ["downloaded", "loaded"])
        self.assertEqual(data["fetch"], {
            "name_prefix": "embedding/", "api_name": True, "fields": [],
            "request": {"method": "POST", "path": "models/download"},
        })
        self.assertEqual(data["roots"], [str((self.model_dir / "embedding").resolve())])
        ruri = items["embedding/ruri-v3-310m"]
        self.assertEqual((ruri["title"], ruri["family"], ruri["version"]), ("ruri-v3-310m", "Ruri", "埋め込み"))
        self.assertEqual((ruri["local_name"], ruri["api_name"]), ("embedding/ruri-v3-310m", "embedding/ruri-v3-310m"))
        self.assertEqual(ruri["source"], "https://huggingface.co/cl-nagoya/ruri-v3-310m")
        self.assertEqual(items["embedding/Qwen3-Reranker-0.6B"]["version"], "リランカー")
        self.assertNotIn("claims", ruri)

    def test_overview_catalog_item_follows_download_load_unload_states(self):
        name = "embedding/ruri-v3-310m"
        _, items = self.overview()
        acts = self.actions(items[name])
        self.assertEqual(items[name]["lifecycle"], {"downloaded": False, "loaded": False})
        self.assertTrue(acts["download"]["enabled"] and acts["download"]["primary"])
        self.assertEqual(acts["download"]["request"], {
            "method": "POST", "path": "models/download",
            "body": {"repo_id": "cl-nagoya/ruri-v3-310m", "local_name": name, "force": False},
        })
        self.assertFalse(acts["load"]["enabled"])
        self.assertIn("取得", acts["load"]["reason"])

        write_model(self.model_dir, name, ["ModernBertModel"])
        _, items = self.overview()
        acts = self.actions(items[name])
        self.assertEqual(items[name]["lifecycle"], {"downloaded": True, "loaded": False})
        self.assertGreater(items[name]["size_bytes"], 0)
        self.assertFalse(acts["download"]["enabled"])
        self.assertFalse(acts["download"]["primary"])
        self.assertTrue(acts["load"]["enabled"] and acts["load"]["primary"])
        self.assertEqual(acts["load"]["request"], {"method": "POST", "path": "models/embedding/ruri-v3-310m/load"})
        self.assertFalse(acts["unload"]["enabled"])
        self.assertEqual(acts["unload"]["tone"], "danger")

        self.loaded[name] = "embedding"
        _, items = self.overview()
        acts = self.actions(items[name])
        self.assertTrue(items[name]["lifecycle"]["loaded"])
        self.assertFalse(acts["load"]["primary"])
        self.assertTrue(acts["unload"]["enabled"] and acts["unload"]["primary"])
        self.assertEqual(acts["unload"]["request"], {"method": "POST", "path": "models/embedding/ruri-v3-310m/unload"})

    def test_overview_request_paths_encode_each_segment(self):
        self.assertEqual(embedding_overview.admin_path("models", "embedding/a b#c", "load"), "models/embedding/a%20b%23c/load")
        write_model(self.model_dir, "embedding/a b#c", ["BertModel"])
        _, items = self.overview()
        load = self.actions(items["local:embedding/a b#c"])["load"]
        self.assertEqual(load["request"]["path"], "models/embedding/a%20b%23c/load")

    def test_overview_load_is_blocked_by_an_unfinished_or_failed_download(self):
        name = "embedding/ruri-v3-310m"
        write_model(self.model_dir, name, ["ModernBertModel"])
        runner = MagicMock()
        runner.run.side_effect = DownloadFailed("synthetic network error")
        failing = DownloadJobRegistry(runner, spawn=lambda run: run())
        failing.start("cl-nagoya/ruri-v3-310m", name, self.model_dir / name)
        overview = self.build_overview(failing)
        items = {item["id"]: item for item in asyncio.run(overview.snapshot())["items"]}
        acts = self.actions(items[name])
        self.assertFalse(acts["load"]["enabled"])
        self.assertIn("失敗", acts["load"]["reason"])
        self.assertEqual(items[name]["job"]["status"], "failed")
        # A failed job also turns download into a forced retry.
        self.assertTrue(acts["download"]["enabled"])
        self.assertTrue(acts["download"]["request"]["body"]["force"])

    def test_overview_active_job_attaches_and_disables_download(self):
        name = "embedding/ruri-v3-310m"
        self.jobs.start("cl-nagoya/ruri-v3-310m", name, self.model_dir / name)
        _, items = self.overview()
        item = items[name]
        self.assertEqual(item["job"]["status"], "queued")
        self.assertFalse(self.actions(item)["download"]["enabled"])
        self.assertEqual(self.actions(item)["download"]["reason"], "取得中です。")
        # Free-form job outside the catalog becomes its own pending row.
        self.jobs.start("o/other", "embedding/other", self.model_dir / "embedding/other")
        _, items = self.overview()
        self.assertEqual(items["job:" + next(j["id"] for j in self.jobs.all() if j["local_name"] == "embedding/other")]["family"], "カタログ外")

    def test_overview_roots_filter_and_local_rows(self):
        write_model(self.model_dir, "embedding/harrier-oss-v1-0.6b", ["BertModel"])
        write_model(self.model_dir, "embedding/my-qwen3-reranker", ["Qwen3ForSequenceClassification"])
        write_model(self.model_dir, "stt/onnx-whisper", ["WhisperModel"])
        write_model(self.model_dir, "hf-cache/models--x/snapshots/abc", ["BertModel"])
        write_model(self.model_dir, "Qwen3-Reranker-0.6B", ["Qwen3ForCausalLM"])  # root level: not the catalog name
        _, items = self.overview()
        self.assertEqual(
            {i for i, item in items.items() if item["support"] == "local"},
            {"local:embedding/harrier-oss-v1-0.6b", "local:embedding/my-qwen3-reranker"},
        )
        harrier = items["local:embedding/harrier-oss-v1-0.6b"]
        self.assertEqual(harrier["family"], "カタログ外")
        self.assertEqual(harrier["api_name"], "embedding/harrier-oss-v1-0.6b")
        self.assertEqual(items["local:embedding/my-qwen3-reranker"]["family"], "Qwen3-Reranker")
        self.assertEqual(harrier["version"], "埋め込み")
        self.assertEqual(items["local:embedding/my-qwen3-reranker"]["version"], "リランカー")
        self.assertEqual(set(self.actions(harrier)), {"load", "unload"})
        self.assertTrue(self.actions(harrier)["load"]["primary"])
        self.assertFalse(items["embedding/Qwen3-Reranker-0.6B"]["lifecycle"]["downloaded"])
        # The classic listing keeps its unscoped behaviour.
        listed = {row["local_name"] for row in self.client.get("/admin/models").json()["items"]}
        self.assertIn("stt/onnx-whisper", listed)

    def test_overview_present_without_config_is_unverified_and_not_downloadable(self):
        name = "embedding/ruri-v3-310m"
        (self.model_dir / name).mkdir(parents=True)
        (self.model_dir / name / "model.safetensors").write_bytes(b"x")
        _, items = self.overview()
        self.assertTrue(items[name]["lifecycle"]["downloaded"])
        self.assertTrue(items[name]["unverified"])
        self.assertFalse(self.actions(items[name])["download"]["enabled"])
        self.assertTrue(self.actions(items[name])["load"]["enabled"])  # requires "downloaded", settled by the merge

    def test_overview_roots_parsing(self):
        parse = embedding_overview.parse_roots
        base = self.model_dir
        self.assertEqual(parse("", base), ((base / "embedding").resolve(),))
        self.assertEqual(parse(" a , b/c ,a", base), ((base / "a").resolve(), (base / "b/c").resolve()))
        self.assertEqual(parse(str(base / "abs"), base), ((base / "abs").resolve(),))
        for bad in ("..", "../x", "/etc", "."):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad, base)
        write_model(self.model_dir, "tts/voice", ["BertModel"])
        write_model(self.model_dir, "embedding/e1", ["BertModel"])
        wide = self.build_overview(self.jobs, roots="embedding,tts")
        names = {i["id"] for i in asyncio.run(wide.snapshot())["items"]}
        self.assertTrue({"local:tts/voice", "local:embedding/e1"} <= names)

    def test_overview_http_has_no_authentication_and_does_not_load(self):
        write_model(self.model_dir, "embedding/e1", ["BertModel"])
        self.assertEqual(self.client.get("/admin/models/overview").status_code, 200)
        self.assertEqual(self.load_calls, [])
        self.assertIn("overview", self.hub.keys)

    def test_interactions_columns_and_rows_newest_first(self):
        self.logs.extend([
            {"timestamp": 1.0, "event": "embedding", "model": "m", "duration_ms": 3.5, "details": {"status": "ok"}},
            {"timestamp": 2.0, "event": "rerank", "model": "r", "duration_ms": 9.0, "details": {"status": "error", "error": "x"}, "level": "error"},
        ])
        data = self.client.get("/admin/interactions?limit=5").json()
        self.assertEqual(data["columns"], [
            {"key": "timestamp", "label": "Time", "kind": "time"},
            {"key": "event", "label": "Event", "kind": "mono"},
            {"key": "model", "label": "Model", "kind": "mono"},
            {"key": "status", "label": "Status", "kind": "status"},
            {"key": "duration_ms", "label": "Duration", "kind": "duration"},
        ])
        self.assertEqual([(r["event"], r["status"]) for r in data["items"]], [("rerank", "error"), ("embedding", "ok")])
        self.assertEqual(len(self.client.get("/admin/interactions?limit=1").json()["items"]), 1)

    def test_admin_health_route_returns_the_provider_payload(self):
        self.loaded["m"] = "embedding"
        self.assertEqual(self.client.get("/admin/health").json(), {"status": "ok", "loaded": {"m": "embedding"}})


if __name__ == "__main__":
    unittest.main()
