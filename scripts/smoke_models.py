"""実モデルの取得・ロード・両側のテキスト推論を 1 モデルずつ隔離して確認する。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_MODELS = (
    "google/embeddinggemma-2",
    "cl-nagoya/ruri-v3-310m",
    "microsoft/harrier-oss-v1-0.6b",
    "perplexity-ai/pplx-embed-v1-0.6b",
    "pfnet/plamo-embedding-1b",
    "Alibaba-NLP/gte-multilingual-base",
    "voyageai/voyage-4-nano",
    "jinaai/jina-embeddings-v5-text-nano",
    "Qwen/Qwen3-Reranker-0.6B",
    "BAAI/bge-reranker-v2-m3",
)


def _one_model(repo_id: str, models_dir: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    import numpy as np
    import torch

    from model_profiles import profile_for
    import server

    profile = profile_for(repo_id, [])
    if profile is None:
        raise ValueError(f"not a built-in profile: {repo_id}")
    local_name = "embedding/" + repo_id.rsplit("/", 1)[-1]
    target = models_dir / local_name
    receipt = target / ".smoke-model.json"
    sha = None
    # 既存重みは再取得せず検証する。既存 config の検出は来歴や完全性の証明ではない。
    if not (target / "config.json").exists():
        sha = HfApi().model_info(repo_id, token=server.HF_TOKEN).sha
        snapshot_download(
            repo_id, revision=sha, local_dir=target, token=server.HF_TOKEN,
            ignore_patterns=["onnx/*", "openvino/*", "*.gguf", "*.onnx", "*.ot", "*.h5", "*.tflite"],
        )
        receipt.write_text(json.dumps({"repo_id": repo_id, "revision": sha}))
    elif receipt.exists():
        sha = json.loads(receipt.read_text()).get("revision")

    use_bf16 = server.DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    server.EMBEDDING_KWARGS = {"model_kwargs": {"dtype": "bfloat16" if use_bf16 else "float32"}}
    server.RERANKER_KWARGS = {"model_kwargs": {"dtype": "bfloat16" if use_bf16 else "float32"}}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "sentence-transformers", "transformers", "huggingface_hub")}
    server._load_into_registry(local_name, profile.model_type)
    try:
        if profile.model_type == "embedding":
            query, _ = server._encode_embeddings(local_name, ["日本の首都はどこですか？"], input_type="query")
            documents, _ = server._encode_embeddings(local_name, ["東京は日本の首都です。", "バナナは黄色い果物です。"], input_type="document")
            shortened, _ = server._encode_embeddings(local_name, ["日本語の検索テスト"], dimensions=128, input_type="document")
            q, d, short = (np.asarray(value) for value in (query, documents, shortened))
            assert q.shape == (1, profile.dimensions), q.shape
            assert d.shape == (2, profile.dimensions), d.shape
            assert short.shape == (1, 128), short.shape
            norms = {}
            for label, array in (("query", q), ("documents", d), ("truncated", short)):
                assert np.isfinite(array).all(), f"{label}: non-finite vector"
                norms[label] = np.linalg.norm(array, axis=1).tolist()
                assert np.allclose(norms[label], 1, atol=2e-3), f"{label}: not L2 normalized: {norms[label]}"
            result = {"dimensions": q.shape[1], "scores": (q @ d.T).tolist()[0], "truncated_dimensions": short.shape[1], "norms": norms}
        else:
            scores = np.asarray(server.registry[local_name].model.predict([
                ["日本の首都はどこですか？", "東京は日本の首都です。"],
                ["日本の首都はどこですか？", "バナナは黄色い果物です。"],
            ]))
            assert scores.shape == (2,), scores.shape
            assert np.isfinite(scores).all(), "non-finite rerank scores"
            assert scores[0] > scores[1], "expected relevant document to rank first"
            result = {"scores": scores.tolist()}
        return {"status": "passed", "repo_id": repo_id, "local_name": local_name, "revision": sha, "versions": versions, "device": server.DEVICE, **result}
    finally:
        server._unload(local_name)


def main() -> int:
    """JSONL を stdout へ返す。失敗も記録し、残るモデルの検証を続ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = str(args.models_dir.resolve())
    if args.one:
        try:
            result = _one_model(args.one, args.models_dir.resolve())
        except Exception as exc:
            result = {"status": "failed", "repo_id": args.one, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["status"] == "passed" else 1
    failures = 0
    for model in args.models:
        process = subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--models-dir", str(args.models_dir), "--one", model,
        ])
        failures += process.returncode != 0
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
