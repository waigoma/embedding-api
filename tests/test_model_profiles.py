"""モデル追加の観点: ID 解決、公式 prefix、ロード設定、HTTP 互換性と数値境界。

CPU stub は実モデルの互換性を保証しない。GPU 確認は scripts/smoke_models.py を使う。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
from embedding_admin import guess_model_type
from model_profiles import PROFILES, encode_options, local_profile_for, profile_for
from plamo_embedding import PlamoEmbedding


@pytest.mark.parametrize("profile", PROFILES, ids=lambda profile: profile.repo_id)
def test_profiles_resolve_remote_and_downloaded_names(profile):
    assert profile_for(profile.repo_id, []) == profile
    assert local_profile_for("embedding/" + profile.repo_id.rsplit("/", 1)[-1], []) == profile
    assert profile_for("embedding/custom", [{"repo_id": profile.repo_id, "local_name": "embedding/custom"}]) == profile
    assert profile.catalog_entry()["type"] == profile.model_type


def test_catalog_has_no_duplicate_repository_or_local_names():
    assert len(PROFILES) == len({profile.repo_id for profile in PROFILES})
    assert len(PROFILES) == len({profile.repo_id.rsplit("/", 1)[-1] for profile in PROFILES})
    assert profile_for("unknown/random", []) is None
    assert profile_for("untrusted/voyage-4-nano", []) is None
    assert profile_for("voyage-4-nano", []) is None


def test_downloaded_local_name_resolves_only_when_present(tmp_path):
    with patch.object(server, "MODEL_DIR", str(tmp_path)):
        assert server._model_profile("embedding/voyage-4-nano") is None
        (tmp_path / "embedding" / "voyage-4-nano").mkdir(parents=True)
        assert server._model_profile("embedding/voyage-4-nano").repo_id == "voyageai/voyage-4-nano"


def test_qwen_cross_encoder_metadata_identifies_arbitrary_save_name(tmp_path):
    (tmp_path / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
    (tmp_path / "config_sentence_transformers.json").write_text('{"model_type": "CrossEncoder"}')
    assert guess_model_type(tmp_path) == "reranker"


@pytest.mark.parametrize(
    "repo,query_options,document_options",
    [
        ("google/embeddinggemma-2", {"prompt_name": "SearchQuery"}, {"prompt_name": "Document"}),
        ("microsoft/harrier-oss-v1-270m", {"prompt_name": "web_search_query"}, {"prompt": ""}),
        ("Qwen/Qwen3-Embedding-0.6B", {"prompt_name": "query"}, {"prompt_name": "document"}),
        ("jinaai/jina-embeddings-v5-text-nano", {"task": "retrieval", "prompt_name": "query"}, {"task": "retrieval", "prompt_name": "document"}),
        ("intfloat/multilingual-e5-large-instruct", {"prompt": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "}, {"prompt": ""}),
    ],
)
def test_query_and_document_use_official_distinct_prompts(repo, query_options, document_options):
    profile = profile_for(repo, [])
    assert encode_options(profile, {}, "query") == query_options
    assert encode_options(profile, {}, "document") == document_options


def test_prefix_free_models_and_unknown_model_keep_generic_behavior():
    for repo in ("cl-nagoya/ruri-v3-30m", "BAAI/bge-m3", "perplexity-ai/pplx-embed-v1-0.6b"):
        assert encode_options(profile_for(repo, []), {}, "query") == {}
        assert encode_options(profile_for(repo, []), {}, "document") == {"prompt": ""}
    assert encode_options(None, {"query": "old prefix"}, "query") == {"prompt_name": "query"}
    assert encode_options(None, {"query": "old prefix"}, "document") == {"prompt": ""}


def test_model_specific_load_options_are_merged_without_mutating_defaults():
    profile = profile_for("google/embeddinggemma-2", [])
    options = server._embedding_load_options(profile, {"config_kwargs": {"attn_implementation": "sdpa"}, "model_kwargs": {"dtype": "bfloat16"}})
    assert options["config_kwargs"] == {"vision_config": None, "audio_config": None, "attn_implementation": "sdpa"}
    assert options["model_kwargs"] == {"dtype": "bfloat16"}
    assert profile.load_kwargs == {"config_kwargs": {"vision_config": None, "audio_config": None}}
    assert server._embedding_load_options(profile, {})["model_kwargs"] == {"dtype": "float32"}


@pytest.mark.parametrize("repo", ["google/embeddinggemma-2", "google/embeddinggemma-300m"])
@pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
def test_embedding_gemma_rejects_fp16_before_loading(repo, key):
    with pytest.raises(HTTPException, match="float16") as error:
        server._embedding_load_options(profile_for(repo, []), {"model_kwargs": {key: "float16"}})
    assert error.value.status_code == 400


def test_known_reranker_cannot_be_loaded_as_embedding():
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
        with pytest.raises(HTTPException, match="reranker"):
            server._load_embedding("Qwen/Qwen3-Reranker-0.6B")


def test_custom_model_code_is_enabled_only_for_known_profiles():
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}) as modules, patch("voyage_compat.configure_voyage_mask"):
        constructor = modules["sentence_transformers"].SentenceTransformer
        server._load_embedding("voyageai/voyage-4-nano")
        assert constructor.call_args.kwargs["trust_remote_code"] is True
        server._load_embedding("unknown/model")
        assert "trust_remote_code" not in constructor.call_args.kwargs


@pytest.mark.parametrize("route", ["/v1/embeddings", "/embeddings", "/v1/responses", "/responses"])
def test_input_type_default_and_document_preserve_response_shape(route):
    client = TestClient(server.app)
    with patch.object(server, "_encode_embeddings", return_value=([[0.6, 0.8]], 1)) as encode:
        for input_type in (None, "query", "document"):
            body = {"model": "demo", "input": "sample"}
            if input_type:
                body["input_type"] = input_type
            response = client.post(route, json=body)
            assert response.status_code == 200
            assert response.json()["data"][0]["embedding"] == [0.6, 0.8]
            assert "input_type" not in response.json()
            actual = encode.call_args.kwargs.get("input_type") or encode.call_args.args[-1]
            assert actual == (input_type or "query")
        assert client.post(route, json={"model": "demo", "input": "x", "input_type": "image"}).status_code == 422


def test_encode_uses_document_prefix_and_refuses_nonfinite_output():
    model = MagicMock(prompts={})
    model.encode.return_value = np.array([[np.nan, 0.0]])
    entry = server.ModelEntry(model, "embedding")
    with patch.object(server, "_get_model", return_value=entry):
        with pytest.raises(HTTPException, match="non-finite"):
            server._encode_embeddings("google/embeddinggemma-2", ["hello"], input_type="document")
    assert model.encode.call_args.kwargs["prompt_name"] == "Document"
    assert entry.last_inference_at is None


def test_truncation_fallback_normalizes_and_does_not_hide_other_type_errors():
    model = MagicMock(prompts={})
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.side_effect = [TypeError("unexpected keyword 'truncate_dim'"), np.array([[3.0, 4.0, 9.0]])]
    entry = server.ModelEntry(model, "embedding")
    with patch.object(server, "_get_model", return_value=entry):
        assert server._encode_embeddings("demo", ["x"], 2)[0] == [[0.6, 0.8]]
        model.encode.side_effect = TypeError("broken custom model")
        with pytest.raises(TypeError, match="broken custom model"):
            server._encode_embeddings("demo", ["x"], 2)


def test_custom_quantizer_and_bf16_norm_error_are_corrected_at_api_boundary():
    model = MagicMock(prompts={})
    model.encode.return_value = np.array([[3.0, 4.0]])
    with patch.object(server, "_get_model", return_value=server.ModelEntry(model, "embedding")):
        assert server._encode_embeddings("demo", ["x"])[0] == [[0.6, 0.8]]


def test_plamo_loading_uses_official_remote_model_and_tokenizer():
    from types import SimpleNamespace
    transformers = MagicMock()
    model = transformers.AutoModel.from_pretrained.return_value.to.return_value.eval.return_value
    model.config = SimpleNamespace(hidden_size=2048, max_position_embeddings=4096)
    with patch.dict("sys.modules", {"transformers": transformers}), patch("plamo_tokenizer_compat.configure_plamo_tokenizer"):
        adapter = PlamoEmbedding("pfnet/plamo-embedding-1b", device="cuda", trust_remote_code=True, model_kwargs={"dtype": "bfloat16"})
    assert transformers.AutoTokenizer.from_pretrained.call_args.kwargs == {"trust_remote_code": True}
    assert transformers.AutoModel.from_pretrained.call_args.kwargs == {"trust_remote_code": True, "dtype": "bfloat16"}
    assert adapter.get_sentence_embedding_dimension() == 2048
    assert model.config.max_length == 4096
    with pytest.raises(ValueError, match="options"):
        PlamoEmbedding("demo", backend="onnx")


def test_plamo_adapter_dispatches_both_sides_and_batches_long_requests():
    adapter = object.__new__(PlamoEmbedding)
    adapter.model = MagicMock()
    adapter.model.config.hidden_size = 2048
    adapter.tokenizer = object()
    adapter.model.encode_document.return_value = MagicMock()
    adapter.encode(["x"] * 9, prompt_name="document")
    assert adapter.model.encode_document.call_count == 2
    assert adapter.model.encode_document.call_args.args == (["x"], adapter.tokenizer)
    assert adapter.model.encode_query.call_count == 0
    adapter.encode(["x"])
    assert adapter.model.encode_query.call_args.args == (["x"], adapter.tokenizer)
    assert adapter.encode([]).shape == (0, 2048)


def test_voyage_compat_only_renames_mask_keywords_and_preserves_bidirectional_callable():
    from voyage_compat import configure_voyage_mask
    namespace = {}
    exec("def forward(self):\n    return create_causal_mask(input_embeds='embeddings', cache_position='positions', or_mask_function='bidirectional')", namespace)

    def mask(inputs_embeds, or_mask_function):
        return inputs_embeds, or_mask_function

    namespace["create_causal_mask"] = mask
    model_class = type("Qwen3BidirectionalModel", (), {"forward": namespace["forward"]})
    transformer = MagicMock(auto_model=model_class())
    configure_voyage_mask([transformer])
    assert transformer.auto_model.forward() == ("embeddings", "bidirectional")
    assert namespace["create_causal_mask"] is not mask


def test_gte_compat_reinitializes_only_nonpersistent_buffers():
    from gte_compat import initialize_gte_buffers
    embeddings_class = type("NewEmbeddings", (MagicMock,), {})
    embeddings = embeddings_class()
    auto_model = MagicMock(embeddings=embeddings)
    transformer = MagicMock(auto_model=auto_model)
    initialize_gte_buffers([transformer])
    assert embeddings.register_buffer.call_args.args[0] == "position_ids"
    assert embeddings.register_buffer.call_args.kwargs == {"persistent": False}
    embeddings._init_rope.assert_called_once_with(auto_model.config)
    assert embeddings.rotary_emb.to.call_args.kwargs == {
        "device": embeddings.word_embeddings.weight.device,
        "dtype": embeddings.rotary_emb.cos_cached.dtype,
    }


def test_plamo_tokenizer_compat_preserves_vocab_threads_and_processor_options():
    from types import SimpleNamespace
    from plamo_tokenizer_compat import configure_plamo_tokenizer

    class Processor:
        def __init__(self, **kwargs):
            self.options = kwargs

    namespace = {"spm": SimpleNamespace(SentencePieceProcessor=Processor)}
    exec("def init(self): pass", namespace)
    tokenizer_class = type("PlamoTokenizer", (), {"__init__": namespace["init"]})
    dynamic_module = MagicMock()
    dynamic_module.get_class_from_dynamic_module.return_value = tokenizer_class
    with patch.dict("sys.modules", {
        "sentencepiece": SimpleNamespace(SentencePieceProcessor=Processor),
        "transformers.dynamic_module_utils": dynamic_module,
    }):
        configure_plamo_tokenizer("demo", {"trust_remote_code": True, "revision": "sha"})
    processor = namespace["spm"].SentencePieceProcessor(add_bos=True)
    processor.Init(model_file="tokenizer.model", num_threads=4)
    assert processor.options == {"add_bos": True, "model_file": "tokenizer.model", "num_threads": 4}


def test_gte_legacy_padding_mask_matches_encoder_additive_bias():
    from gte_compat import _extended_attention_bias

    class Mask(np.ndarray):
        def to(self, *, dtype):
            return self.astype(dtype)

    mask = np.array([[1, 1, 0]], dtype=np.float32).view(Mask)
    with patch("gte_compat.torch.finfo", return_value=np.finfo(np.float32)):
        result = _extended_attention_bias(MagicMock(dtype=np.float32), mask, (1, 3))
    assert result.shape == (1, 1, 1, 3)
    assert result.tolist() == [[[[0.0, 0.0, float(np.finfo(np.float32).min)]]]]


def test_jina_declared_dimension_fallback_is_limited_to_known_profile():
    model = MagicMock(prompts={"query": "Query: "})
    model.get_sentence_embedding_dimension.return_value = None
    model.encode.return_value = np.ones((1, 128), dtype=np.float32)
    with patch.object(server, "_get_model", return_value=server.ModelEntry(model, "embedding")):
        assert len(server._encode_embeddings("jinaai/jina-embeddings-v5-text-nano", ["x"], 128)[0][0]) == 128
        with pytest.raises(HTTPException, match="does not report"):
            server._encode_embeddings("unknown/model", ["x"], 128)
        model.encode.return_value = np.ones((1, 768), dtype=np.float32)
        with pytest.raises(HTTPException, match="requested embedding dimension"):
            server._encode_embeddings("jinaai/jina-embeddings-v5-text-nano", ["x"], 128)


def test_plamo_rope_uses_official_formula_and_original_cache_dtype():
    from plamo_embedding import _restore_plamo_rope
    rotary_class = type("RotaryEmbedding", (MagicMock,), {})
    rotary = rotary_class()
    rotary.dim = 4
    rotary.base = 10000
    rotary.max_position_embeddings = 4096
    model = MagicMock()
    model.modules.return_value = [rotary]
    with patch("plamo_embedding.torch.arange", return_value=np.array([0.0, 2.0])):
        _restore_plamo_rope(model)
    assert np.allclose(rotary.register_buffer.call_args.args[1], [1.0, 0.01])
    assert rotary.register_buffer.call_args.kwargs == {"persistent": False}
    assert rotary._set_cos_sin_cache.call_args.args == (4096, rotary.cos_cached.device, rotary.cos_cached.dtype)
