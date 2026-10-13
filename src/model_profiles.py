"""公式推論設定を持つテキストモデルのカタログ。重みの検証結果とは分けて扱う。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

InputType = Literal["query", "document"]


@dataclass(frozen=True)
class ModelProfile:
    """モデル ID、ロード方式、検索側ごとの prefix を宣言する。

    dimensions は公式の標準次元であり、実行時の上限はロード済みモデルから取得する。
    load_kwargs は明示的な環境設定で上書きできる (Gemma の FP16 は除く)。
    """

    repo_id: str
    family: str
    model_type: Literal["embedding", "reranker"] = "embedding"
    dimensions: int | None = None
    engine: Literal["sentence_transformer", "plamo"] = "sentence_transformer"
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    encode_kwargs: dict[str, Any] = field(default_factory=dict)
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    query_prompt: str | None = None
    document_prompt: str | None = None
    license: str = "Apache-2.0"
    compatibility: Literal["none", "voyage_mask", "gte_buffers"] = "none"
    precision_policy: Literal["default", "bf16_or_fp32"] = "default"
    dimension_fallback: Literal["none", "profile"] = "none"

    def catalog_entry(self) -> dict[str, Any]:
        """既存の repo_id / family / type 契約に公式次元とライセンスを添える。"""
        return {
            "repo_id": self.repo_id,
            "family": self.family,
            "type": self.model_type,
            "dimensions": self.dimensions,
            "license": self.license,
        }


_SEARCH_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
_SARASHINA_INSTRUCTION = "task: 質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。\nquery: "

PROFILES = (
    ModelProfile("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding", dimensions=1024, query_prompt_name="query", document_prompt_name="document"),
    ModelProfile("Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding", dimensions=2560, query_prompt_name="query", document_prompt_name="document"),
    ModelProfile("Qwen/Qwen3-Embedding-8B", "Qwen3-Embedding", dimensions=4096, query_prompt_name="query", document_prompt_name="document"),
    ModelProfile("Qwen/Qwen3-Reranker-0.6B", "Qwen3-Reranker", model_type="reranker"),
    ModelProfile("Qwen/Qwen3-Reranker-4B", "Qwen3-Reranker", model_type="reranker"),
    ModelProfile("cl-nagoya/ruri-v3-310m", "Ruri", dimensions=768),
    ModelProfile("cl-nagoya/ruri-v3-130m", "Ruri", dimensions=512),
    ModelProfile("cl-nagoya/ruri-v3-70m", "Ruri", dimensions=384),
    ModelProfile("cl-nagoya/ruri-v3-30m", "Ruri", dimensions=256),
    ModelProfile("cl-nagoya/ruri-v3-reranker-310m", "Ruri", model_type="reranker"),
    ModelProfile("pfnet/plamo-embedding-1b", "PLaMo", dimensions=2048, engine="plamo", load_kwargs={"trust_remote_code": True}),
    ModelProfile("sbintuitions/sarashina-embedding-v2-1b", "Sarashina", dimensions=1792, query_prompt=_SARASHINA_INSTRUCTION, document_prompt="text: ", license="Sarashina NonCommercial"),
    ModelProfile("microsoft/harrier-oss-v1-0.6b", "Harrier", dimensions=1024, query_prompt_name="web_search_query", document_prompt="", license="MIT"),
    ModelProfile("microsoft/harrier-oss-v1-270m", "Harrier", dimensions=640, query_prompt_name="web_search_query", document_prompt="", license="MIT"),
    ModelProfile("ibm-granite/granite-embedding-311m-multilingual-r2", "Granite", dimensions=768),
    ModelProfile("ibm-granite/granite-embedding-97m-multilingual-r2", "Granite", dimensions=384),
    ModelProfile("voyageai/voyage-4-nano", "Voyage", dimensions=2048, load_kwargs={"trust_remote_code": True}, query_prompt_name="query", document_prompt_name="document", compatibility="voyage_mask"),
    ModelProfile("perplexity-ai/pplx-embed-v1-0.6b", "pplx", dimensions=1024, load_kwargs={"trust_remote_code": True}, license="MIT"),
    ModelProfile("perplexity-ai/pplx-embed-v1-4b", "pplx", dimensions=2560, load_kwargs={"trust_remote_code": True}, license="MIT"),
    ModelProfile("nvidia/Nemotron-3-Embed-1B-BF16", "Nemotron", dimensions=2048, query_prompt_name="query", document_prompt_name="document", license="OpenMDW-1.1"),
    ModelProfile("google/embeddinggemma-2", "EmbeddingGemma", dimensions=768, load_kwargs={"config_kwargs": {"vision_config": None, "audio_config": None}}, query_prompt_name="SearchQuery", document_prompt_name="Document", precision_policy="bf16_or_fp32"),
    ModelProfile("google/embeddinggemma-300m", "EmbeddingGemma", dimensions=768, query_prompt_name="query", document_prompt_name="document", license="Gemma", precision_policy="bf16_or_fp32"),
    ModelProfile("jinaai/jina-embeddings-v5-text-nano", "Jina v5 Text", dimensions=768, load_kwargs={"trust_remote_code": True}, encode_kwargs={"task": "retrieval"}, query_prompt_name="query", document_prompt_name="document", license="CC-BY-NC-4.0", dimension_fallback="profile"),
    ModelProfile("BAAI/bge-m3", "BGE", dimensions=1024, license="MIT"),
    ModelProfile("Alibaba-NLP/gte-multilingual-base", "GTE", dimensions=768, load_kwargs={"trust_remote_code": True}, compatibility="gte_buffers"),
    ModelProfile("hotchpotch/bekko-embedding-v1-a25m", "Bekko", dimensions=384, license="MIT"),
    ModelProfile("hotchpotch/bekko-embedding-v1-a8m", "Bekko", dimensions=384, license="MIT"),
    ModelProfile("intfloat/multilingual-e5-large-instruct", "E5", dimensions=1024, query_prompt=_SEARCH_INSTRUCTION, document_prompt="", license="MIT"),
    ModelProfile("BAAI/bge-reranker-v2-m3", "BGE", model_type="reranker"),
)
_BY_REPO = {profile.repo_id: profile for profile in PROFILES}
_BY_NAME = {profile.repo_id.rsplit("/", 1)[-1]: profile for profile in PROFILES}


def profile_for(model_id: str, catalog: list[dict[str, Any]]) -> ModelProfile | None:
    """HF の正式 ID、カタログの明示的 local_name を解決する。

    local_name の basename への変換は、呼び出し側でディスク上の存在確認後に行う。
    未知のリモート owner を basename だけで対応モデルと認定しない。
    """
    for entry in catalog:
        if entry.get("local_name") == model_id:
            return _BY_REPO.get(str(entry.get("repo_id")))
    return _BY_REPO.get(model_id)


def local_profile_for(local_name: str, catalog: list[dict[str, Any]]) -> ModelProfile | None:
    """存在確認済みのローカル保存名から公式プロファイルを引く。

    ディスクの存在確認は呼び出し側の責任。未知の保存名は None を返す。
    """
    return profile_for(local_name, catalog) or _BY_NAME.get(PurePosixPath(local_name).name)


def encode_options(profile: ModelProfile | None, prompts: dict[str, str], input_type: InputType) -> dict[str, Any]:
    """検索側に合う公式 prefix を選ぶ。文書側の未知モデルには query prefix を付けない。"""
    options = dict(profile.encode_kwargs) if profile else {}
    prompt_name = getattr(profile, f"{input_type}_prompt_name", None)
    prompt = getattr(profile, f"{input_type}_prompt", None)
    if prompt_name is not None:
        options["prompt_name"] = prompt_name
    elif prompt is not None:
        options["prompt"] = prompt
    elif input_type in prompts:
        options["prompt_name"] = input_type
    elif input_type == "document":
        # default_prompt_name=query のモデルでも文書に query prefix が混入しないようにする。
        options["prompt"] = ""
    return options
