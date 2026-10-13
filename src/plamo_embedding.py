"""PLaMo の公式 instruction masking と pooling を再利用するテキスト adapter。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class PlamoEmbedding:
    """公式 encode_query / encode_document を SentenceTransformer と同じ境界へ適合する。

    encode() は list[str] を受け、float32 の密ベクトル配列を返す。
    input_type は prompt_name に query / document として渡す。
    """

    prompts = {"query": "", "document": ""}

    def __init__(self, path: str, **kwargs: Any) -> None:
        options = dict(kwargs)
        device = options.pop("device", "cpu")
        model_kwargs = options.pop("model_kwargs", {})
        tokenizer_kwargs = options.pop("tokenizer_kwargs", {})
        allowed = {"trust_remote_code", "revision", "token", "local_files_only", "cache_folder"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"PLaMo does not support SentenceTransformer options: {sorted(unknown)}")
        from transformers import AutoModel, AutoTokenizer

        if "cache_folder" in options:
            options["cache_dir"] = options.pop("cache_folder")
        from plamo_tokenizer_compat import configure_plamo_tokenizer
        tokenizer_options = options | tokenizer_kwargs
        configure_plamo_tokenizer(path, tokenizer_options)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_options)
        self.model = AutoModel.from_pretrained(path, **(options | model_kwargs)).to(device).eval()
        if not hasattr(self.model.config, "max_length"):
            # TF 5 は generation 専用 field を config から除く。公式 embedding はこの長さを tokenizer に渡す。
            self.model.config.max_length = self.model.config.max_position_embeddings
        _restore_plamo_rope(self.model)

    def get_sentence_embedding_dimension(self) -> int:
        """公式 checkpoint の hidden_size (2048) を返す。"""
        return self.model.config.hidden_size

    def encode(
        self,
        inputs: list[str],
        *,
        prompt_name: str = "query",
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        truncate_dim: int | None = None,
    ) -> np.ndarray:
        """prefix 部分を pooling から除外し、必要なら次元短縮後に L2 正規化する。"""
        if prompt_name not in self.prompts:
            raise ValueError("PLaMo prompt_name must be query or document")
        encode = {"query": self.model.encode_query, "document": self.model.encode_document}[prompt_name]
        rows = []
        # 公式 encode はバッチ長に比例して padding するため、大きな入力列を小分けにする。
        with torch.inference_mode():
            for start in range(0, len(inputs), 8):
                vectors = encode(inputs[start:start + 8], self.tokenizer)
                if truncate_dim is not None:
                    vectors = vectors[:, :truncate_dim]
                if normalize_embeddings:
                    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                rows.append(vectors.float().cpu().numpy())
        dimension = truncate_dim or self.get_sentence_embedding_dimension()
        return np.concatenate(rows) if rows else np.empty((0, dimension), dtype=np.float32)


def _restore_plamo_rope(model) -> None:
    # 旧 remote code の _init_weights は checkpoint に保存しない RoPE buffer を復元しない。
    for module in model.modules():
        if type(module).__name__ != "RotaryEmbedding":
            continue
        device = module.cos_cached.device
        cache_dtype = module.cos_cached.dtype
        inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2, device=device, dtype=torch.float32) / module.dim))
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module._set_cos_sin_cache(module.max_position_embeddings, device, cache_dtype)
