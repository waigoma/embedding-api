"""GTE の旧 remote code が保存しない位置・RoPE buffer を公式式で再生成する。"""

from __future__ import annotations

from typing import Any
from types import MethodType

import torch


def initialize_gte_buffers(model: Any) -> None:
    """Transformers 5 の meta load 後に GTE の非永続 buffer を復元する。

    学習済みパラメータは変更しない。公式 NewEmbeddings._init_rope を呼び直して
    inv_freq / cos_cached / sin_cached を復元し、position_ids は公式の arange に戻す。
    """
    auto_model = model[0].auto_model
    embeddings = auto_model.embeddings
    if type(embeddings).__name__ != "NewEmbeddings":
        raise ValueError("GTE compatibility requires the official NewEmbeddings")
    device = embeddings.word_embeddings.weight.device
    cache_dtype = embeddings.rotary_emb.cos_cached.dtype
    embeddings.register_buffer(
        "position_ids", torch.arange(auto_model.config.max_position_embeddings, device=device), persistent=False,
    )
    # 旧 _init_weights は非永続 buffer を初期化しないため、meta からの空メモリを使わせない。
    embeddings._init_rope(auto_model.config)
    embeddings.rotary_emb.to(device=device, dtype=cache_dtype)
    if not hasattr(auto_model, "get_extended_attention_mask"):
        auto_model.get_extended_attention_mask = MethodType(_extended_attention_bias, auto_model)


def _extended_attention_bias(model: Any, attention_mask, input_shape):
    # GTE は encoder。旧 Transformers の「padding のみ」の additive mask を再現する。
    if attention_mask.ndim == 2:
        expanded = attention_mask[:, None, None, :]
    elif attention_mask.ndim == 3:
        expanded = attention_mask[:, None, :, :]
    else:
        raise ValueError("GTE attention_mask must have 2 or 3 dimensions")
    dtype = model.dtype
    return (1.0 - expanded.to(dtype=dtype)) * torch.finfo(dtype).min
