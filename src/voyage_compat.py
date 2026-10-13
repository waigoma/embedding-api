"""Voyage 公式 remote code の mask キーワードを Transformers 5.19 に適合する。"""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any


def configure_voyage_mask(model: Any) -> None:
    """公式 bidirectional mask の意味を保持して旧引数名だけを変換する。

    Voyage の remote module namespace 内だけを更新し、Transformers の関数は変更しない。
    未知の module 構成は互換性を推測せず ValueError でロードを失敗させる。
    """
    auto_model = model[0].auto_model
    if type(auto_model).__name__ != "Qwen3BidirectionalModel":
        raise ValueError("Voyage compatibility requires the official Qwen3BidirectionalModel")
    namespace = auto_model.forward.__func__.__globals__
    create_mask = namespace["create_causal_mask"]
    if getattr(create_mask, "_embedding_api_compatible", False):
        return
    if "inputs_embeds" not in inspect.signature(create_mask).parameters:
        return

    @wraps(create_mask)
    def compatible_mask(**kwargs: Any):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        # 公式コードの cache_position は 0 始まりの arange。5.19 は入力長から同じ位置を導出する。
        kwargs.pop("cache_position", None)
        return create_mask(**kwargs)

    compatible_mask._embedding_api_compatible = True
    namespace["create_causal_mask"] = compatible_mask
