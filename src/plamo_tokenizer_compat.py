"""PLaMo tokenizer の旧 SentencePiece Init を現行 constructor へ適合する。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def configure_plamo_tokenizer(path: str, options: dict[str, Any]) -> None:
    """PLaMo の remote module 内だけで Init(model_file, num_threads) を復元する。

    SentencePiece 全体は変更しない。trust_remote_code が無効なら remote class を読み込まない。
    現行公開 constructor に語彙ファイル・スレッド数・元の設定を渡し、同じ tokenizer を再初期化する。
    """
    if not options.get("trust_remote_code"):
        return
    import sentencepiece as spm
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if hasattr(spm.SentencePieceProcessor, "Init"):
        return
    class_options = {key: value for key, value in options.items() if key in {"revision", "token", "local_files_only", "cache_dir"}}
    tokenizer_class = get_class_from_dynamic_module("tokenization_plamo.PlamoTokenizer", path, **class_options)
    namespace = tokenizer_class.__init__.__globals__
    if hasattr(namespace["spm"].SentencePieceProcessor, "Init"):
        return

    class CompatibleProcessor(spm.SentencePieceProcessor):
        def __init__(self, **kwargs: Any) -> None:
            self._plamo_options = dict(kwargs)
            super().__init__(**kwargs)

        def Init(self, *, model_file: str, num_threads: int = -1) -> None:
            super().__init__(**(self._plamo_options | {"model_file": model_file, "num_threads": num_threads}))

    namespace["spm"] = SimpleNamespace(SentencePieceProcessor=CompatibleProcessor)
