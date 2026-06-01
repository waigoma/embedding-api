"""Tests for the `dimensions` parameter in _encode_embeddings / POST /v1/embeddings.

Run inside the container (or any env with dependencies):
    python -m pytest tests/test_dimensions.py -v
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Mock torch so the module can be imported without a GPU env
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False
_torch_mock.version.hip = None
_torch_mock.version.cuda = None
sys.modules.setdefault("torch", _torch_mock)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server  # noqa: E402  (imported after sys.path fixup)
from fastapi import HTTPException  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NATIVE_DIM = 1024


def _make_entry(supports_truncate_dim: bool = True) -> MagicMock:
    """Return a fake ModelEntry whose model behaves like a SentenceTransformer."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = NATIVE_DIM
    model.prompts = {}

    def _encode(inputs, **kwargs):
        if "truncate_dim" in kwargs and not supports_truncate_dim:
            raise TypeError("unexpected keyword argument 'truncate_dim'")
        out_dim = kwargs.get("truncate_dim", NATIVE_DIM)
        return np.ones((len(inputs), out_dim), dtype=np.float32)

    model.encode.side_effect = _encode

    entry = MagicMock()
    entry.model = model
    entry.model_type = "embedding"
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestEncodeEmbeddingsDimensions(unittest.TestCase):

    def _patch_get_model(self, entry: MagicMock):
        return patch.object(server, "_get_model", return_value=entry)

    # -- validation ----------------------------------------------------------

    def test_dimensions_zero_raises_400(self):
        entry = _make_entry()
        with self._patch_get_model(entry):
            with self.assertRaises(HTTPException) as ctx:
                server._encode_embeddings("model", ["hello"], dimensions=0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_dimensions_negative_raises_400(self):
        entry = _make_entry()
        with self._patch_get_model(entry):
            with self.assertRaises(HTTPException) as ctx:
                server._encode_embeddings("model", ["hello"], dimensions=-1)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_dimensions_above_max_raises_400(self):
        entry = _make_entry()
        with self._patch_get_model(entry):
            with self.assertRaises(HTTPException) as ctx:
                server._encode_embeddings("model", ["hello"], dimensions=NATIVE_DIM + 1)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_dimensions_equal_max_is_valid(self):
        entry = _make_entry()
        with self._patch_get_model(entry):
            embeddings, _ = server._encode_embeddings("model", ["hello"], dimensions=NATIVE_DIM)
        self.assertEqual(len(embeddings[0]), NATIVE_DIM)

    # -- truncation via truncate_dim -----------------------------------------

    def test_dimensions_truncates_with_truncate_dim(self):
        entry = _make_entry(supports_truncate_dim=True)
        with self._patch_get_model(entry):
            embeddings, _ = server._encode_embeddings("model", ["a", "b"], dimensions=256)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 256)
        self.assertEqual(len(embeddings[1]), 256)

    # -- fallback slicing when truncate_dim is not supported -----------------

    def test_dimensions_fallback_slice(self):
        entry = _make_entry(supports_truncate_dim=False)
        with self._patch_get_model(entry):
            embeddings, _ = server._encode_embeddings("model", ["hello"], dimensions=512)
        self.assertEqual(len(embeddings[0]), 512)

    # -- no dimensions → native dim ------------------------------------------

    def test_no_dimensions_returns_native_dim(self):
        entry = _make_entry()
        with self._patch_get_model(entry):
            embeddings, _ = server._encode_embeddings("model", ["hello"])
        self.assertEqual(len(embeddings[0]), NATIVE_DIM)

    # -- base64 output is consistent with dimensions -------------------------

    def test_base64_output_matches_dimensions(self):
        import base64, struct
        entry = _make_entry()
        with self._patch_get_model(entry):
            embeddings, _ = server._encode_embeddings("model", ["hello"], dimensions=128)
        b64 = server._format_embedding_output(embeddings[0], "base64")
        decoded = struct.unpack(f"<{len(embeddings[0])}f", base64.b64decode(b64))
        self.assertEqual(len(decoded), 128)


if __name__ == "__main__":
    unittest.main()
