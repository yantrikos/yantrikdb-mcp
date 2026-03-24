"""Fast ONNX-based embedder for YantrikDB MCP server.

Uses onnxruntime + tokenizers directly — loads in ~2s vs ~5min for PyTorch.
Falls back to sentence-transformers if ONNX deps are missing.
"""

import logging
import time

import numpy as np

log = logging.getLogger("yantrikdb.mcp")


class OnnxEmbedder:
    """Minimal ONNX embedder that implements the YantrikDB Embedder interface."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        t0 = time.time()
        model_path = hf_hub_download(model_name, "onnx/model.onnx")
        tokenizer_path = hf_hub_download(model_name, "tokenizer.json")

        self._session = ort.InferenceSession(model_path)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=256)
        self._dim = 384
        log.info("ONNX embedder loaded in %.1fs", time.time() - t0)

    def embed(self, text: str) -> list[float]:
        enc = self._tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        # Mean pooling + L2 normalize
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (token_embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / norm

        return pooled[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        encodings = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask
            token_type_ids[i, :length] = enc.type_ids

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (token_embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / norms

        return pooled.tolist()

    def dim(self) -> int:
        return self._dim


def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Load the fastest available embedder — ONNX first, PyTorch fallback."""
    # Normalize model name for huggingface hub
    hf_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"

    try:
        embedder = OnnxEmbedder(hf_name)
        return embedder
    except ImportError:
        log.info("ONNX deps not available, falling back to sentence-transformers (slow)")
    except Exception as e:
        log.warning("ONNX load failed (%s), falling back to sentence-transformers", e)

    # Fallback to PyTorch sentence-transformers
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    log.info("Loading sentence-transformers model: %s", model_name)
    model = SentenceTransformer(model_name)
    log.info("Model loaded in %.1fs", time.time() - t0)
    return model
