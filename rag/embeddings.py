"""Chunk embeddings -- paper Sec. III-B-2(c): M3E-Base, 768-d, L2 normalised.

Hash embeddings remain available only as an explicitly selected offline/test backend.
"""
import hashlib
import re

import numpy as np

import config


class HashEmbedder:
    def __init__(self, dim=config.EMBEDDING_DIM):
        self.dim = dim
        self.name = "hash-ngram"

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        t = re.sub(r"\s+", " ", text.lower())
        words = re.findall(r"[一-鿿]|[a-z0-9]+(?:\.\d+)?", t)
        feats = [("w", w, 2.0) for w in words]
        feats += [("b", a + " " + b, 1.0) for a, b in zip(words, words[1:])]
        for w in words:                       # character 3-grams give robustness to morphology
            p = f"#{w}#"
            feats += [("c", p[i:i + 3], 0.5) for i in range(max(len(p) - 2, 1))]
        for kind, f, wt in feats:
            h = int(hashlib.md5(f"{kind}|{f}".encode()).hexdigest(), 16)
            v[h % self.dim] += wt * (1 if (h >> 64) & 1 else -1)
        return v

    def encode(self, texts):
        m = np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


class M3EEmbedder:
    def __init__(self, model=config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        if not callable(get_dimension):
            get_dimension = getattr(self.model, "get_sentence_embedding_dimension", None)
        if not callable(get_dimension):
            raise RuntimeError("SentenceTransformer does not expose its embedding dimension.")
        self.dim = get_dimension()
        if self.dim != config.EMBEDDING_DIM:
            raise ValueError(f"{model} returned {self.dim} dimensions; expected {config.EMBEDDING_DIM}")
        self.name = model

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=32), dtype=np.float32)


class Embedder:
    """Select the configured embedding backend without silently changing retrieval semantics."""

    def __init__(self, backend: str = None):
        backend = backend or config.EMBEDDING_BACKEND
        if backend == "m3e":
            try:
                self.impl = M3EEmbedder()
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "M3E embeddings are required but unavailable. Install sentence-transformers and ensure "
                    "moka-ai/m3e-base is downloadable, or explicitly set EMBEDDING_BACKEND=hash for offline tests."
                ) from exc
        elif backend == "hash":
            self.impl = HashEmbedder()
        else:
            raise ValueError(f"Unsupported EMBEDDING_BACKEND {backend!r}; choose 'm3e' or 'hash'.")
        self.name, self.dim = self.impl.name, self.impl.dim

    def encode(self, texts):
        out = self.impl.encode(list(texts))
        if out.ndim != 2 or out.shape[1] != self.dim:
            raise ValueError(f"Embedding backend {self.name!r} returned shape {out.shape}; expected (*, {self.dim})")
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)   # ||v||_2 = 1

    def encode_one(self, text):
        return self.encode([text])[0]
