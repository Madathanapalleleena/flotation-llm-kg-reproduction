"""Vector database -- paper Sec. III-B-2(d).

Chroma + HNSW (M = 8, ef_construction = 40), indexed in batches of 100.
Falls back to an exact NumPy store when chromadb is not installed.

Design objective, Eq. (12):
    L(D) = 1/N * sum_i (1 - sim_fuzzy(v_i, q_i))^2  +  lambda * S_index(D) / N
"""
import shutil
from pathlib import Path

import numpy as np

import config


def _dir_size_kb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024.0 if Path(path).exists() else 0.0


class NumpyStore:
    kind = "numpy-exact"

    def __init__(self, path: Path = None, reset=False):
        self.ids, self.texts, self.metas, self.mat = [], [], [], None

    def add(self, ids, texts, embs, metas):
        self.ids += list(ids)
        self.texts += list(texts)
        self.metas += list(metas)
        embs = np.asarray(embs, dtype=np.float32)
        self.mat = embs if self.mat is None else np.vstack([self.mat, embs])

    def query(self, emb, n):
        sims = self.mat @ np.asarray(emb, dtype=np.float32)
        idx = np.argsort(-sims)[:n]
        return [dict(id=self.ids[i], text=self.texts[i], meta=self.metas[i], cos=float(sims[i]), emb=self.mat[i]) for i in idx]

    def count(self):
        return len(self.ids)

    def index_size_kb(self):
        return 0.0 if self.mat is None else self.mat.nbytes / 1024.0


class ChromaStore:
    kind = "chroma-hnsw"

    def __init__(self, path: Path = config.CHROMA_DIR, reset=False, name: str = config.COLLECTION):
        import chromadb
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        if reset:
            try:
                self.client.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass
        # HNSW parameters differ by chroma version: try the new configuration API, then legacy metadata.
        try:
            self.col = self.client.get_or_create_collection(
                name, configuration={"hnsw": {"space": "cosine", "ef_construction": config.HNSW_EF_CONSTRUCTION,
                                              "max_neighbors": config.HNSW_M}})
        except Exception:  # noqa: BLE001
            self.col = self.client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine", "hnsw:M": config.HNSW_M,
                                "hnsw:construction_ef": config.HNSW_EF_CONSTRUCTION})

    def add(self, ids, texts, embs, metas):
        ids, texts, metas = list(ids), list(texts), list(metas)
        embs = np.asarray(embs, dtype=np.float32)
        for s in range(0, len(ids), config.INDEX_BATCH):          # batches of 100 [paper]
            e = s + config.INDEX_BATCH
            self.col.add(ids=ids[s:e], documents=texts[s:e], embeddings=embs[s:e].tolist(),
                         metadatas=[m or {"_": 0} for m in metas[s:e]])

    def query(self, emb, n):
        n = min(n, self.col.count())
        if n == 0:
            return []
        r = self.col.query(query_embeddings=[np.asarray(emb, dtype=float).tolist()], n_results=n,
                           include=["documents", "metadatas", "distances", "embeddings"])
        out = []
        for i in range(len(r["ids"][0])):
            out.append(dict(id=r["ids"][0][i], text=r["documents"][0][i], meta=r["metadatas"][0][i],
                            cos=1.0 - float(r["distances"][0][i]), emb=np.asarray(r["embeddings"][0][i], dtype=np.float32)))
        return out

    def count(self):
        return self.col.count()

    def index_size_kb(self):
        return _dir_size_kb(self.path)


def open_store(reset=False):
    try:
        import chromadb  # noqa: F401
        return ChromaStore(reset=reset)
    except ImportError:
        print("[vector_store] chromadb not installed -> exact NumPy store (not persisted)")
        return NumpyStore()


def storage_objective(sims, index_size_kb: float, n: int, lam: float = config.LAMBDA_STORAGE) -> float:
    """Eq. (12).  `sims` = sim_fuzzy(v_i, q_i) for each representative query vector."""
    sims = np.asarray(sims, dtype=float)
    n = max(n, 1)
    return float(np.mean((1.0 - sims) ** 2) + lam * index_size_kb / n)
