"""RAG index builder + retrievers.

Step 5  normal RAG : fixed-size chunks, plain cosine ranking.
Steps 6-8 fuzzy RAG : TF-IDF/Gaussian term weights -> entropy chunking -> M3E vectors ->
                      Chroma/HNSW candidates -> re-ranked with sim_fuzzy (Eq. 11).
"""
import json
import time

import config
from rag.embeddings import Embedder
from rag.entropy_chunking import EntropyChunker, FixedChunker, split_sentences
from rag.fuzzy_membership import sim_fuzzy
from rag.term_weighting import TermWeighter
from rag.vector_store import open_store, storage_objective


def corpus_units(documents):
    """Flatten parsed units while retaining source and location metadata for each indexed item."""
    units = []
    for d in documents:
        if d.get("units"):
            for unit in d["units"]:
                text = unit.get("content", "").strip()
                if not text:
                    continue
                kind = str(unit.get("modality", "text")).casefold()
                if kind in ("image", "caption"):
                    kind = "caption"
                elif kind in ("table", "tablerow", "record"):
                    kind = "record"
                units.append({
                    "doc_id": unit.get("doc_id", d["doc_id"]),
                    "kind": kind,
                    "text": text,
                    "source": unit.get("source", d.get("source", d["doc_id"])),
                    "source_path": unit.get("source_path", d.get("source_path", d.get("source", d["doc_id"]))),
                    "page_number": unit.get(
                        "page_number",
                        unit.get("position", 0) + 1 if d.get("modality") == "pdf" and
                        isinstance(unit.get("position", 0), int) else None,
                    ),
                    "row_number": unit.get("row_number"),
                    "section": unit.get("section", ""),
                    "evidence": unit.get("evidence", text),
                })
            continue
        for s in d.get("S_D", []):
            units.append(dict(doc_id=d["doc_id"], kind="text", text=s, source=d.get("source", d["doc_id"]),
                              source_path=d.get("source", d["doc_id"]), evidence=s))
        for c in d.get("C_I", []):
            if c:
                units.append(dict(doc_id=d["doc_id"], kind="caption", text=c, source=d.get("source", d["doc_id"]),
                                  source_path=d.get("source", d["doc_id"]), evidence=c))
        for r in d.get("records", []):
            units.append(dict(doc_id=d["doc_id"], kind="record", text=r, source=d.get("source", d["doc_id"]),
                              source_path=d.get("source", d["doc_id"]), evidence=r))
    return units


def build_index(mode: str = "fuzzy", documents=None, verbose=True):
    """Build and persist the vector library.  mode = 'fuzzy' | 'normal'."""
    t0 = time.time()
    if documents is None:
        documents = json.loads(config.PARSED_JSON.read_text(encoding="utf-8"))
    units = corpus_units(documents)
    sentence_corpus = [sentence for unit in units for sentence in split_sentences(unit["text"])]
    weighter = TermWeighter(sentence_corpus)
    chunker = EntropyChunker(weighter) if mode == "fuzzy" else FixedChunker()

    chunks = []
    for unit in units:
        for c in chunker.chunk(unit["text"]):
            chunks.append(dict(
                id=f"c{len(chunks):05d}",
                **{key: value for key, value in unit.items() if key != "text"},
                text=c,
            ))
    emb = Embedder()
    vecs = emb.encode([c["text"] for c in chunks])
    store = open_store(reset=True)
    store.add([c["id"] for c in chunks], [c["text"] for c in chunks], vecs,
              [{key: c[key] for key in ("doc_id", "kind", "source", "source_path", "page_number", "row_number", "section")
                if c.get(key) is not None} for c in chunks])

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_JSON.write_text(json.dumps(dict(mode=mode, embedder=emb.name, embedding_dim=emb.dim,
                                                  chunker="fuzzy-entropy" if mode == "fuzzy" else "fixed-baseline",
                                                  corpus=sentence_corpus, chunks=chunks),
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    lens = [len(c["text"]) for c in chunks]
    stats = dict(mode=mode, n_units=len(units), n_chunks=len(chunks),
                 avg_chunk_chars=round(sum(lens) / max(len(lens), 1), 2),
                 embedder=emb.name, store=store.kind, index_kb=round(store.index_size_kb(), 1),
                 top_terms=[(w, round(m, 3)) for w, m in weighter.top_terms(8)],
                 build_seconds=round(time.time() - t0, 2))
    if mode == "fuzzy" and chunks:      # Eq. (12), using every indexed vector as a representative query
        sims = []
        for c, v in zip(chunks, vecs):
            hits = store.query(v, 1)
            sims.append(sim_fuzzy(v, hits[0]["emb"], c["text"], hits[0]["text"], weighter) if hits else 0.0)
        stats["L_D"] = round(storage_objective(sims, store.index_size_kb(), len(chunks)), 5)
        stats["objective_queries"] = len(sims)
    if verbose:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


class Retriever:
    """Loads the persisted index and answers top-k queries (mode 'fuzzy' or 'normal')."""

    def __init__(self):
        meta = json.loads(config.CHUNKS_JSON.read_text(encoding="utf-8"))
        self.mode = meta["mode"]
        self.weighter = TermWeighter(meta["corpus"])
        self.emb = Embedder()
        if meta.get("embedder") != self.emb.name or meta.get("embedding_dim", self.emb.dim) != self.emb.dim:
            raise RuntimeError(
                f"Vector index uses {meta.get('embedder')} ({meta.get('embedding_dim')} dimensions), but the "
                f"configured embedder is {self.emb.name} ({self.emb.dim}). Rebuild it with `python main.py build-rag`."
            )
        self.store = open_store(reset=False)
        if self.store.count() == 0:       # NumPy fallback is not persisted: rebuild from chunks.json
            chunks = meta["chunks"]
            self.store.add([c["id"] for c in chunks], [c["text"] for c in chunks],
                           self.emb.encode([c["text"] for c in chunks]),
                           [{key: c[key] for key in ("doc_id", "kind", "source", "source_path", "page_number",
                                                     "row_number", "section") if c.get(key) is not None}
                            for c in chunks])

    def retrieve(self, query: str, top_k: int = config.TOP_K, fuzzy: bool = True, exclude_self: bool = False):
        """exclude_self: drop chunks that contain the query itself (used when the query is a corpus sentence)."""
        t0 = time.time()
        qv = self.emb.encode_one(query)
        n = top_k * config.CANDIDATE_FACTOR if fuzzy else top_k
        cands = self.store.query(qv, n + (3 if exclude_self else 0))
        if exclude_self:
            q = query.strip().lower()
            cands = [c for c in cands if q not in c["text"].lower()]
        for c in cands:
            c["score"] = sim_fuzzy(qv, c["emb"], query, c["text"], self.weighter) if fuzzy else c["cos"]
        cands.sort(key=lambda c: -c["score"])
        res = [dict(id=c["id"], text=c["text"], score=round(c["score"], 4), cos=round(c["cos"], 4),
                    **{key: c["meta"].get(key) for key in
                       ("doc_id", "kind", "source", "source_path", "page_number", "row_number", "section")})
               for c in cands[:top_k]]
        for r in res:
            r["ms"] = round((time.time() - t0) * 1000, 1)
        return res

    def context(self, query, top_k=config.TOP_K, fuzzy=True, exclude_self=True):
        return "\n".join(
            f"- [{r.get('source', r.get('doc_id', 'unknown'))}"
            f"{', page ' + str(r['page_number']) if r.get('page_number') is not None else ''}] {r['text']}"
            for r in self.retrieve(query, top_k, fuzzy, exclude_self)
        )
