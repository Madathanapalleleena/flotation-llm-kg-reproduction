import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rag.embeddings import Embedder
from rag.entropy_chunking import EntropyChunker
from rag.retriever import Retriever, corpus_units
from rag.term_weighting import TermWeighter


class CandidateStore:
    def query(self, embedding, n):
        return [
            dict(id="cosine-first", text="unrelated phrase", meta={}, cos=0.95,
                 emb=np.array([0.95, (1 - 0.95 ** 2) ** 0.5], dtype=np.float32)),
            dict(id="fuzzy-first", text="aeration rate affects froth depth", meta={}, cos=0.80,
                 emb=np.array([0.80, 0.60], dtype=np.float32)),
        ][:n]


class EmbeddingAndRetrievalTests(unittest.TestCase):
    def test_hash_backend_is_only_used_when_explicitly_selected(self):
        embedder = Embedder("hash")
        vectors = embedder.encode(["aeration rate", "froth depth"])
        self.assertEqual(embedder.name, "hash-ngram")
        self.assertEqual(vectors.shape, (2, 768))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)

    def test_m3e_backend_validates_768_dimensions_and_normalizes_vectors(self):
        model = Mock()
        model.get_embedding_dimension = None
        model.get_sentence_embedding_dimension.return_value = 768
        model.encode.return_value = np.full((1, 768), 2.0, dtype=np.float32)
        sentence_transformers = SimpleNamespace(SentenceTransformer=Mock(return_value=model))
        with patch.dict(sys.modules, {"sentence_transformers": sentence_transformers}):
            embedder = Embedder("m3e")

        vector = embedder.encode(["sample"])

        self.assertEqual(embedder.name, "moka-ai/m3e-base")
        self.assertEqual(vector.shape, (1, 768))
        self.assertAlmostEqual(float(np.linalg.norm(vector[0])), 1.0)
        model.encode.assert_called_once_with(["sample"], normalize_embeddings=True, batch_size=32)

    @patch("rag.embeddings.M3EEmbedder", side_effect=RuntimeError("model download failed"))
    def test_m3e_failure_does_not_silently_fall_back(self, _):
        with self.assertRaisesRegex(RuntimeError, "model download failed"):
            Embedder("m3e")

    def test_retriever_reranks_cosine_candidates_with_fuzzy_similarity(self):
        retriever = Retriever.__new__(Retriever)
        retriever.mode = "fuzzy"
        retriever.emb = Mock()
        retriever.emb.encode_one.return_value = np.array([1.0, 0.0], dtype=np.float32)
        retriever.weighter = TermWeighter(["aeration rate affects froth depth"] * 5)
        retriever.store = CandidateStore()

        result = retriever.retrieve("aeration rate", top_k=1, fuzzy=True)

        self.assertEqual(result[0]["id"], "fuzzy-first")
        self.assertGreater(result[0]["score"], 1.0)

    def test_entropy_chunker_merges_low_entropy_text_until_threshold(self):
        chunker = EntropyChunker(TermWeighter(["first sentence second sentence"] * 3))
        with patch.object(chunker, "H", side_effect=[0.0, 10.0, 10.0]) as entropy:
            chunks = chunker.chunk("First sentence. Second sentence.")

        self.assertEqual(chunks, ["First sentence. Second sentence."])
        self.assertEqual(entropy.call_count, 3)

    def test_rag_corpus_units_preserve_page_and_document_provenance(self):
        units = corpus_units([{
            "doc_id": "paper",
            "source": "paper.pdf",
            "modality": "pdf",
            "units": [{
                "doc_id": "paper",
                "source": "paper.pdf",
                "source_path": "data/raw/pdf/paper.pdf",
                "modality": "Text",
                "position": 2,
                "content": "Flotation recovery increases with improved aeration.",
                "evidence": "full source paragraph",
            }],
        }])

        self.assertEqual(units[0]["page_number"], 3)
        self.assertEqual(units[0]["source"], "paper.pdf")
        self.assertEqual(units[0]["evidence"], "full source paragraph")


if __name__ == "__main__":
    unittest.main()
