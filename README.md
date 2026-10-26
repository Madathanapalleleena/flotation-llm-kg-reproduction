# Multimodal Industrial Knowledge Graph (froth flotation)

See **RUN.md** for step-by-step run instructions and the dataset guide.

Re-implementation of *"LLM-Driven Multimodal Knowledge Graph Construction for Industrial Process With Prompt
Optimization and Fuzzy RAG"* (Xie et al., IEEE Trans. Fuzzy Systems 34(4), 2026) — **without the LoRA stage**.

## Quick start

Requires **Python 3.10+** (the local `parser/` package would clash with the stdlib `parser` module on older versions).

```bash
cd multimodal-industrial-kg
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# A) 2-minute offline smoke test (no API key, no model download, no Neo4j) -- uses SYNTHETIC data + a rule-based mock LLM
EMBEDDING_BACKEND=hash LLM_BACKEND=mock USE_NEO4J=0 OPT_MIN_ITERS=0 python main.py all
python main.py serve                                     # open http://127.0.0.1:5000  (any --port works; image thumbnails follow it)

# B) The real thing
cp .env.example .env                                     # put your DEEPSEEK_API_KEY in it
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
python main.py all                                       # downloads moka-ai/m3e-base on first use (~400 MB)
python main.py serve
```

Step by step (same order as the paper's flow):

| Step | Command | Code |
|---|---|---|
| 1 dataset | `python main.py init-data` (sample only) | `data/generate_full_dataset.py` |
| 2–3 parser → JSON | `python main.py parse` | `parser/mineru_parser.py` |
| 4 ontology | (loaded automatically) | `ontology/ontology.json`, `ontology/knowledge_tree.py` |
| 5–7 normal RAG, fuzzy TF-IDF, entropy chunking | `python main.py build-rag` | `rag/term_weighting.py`, `fuzzy_membership.py`, `entropy_chunking.py`, `embeddings.py`, `vector_store.py`, `retriever.py` |
| 8 fuzzy retrieval | `python main.py query "foam colour changes from dark to lead gray"` | `rag/retriever.py` |
| 9–10 prompt + optimisation | `python main.py optimize-prompt` | `prompting/base_prompt.py`, `prompt_optimizer.py` |
| 11–12 triples + LLM refinement | `python main.py extract` (`--max-units 20` for a cheap API test) | `llm/extraction.py`, `llm/refinement.py` |
| 13–14 ontology mapping + Neo4j | `python main.py build-kg` | `kg/*.py` |
| 15 D3 visualisation | `python main.py serve` | `backend/api.py`, `frontend/graph.html` |
| 16–17 evaluation | `python main.py evaluate [--full-semantic]` | `evaluation/*.py` |
| 18 LoRA | **removed** | `llm/lora/README.md` |

Formula tests: `python -m unittest tests.test_formulas -v` (reproduces the paper's S₁ = 0.9774 and S_Final = 0.9195 from its component values).

## Datasets you must upload yourself

The sample data created by `init-data` is **synthetic and tiny** — it exists only so the code runs. To reproduce the
paper you need to supply (I cannot download anything from the internet in my build environment, and the paper's own
dataset is not distributed with it):

| What | Where | Format |
|---|---|---|
| Flotation PDFs: standards, books, papers, equipment manuals, accident reports, regulations (paper §II) | `data/raw/pdf/*.pdf` | any text/figure PDF; delete the sample ones first |
| Froth images / flowsheets not inside PDFs | `data/raw/images/` | image + optional same-name `.txt` caption sidecar |
| Semi-structured HTML documents (optional) | `data/raw/html/*.html` | headings, tables with captions, figure + figcaption |
| Plant inspection logs (paper Fig. 3) | `data/raw/csv/*.csv` | columns `Time, Process, Equipment, Operating status, Problem description` (other column names work if you edit `record_to_sentence`) |
| **~100 expert-annotated sentences** with gold triples (paper §IV-A) | `data/training/flotation_train.jsonl` | one JSON per line: `{"id":0,"system":"…","user":"<sentence>","assistant":"{\"triples\":[{\"subject\":…,\"subject_type\":…,\"relation\":…,\"object\":…,\"object_type\":…}]}"}` — drives prompt optimisation (first 40) and held-out evaluation (rest) |
| API access | `.env` → `DEEPSEEK_API_KEY` | DeepSeek (or any OpenAI-compatible endpoint) |
| Embedding model | auto-downloaded | `moka-ai/m3e-base` from Hugging Face |
| MinerU (optional) | `pip install -U "mineru[all]"` | if absent, a PyMuPDF fallback with the same JSON schema is used |

Paper corpus is Chinese. For Chinese text the tokenizer works per character, entropies land around 4–5 bits (paper:
4.2–5.4), so set `H_MIN≈4.0` in `.env`; the default 2.6 is calibrated for short English sentences.

## Where each formula lives

| Eq. | What | File |
|---|---|---|
| 1–3 | D_stru, D_unstru, D_JSON | `parser/mineru_parser.py` |
| 4–5 | knowledge tree T = (V, E, A) | `ontology/knowledge_tree.py` |
| 6 | P₀ = {R₀, C₀, E₀, O₀, T₀} | `prompting/base_prompt.py` |
| 7 | P, R, F1, C_sis, S_triplet (0.3/0.3/0.25/0.15, thr 0.85) | `evaluation/extraction_metrics.py` |
| – | update rules F1<0.6, C_sis<0.8, R<P; stop F1≥0.90 ∧ C_sis≥0.8 | `prompting/prompt_optimizer.py` |
| 8 | Gaussian TF-IDF membership | `rag/term_weighting.py`, `rag/fuzzy_membership.py` |
| 9–10 | fuzzy entropy H(c), μ_h(w) | `rag/fuzzy_membership.py` |
| – | H_min^fuzzy, merge rule (≤50 / ≤100 chars) | `rag/fuzzy_membership.py`, `rag/entropy_chunking.py` |
| 11 | sim_fuzzy, M_fuzzy | `rag/fuzzy_membership.py`, `rag/retriever.py` |
| 12 | storage objective L(D); HNSW M=8, ef=40, batch 100 | `rag/vector_store.py` |
| 13–16 | LoRA | removed |
| 17 | T_m = f_LLM(x, f_Fine(x, C_RAG)) | `llm/refinement.py` |
| 18 | v* = argmax deg | `kg/representative_node.py` |
| 19–21 | E = E_triples ∪ E_img ∪ E_onto | `kg/ontology_mapper.py` |
| 22–28 | I_s, C_s, B_s, Z_s, P_s, R_s, S₁ | `evaluation/structural_metrics.py` |
| 29–30 | dynamic sampling, S₂, S_Final | `evaluation/semantic_metrics.py` |

## Things the paper does not specify (my choices — all in `config.py`, marked `[assumed]`)

* Nine relation types and six entity types (the prompt in Fig. 6 gives the counts only). Mine: entities `RawOre, ProcessStage, Equipment, Parameter, Reagent, Product`; relations `affects, increases, decreases, keeps, causes, uses, produces, follows, has_property`. Edit `ontology/ontology.json`.
* `H_min` (base entropy threshold), `σ_s` (Eq. 29, default 0.1; result is clipped to 1), `λ` (Eq. 12), the cut-off for the `terms` set (μ ≥ 0.5), max optimisation iterations.
* Eq. 27: `T_mid` and `T_target` are undefined in the paper → midpoint of [T_min, T_max]; `N_process` = simple cycles of the ontology process sub-graph.
* Eq. 23: a core stage counts as covered when extracted knowledge is attached to it. With a small KG expect C_s < 1.
* "Triple label" in S_triplet = agreement of subject/object ontology types.
* TF-IDF "document" = one sentence; W_term uses μ_h of shared terms; B_context = shared terms / terms in the query.

## Honest limitations

* **LoRA removed:** stage 1 is a plain chat model (default the same DeepSeek model as stage 2; point `STAGE1_*` at a local 8B model to get closer to the paper).
* **Django → Flask** for the web layer (same JSON + D3 force layout, thumbnails for image nodes, legend filter, zoom, drag).
* **Baselines UIE and Rule+LTP** from Table III need external models and are not included; `evaluate` compares the proposed pipeline against the LLM-FewShot baseline (P₀, no RAG).
* **The offline MockLLM is not a language model.** It is a lexicon + regex extractor that reacts to the optimiser's rules so the loop can be tested. Scores produced with it (and with the synthetic data) are *not* comparable to the paper's (F1 0.85, S₁ 0.97, S₂ 0.88, S_Final 0.92). Real numbers need real data + a real LLM.
* Neo4j: if no server is reachable the graph is kept in `outputs/kg_local.json` and everything else works the same; with Neo4j up (`USE_NEO4J=1` to force) it is written to the database and read back from it.
* Paper-aligned retrieval uses M3E-Base embeddings (768 dimensions), fuzzy-entropy chunking, and fuzzy-similarity reranking. Hash embeddings are only an explicit offline/test option. Prompt optimization uses feedback to revise rules/examples and records every evaluated version; each extracted unit records the backend/model used for both extraction stages. Evaluation reports extraction quality, triple conflicts, graph structure, graph integrity, and semantic scores.
* Parsed PDF, HTML, and CSV units retain source, page/row, section, and evidence metadata into the vector chunks, extracted triples, and graph relationships. `python main.py build-kg --dry-run` validates without writing; a normal Neo4j rebuild is scoped and transactional, and saves the previous graph to `outputs/kg_backup.json`.

## What was verified in the build environment (and what was not)

| Verified here (offline) | Not verifiable here -- try on your machine |
|---|---|
| `python main.py all` end to end with `EMBEDDING_BACKEND=hash LLM_BACKEND=mock USE_NEO4J=0` | Real DeepSeek calls (no API key / network in the sandbox) |
| 51 unit tests incl. paper formulas, embeddings, fuzzy retrieval, prompt feedback, parsing provenance, and graph integrity | Live M3E model download/inference (requires Hugging Face model access) |
| Chroma HNSW created with M = 8, ef_construction = 40, batches of 100 | Live Neo4j write/read (Cypher reviewed, not executed against a server) |
| Flask API + D3 page rendered in headless Chromium: ontology backbone, typed nodes, 3 image thumbnails | MinerU parsing (PyMuPDF fallback is what ran) |
