# How to run (RUN.md)

Python **3.10+**. Commands are for Linux/macOS; on Windows activate with `.venv\Scripts\activate`.

## 1. Install
```bash
cd multimodal-industrial-kg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
`sentence-transformers` pulls in PyTorch (~2 GB). The paper-aligned default is M3E-Base (`moka-ai/m3e-base`, 768 dimensions); the first build downloads the model. If it cannot load, the run fails instead of silently using different embeddings. `EMBEDDING_BACKEND=hash` is an explicit offline/test-only option, not a paper-equivalent embedding.

## 2. Quick check with the bundled sample data (2 minutes, offline)
```bash
EMBEDDING_BACKEND=hash LLM_BACKEND=mock USE_NEO4J=0 python main.py all
python main.py serve            # open http://127.0.0.1:5000
```
The sample data is synthetic and the mock LLM is a rule-based stand-in, so the printed scores only prove that the code runs.
They are **not** comparable with the paper.

## 3. Real run
```bash
cp .env.example .env            # set DEEPSEEK_API_KEY=...
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5   # optional
# delete the sample files in data/raw/* and data/training/*, add your own (section 4), then:
python main.py all
python main.py serve
```
Run steps one by one:

| Command | Paper steps |
|---|---|
| `python main.py init-data` | 1 (sample data only; skip with real data) |
| `python main.py parse` | 2-3 MinerU parser -> `data/parsed/documents.json` |
| `python main.py build-rag` | 5-7 normal RAG, fuzzy TF-IDF, entropy chunking, M3E, Chroma/HNSW |
| `python main.py query "foam color changes from dark to lead gray"` | 8 fuzzy retrieval (plain vs fuzzy, top-3) |
| `python main.py optimize-prompt` | 9-10 prompt + adaptive optimisation |
| `python main.py extract [--max-units 20]` | 11-12 stage-1 triples + LLM refinement |
| `python main.py build-kg` | 13-14 ontology mapping + Neo4j |
| `python main.py evaluate [--full-semantic]` | 16-17 extraction / structural / semantic metrics |
| `python main.py serve [--port 5000]` | 15 D3 visualisation |

Tests: `python -m unittest discover -s tests -t .`

`python main.py build-kg --dry-run` assembles and validates the graph without writing it. A normal rebuild is
transactional within the configured `NEO4J_GRAPH_SCOPE` (default `default`): it replaces only that managed subgraph,
keeps other graph scopes untouched, and writes a snapshot of the prior graph to `outputs/kg_backup.json` first.
The local JSON graph is atomically replaced and its previous version is retained as `outputs/kg_local.json.bak`.

The parser filters short, repeated page-furniture, and visibly corrupted prose fragments. Semantic assessment validates
triples with the same ontology validator used by graph assembly and reports input, accepted, and rejected triple counts.
After changing source documents or parser behavior, rerun `parse`, `build-rag`, `optimize-prompt`, `extract`, `build-kg`,
and `evaluate` so downstream files and metrics reflect the new inputs.

### Bounded end-to-end presentation demo

First refresh the parsed data and RAG index if the source files have changed:

```powershell
python main.py parse
python main.py build-rag
```

Then run one document through prompt optimization, two-stage extraction (including refinement), graph building, and
evaluation:

```powershell
python main.py demo --source "Froth Images from Flotation Laboratory Test in Magotteaux Cell - data-08-00069.pdf" --max-units 12 --eval-samples 5 --replace-current
python main.py serve
```

The demo runs prompt optimization on the configured annotated training subset, then initial extraction and refinement
for each selected unit, builds the graph, and evaluates a deterministic sample of held-out annotations plus dynamically
sampled demo units. It replaces `optimized_prompt.json`, `outputs/triples.json`, the current graph (including Neo4j
`KGNode` data when enabled), and `outputs/metrics.json`. The sample-scoped metrics are for demonstrating the workflow,
not full-corpus results. Increase `--max-units` and `--eval-samples` for a longer presentation run; use the separate
full-pipeline commands when you need full-corpus outputs.

### Optional LoRA training and separate refinement model

Use expert-annotated JSONL training data (the bundled set is synthetic and intended only for smoke tests). Install
the optional `peft` dependency with `pip install -r requirements.txt`, then train on a CUDA bf16-capable GPU:

```powershell
python main.py train-lora
```

This trains the paper-configured DeepSeek-R1-Distill-Llama-8B stage-one adapter and saves it under
`outputs/lora_adapter`. With the default `STAGE1_BACKEND=auto`, extraction uses that adapter once the saved adapter is
present. `STAGE1_BACKEND=lora` requires the adapter; `STAGE1_BACKEND=default` bypasses it.

### Local Ollama model

For a local Ollama server, set these in `.env`. Stage 1 generates initial triples; stage 2 receives those triples plus the original source and refines them. They are separate model calls and may use different models; the backend/model used for each unit is saved in `outputs/triples.json`. Choose a distinct larger model for stage two to follow the paper's two-model design. The small-model example is only
for machines that cannot run a larger model locally.

```dotenv
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
STAGE1_MODEL=qwen2.5:1.5b
OLLAMA_STAGE2_MODEL=deepseek-r1:14b
OLLAMA_NUM_PREDICT=256
LLM_TIMEOUT=180
OPT_DEV_SIZE=40
OPT_MAX_ITERS=3
OPT_USE_RAG=1
```

Pull each configured model with `ollama pull <model>`. The stage-two model requires substantially more memory; if
unavailable, use the DeepSeek API backend with a configured key or deliberately select a smaller refinement model.
Prompt optimization evaluates the current prompt on the configured annotated development set,
uses missed/spurious triples and P/R/F1/consistency to update its rules/examples, and saves per-version scores in
`prompting/prompts/history.json`. `OPT_MAX_ITERS=3` allows up to three feedback updates (four prompt evaluations);
optimization stops early when the current version reaches the paper's F1 and consistency thresholds. Optimization
quality depends on representative expert-annotated examples;
synthetic training data only tests the workflow. Each optimization run has a unique folder under
`prompting/prompts/optimization_runs/`; every iteration saves the exact prompt, test inputs, gold and generated
triples, per-example and aggregate scores, missed/spurious triples, and the next updated prompt.

The fuzzy retrieval path uses fuzzy-entropy chunking, M3E vectors, cosine-selected HNSW candidates, then fuzzy
similarity reranking. `python main.py build-rag` builds the fixed-chunk baseline first and the fuzzy index second;
the fuzzy index is the one used by extraction. Evaluation reports held-out extraction metrics, triple validation and
contradiction consistency, graph-structure metrics, graph-integrity checks, and LLM-judged semantic scores.

## 3b. Frontend (yes, there is one)
`python main.py serve` starts the Flask backend (`backend/api.py`) and the D3 page (`frontend/graph.html`) at
http://127.0.0.1:5000. It shows a force-directed graph with nodes coloured by type, arrows for direction, image nodes as
thumbnails, a clickable legend filter, search, zoom/drag, an edge-label toggle, a details panel and the S1/S2/S_Final scores.
It reads from Neo4j if reachable, otherwise from `outputs/kg_local.json`. Neo4j's own browser is at http://localhost:7474.

## 4. Data

### 4a. What is bundled (generated, synthetic)
`python main.py init-data` (also run automatically by `all` when `data/raw/` is empty) writes a complete but **invented**
dataset with every data type from the paper (Sec. II). It exists so the full pipeline runs end to end; it is not real plant
data and not the paper's data.

| Data type (paper) | Generated files | How it is parsed |
|---|---|---|
| **Structured data** (inspection records, production logs, timestamps, equipment, status, problems) | `data/raw/csv/inspection_records.csv` (150 rows, Fig. 3 format), `data/raw/csv/production_log.csv` (72 hourly rows) | each row -> a sentence (Eq. 1) |
| **Unstructured text** (accident reports, regulations, standards, equipment manuals, academic papers, technical reports) | 6 PDFs in `data/raw/pdf/`: `academic_paper_...`, `industry_standard_...`, `equipment_manual_...`, `accident_report_...`, `production_regulations`, `technical_report_...` | text fragments `S_D` |
| **Images and figures** (process diagrams, equipment images, figures from documents) | figures inside the PDFs (flowsheets, equipment schematic, froth image, raster + vector charts) and 5 loose images in `data/raw/images/` with same-name `.txt` captions (froth normal / overflow / thin layer, flotation cell, flowsheet) | images `I_D` + captions `C_I` |
| **Tables and formulas** | numbered tables and numbered equations in every PDF and in the HTML bulletin | tables -> image + caption **and** one sentence per row; equations -> `formulas` + text |
| **Semi-structured data** (identifiable sections, captions, tables, mixed formats) | the sectioned PDFs and `data/raw/html/maintenance_bulletin.html` (headings, table with caption, figure + figcaption, formula) | `sections`, tables, figures |
| **Annotated sentences** | `data/training/flotation_train.jsonl` (100 template sentences with gold triples) | prompt optimisation + evaluation |

`data/parsed/documents.json` shows the result per document: `S_D`, `I_D`, `C_I`, `records`, `formulas`, `sections`.

### 4b. What the paper used, and where to get substitutes
The paper's own data is **not public**. The table lists what it describes and what you can download instead. Download the
files yourself (I could not fetch them from my build environment) and drop them in the folder shown. Delete the synthetic
files in `data/raw/*` and `data/training/*` first.

| Data type | Folder | External resources |
|---|---|---|
| Structured plant records | `data/raw/csv/` | The paper's logs are private; use your own plant logs in the Fig. 3 format (`Time, Process, Equipment, Operating status, Problem description`). Public numeric stand-in: iron-ore flotation plant time series (Mar-Sep 2017), Mendeley Data DOI 10.17632/2vsrjbtfp5.1 / Kaggle "Quality Prediction in a Mining Process". It is numeric only (no problem text) and iron ore, so it adds little to the graph; set `CSV_MAX_ROWS=50`. |
| Academic papers, technical reports, books | `data/raw/pdf/` | Open-access flotation literature: the MDPI books *Advances in Flotation of Copper, Lead and Zinc Minerals* and *Design, Modeling, Optimization and Control of Flotation Process*; open-access articles in MDPI *Minerals* and *Processes*. |
| Accident reports | `data/raw/pdf/` | MSHA publishes metal/nonmetal mine accident and fatality investigation reports (msha.gov/fatality-reports). Choose the ones about processing plants / mills. |
| Industry standards, regulations | `data/raw/pdf/` | Most standards are paywalled; use the ones you have licensed access to, or public safety regulations. |
| Equipment manuals | `data/raw/pdf/` | Public product manuals / datasheets from flotation-machine vendors. |
| Images and figures | automatic | The paper's images are the figures inside its PDFs (process diagrams, tables rendered as images, chemical structures, charts, equipment pictures, each with a "Fig. N" caption), so real PDFs supply them. Extra loose images: `data/raw/images/` + same-name `.txt` caption. Public froth photos: MDPI *Data* 2023, 8(4), 69 "Froth Images from Flotation Laboratory Test in Magotteaux Cell" (check its licence and file list). |
| Tables and formulas | automatic | come with the PDFs; install MinerU (`pip install -U "mineru[all]"`) for the best table, formula and figure extraction on real documents. |
| Semi-structured | `data/raw/html/` | Any `.html` article / manual with `<h1-3>`, `<p>`, `<table><caption>`, `<figure><img><figcaption>`, `<div class="formula">`. Journal HTML pages saved to disk work if they use these tags. |
| Annotated sentences (~100) | `data/training/flotation_train.jsonl` | Must be created by you (draft with an LLM, correct by hand), same JSONL format as the sample. First 40 lines drive prompt optimisation, the rest is the held-out test. |
| LLM / embeddings | `.env` / automatic | `DEEPSEEK_API_KEY`; `moka-ai/m3e-base` is downloaded from Hugging Face on first use. |

## 5. Troubleshooting
* *`No module named 'parser'` or odd import errors*: use Python 3.10+ and run from the project root.
* *Neo4j not reachable*: nothing breaks, the graph is stored in `outputs/kg_local.json`. Force Neo4j with `USE_NEO4J=1`.
* *M3E download fails*: set `EMBEDDING_BACKEND=hash` or pre-download the model on a machine with internet.
* *Images missing in the graph*: the web page serves them from `data/parsed/images/`; re-run `parse` and `build-kg`.
* *Chinese corpus*: set `H_MIN=4.0` in `.env` (default 2.6 is calibrated for English sentences).
