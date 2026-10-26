"""Central configuration.

Every constant that comes from the paper (IEEE TFS vol.34 no.4, 2026, Xie et al.)
is marked [paper].  Values the paper does NOT specify are marked [assumed] so
they are easy to find and change.  Anything can be overridden through
environment variables or a ``.env`` file in the project root.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if v[:1] not in ('"', "'"):          # drop inline comments:  KEY=value   # comment
                    v = v.split(" #", 1)[0].split("\t#", 1)[0].strip()
                os.environ.setdefault(k.strip(), v.strip('"').strip("'"))


_load_dotenv()


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


# ----------------------------------------------------------------- paths
RAW_PDF_DIR = ROOT / "data" / "raw" / "pdf"
RAW_IMG_DIR = ROOT / "data" / "raw" / "images"
RAW_CSV_DIR = ROOT / "data" / "raw" / "csv"
RAW_HTML_DIR = ROOT / "data" / "raw" / "html"   # semi-structured HTML/XML documents (optional)
PARSED_JSON = ROOT / "data" / "parsed" / "documents.json"
MINERU_OUT_DIR = ROOT / "data" / "parsed" / "mineru_out"
TRAIN_JSONL = ROOT / "data" / "training" / "flotation_train.jsonl"
ONTOLOGY_JSON = ROOT / "ontology" / "ontology.json"
PROMPT_DIR = ROOT / "prompting" / "prompts"
LORA_DIR = ROOT / "llm" / "lora"
OUTPUT_DIR = ROOT / "outputs"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
CHUNKS_JSON = OUTPUT_DIR / "chunks.json"
TRIPLES_JSON = OUTPUT_DIR / "triples.json"
KG_JSON = OUTPUT_DIR / "kg_local.json"       # used when Neo4j is unavailable
KG_BACKUP_JSON = OUTPUT_DIR / "kg_backup.json"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
# Images are "hosted on a repository to obtain unique URLs" [paper].  Point this
# at your GitHub raw URL prefix; default serves them from the local Flask API.
CSV_MAX_ROWS = _i("CSV_MAX_ROWS", 300)
# image filtering in the PDF parser: logos/icons/page decorations are dropped, and (default) so are images without a
# caption, because an image only joins the graph through the entities named in its caption (Eq. 21).
MIN_IMAGE_PX = _i("MIN_IMAGE_PX", 120)                    # [assumed] min(width, height) in pixels
IMAGE_REQUIRE_CAPTION = _i("IMAGE_REQUIRE_CAPTION", 1)    # [assumed] 1 = keep only captioned figures/tables
# built-in local LLM (LLM_BACKEND=local): a small open model run by the project itself through transformers
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")   # try Qwen/Qwen2.5-0.5B-Instruct on a weak PC
LOCAL_DEVICE = os.getenv("LOCAL_DEVICE", "")              # "" = cuda if available else cpu
LOCAL_MAX_NEW_TOKENS = _i("LOCAL_MAX_NEW_TOKENS", 400)   # [assumed] long plant logs are sub-sampled evenly to this many records
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://127.0.0.1:5000/images")

# ------------------------------------------------------------------- LLM
# "auto" -> deepseek if DEEPSEEK_API_KEY is set, otherwise the offline mock.
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# Stage 1 = initial triples (paper: LoRA-tuned DeepSeek-R1-Distill-Llama-8B).
# LoRA is removed in this build, so stage 1 uses a plain model.  Point it at a
# local OpenAI-compatible server (e.g. Ollama deepseek-r1:8b) if you want a
# small model here, otherwise it defaults to the same DeepSeek API model.
STAGE1_MODEL = os.getenv("STAGE1_MODEL", "deepseek-chat")
STAGE1_BASE_URL = os.getenv("STAGE1_BASE_URL", DEEPSEEK_BASE_URL)
STAGE1_API_KEY = os.getenv("STAGE1_API_KEY", DEEPSEEK_API_KEY)
# Stage 2 = refinement with the large-scale model [paper: DeepSeek].
STAGE2_MODEL = os.getenv("STAGE2_MODEL", "deepseek-chat")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_STAGE2_MODEL = os.getenv("OLLAMA_STAGE2_MODEL", STAGE2_MODEL)
OLLAMA_NUM_PREDICT = _i("OLLAMA_NUM_PREDICT", 256)
LLM_TEMPERATURE = _f("LLM_TEMPERATURE", 0.0)
LLM_TIMEOUT = _i("LLM_TIMEOUT", 120)
LLM_MAX_TOKENS = _i("LLM_MAX_TOKENS", 512)

# ------------------------------------------------------------ embeddings
# "m3e" -> sentence-transformers moka-ai/m3e-base (768-d) [paper]
# "hash" -> deterministic offline n-gram hashing embedder (fallback / tests)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "m3e")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "moka-ai/m3e-base")
EMBEDDING_DIM = 768  # [paper]

# --------------------------------------------- fuzzy term weighting  (Eq. 8)
GAUSS_CENTER = 0.8   # [paper] Eq.8
GAUSS_SIGMA = 0.2    # [paper] Eq.8
TERM_MU_THRESHOLD = _f("TERM_MU_THRESHOLD", 0.5)   # [assumed] membership needed to enter set `terms`
DYN_STOP_DF_RATIO = _f("DYN_STOP_DF_RATIO", 0.5)   # [assumed] dynamic stop-word df ratio

# ------------------------------------------- fuzzy entropy      (Eq. 9, 10)
ENTROPY_MU_COEF = 0.3     # [paper] Eq.9
NUM_BOOST = 0.2           # [paper] Eq.10  (1.0 + 0.2*I_num)
MUH_CENTER = 0.01         # [paper] Eq.10
MUH_SIGMA = 0.05          # [paper] Eq.10
NUM_CONTEXT_WINDOW = 2    # [assumed] tokens on each side that count as "numerical context"

# --------------------------------- dynamic entropy threshold + merge logic
H_MIN = _f("H_MIN", 2.6)  # [assumed] base threshold H_min (paper gives no value; tune per language)
HF_AMP = 0.2              # [paper] 1.0 + 0.2*exp(...)
HF_SIGMA = 0.5            # [paper] 4*(0.5)^2
SOFT_MAX_LEN = 50         # [paper] |c'| <= 50
HARD_MAX_LEN = 100        # [paper] |c| <= 100

# --------------------------------------------- fuzzy similarity     (Eq. 11)
M_FUZZY_SCALE = 2.0       # [paper] M = 1.0 + (W_term + B_context) * 2
FUZZY_M_MAX = 1.0 + M_FUZZY_SCALE * (2.0 + NUM_BOOST)
TOP_K = 3                 # [paper] Fig. 7(c) top-3
CANDIDATE_FACTOR = 5      # [assumed] ANN candidates = TOP_K*factor, then fuzzy re-rank

# ------------------------------------------------ vector store    (Eq. 12)
HNSW_M = 8                # [paper]
HNSW_EF_CONSTRUCTION = 40  # [paper]
INDEX_BATCH = 100         # [paper]
LAMBDA_STORAGE = _f("LAMBDA_STORAGE", 0.01)  # [assumed] lambda in Eq.12
COLLECTION = "flotation_chunks"

# --------------------------------------------- prompt optimisation (Eq. 6,7)
MATCH_THRESHOLD = 0.85    # [paper]
W_SUBJECT, W_OBJECT, W_RELATION, W_LABEL = 0.3, 0.3, 0.25, 0.15  # [paper] Eq.7
OPT_F1_FLOOR = 0.6        # [paper] F1 < 0.6  -> decompose
OPT_CSIS_FLOOR = 0.8      # [paper] Csis < 0.8 -> consistency
OPT_STOP_F1 = 0.90        # [paper]
OPT_STOP_CSIS = 0.8       # [paper]
OPT_MAX_ITERS = _i("OPT_MAX_ITERS", 3)   # [assumed] maximum feedback updates; loop evaluates versions 0..N
OPT_MIN_ITERS = _i("OPT_MIN_ITERS", 1)   # [assumed] require at least one feedback update before convergence
OPT_DEV_SIZE = _i("OPT_DEV_SIZE", 40)    # [assumed] how many annotated sentences drive optimisation
OPT_USE_RAG = os.getenv("OPT_USE_RAG", "1") == "1"

# ------------------------------------------------------------- KG / eval
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"   # Aura files use NEO4J_USERNAME
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "")   # Aura Free: the database is named like the instance id; empty = server default
NEO4J_GRAPH_SCOPE = os.getenv("NEO4J_GRAPH_SCOPE", "default")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
USE_NEO4J = os.getenv("USE_NEO4J", "auto")   # auto | 1 | 0

# structural score weights Eq.28 [paper]
S1_WEIGHTS = {"I": 0.25, "C": 0.25, "B": 0.2, "Z": 0.15, "P": 0.1, "R": 0.05}
R_TMIN_NUM, R_TMAX_NUM = 20.0, 60.0   # [paper] Tmin ~ 20/N_edges, Tmax ~ 60/N_edges
# semantic sampling [paper]
SEM_BATCH = 100
SEM_RHO = 0.2
SEM_PHI = 0.618
FIBONACCI = [1, 2, 3, 5, 8, 13]
SEM_LOW_QUALITY = 0.5
SIGMA_S = _f("SIGMA_S", 0.1)   # [assumed] variance weight sigma_s in Eq.29 (value not given)
FINAL_W_STRUCT, FINAL_W_SEM = 0.4, 0.6   # [paper] Eq.30
