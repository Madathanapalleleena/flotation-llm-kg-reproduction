"""Stage 1 of the two-stage triple generation -- paper Sec. III-C, Eq. (17):

    T_m = f_LLM( x_m , f_FineLLM( x_m , C_RAG ) ),   m in {text, caption}

This file holds f_FineLLM (initial triples from the optimised prompt + fuzzy-RAG context).
LoRA fine-tuning is REMOVED in this build: stage 1 uses a plain chat model (config.STAGE1_MODEL).

LLM back-ends
  deepseek / any OpenAI-compatible server : real model (needs DEEPSEEK_API_KEY or a local endpoint)
  mock                                    : OFFLINE rule-based stand-in so the pipeline can be tested
                                            without an API key.  NOT a language model - results from it
                                            say nothing about the paper's accuracy.
"""
import json
import re
import time

import requests

import config
from prompting.base_prompt import RULE_COVERAGE, RULE_DECOMPOSE

# ------------------------------------------------------------------ helpers
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_triples(raw: str):
    """Extract the {"triples": [...]} JSON from an LLM answer (tolerates ``` fences and <think> blocks)."""
    if not raw:
        return []
    raw = _THINK.sub("", raw)
    raw = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\{.*\}", raw, re.S) or re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        obj = obj.get("triples", [])
    return [t for t in obj if isinstance(t, dict)] if isinstance(obj, list) else []


# ------------------------------------------------------------- real client
class ChatLLM:
    kind = "openai-compatible"

    def __init__(self, model, base_url, api_key):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    def chat(self, messages, temperature=None):
        body = dict(model=self.model, messages=messages,
                    temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
                    max_tokens=config.LLM_MAX_TOKENS)
        last = None
        for attempt in range(4):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=config.LLM_TIMEOUT,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last}")


class OllamaLLM:
    kind = "ollama"

    def __init__(self, model):
        self.model = model

    def chat(self, messages, temperature=None):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
                "num_predict": config.OLLAMA_NUM_PREDICT,
            },
        }
        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                json=body,
                timeout=config.LLM_TIMEOUT,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama model {self.model!r} did not finish within {config.LLM_TIMEOUT}s. "
                "Use a smaller Ollama model or increase LLM_TIMEOUT."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed at {config.OLLAMA_BASE_URL}/api/chat "
                f"for model {self.model!r}: {exc}"
            ) from exc

        try:
            return response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an invalid chat response.") from exc


# ------------------------------------------------- built-in local model
class LocalLLM:
    """A small open chat model run by the project itself (transformers). No key, no server, no network after the
    first download.  Select with LLM_BACKEND=local, choose the model with LOCAL_MODEL."""
    kind = "local-transformers"
    _cache = {}

    def __init__(self, model=None):
        self.name = model or config.LOCAL_MODEL
        if self.name not in LocalLLM._cache:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:  # noqa: BLE001
                raise RuntimeError("LLM_BACKEND=local needs:  pip install transformers torch accelerate") from e
            dev = config.LOCAL_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[local-llm] loading {self.name} on {dev} (first run downloads the model) ...")
            tok = AutoTokenizer.from_pretrained(self.name)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
            LocalLLM._cache[self.name] = (tok, mdl, dev)
        self.tok, self.model, self.dev = LocalLLM._cache[self.name]

    def chat(self, messages, temperature=None):          # greedy decoding: temperature is ignored
        import torch
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=config.LOCAL_MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------ offline mock
_LEX = {
    "Parameter": ["pulp concentration", "ph value", "aeration rate", "froth depth", "impeller speed", "pulp level",
                  "flotation time", "grinding fineness", "reagent dosage", "feed flow rate", "particle size",
                  "foam color", "foam size", "flow rate", "residence time", "bubble size", "froth layer",
                  "lead gray", "dark", "over-flotation state", "operating status"],
    "Product": ["concentrate grade", "recovery rate", "tailing grade", "froth stability", "production efficiency",
                "concentrate", "tailing"],
    "Equipment": ["flotation cell", "agitator", "froth scraper", "air blower", "slurry pump", "level sensor",
                  "mixing bowl", "motor", "pipeline"],
    "Reagent": ["xanthate", "copper sulfate", "lime", "pine oil", "mibc", "sodium silicate", "collector", "frother"],
    "ProcessStage": ["fast roughing", "roughing", "scavenging", "cleaning"],
    "RawOre": ["lead-zinc ore", "galena", "sphalerite", "pyrite", "raw ore", "ore", "gangue"],
}
_PHRASES = sorted(((p, t) for t, ps in _LEX.items() for p in ps), key=lambda x: -len(x[0]))
_CUES_BASE = [("increases", r"\b(increases?|raises?)\b"), ("decreases", r"\b(decreases?|reduces?|lowers?)\b"),
              ("keeps", r"\b(keeps?|maintains?)\b"), ("affects", r"\b(affects?|influences?)\b"),
              ("uses", r"\b(uses?|adds?)\b"), ("produces", r"\b(produces?|yields?)\b")]
_CUES_EXT = [("causes", r"\b(causes?|leads? to|results? in)\b"), ("follows", r"\bfollows?\b"),
             ("has_property", r"\b(has|have)\b")]


_VALUE_TERMS = {"lead gray", "dark", "over-flotation state"}
_CHAIN_RE = re.compile(r"from ([a-z\- ]+?) (?:through|via) ([a-z,\- ]+?) to ([a-z\- ]+?)(?:[.,;]|$)")


def _lookup_type(phrase):
    for p, t in _PHRASES:
        if p == phrase:
            return t
    return None


def _mock_caption_patterns(low: str):
    """Rules that only fire when the prompt asks for full coverage (RULE_COVERAGE)."""
    out = []
    # "<parameter> is/are <value>"  (adjacent entities only, e.g. 'the foam color is lead gray')
    for ph, ty in _PHRASES:
        for m in re.finditer(r"\b" + re.escape(ph) + r"\s+(?:is|are)\s+(?:now\s+)?", low):
            for v in _VALUE_TERMS:
                if low.startswith(v, m.end()) and ty == "Parameter" and ph not in _VALUE_TERMS:
                    out.append(dict(subject=ph, subject_type=ty, relation="has_property", object=v, object_type="Parameter"))
    # flow chain "from raw ore through fast roughing, roughing, scavenging and cleaning to concentrate"
    m = _CHAIN_RE.search(low)
    if m:
        seq = [m.group(1).strip()] + [x.strip() for x in re.split(r",\s*|\s+and\s+", m.group(2)) if x.strip()] + [m.group(3).strip()]
        known = [(x, _lookup_type(x)) for x in seq]
        known = [(x, t) for x, t in known if t]
        for (a, ta), (b, tb) in zip(known, known[1:]):
            out.append(dict(subject=b, subject_type=tb, relation="follows", object=a, object_type=ta))
    return out


def _mock_extract(text: str, decompose: bool, coverage: bool):
    low = text.lower()
    clauses = re.split(r",\s*(?:and|while)\s+|;\s*", low)
    clauses = clauses if decompose else clauses[:1]
    cues = _CUES_BASE + (_CUES_EXT if coverage else [])
    out = []
    for cl in clauses:
        ents, taken = [], [False] * len(cl)
        for ph, ty in _PHRASES:
            for m in re.finditer(r"\b" + re.escape(ph) + r"\b", cl):
                if not any(taken[m.start():m.end()]):
                    for k in range(m.start(), m.end()):
                        taken[k] = True
                    ents.append((m.start(), m.end(), ph, ty))
        ents.sort()
        for (s1, e1, p1, t1), (s2, e2, p2, t2) in zip(ents, ents[1:]):
            between = cl[e1:s2]
            for rel, rx in cues:
                if re.search(rx, between):
                    out.append(dict(subject=p1, subject_type=t1, relation=rel, object=p2, object_type=t2))
                    break
    if coverage:
        out += _mock_caption_patterns(low)
    return out


def _mock_score(text: str, triples: list) -> dict:
    if not triples:
        return dict(score=0.0, reason="no triples")
    low = text.lower()
    s = []
    for t in triples:
        g = (t["subject"].lower() in low) + (t["object"].lower() in low)
        s.append(g / 2.0)
    sc = round(0.35 + 0.65 * sum(s) / len(s), 3)
    return dict(score=sc, reason="heuristic: fraction of triple entities grounded in the sentence")


class MockLLM:
    kind = "mock-offline"
    model = "mock-rule-based"

    def chat(self, messages, temperature=None):
        sysm = messages[0]["content"]
        user = messages[-1]["content"]
        if "[SEMANTIC-CHECK]" in sysm:
            text = re.search(r"Sentence: (.*)", user).group(1)
            trip = parse_triples(user.split("Triples:", 1)[1])
            return json.dumps(_mock_score(text, trip))
        if "[REFINE]" in sysm:
            init = parse_triples(user.split("Initial triples:", 1)[1].split("### ", 1)[0])
            return json.dumps({"triples": init})
        text = user.split("### Pending text", 1)[1].strip()
        trip = _mock_extract(text, decompose=RULE_DECOMPOSE in sysm, coverage=RULE_COVERAGE in sysm)
        return json.dumps({"triples": trip})


def get_client(stage: int = 1):
    backend = config.LLM_BACKEND
    if backend == "auto":
        backend = "deepseek" if (config.STAGE1_API_KEY or config.DEEPSEEK_API_KEY) else "mock"
    if backend == "mock":
        return MockLLM()
    if backend == "local":
        return LocalLLM()
    if backend == "ollama":
        model = config.STAGE1_MODEL if stage == 1 else config.OLLAMA_STAGE2_MODEL
        return OllamaLLM(model)
    if stage == 1:
        return ChatLLM(config.STAGE1_MODEL, config.STAGE1_BASE_URL, config.STAGE1_API_KEY)
    return ChatLLM(config.STAGE2_MODEL, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY)


# --------------------------------------------------------------- stage 1
class InitialExtractor:
    """f_FineLLM(x_m, C_RAG): optimised prompt + fuzzy-RAG context -> initial triples."""

    def __init__(self, template, retriever=None, client=None):
        self.template, self.retriever = template, retriever
        self.client = client or get_client(1)

    def context_for(self, text):
        return self.retriever.context(text) if self.retriever is not None else ""

    def extract(self, text: str, context: str = None):
        ctx = self.context_for(text) if context is None else context
        raw = self.client.chat(self.template.messages(text, ctx))
        return parse_triples(raw), ctx
