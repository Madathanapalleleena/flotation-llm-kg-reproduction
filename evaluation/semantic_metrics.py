"""Semantic assessment of the KG -- paper Sec. IV-B-2, Eq. (29), (30).

Reverse verification: an LLM scores the consistency between a sentence and the triples extracted from it
(score in [0,1]).  Records are chosen by a dynamic sampling strategy:
  * batches of B = 100 records, base sample size s = rho * B (rho = 0.2)
  * complex / simple split at the phi-quantile (phi = 0.618, golden ratio) of the batch's triple counts c_phi
  * s_c = min(s * phi, #complex),  s_s = s - s_c
  * triple counts are cut into intervals, picked with Fibonacci weights  p_i = F_i / sum F ,  F = [1,2,3,5,8,13]
Overall semantic score:
    S_2 = s_bar + sigma_s * sqrt( 1/(N_s-1) * sum (s_i - s_bar)^2 )          (29)
Final:  S_Final = 0.4 S_1 + 0.6 S_2                                          (30)
"""
import json
import math
import random
import re
import time

import numpy as np

import config
from llm.extraction import get_client
from kg.triple_validator import TripleValidator

SEM_SYSTEM = ("[SEMANTIC-CHECK] You are a strict flotation-domain reviewer. Given a source sentence and triples "
              "(subject, relation, object) extracted from it, rate how faithfully and completely the triples express the "
              'sentence. Answer ONLY with JSON: {"score": <float 0..1>, "reason": "<short rationale>"}.')


def llm_score(client, text, triples):
    """score = f_LLM(prompt(text, T_m; theta))"""
    user = f"Sentence: {text}\nTriples: {json.dumps({'triples': triples}, ensure_ascii=False)}"
    raw = client.chat([{"role": "system", "content": SEM_SYSTEM}, {"role": "user", "content": user}])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        obj = json.loads(m.group(0))
        return float(min(1.0, max(0.0, obj["score"]))), str(obj.get("reason", ""))
    except Exception:  # noqa: BLE001
        return 0.0, "unparseable judge output"


def dynamic_sample(records, seed=7):
    """records: list of dict with key 'n_count'.  Returns sampled records."""
    rnd = random.Random(seed)
    B, rho, phi, F = config.SEM_BATCH, config.SEM_RHO, config.SEM_PHI, config.FIBONACCI
    picked = []
    for b0 in range(0, len(records), B):
        batch = records[b0:b0 + B]
        s = max(1, round(rho * len(batch)))
        counts = np.array([r["n_count"] for r in batch])
        c_phi = float(np.quantile(counts, phi))
        complex_ = [r for r in batch if r["n_count"] >= c_phi and r["n_count"] > 0] or []
        simple = [r for r in batch if r not in complex_]
        s_c = min(int(round(s * phi)), len(complex_))
        s_s = min(s - s_c, len(simple))
        s_c = min(s - s_s, len(complex_))                       # give leftover quota back to the complex group
        for group, k in ((complex_, s_c), (simple, s_s)):
            picked += _fib_pick(group, k, rnd, F)
    return picked


def _fib_pick(group, k, rnd, F):
    if k <= 0 or not group:
        return []
    lo, hi = min(r["n_count"] for r in group), max(r["n_count"] for r in group)
    n_int = min(len(F), hi - lo + 1)
    width = (hi - lo + 1) / n_int
    bins = [[] for _ in range(n_int)]
    for r in group:
        bins[min(int((r["n_count"] - lo) / width), n_int - 1)].append(r)
    out = []
    while len(out) < k and any(bins):
        idx = [i for i, b in enumerate(bins) if b]
        w = [F[i] for i in idx]                                  # p_i = F_i / sum F  (renormalised over non-empty intervals)
        i = rnd.choices(idx, weights=w)[0]
        out.append(bins[i].pop(rnd.randrange(len(bins[i]))))
    return out


def semantic_assessment(units, client=None, full=False, seed=7, verbose=False):
    """Judge only schema-valid triples; report validation counts alongside the semantic score."""
    client = client or get_client(2)
    validator = TripleValidator()
    records = []
    input_triples = valid_triples = rejected_triples = 0
    for unit in units:
        raw = unit.get("triples", [])
        validated, rejected = validator.validate(raw)
        input_triples += len(raw)
        valid_triples += len(validated)
        rejected_triples += len(rejected)
        records.append(dict(text=unit["text"], triples=validated, n_count=len(validated),
                            unit_id=unit.get("unit_id"), n_input_triples=len(raw),
                            n_rejected_triples=len(rejected)))
    sample = records if full else dynamic_sample(records, seed)
    scores = []
    for index, r in enumerate(sample, 1):
        started = time.monotonic()
        if verbose:
            print(f"  semantic check {index}/{len(sample)} (unit={r.get('unit_id')}) ...", flush=True)
        sc, why = llm_score(client, r["text"], r["triples"])
        r["score"], r["reason"] = sc, why
        scores.append(sc)
        if verbose:
            print(f"  finished in {time.monotonic() - started:.1f}s", flush=True)
    n = len(scores)
    s_bar = sum(scores) / max(n, 1)
    sd = math.sqrt(sum((x - s_bar) ** 2 for x in scores) / (n - 1)) if n > 1 else 0.0
    S2_raw = s_bar + config.SIGMA_S * sd                         # Eq. (29) as printed
    return dict(n_records=len(records), n_sampled=n, input_triples=input_triples,
                validated_triples=valid_triples, rejected_triples=rejected_triples,
                validation_rate=valid_triples / max(input_triples, 1),
                mean_score=s_bar, std=sd, sigma_s=config.SIGMA_S,
                S2_raw=S2_raw, S2=S2_raw,
                frac_ge_0_8=sum(x >= 0.8 for x in scores) / max(n, 1),
                n_low_quality=sum(x < config.SEM_LOW_QUALITY for x in scores),
                judge=getattr(client, "kind", "?"), samples=sample)


def final_score(S1, S2):
    return config.FINAL_W_STRUCT * S1 + config.FINAL_W_SEM * S2      # Eq. (30)
