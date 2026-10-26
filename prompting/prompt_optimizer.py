"""Adaptive prompt optimisation -- paper Sec. III-B-1(b)(c).

Loop (three stages):  P_0 -> generate triples -> dynamic feedback (P, R, F1, C_sis) -> P_{n+1}
Update rules (verbatim from the paper):
    F1   < 0.6        -> decompose complex sentences / supplement missing entities & relations
    C_sis < 0.8       -> strengthen grammatical coherence & directional consistency
    R_ec < P_re       -> expand entity / relationship coverage
Each iteration replaces the best-so-far prompt only when F1 improves.
Stops when the current evaluation reaches F1 >= 0.90 and C_sis >= 0.8, or at the maximum iteration count.
"""
import json
import time
from pathlib import Path

import config
from evaluation.extraction_metrics import evaluate
from kg.triple_validator import TripleValidator
from llm.extraction import InitialExtractor
from prompting.base_prompt import (RULE_CONSISTENCY, RULE_COVERAGE, RULE_DECOMPOSE, PromptTemplate,
                                   build_base_prompt)


def load_training(path: Path = config.TRAIN_JSONL):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["gold"] = json.loads(r["assistant"])["triples"]
    return rows


def split_dev_test(rows, dev_size=config.OPT_DEV_SIZE):
    return rows[:dev_size], rows[dev_size:]


def run_extraction(template, rows, retriever=None, client=None, verbose=True):
    ex = InitialExtractor(template, retriever, client)
    val = TripleValidator()
    preds = {}
    for index, r in enumerate(rows, 1):
        started = time.monotonic()
        if verbose:
            print(f"    extracting example {index}/{len(rows)} (id={r['id']}) ...", flush=True)
        trip, _ = ex.extract(r["user"])
        preds[r["id"]] = val.validate(trip)[0]
        if verbose:
            print(f"    finished in {time.monotonic() - started:.1f}s", flush=True)
    return preds


def score(template, rows, retriever=None, client=None, verbose=True):
    golds = {r["id"]: r["gold"] for r in rows}
    return evaluate(run_extraction(template, rows, retriever, client, verbose), golds)


def _failure_examples(res, rows, kind="missed", k=2):
    by_id = {r["id"]: r for r in rows}
    out = []
    for sid, items in res[kind].items():
        if items and len(out) < k:
            out.append(dict(input=by_id[sid]["user"], output={"triples": by_id[sid]["gold"]}))
    return out


def next_prompt(cur: PromptTemplate, res, rows) -> PromptTemplate:
    """Dynamic adjustment P_n -> P_{n+1}."""
    nxt = cur.clone(version=cur.version + 1)
    f1, cs, p, r = res["f1"], res["c_sis"], res["precision"], res["recall"]
    if f1 < config.OPT_F1_FLOOR:                                  # insufficient extraction capability
        if RULE_DECOMPOSE not in nxt.extra_rules:
            nxt.extra_rules.append(RULE_DECOMPOSE)
        for ex in _failure_examples(res, rows, "missed"):          # supplement missing entities / relations
            if ex not in nxt.examples:
                nxt.examples.append(ex)
    if cs < config.OPT_CSIS_FLOOR:                                # logical inconsistency
        if RULE_CONSISTENCY not in nxt.extra_rules:
            nxt.extra_rules.append(RULE_CONSISTENCY)
    if r < p:                                                     # recall lags precision
        if RULE_COVERAGE not in nxt.extra_rules:
            nxt.extra_rules.append(RULE_COVERAGE)
        for ex in _failure_examples(res, rows, "missed", k=1):
            if ex not in nxt.examples:
                nxt.examples.append(ex)
    if nxt.extra_rules == cur.extra_rules and nxt.examples == cur.examples:
        # no rule fired but the stop criterion is unmet -> add one more hard example for diversity
        for ex in _failure_examples(res, rows, "missed", k=3):
            if ex not in nxt.examples:
                nxt.examples.append(ex)
                break
    return nxt


def optimize(rows=None, retriever=None, client=None, max_iters=config.OPT_MAX_ITERS, verbose=True):
    rows = rows or load_training()
    dev, _ = split_dev_test(rows)
    if not dev:
        raise ValueError("Prompt optimization requires at least one development example.")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative.")
    cur = build_base_prompt()
    best, best_res, history = None, None, []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.PROMPT_DIR / "optimization_runs" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for it in range(max_iters + 1):
        if verbose:
            print(f"  starting iteration {it} ({len(dev)} examples)", flush=True)
        predictions = run_extraction(cur, dev, retriever, client, verbose)
        golds = {row["id"]: row["gold"] for row in dev}
        res = evaluate(predictions, golds)
        history.append(dict(iteration=it, version=cur.version, precision=round(res["precision"], 4),
                            recall=round(res["recall"], 4), f1=round(res["f1"], 4), c_sis=round(res["c_sis"], 4),
                            n_rules=len(cur.extra_rules), n_examples=len(cur.examples)))
        version_path = config.PROMPT_DIR / f"prompt_v{cur.version}.json"
        cur.save(version_path)
        if verbose:
            h = history[-1]
            print(f"  iter {it}: P={h['precision']:.3f} R={h['recall']:.3f} F1={h['f1']:.3f} Csis={h['c_sis']:.3f} "
                  f"(rules={h['n_rules']}, examples={h['n_examples']})")
        if best is None or res["f1"] > best_res["f1"]:
            best, best_res = cur, res
        stop = res["f1"] >= config.OPT_STOP_F1 and res["c_sis"] >= config.OPT_STOP_CSIS
        updated = None if stop else next_prompt(cur, res, dev)
        artifact = {
            "iteration": it,
            "version": cur.version,
            "prompt": cur.to_dict(),
            "prompt_path": version_path.name,
            "examples": [
                {
                    "id": row["id"],
                    "input": row["user"],
                    "gold": row["gold"],
                    "generated": predictions.get(row["id"], []),
                    "scores": res["per_sample"][row["id"]],
                }
                for row in dev
            ],
            "evaluation": {key: value for key, value in res.items()
                           if key not in ("per_sample", "missed", "spurious")},
            "missed": res["missed"],
            "spurious": res["spurious"],
            "updated_prompt": updated.to_dict() if updated else None,
        }
        (run_dir / f"iteration_{it:02d}_version_{cur.version:02d}.json").write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        if stop:
            break
        cur = updated
    best.save(config.PROMPT_DIR / "optimized_prompt.json")
    (config.PROMPT_DIR / "history.json").write_text(json.dumps(
        dict(run_id=run_id, artifact_dir=str(run_dir.relative_to(config.PROMPT_DIR)),
             max_updates=max_iters, selected_version=best.version,
             history=history), indent=2), encoding="utf-8")
    return best, history
