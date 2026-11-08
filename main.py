"""LLM-driven multimodal KG construction for the flotation process -- end-to-end driver.

  python main.py init-data         Step 1     create the SYNTHETIC sample dataset (skip if you uploaded real data  )
  if parsed.get("unit_id") is not None:
      item["source_unit_id"] = parsed["unit_id"]
  python main.py parse             Steps 2-3  MinerU parser -> data/parsed/documents.json
  python main.py build-rag         Steps 5-7  normal index, then fuzzy TF-IDF + entropy chunking + M3E + Chroma
  python main.py query "..."       Step 8     fuzzy retrieval (normal vs fuzzy, top-3)
  python main.py optimize-prompt   Steps 9-10 adaptive prompt optimisation
  python main.py train-lora        Step 10   fine-tune the paper's stage-1 LoRA adapter
  python main.py demo --source <file.pdf>  Fast single-document graph demo
  python main.py extract           Steps 11-12 two-stage triple generation (initial -> refinement)
  python main.py build-kg          Steps 13-14 ontology mapping + Neo4j (or local JSON) storage
  python main.py evaluate          Steps 16-17 extraction / structural / semantic metrics
  python main.py serve             Step 15    Flask API + D3 visualisation  (http://127.0.0.1:5000)
  python main.py all               everything above except `serve`
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def banner(step, title):
    print(f"\n{'=' * 70}\nSTEP {step}: {title}\n{'=' * 70}")


# ---------------------------------------------------------------------------- steps
def cmd_init_data(_):
    banner("1", "Prepare industrial / flotation dataset (synthetic, all data types)")
    from data.generate_full_dataset import main as gen
    gen()


def cmd_parse(_):
    banner("2-3", "MinerU parser -> JSON multimodal dataset")
    from parser.mineru_parser import run
    run()


def cmd_build_rag(args):
    from rag.retriever import build_index
    banner("5", "Normal RAG (fixed chunks, cosine)")
    build_index("normal")
    banner("6-7", "Fuzzy TF-IDF weighting + entropy-based chunking -> vector library")
    build_index("fuzzy")


def cmd_query(args):
    banner("8", "Fuzzy retrieval")
    from rag.retriever import Retriever
    r = Retriever()
    for fuzzy in (False, True):
        print(f"\n[{'fuzzy sim_fuzzy (Eq.11)' if fuzzy else 'plain cosine'}]")
        for i, h in enumerate(r.retrieve(args.text, config.TOP_K, fuzzy), 1):
            print(f"  {i}. score={h['score']:.4f} cos={h['cos']:.4f} ({h['ms']} ms) [{h['kind']}] {h['text']}")


def _retriever():
    from rag.retriever import Retriever
    return Retriever() if config.CHUNKS_JSON.exists() else None


def cmd_optimize(_):
    banner("9-10", "Extraction prompt + adaptive prompt optimisation")
    from llm.extraction import get_client
    print(f"LLM backend: {get_client(1).kind}")
    from prompting.prompt_optimizer import optimize
    best, hist = optimize(retriever=_retriever() if config.OPT_USE_RAG else None)
    print(f"best prompt = version {best.version}, saved to {config.PROMPT_DIR / 'optimized_prompt.json'}")


def cmd_train_lora(_):
    banner("10", "LoRA fine-tuning for stage-1 triple generation")
    from llm.lora.training import train_lora
    result = train_lora()
    print(f"LoRA adapter saved to {config.LORA_ADAPTER_DIR}")
    print(f"training loss={result['train_loss']}, examples={result['examples']}, "
          f"base model={result['base_model']}")


def _load_prompt():
    from prompting.base_prompt import PromptTemplate, build_base_prompt
    p = config.PROMPT_DIR / "optimized_prompt.json"
    if p.exists():
        return PromptTemplate.load(p)
    print("[warn] no optimized prompt found, using base prompt P0 (run optimize-prompt)")
    return build_base_prompt()


def build_units(documents=None):
    from rag.entropy_chunking import split_sentences
    docs = documents if documents is not None else json.loads(config.PARSED_JSON.read_text(encoding="utf-8"))
    units = []
    seen = set()

    def add(doc, parsed, modality, text, image_url=None):
        text = " ".join(str(text or "").split())
        if not text:
            return
        key = (doc["doc_id"], modality, text.casefold(), image_url)
        if key in seen:
            return
        seen.add(key)
        item = dict(
            unit_id=f"u{len(units):05d}",
            doc_id=parsed.get("doc_id", doc["doc_id"]),
            source=parsed.get("source", doc.get("source", doc["doc_id"])),
            source_path=parsed.get("source_path", doc.get("source_path", doc.get("source", doc["doc_id"]))),
            source_modality=parsed.get("source_modality", doc.get("modality", "unknown")),
            modality=modality,
            text=text,
            evidence=parsed.get("evidence", text),
            page_number=parsed.get("page_number"),
            row_number=parsed.get("row_number"),
            section=parsed.get("section", ""),
            image_url=image_url or parsed.get("url") or parsed.get("image_url"),
        )
        if parsed.get("unit_id") is not None:
            item["source_unit_id"] = parsed["unit_id"]
        if isinstance(item["evidence"], (dict, list)):
            item["evidence"] = json.dumps(item["evidence"], ensure_ascii=False, sort_keys=True)
        source_evidence = parsed.get("source_evidence")
        if isinstance(source_evidence, (dict, list)):
            source_evidence = json.dumps(source_evidence, ensure_ascii=False, sort_keys=True)
        if source_evidence:
            item["source_evidence"] = source_evidence
        if parsed.get("extraction_method"):
            item["extraction_method"] = parsed["extraction_method"]
        if parsed.get("caption_id"):
            item["caption"] = text
        for field in (
            "image_id",
            "figure_id",
            "caption_id",
            "image_ids",
            "image_urls",
            "panel_index",
            "panel_count",
            "related_text_ids",
            "related_figure_ids",
        ):
            if parsed.get(field) is not None:
                item[field] = parsed[field]
        for location_field in ("duplicate_page_numbers", "duplicate_row_numbers"):
            if parsed.get(location_field):
                item[location_field] = parsed[location_field]
        if parsed.get("table_caption"):
            item["table_caption"] = parsed["table_caption"]
        units.append(item)

    for d in docs:
        for parsed in d.get("units", []):
            raw_modality = str(parsed.get("modality", "Text")).casefold()
            if raw_modality in ("text", "formula"):
                modality = raw_modality
                segments = split_sentences(parsed.get("content", ""))
            elif raw_modality == "table" and not parsed.get("url"):
                modality = "caption"
                segments = [parsed.get("content", "")]
            elif raw_modality in ("table", "tablerow", "record"):
                modality = "record"
                segments = [parsed.get("content", "")]
            elif raw_modality in ("image", "caption"):
                modality = "caption"
                segments = [parsed.get("content", "")]
            else:
                modality = raw_modality
                segments = [parsed.get("content", "")]
            for segment in segments:
                if modality == "text" and len(segment.split()) < 4:
                    continue
                item = dict(parsed)
                if modality == "text":
                    item["source_evidence"] = parsed.get("evidence", parsed.get("content", ""))
                    item["evidence"] = segment
                add(d, item, modality, segment, parsed.get("url"))

        if not d.get("units"):
            for fragment in d.get("S_D", []):
                for sentence in split_sentences(fragment):
                    if len(sentence.split()) >= 4:
                        add(d, {}, "text", sentence)
            for url, caption in zip(d.get("I_D", []), d.get("C_I", [])):
                if caption:
                    add(d, {}, "caption", caption, url)
            for record in d.get("records", []):
                add(d, {}, "record", record)
    return units


def select_demo_units(source, max_units=12, documents=None):
    """Select a bounded, evenly sampled, modality-balanced set from one parsed document."""
    if max_units < 1:
        raise ValueError("--max-units must be at least 1")
    docs = documents if documents is not None else json.loads(config.PARSED_JSON.read_text(encoding="utf-8"))
    matches = [d for d in docs if source in (d.get("source"), d.get("doc_id"))]
    if not matches:
        available = ", ".join(d.get("source", d.get("doc_id", "?")) for d in docs)
        raise ValueError(f"No parsed document matches {source!r}. Available sources: {available}")
    if len(matches) > 1:
        raise ValueError(f"Source {source!r} matches multiple parsed documents; use an exact source filename.")
    units = build_units(matches)
    if not units:
        raise ValueError(f"Parsed document {source!r} contains no extractable text, captions, or records.")
    if len(units) <= max_units:
        return units

    text_terms = (
        "flotation", "froth", "ore", "pulp", "mineral", "collector", "reagent", "aeration", "air flow",
        "recovery", "concentrate", "tailing", "roughing", "cleaning", "scavenging", "bubble", "dosage",
        "impeller", "particle", "grade", "ph ",
    )
    relation_cue = re.compile(
        r"\b(?:increas\w*|decreas\w*|reduc\w*|affect\w*|influenc\w*|caus\w*|"
        r"lead\w*\s+to|result\w*\s+in|us(?:e|es|ed|ing)|appl\w*|produc\w*|"
        r"yield\w*|follow\w*|maintain\w*)\b",
        re.IGNORECASE,
    )
    indexed = list(enumerate(units))
    text_units = [
        item for item in indexed
        if item[1]["modality"] == "text"
        and len(item[1]["text"].split()) >= 8
        and relation_cue.search(item[1]["text"])
        and any(term in item[1]["text"].casefold() for term in text_terms)
    ]
    if not text_units:
        text_units = [item for item in indexed if item[1]["modality"] == "text"]
    caption_units = [item for item in indexed if item[1]["modality"] == "caption"]
    record_units = [item for item in indexed if item[1]["modality"] == "record"]

    if max_units < 4:
        return [unit for _, unit in _even_sample(indexed, max_units)]

    caption_quota = min(len(caption_units), max(1, max_units // 5))
    record_quota = min(len(record_units), max_units // 10)
    text_quota = min(len(text_units), max_units - caption_quota - record_quota)
    remaining = max_units - caption_quota - record_quota - text_quota
    if remaining:
        extra_records = min(remaining, len(record_units) - record_quota)
        record_quota += extra_records
        remaining -= extra_records
    if remaining:
        extra_captions = min(remaining, len(caption_units) - caption_quota)
        caption_quota += extra_captions
        remaining -= extra_captions
    if remaining:
        text_quota += min(remaining, len(text_units) - text_quota)

    selected = (
        _even_sample(text_units, text_quota)
        + _even_sample(record_units, record_quota)
        + _even_sample(caption_units, caption_quota)
    )
    return [unit for _, unit in sorted(selected, key=lambda item: item[0])]


def _even_sample(items, max_items):
    if max_items < 1 or len(items) <= max_items:
        return list(items)
    step = len(items) / max_items
    return [items[int(i * step)] for i in range(max_items)]


def _evaluation_units(rows, predictions):
    return [
        {
            "unit_id": str(row["id"]),
            "doc_id": str(row["id"]),
            "modality": "text",
            "text": row["user"],
            "triples": predictions[row["id"]],
        }
        for row in rows
    ]


def cmd_extract(args):
    banner("11-12", "Generate triples (stage 1) + LLM refinement (stage 2)")
    from llm.extraction import get_client
    from llm.refinement import TwoStageExtractor
    print(f"stage-1 LLM: {get_client(1).kind} | stage-2 LLM: {get_client(2).kind}")
    ext = TwoStageExtractor(_load_prompt(), _retriever())
    units = build_units()
    if args.max_units:
        if len(units) > args.max_units:       # even sample over ALL documents (not just the first N units)
            step = len(units) / args.max_units
            units = [units[int(i * step)] for i in range(args.max_units)]
    progress_path = config.OUTPUT_DIR / "triples.progress.json"
    if args.resume:
        if not progress_path.exists():
            raise SystemExit("No extraction checkpoint found to resume.")
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(saved, list):
            raise ValueError(f"Invalid extraction checkpoint format: {progress_path}")
        current_units = {unit["unit_id"]: unit for unit in units}
        out_by_id = {}
        for item in saved:
            unit_id = item.get("unit_id")
            current = current_units.get(unit_id)
            if current is None or item.get("text") != current.get("text"):
                raise ValueError(
                    f"Checkpoint unit {unit_id!r} does not match the current parsed corpus. "
                    "Move the checkpoint aside before starting a new extraction."
                )
            out_by_id[unit_id] = item
        print(f"Resuming from checkpoint: {len(out_by_id)}/{len(units)} units already complete.")
    else:
        if progress_path.exists():
            raise SystemExit(
                f"An extraction checkpoint already exists at {progress_path}. "
                "Use `python main.py extract --resume` to continue it."
            )
        out_by_id = {}

    def save_checkpoint():
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = [out_by_id[unit["unit_id"]] for unit in units if unit["unit_id"] in out_by_id]
        temporary_path = progress_path.with_name(progress_path.name + ".tmp")
        temporary_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary_path, progress_path)

    t0 = time.time()
    for i, u in enumerate(units, 1):
        if u["unit_id"] in out_by_id:
            continue
        print(f"  extracting {i}/{len(units)} (unit={u['unit_id']}) ...", flush=True)
        unit_started = time.time()
        r = ext.extract(u["text"])
        out_by_id[u["unit_id"]] = dict(u, **{k: v for k, v in r.items() if k != "text"})
        print(f"  finished in {time.time() - unit_started:.1f}s", flush=True)
        if len(out_by_id) % 10 == 0 or len(out_by_id) == len(units):
            save_checkpoint()
            print(f"  {len(out_by_id)}/{len(units)} units  ({time.time() - t0:.1f}s)", flush=True)

    out = [out_by_id[unit["unit_id"]] for unit in units]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = config.TRIPLES_JSON.with_name(config.TRIPLES_JSON.name + ".tmp")
    temporary_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary_path, config.TRIPLES_JSON)
    progress_path.unlink(missing_ok=True)
    n = sum(len(u["triples"]) for u in out)
    print(f"{n} triples from {len(out)} units -> {config.TRIPLES_JSON}")


def cmd_demo(args):
    """Run a bounded end-to-end demo, including optimization, refinement, and evaluation."""
    banner("DEMO", "Bounded end-to-end pipeline demo")
    from llm.refinement import TwoStageExtractor
    from prompting.prompt_optimizer import optimize

    if not args.replace_current:
        raise SystemExit(
            "This demo updates optimized_prompt.json, outputs/triples.json, the current graph, and metrics.json; "
            "rerun with --replace-current to continue."
        )
    if args.eval_samples < 1:
        raise SystemExit("--eval-samples must be at least 1.")
    if not config.CHUNKS_JSON.exists():
        raise SystemExit("Missing RAG index. Run `python main.py parse` and `python main.py build-rag` first.")

    units = select_demo_units(args.source, args.max_units)
    retriever = _retriever()
    print(f"source: {args.source} | selected document units: {len(units)}")

    print("\n[1/5] Optimizing the prompt on the configured annotated training subset...")
    best, history = optimize(retriever=retriever)
    print(f"optimized prompt version {best.version} after {len(history)} iteration(s)")

    print("\n[2/5] Extracting and refining triples (two LLM stages per unit)...")
    extractor = TwoStageExtractor(_load_prompt(), retriever)
    output, started = [], time.monotonic()
    for index, unit in enumerate(units, 1):
        unit_started = time.monotonic()
        print(f"  extracting {index}/{len(units)} (unit={unit['unit_id']}) ...", flush=True)
        result = extractor.extract(unit["text"])
        output.append(dict(unit, **{key: value for key, value in result.items() if key != "text"}))
        print(f"  finished in {time.monotonic() - unit_started:.1f}s "
              f"({time.monotonic() - started:.1f}s total)", flush=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.TRIPLES_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"demo triples saved to {config.TRIPLES_JSON}")

    print("\n[3/5] Building the knowledge graph...")
    cmd_build_kg(None)

    print(f"\n[4/5] Evaluating on up to {args.eval_samples} held-out examples "
          "and dynamically sampled demo units...")
    cmd_evaluate(argparse.Namespace(full_semantic=False, max_samples=args.eval_samples))

    print("\n[5/5] Demo pipeline complete.")
    print("The graph is ready. Start the viewer with: python main.py serve")
    print("Metrics are sample-scoped; they are not full-corpus scores.")


def cmd_build_kg(_):
    banner("13-14", "Map triples to ontology and store in Neo4j")
    from evaluation.graph_metrics import graph_consistency_assessment
    from kg.neo4j_manager import LocalGraphStore, get_store
    from kg.ontology_mapper import assemble
    units = json.loads(config.TRIPLES_JSON.read_text(encoding="utf-8"))
    graph, st = assemble(units)
    integrity = graph_consistency_assessment(graph)
    if integrity["n_issues"]:
        raise ValueError(f"Refusing to store invalid graph: {integrity}")
    print(f"validated graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges; "
          f"{st['rejected']} invalid/conflicting triples rejected")
    if getattr(_, "dry_run", False):
        print("dry run: graph was validated but not written")
        return

    store = get_store()
    try:
        try:
            previous = store.read_graph()
        except FileNotFoundError:
            previous = {"nodes": [], "edges": []}
        if previous.get("nodes"):
            LocalGraphStore(config.KG_BACKUP_JSON).write_graph(previous)
            print(f"previous graph snapshot saved to {config.KG_BACKUP_JSON}")
        store.write_graph(graph)
        if store.kind != "local-json":
            LocalGraphStore().write_graph(graph)
        print(f"graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} relationships | "
              f"{st} | stored in {store.describe()} (scope={config.NEO4J_GRAPH_SCOPE})")
    finally:
        close = getattr(store, "close", None)
        if close:
            close()


def _mock_warning():
    from llm.extraction import get_client
    if get_client(2).kind == "mock-offline":
        print("\n" + "!" * 70 +
              "\n  OFFLINE MOCK LLM IN USE: the extractor is a rule-based stand-in and the sample gold set is\n"
              "  synthetic and built from the same lexicon, so scores below only prove the code runs end to end.\n"
              "  They are NOT comparable with the paper (F1 0.85, S1 0.97, S2 0.88). Set DEEPSEEK_API_KEY and\n"
              "  upload your own annotated sentences for real numbers.\n" + "!" * 70)


def _print_final_metrics(metrics):
    extraction = metrics["extraction"]["Proposed_pipeline"]
    structural = metrics["structural"]
    detail = structural["detail"]
    print("\nFinal knowledge graph metrics:")
    print(f"  Evaluation scope: {metrics['evaluation_scope']}")
    print(f"  1. Precision: {extraction['precision']:.4f}")
    print(f"  2. Recall: {extraction['recall']:.4f}")
    print(f"  3. F1-score: {extraction['f1']:.4f}")
    print(f"  4. Consistency score (C_sis): {extraction['c_sis']:.4f}")
    print(f"  5. Isolated node ratio (I_s): {structural['I_s']:.4f}")
    print(f"  6. Process coverage rate (C_s): {structural['C_s']:.4f}")
    print(f"  7. Bidirectional relationship balance (B_s): {structural['B_s']:.4f}")
    print(f"  8. Process node knowledge coverage (Z_s): {structural['Z_s']:.4f}")
    print(f"  9. Longest process path length (L_max): {detail['L_max']}")
    print(f" 10. Process loop rationality (R_s): {structural['R_s']:.4f}")
    print(f" 11. Structural score (S1): {structural['S1']:.4f}")
    print(f" 12. Semantic score (S2): {metrics['semantic']['S2']:.4f}")
    print(f" 13. Final knowledge graph quality score (S_final): {metrics['S_final']:.4f}")
    print(f"  Graph consistency check: {metrics['graph_consistency']['consistency']:.4f} "
          f"({metrics['graph_consistency']['n_issues']} issues)")


def cmd_evaluate(args):
    banner("16-17", "Structural + semantic evaluation (and extraction metrics)")
    _mock_warning()
    from evaluation.extraction_metrics import evaluate as ext_eval
    from evaluation.graph_metrics import graph_consistency_assessment
    from evaluation.semantic_metrics import final_score, semantic_assessment
    from evaluation.structural_metrics import structural_assessment
    from kg.ontology_mapper import assemble
    from llm.extraction import get_client
    from llm.refinement import TwoStageExtractor
    from prompting.base_prompt import build_base_prompt
    from prompting.prompt_optimizer import load_training, run_extraction, split_dev_test
    from kg.triple_validator import TripleValidator

    metrics = {}
    rows = load_training()
    _, held_out = split_dev_test(rows)
    max_samples = getattr(args, "max_samples", 0)
    test = _even_sample(held_out, max_samples)
    if len(test) < len(held_out):
        print(f"Evaluating a deterministic sample of {len(test)}/{len(held_out)} held-out examples.")
    golds = {r["id"]: r["gold"] for r in test}
    retr = _retriever()
    val = TripleValidator()
    base = ext_eval(run_extraction(build_base_prompt(), test), golds)            # LLM-FewShot baseline (P0, no RAG)
    two = TwoStageExtractor(_load_prompt(), retr)
    preds = {}
    for index, row in enumerate(test, 1):
        started = time.monotonic()
        print(f"  proposed extraction {index}/{len(test)} (id={row['id']}) ...", flush=True)
        preds[row["id"]] = val.validate(two.extract(row["user"])["triples"])[0]
        print(f"  finished in {time.monotonic() - started:.1f}s", flush=True)
    prop = ext_eval(preds, golds)
    keep = ("precision", "recall", "f1", "c_sis", "micro_precision", "micro_recall", "micro_f1",
            "n_pred", "n_exp", "n_valid", "n_conflict")
    metrics["extraction"] = dict(held_out_samples=len(test), held_out_total=len(held_out),
                                 test_ids=[row["id"] for row in test],
                                 LLM_FewShot_baseline={**{k: v for k, v in base.items() if k in keep},
                                                       "per_sample": base["per_sample"]},
                                 Proposed_pipeline={**{k: v for k, v in prop.items() if k in keep},
                                                    "per_sample": prop["per_sample"]})
    metrics["evaluation_protocol"] = {
        "training_data": str(config.TRAIN_JSONL.relative_to(config.ROOT)),
        "development_examples": config.OPT_DEV_SIZE,
        "held_out_test_ids": [row["id"] for row in test],
        "semantic_sampling_seed": 7,
        "semantic_full": bool(args.full_semantic),
        "quality_graph_scope": "assembled from Proposed_pipeline predictions on the held-out sample",
    }
    metrics["evaluation_scope"] = ("full held-out set" if len(test) == len(held_out)
                                   else f"sampled {len(test)} of {len(held_out)} held-out examples")
    print("\nTable-III style comparison on held-out annotated sentences "
          "(UIE / Rule+LTP baselines need external models and are not included):")
    print(f"  {'Method':22s} {'Recall':>7s} {'Precision':>10s} {'F1':>7s}")
    for name, m in (("LLM-FewShot (P0)", base), ("Proposed (opt+RAG+2stage)", prop)):
        print(f"  {name:26s} {m['recall']:7.4f} {m['precision']:10.4f} {m['f1']:7.4f}")

    evaluation_units = _evaluation_units(test, preds)
    graph, _ = assemble(evaluation_units)
    S = structural_assessment(graph)
    metrics["structural"] = S
    consistency = graph_consistency_assessment(graph)
    metrics["graph_consistency"] = consistency
    print("\nStructural assessment (Eq. 22-28):")
    print("  " + "  ".join(f"{k}={S[k]:.4f}" for k in ("I_s", "C_s", "B_s", "Z_s", "P_s", "R_s")))
    print(f"  S1 = {S['S1']:.4f}   ({S['n_nodes']} nodes, {S['n_edges']} edges)")
    print(f"  Graph consistency = {consistency['consistency']:.4f} "
          f"({consistency['n_issues']} issues across {consistency['n_nodes']} nodes / "
          f"{consistency['n_relationships']} relationships)")

    sem = semantic_assessment(evaluation_units, get_client(2), full=args.full_semantic, verbose=True)
    sem_out = {k: v for k, v in sem.items() if k != "samples"}
    sem_out["input_scope"] = "Proposed_pipeline predictions on the held-out sample"
    metrics["semantic"] = dict(sem_out, samples=sem["samples"][:50])
    print(f"\nSemantic assessment (Eq. 29): judge={sem['judge']}, sampled {sem['n_sampled']}/{sem['n_records']}, "
          f"mean={sem['mean_score']:.4f}, S2={sem['S2']:.4f}")
    metrics["S_final"] = final_score(S["S1"], sem["S2"])
    print(f"\nS_Final = 0.4*S1 + 0.6*S2 = {metrics['S_final']:.4f}")
    _print_final_metrics(metrics)
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"metrics written to {config.METRICS_JSON}")


def cmd_serve(args):
    banner("15", "D3 visualisation + API")
    from backend.api import app
    app.run(host=args.host, port=args.port, debug=False)


def cmd_all(args):
    args.max_units = getattr(args, "max_units", 0)
    args.full_semantic = getattr(args, "full_semantic", False)
    if not list(config.RAW_PDF_DIR.glob("*.pdf")) and not list(config.RAW_CSV_DIR.glob("*.csv")) and not list(config.RAW_HTML_DIR.glob("*.htm*")):
        cmd_init_data(args)
    cmd_parse(args)
    cmd_build_rag(args)
    cmd_optimize(args)
    cmd_extract(args)
    cmd_build_kg(args)
    cmd_evaluate(args)
    print("\nAll done. Start the visualisation with:  python main.py serve")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-data").set_defaults(fn=cmd_init_data)
    sub.add_parser("parse").set_defaults(fn=cmd_parse)
    sub.add_parser("build-rag").set_defaults(fn=cmd_build_rag)
    q = sub.add_parser("query")
    q.add_argument("text")
    q.set_defaults(fn=cmd_query)
    sub.add_parser("optimize-prompt").set_defaults(fn=cmd_optimize)
    sub.add_parser("train-lora", help="train the paper-configured stage-1 LoRA adapter").set_defaults(fn=cmd_train_lora)
    e = sub.add_parser("extract")
    e.add_argument("--max-units", type=int, default=0, help="limit units (0 = all extractable units)")
    e.add_argument("--resume", action="store_true", help="continue from outputs/triples.progress.json")
    e.set_defaults(fn=cmd_extract)
    demo = sub.add_parser("demo", help="run the bounded pipeline from one parsed document")
    demo.add_argument("--source", required=True, help="exact parsed source filename or document id")
    demo.add_argument("--max-units", type=int, default=12, help="maximum units to send to the LLM (default: 12)")
    demo.add_argument("--eval-samples", type=int, default=5,
                      help="held-out examples used for evaluation (default: 5)")
    demo.add_argument("--replace-current", action="store_true",
                      help="replace optimized prompt, triples, graph, and metrics with this demo run")
    demo.set_defaults(fn=cmd_demo)
    kg = sub.add_parser("build-kg")
    kg.add_argument("--dry-run", action="store_true", help="assemble and validate without changing stored graphs")
    kg.set_defaults(fn=cmd_build_kg)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--full-semantic", action="store_true", help="score every record instead of the dynamic sample")
    ev.add_argument("--max-samples", type=int, default=0,
                    help="cap held-out extraction evaluation examples (0 = all)")
    ev.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.set_defaults(fn=cmd_serve)
    a = sub.add_parser("all")
    a.add_argument("--max-units", type=int, default=0)
    a.add_argument("--full-semantic", action="store_true")
    a.set_defaults(fn=cmd_all)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
