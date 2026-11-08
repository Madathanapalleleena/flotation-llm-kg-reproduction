"""Backend for the D3 visualisation (paper Sec. IV-C).  The paper used Django; Flask is used here for a lighter setup.

GET /                 -> frontend/graph.html
GET /api/graph        -> {"nodes":[...], "links":[...]}   from Neo4j (if reachable) or outputs/kg_local.json
GET /api/metrics      -> outputs/metrics.json
GET /api/retrieve?q=  -> top-3 fuzzy-RAG chunks (demo of Step 8)
GET /images/<file>    -> image files (parsed/images first, then raw/images)
"""
import json
import sys
from collections import Counter
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from kg.neo4j_manager import LocalGraphStore, get_store  # noqa: E402
from ontology.knowledge_tree import KnowledgeTree  # noqa: E402

app = Flask(__name__)
FRONTEND = config.ROOT / "frontend"
_retriever = None


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def load_graph():
    try:
        g = get_store(verbose=False).read_graph()
    except Exception:  # noqa: BLE001
        g = None
    if not g or not g.get("nodes"):
        g = LocalGraphStore().read_graph()
    return g


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "graph.html")


LOCAL_IMG_PREFIX = "http://127.0.0.1:5000/images/"


def public_url(u):
    """Images parsed locally are stored with the default local prefix; serve them relative to whichever host/port
    is running.  A custom IMAGE_BASE_URL (e.g. a GitHub raw URL) is passed through untouched."""
    if u and u.startswith(LOCAL_IMG_PREFIX):
        return "/images/" + u[len(LOCAL_IMG_PREFIX):]
    return u


@app.route("/d3.min.js")
def d3_local():
    return send_from_directory(FRONTEND, "d3.min.js", mimetype="application/javascript")


@app.route("/api/graph")
def api_graph():
    try:
        g = load_graph()
    except FileNotFoundError:
        return jsonify(error="No graph yet - run `python main.py build-kg` first."), 404
    deg = Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    nodes = [dict(id=n["id"], name=n.get("name", n["id"]), kind=n.get("kind"), type=n.get("type"),
                  url=public_url(n.get("url")), level=n.get("level"), degree=deg[n["id"]]) for n in g["nodes"]]
    ids = {n["id"] for n in nodes}
    links = [dict(source=e["source"], target=e["target"], type=e["type"], text=e.get("text"), doc=e.get("doc"))
             for e in g["edges"] if e["source"] in ids and e["target"] in ids]
    if request.args.get("scope") == "corpus":
        if not config.PARSED_JSON.exists():
            return jsonify(error="No parsed corpus yet - run `python main.py parse` first."), 404
        documents = json.loads(config.PARSED_JSON.read_text(encoding="utf-8"))
        triples = json.loads(config.TRIPLES_JSON.read_text(encoding="utf-8")) if config.TRIPLES_JSON.exists() else []
        graph = _corpus_graph(g, documents, triples)
        return jsonify(graph)
    return jsonify(nodes=nodes, links=links)


def _corpus_graph(graph, documents, units):
    nodes = {}
    edges = {}
    original_ids = {}

    for node in graph["nodes"]:
        original_id = node["id"]
        node_id = node.get("source_id", original_id)
        original_ids[original_id] = node_id
        nodes[node_id] = {
            **node,
            "id": node_id,
            "name": node.get("name", node_id),
            "kind": node.get("kind", "Entity"),
            "type": node.get("type", "Other"),
            "url": public_url(node.get("url")),
        }

    def add_edge(source, target, relation, **properties):
        if source not in nodes or target not in nodes:
            return
        key = (source, relation, target)
        if key not in edges:
            edges[key] = {"source": source, "target": target, "type": relation, **properties}

    for edge in graph["edges"]:
        add_edge(
            original_ids.get(edge["source"], edge["source"]),
            original_ids.get(edge["target"], edge["target"]),
            edge["type"],
            **{key: value for key, value in edge.items() if key not in {"source", "target", "type"}},
        )

    tree = KnowledgeTree()
    for document in documents:
        doc_id = str(document.get("doc_id") or document.get("source") or "unknown")
        document_id = f"doc:{doc_id}"
        source = document.get("source") or doc_id
        nodes[document_id] = {
            "id": document_id,
            "name": Path(source).name,
            "kind": "Document",
            "type": document.get("modality") or "Document",
            "source": source,
            "source_path": document.get("source_path"),
            "degree": 0,
        }
        source_concept, _ = tree.match_node(source)
        add_edge(f"onto:{source_concept or 'FrothFlotation'}", document_id, "HAS_SOURCE", doc=source)

        for index, unit in enumerate(document.get("units", [])):
            content = (unit.get("content") or "").strip()
            if not content:
                continue
            unit_id = str(unit.get("unit_id") or f"position-{index}")
            evidence_id = f"evidence:{doc_id}:{unit_id}"
            modality = str(unit.get("modality") or "Text")
            nodes[evidence_id] = {
                "id": evidence_id,
                "name": content[:100] + ("…" if len(content) > 100 else ""),
                "kind": "Evidence",
                "type": modality,
                "source": unit.get("source") or source,
                "source_path": unit.get("source_path") or document.get("source_path"),
                "page_number": unit.get("page_number"),
                "row_number": unit.get("row_number"),
                "section": unit.get("section"),
                "unit_id": unit_id,
                "evidence": unit.get("evidence") or content,
                "text": content,
                "degree": 0,
            }
            add_edge(document_id, evidence_id, "HAS_EVIDENCE", doc=source)
            concept, _ = tree.match_node(content)
            if concept:
                add_edge(f"onto:{concept}", evidence_id, "HAS_EVIDENCE", doc=source)

        for index, image in enumerate(document.get("image_records", [])):
            url = public_url(image.get("url") or image.get("image_url"))
            if not url:
                continue
            image_id = str(image.get("image_id") or f"{doc_id}:image-{index}")
            graph_id = f"img:{image_id}"
            caption = image.get("caption") or "Uncaptioned figure"
            nodes[graph_id] = {
                "id": graph_id,
                "name": caption[:100],
                "kind": "Image",
                "type": "Image",
                "url": url,
                "image_id": image_id,
                "figure_id": image.get("figure_id"),
                "caption_id": image.get("caption_id"),
                "panel_index": image.get("panel_index"),
                "panel_count": image.get("panel_count"),
                "source": image.get("source") or source,
                "source_path": image.get("source_path") or document.get("source_path"),
                "page_number": image.get("page_number"),
                "related_text_ids": image.get("related_text_ids", []),
                "caption": caption,
                "degree": 0,
            }
            add_edge(document_id, graph_id, "CONTAINS_IMAGE", doc=source)
            concept, _ = tree.match_node(caption)
            add_edge(f"onto:{concept or 'FrothFlotation'}", graph_id, "HAS_IMG", doc=source, text=caption)

    for unit in units:
        source = unit.get("source") or unit.get("doc_id") or "unknown"
        doc_id = str(unit.get("doc_id") or source)
        document_id = f"doc:{doc_id}"
        if document_id not in nodes:
            continue
        for triple in unit.get("initial", []):
            subject = str(triple.get("subject") or "").strip()
            obj = str(triple.get("object") or "").strip()
            subject_type = str(triple.get("subject_type") or "Entity")
            object_type = str(triple.get("object_type") or "Entity")
            if not subject or not obj:
                continue
            source_id = f"candidate:{subject_type}:{subject.casefold()}"
            target_id = f"candidate:{object_type}:{obj.casefold()}"
            nodes.setdefault(source_id, {
                "id": source_id, "name": subject, "kind": "Candidate", "type": subject_type, "degree": 0,
            })
            nodes.setdefault(target_id, {
                "id": target_id, "name": obj, "kind": "Candidate", "type": object_type, "degree": 0,
            })
            add_edge(
                source_id,
                target_id,
                str(triple.get("relation") or "RELATED_TO"),
                kind="candidate",
                is_candidate=True,
                text=unit.get("evidence") or unit.get("text"),
                doc=source,
                page_number=unit.get("page_number"),
            )
            add_edge(document_id, source_id, "SUGGESTS", is_candidate=True, doc=source)
            add_edge(document_id, target_id, "SUGGESTS", is_candidate=True, doc=source)

    degree = Counter()
    for edge in edges.values():
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    for node_id, node in nodes.items():
        node["degree"] = degree[node_id]

    return {
        "nodes": list(nodes.values()),
        "links": list(edges.values()),
        "scope": "corpus",
        "documents": len(documents),
        "evidence_units": sum(node["kind"] == "Evidence" for node in nodes.values()),
        "images": sum(len(document.get("image_records", [])) for document in documents),
    }


def _page_args():
    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return None, None, (jsonify(error="limit and offset must be integers"), 400)
    if limit < 1 or limit > 200 or offset < 0:
        return None, None, (jsonify(error="limit must be 1..200 and offset must be non-negative"), 400)
    return limit, offset, None


@app.route("/api/triples")
def api_triples():
    stage = request.args.get("stage", "final")
    if stage not in {"final", "candidate"}:
        return jsonify(error="stage must be 'final' or 'candidate'"), 400
    limit, offset, error = _page_args()
    if error:
        return error
    if not config.TRIPLES_JSON.exists():
        return jsonify(error="No extraction results yet - run `python main.py extract` first."), 404

    units = json.loads(config.TRIPLES_JSON.read_text(encoding="utf-8"))
    key = "triples" if stage == "final" else "initial"
    records = []
    for unit in units:
        provenance = {
            "unit_id": unit.get("unit_id"),
            "source": unit.get("source"),
            "source_path": unit.get("source_path"),
            "modality": unit.get("modality"),
            "page_number": unit.get("page_number"),
            "row_number": unit.get("row_number"),
            "section": unit.get("section"),
            "evidence": unit.get("evidence") or unit.get("text"),
            "backend": unit.get("stage1_backend" if stage == "candidate" else "stage2_backend"),
            "model": unit.get("stage1_model" if stage == "candidate" else "stage2_model"),
        }
        for triple in unit.get(key, []):
            records.append({**triple, **provenance})
    return jsonify(
        stage=stage,
        total=len(records),
        offset=offset,
        limit=limit,
        units=len(units),
        triples=records[offset:offset + limit],
    )


@app.route("/api/images")
def api_images():
    limit, offset, error = _page_args()
    if error:
        return error
    if not config.PARSED_JSON.exists():
        return jsonify(error="No parsed documents yet - run `python main.py parse` first."), 404

    documents = json.loads(config.PARSED_JSON.read_text(encoding="utf-8"))
    records = []
    for document in documents:
        for image in document.get("image_records", []):
            url = public_url(image.get("url") or image.get("image_url"))
            if not url:
                continue
            records.append({
                "id": image.get("image_id"),
                "url": url,
                "caption": image.get("caption") or "Uncaptioned figure",
                "source": document.get("source") or document.get("doc_id"),
                "page_number": image.get("page_number"),
            })
    return jsonify(total=len(records), offset=offset, limit=limit, images=records[offset:offset + limit])


@app.route("/api/metrics")
def api_metrics():
    if not config.METRICS_JSON.exists():
        return jsonify(error="run `python main.py evaluate` first"), 404
    m = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
    if request.args.get("brief"):                       # keep report summaries, not per-sample payloads
        semantic = m.get("semantic", {})
        sample_scores = [
            sample["score"] for sample in semantic.get("samples", [])
            if isinstance(sample.get("score"), (int, float))
        ]
        m["semantic"] = {k: v for k, v in semantic.items() if k != "samples"}
        if sample_scores:
            labels = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
            counts = [0] * len(labels)
            for score in sample_scores:
                counts[min(max(int(score * len(labels)), 0), len(labels) - 1)] += 1
            m["semantic"]["score_distribution"] = [
                {"range": label, "count": count} for label, count in zip(labels, counts)
            ]
        for result in m.get("extraction", {}).values():
            if isinstance(result, dict):
                result.pop("per_sample", None)
        structural = m.get("structural", {})
        detail = structural.get("detail", {})
        m["structural"] = {k: v for k, v in structural.items() if k != "detail"}
        m["structural"]["detail"] = {"L_max": detail.get("L_max")}
    return jsonify(m)


@app.route("/api/retrieve")
def api_retrieve():
    global _retriever
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify(results=[])
    if _retriever is None:
        from rag.retriever import Retriever
        _retriever = Retriever()
    return jsonify(results=_retriever.retrieve(q, config.TOP_K, fuzzy=True))


@app.route("/images/<path:name>")
def images(name):
    for d in (config.PARSED_JSON.parent / "images", config.RAW_IMG_DIR):
        if (d / name).exists():
            return send_from_directory(d, name)
    abort(404)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
