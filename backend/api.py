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
    return jsonify(nodes=nodes, links=links)


@app.route("/api/metrics")
def api_metrics():
    if not config.METRICS_JSON.exists():
        return jsonify(error="run `python main.py evaluate` first"), 404
    m = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
    if request.args.get("brief"):                       # drop the bulky per-sample details
        m["semantic"] = {k: v for k, v in m.get("semantic", {}).items() if k != "samples"}
        m["structural"] = {k: v for k, v in m.get("structural", {}).items() if k != "detail"}
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
