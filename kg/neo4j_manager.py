"""Step 14 - knowledge storage (Neo4j).   Paper Sec. IV-C.

get_store() returns a Neo4jManager when a Neo4j server is reachable, otherwise a LocalGraphStore that keeps the
same graph in outputs/kg_local.json. Neo4j rebuilds replace only the configured managed scope.
Both stores expose:  write_graph(graph) | read_graph() | clear() | describe()
graph = {"nodes":[{id,name,kind,type,...}], "edges":[{source,target,type,...}]}
"""
import json
import re
import os
import shutil
import tempfile

import config
from ontology.knowledge_tree import KnowledgeTree

_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe(token: str) -> str:
    if not _SAFE.match(token):
        raise ValueError(f"unsafe Cypher identifier: {token!r}")
    return token


class LocalGraphStore:
    kind = "local-json"

    def __init__(self, path=config.KG_JSON):
        self.path = path

    def write_graph(self, graph):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(graph, ensure_ascii=False, indent=1)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                temporary = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                shutil.copy2(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
            os.replace(temporary, self.path)
        except Exception:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def read_graph(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def clear(self):
        if self.path.exists():
            self.path.unlink()

    def describe(self):
        return f"local JSON file {self.path}"


class Neo4jManager:
    kind = "neo4j"

    def __init__(self, uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD, timeout=5):
        from neo4j import GraphDatabase
        self.uri = uri
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)
        self.driver.verify_connectivity()
        self.db = config.NEO4J_DATABASE or None
        kt = KnowledgeTree()
        self.allowed_rel = {r.upper() for r in kt.relation_types} | {e[1] for e in kt.edges} | {"HAS_ENTITIES", "HAS_IMG"}
        self.allowed_label = set(kt.entity_types) | {"Ontology", "Entity", "Image"}

    def close(self):
        self.driver.close()

    def clear(self, scope=config.NEO4J_GRAPH_SCOPE):
        with self.driver.session(database=self.db) as s:
            s.run(
                "MATCH (n:KGNode) WHERE n.kg_scope = $scope OR "
                "(n.kg_scope IS NULL AND $scope = 'default') DETACH DELETE n",
                scope=scope,
            )

    def _replace_graph_tx(self, tx, graph, batch, scope):
        tx.run(
            "MATCH (n:KGNode) WHERE n.kg_scope = $scope OR "
            "(n.kg_scope IS NULL AND $scope = 'default') DETACH DELETE n",
            scope=scope,
        )
        by_label = {}
        for node in graph["nodes"]:
            by_label.setdefault((node["kind"], node["type"] if node["kind"] == "Entity" else None), []).append(node)
        for (kind, typ), nodes in by_label.items():
            labels = ":".join(_safe(label) for label in (kind, typ) if label and label in self.allowed_label)
            if not labels:
                raise ValueError(f"Graph node has no allowed Neo4j labels: {kind!r}, {typ!r}")
            for start in range(0, len(nodes), batch):
                rows = []
                for node in nodes[start:start + batch]:
                    scoped_id = f"{scope}:{node['id']}"
                    props = {key: value for key, value in node.items() if key != "id"}
                    props.update(id=scoped_id, source_id=node["id"], kg_scope=scope)
                    rows.append(dict(id=scoped_id, props=props))
                tx.run(
                    f"UNWIND $rows AS r MERGE (n:KGNode {{id: r.id}}) "
                    f"SET n += r.props SET n:{labels}",
                    rows=rows,
                )

        by_type = {}
        for edge in graph["edges"]:
            relation = str(edge["type"]).upper()
            if relation not in self.allowed_rel:
                raise ValueError(f"Graph contains a disallowed Neo4j relationship type: {relation!r}")
            by_type.setdefault(relation, []).append(edge)
        for relation, edges in by_type.items():
            for start in range(0, len(edges), batch):
                rows = [
                    dict(
                        source=f"{scope}:{edge['source']}",
                        target=f"{scope}:{edge['target']}",
                        props={key: value for key, value in edge.items()
                               if key not in ("source", "target", "type")},
                    )
                    for edge in edges[start:start + batch]
                ]
                tx.run(
                    f"UNWIND $rows AS r MATCH (a:KGNode {{id: r.source, kg_scope: $scope}}) "
                    f"MATCH (b:KGNode {{id: r.target, kg_scope: $scope}}) "
                    f"MERGE (a)-[e:{_safe(relation)}]->(b) SET e += r.props",
                    rows=rows,
                    scope=scope,
                )

    def write_graph(self, graph, batch=500, scope=config.NEO4J_GRAPH_SCOPE):
        if batch < 1:
            raise ValueError("batch must be at least 1")
        with self.driver.session(database=self.db) as s:
            s.run("CREATE CONSTRAINT kgnode_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")
            s.execute_write(self._replace_graph_tx, graph, batch, scope)

    def read_graph(self):
        with self.driver.session(database=self.db) as s:
            nodes = [dict(r["n"]) for r in s.run(
                "MATCH (n:KGNode) WHERE n.kg_scope = $scope OR "
                "(n.kg_scope IS NULL AND $scope = 'default') RETURN n",
                scope=config.NEO4J_GRAPH_SCOPE)]
            edges = [dict(source=r["s"], target=r["t"], type=r["ty"], **{k: v for k, v in r["p"].items()})
                     for r in s.run(
                         "MATCH (a:KGNode)-[e]->(b:KGNode) "
                         "WHERE (a.kg_scope = $scope OR (a.kg_scope IS NULL AND $scope = 'default')) "
                         "AND (b.kg_scope = $scope OR (b.kg_scope IS NULL AND $scope = 'default')) "
                         "RETURN a.id AS s, b.id AS t, type(e) AS ty, properties(e) AS p",
                         scope=config.NEO4J_GRAPH_SCOPE)]
        return dict(nodes=nodes, edges=edges)

    def describe(self):
        return f"Neo4j at {self.uri}"


def get_store(verbose=True):
    mode = config.USE_NEO4J
    if mode in ("auto", "1"):
        try:
            st = Neo4jManager()
            if verbose:
                print(f"[kg] connected to Neo4j at {config.NEO4J_URI}")
            return st
        except Exception as e:  # noqa: BLE001
            if mode == "1":
                raise
            if verbose:
                print(f"[kg] Neo4j not reachable ({type(e).__name__}); using local JSON graph store instead")
    return LocalGraphStore()
