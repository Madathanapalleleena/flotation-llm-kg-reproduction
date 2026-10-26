import json
import tempfile
import unittest
from pathlib import Path

from kg.neo4j_manager import LocalGraphStore, Neo4jManager


class RecordingTransaction:
    def __init__(self):
        self.calls = []

    def run(self, query, **parameters):
        self.calls.append((query, parameters))


class GraphStoreTests(unittest.TestCase):
    def test_local_graph_replacement_keeps_recovery_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            store = LocalGraphStore(path)
            store.write_graph({"nodes": [{"id": "old"}], "edges": []})
            store.write_graph({"nodes": [{"id": "new"}], "edges": []})

            current = json.loads(path.read_text(encoding="utf-8"))
            backup = json.loads(path.with_suffix(".json.bak").read_text(encoding="utf-8"))

        self.assertEqual(current["nodes"][0]["id"], "new")
        self.assertEqual(backup["nodes"][0]["id"], "old")

    def test_neo4j_rebuild_transaction_is_limited_to_graph_scope(self):
        manager = Neo4jManager.__new__(Neo4jManager)
        manager.allowed_label = {"Entity", "Parameter", "Ontology", "Image"}
        manager.allowed_rel = {"INCREASES"}
        graph = {
            "nodes": [
                {"id": "ent:Parameter:a", "name": "a", "kind": "Entity", "type": "Parameter"},
                {"id": "ent:Parameter:b", "name": "b", "kind": "Entity", "type": "Parameter"},
            ],
            "edges": [{"source": "ent:Parameter:a", "target": "ent:Parameter:b", "type": "INCREASES"}],
        }
        transaction = RecordingTransaction()

        manager._replace_graph_tx(transaction, graph, 20, "test-scope")

        self.assertIn("n.kg_scope = $scope", transaction.calls[0][0])
        self.assertEqual(transaction.calls[0][1]["scope"], "test-scope")
        self.assertNotIn("MATCH (n:KGNode) DETACH DELETE n", transaction.calls[0][0])
        node_rows = transaction.calls[1][1]["rows"]
        self.assertTrue(all(row["props"]["kg_scope"] == "test-scope" for row in node_rows))
        self.assertTrue(all(row["id"].startswith("test-scope:") for row in node_rows))
        self.assertEqual(transaction.calls[2][1]["scope"], "test-scope")


if __name__ == "__main__":
    unittest.main()
