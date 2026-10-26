import unittest

from evaluation.graph_metrics import graph_consistency_assessment
from kg.ontology_mapper import assemble


class GraphMetricsTests(unittest.TestCase):
    def test_valid_assembled_graph_has_no_integrity_issues(self):
        unit = {
            "unit_id": "u1",
            "doc_id": "d1",
            "modality": "text",
            "text": "Aeration rate increases froth depth.",
            "triples": [{
                "subject": "aeration rate",
                "subject_type": "Parameter",
                "relation": "increases",
                "object": "froth depth",
                "object_type": "Parameter",
            }],
        }
        graph, _ = assemble([unit])

        result = graph_consistency_assessment(graph)

        self.assertEqual(result["n_issues"], 0)
        self.assertEqual(result["consistency"], 1.0)

    def test_reports_dangling_and_unknown_relationships(self):
        graph = {
            "nodes": [{"id": "n1", "name": "Aeration rate", "kind": "Entity", "type": "Parameter"}],
            "edges": [{"source": "n1", "target": "missing", "type": "UNRECOGNIZED"}],
        }

        result = graph_consistency_assessment(graph)

        self.assertEqual(result["dangling_relationships"], 1)
        self.assertEqual(result["unknown_relationship_types"], 1)
        self.assertEqual(result["n_issues"], 2)

    def test_reports_duplicate_relationships(self):
        edge = {"source": "n1", "target": "n2", "type": "INCREASES"}
        graph = {
            "nodes": [
                {"id": "n1", "name": "a", "kind": "Entity", "type": "Parameter"},
                {"id": "n2", "name": "b", "kind": "Entity", "type": "Parameter"},
            ],
            "edges": [edge, dict(edge)],
        }

        result = graph_consistency_assessment(graph)

        self.assertEqual(result["duplicate_relationships"], 1)

    def test_reports_conflicting_triple_edges_across_source_units(self):
        graph = {
            "nodes": [
                {"id": "a", "name": "Aeration rate", "kind": "Entity", "type": "Parameter"},
                {"id": "b", "name": "Froth depth", "kind": "Entity", "type": "Parameter"},
            ],
            "edges": [
                {"source": "a", "target": "b", "type": "INCREASES", "kind": "triple"},
                {"source": "a", "target": "b", "type": "DECREASES", "kind": "triple"},
            ],
        }

        result = graph_consistency_assessment(graph)

        self.assertEqual(result["contradictory_triples"], 1)


if __name__ == "__main__":
    unittest.main()
