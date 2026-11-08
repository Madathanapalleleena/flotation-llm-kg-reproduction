import unittest

from evaluation.structural_metrics import structural_assessment
from kg.ontology_mapper import assemble


class StructuralMetricsTests(unittest.TestCase):
    def test_reverse_hierarchy_edges_are_not_counted_as_process_loops(self):
        graph, _ = assemble([])

        result = structural_assessment(graph)

        self.assertEqual(result["detail"]["n_process_loops"], 0)

    def test_cycle_in_process_flow_is_counted(self):
        graph, _ = assemble([])
        graph["edges"].append({
            "source": "onto:Cleaning",
            "target": "onto:Roughing",
            "type": "HAS_NEXT_DOWN",
            "kind": "ontology",
        })

        result = structural_assessment(graph)

        self.assertEqual(result["detail"]["n_process_loops"], 1)


if __name__ == "__main__":
    unittest.main()
