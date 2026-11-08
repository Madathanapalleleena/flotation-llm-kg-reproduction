import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import config
from evaluation.graph_metrics import graph_consistency_assessment
from kg.ontology_mapper import assemble
from main import _evaluation_units, cmd_evaluate


class EvaluationScopeTests(unittest.TestCase):
    def test_quality_units_pair_held_out_text_with_proposed_predictions(self):
        rows = [
            {"id": "test-1", "user": "Roughing uses a flotation cell."},
            {"id": "test-2", "user": "Aeration rate increases froth depth."},
        ]
        predictions = {
            "test-1": [{
                "subject": "roughing",
                "subject_type": "ProcessStage",
                "relation": "uses",
                "object": "flotation cell",
                "object_type": "Equipment",
            }],
            "test-2": [{
                "subject": "aeration rate",
                "subject_type": "Parameter",
                "relation": "increases",
                "object": "froth depth",
                "object_type": "Parameter",
            }],
        }

        units = _evaluation_units(rows, predictions)

        self.assertEqual([unit["text"] for unit in units], [row["user"] for row in rows])
        self.assertEqual([unit["triples"] for unit in units],
                         [predictions[row["id"]] for row in rows])
        graph, _ = assemble(units)
        self.assertTrue(any(edge["type"] == "HAS_ENTITIES" for edge in graph["edges"]))

    def test_numeric_held_out_ids_are_serialized_as_graph_string_properties(self):
        row = {"id": 40, "user": "Roughing uses a flotation cell."}
        triple = {
            "subject": "roughing",
            "subject_type": "ProcessStage",
            "relation": "uses",
            "object": "flotation cell",
            "object_type": "Equipment",
        }

        unit = _evaluation_units([row], {40: [triple]})[0]
        graph, _ = assemble([unit])

        self.assertEqual(unit["unit_id"], "40")
        self.assertEqual(unit["doc_id"], "40")
        self.assertEqual(graph_consistency_assessment(graph)["n_issues"], 0)

    def test_evaluation_uses_held_out_predictions_for_all_quality_scores(self):
        row = {
            "id": "test-1",
            "user": "Roughing uses a flotation cell.",
            "gold": [{
                "subject": "roughing",
                "subject_type": "ProcessStage",
                "relation": "uses",
                "object": "flotation cell",
                "object_type": "Equipment",
            }],
        }

        class Judge:
            kind = "test"

            def chat(self, messages, temperature=None):
                return '{"score": 0.9, "reason": "supported"}'

        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            with (
                patch.object(config, "METRICS_JSON", metrics_path),
                patch("main._mock_warning"),
                patch("main._retriever", return_value=None),
                patch("main._load_prompt", return_value="prompt"),
                patch("prompting.prompt_optimizer.load_training", return_value=[row]),
                patch("prompting.prompt_optimizer.split_dev_test", return_value=([], [row])),
                patch("prompting.prompt_optimizer.run_extraction", return_value={"test-1": row["gold"]}),
                patch("llm.refinement.TwoStageExtractor") as extractor_type,
                patch("llm.extraction.get_client", return_value=Judge()),
            ):
                extractor_type.return_value.extract.return_value = {"triples": row["gold"]}
                with contextlib.redirect_stdout(io.StringIO()):
                    cmd_evaluate(Namespace(max_samples=1, full_semantic=True))

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        protocol = metrics["evaluation_protocol"]
        self.assertIn("Proposed_pipeline predictions", protocol["quality_graph_scope"])
        self.assertEqual(metrics["structural"]["n_nodes"], 17)
        self.assertEqual(metrics["semantic"]["n_records"], 1)
        self.assertEqual(metrics["semantic"]["n_sampled"], 1)
        self.assertEqual(metrics["semantic"]["input_triples"], 1)
        self.assertEqual(metrics["semantic"]["input_scope"],
                         "Proposed_pipeline predictions on the held-out sample")
        self.assertEqual(metrics["S_final"], 0.4 * metrics["structural"]["S1"] + 0.6 * metrics["semantic"]["S2"])


if __name__ == "__main__":
    unittest.main()
