import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import config
from main import _even_sample, build_units, cmd_demo, select_demo_units


class DemoSelectionTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {
                "doc_id": "report_a",
                "source": "report_a.pdf",
                "S_D": [
                    "The flotation cell increases recovery when the aeration rate is adjusted.",
                    "A higher pulp level causes the froth depth to increase during operation.",
                    "The cleaning stage follows the roughing stage in the process.",
                ],
                "I_D": ["http://127.0.0.1:5000/images/figure.png"],
                "C_I": ["Fig. 1 Flotation cell and aeration flow."],
                "records": [],
            },
            {
                "doc_id": "report_b",
                "source": "report_b.pdf",
                "S_D": ["The scavenging stage processes tailings from roughing."],
                "I_D": [],
                "C_I": [],
                "records": [],
            },
        ]

    def test_selects_units_from_only_the_exact_source(self):
        units = select_demo_units("report_a.pdf", documents=self.documents)

        self.assertTrue(units)
        self.assertEqual({unit["doc_id"] for unit in units}, {"report_a"})
        self.assertTrue(any(unit["modality"] == "caption" for unit in units))

    def test_respects_unit_limit_with_even_sampling(self):
        units = select_demo_units("report_a.pdf", max_units=2, documents=self.documents)

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["text"], self.documents[0]["S_D"][0])
        self.assertEqual(units[1]["text"], self.documents[0]["S_D"][2])

    def test_larger_demo_selection_includes_text_records_and_caption_images(self):
        document = {
            "doc_id": "balanced",
            "source": "balanced.pdf",
            "units": [
                {"modality": "Text", "content": f"Flotation recovery changes with condition {index}."}
                for index in range(10)
            ] + [
                {"modality": "TableRow", "content": f"Ore condition {index}."}
                for index in range(2)
            ] + [
                {
                    "modality": "Image",
                    "content": f"Figure {index}. Froth flotation image.",
                    "url": f"http://127.0.0.1:5000/images/figure-{index}.png",
                }
                for index in range(4)
            ],
        }

        units = select_demo_units("balanced.pdf", max_units=10, documents=[document])
        modalities = [unit["modality"] for unit in units]

        self.assertEqual(len(units), 10)
        self.assertEqual(modalities.count("text"), 7)
        self.assertEqual(modalities.count("record"), 1)
        self.assertEqual(modalities.count("caption"), 2)
        self.assertTrue(all(unit.get("image_url") for unit in units if unit["modality"] == "caption"))

    def test_unknown_source_reports_available_sources(self):
        with self.assertRaisesRegex(ValueError, "report_a.pdf"):
            select_demo_units("missing.pdf", documents=self.documents)

    def test_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            select_demo_units("report_a.pdf", max_units=0, documents=self.documents)

    def test_evaluation_sample_is_evenly_spaced(self):
        items = list(range(10))

        self.assertEqual(_even_sample(items, 3), [0, 3, 6])

    def test_non_positive_evaluation_limit_means_no_cap(self):
        items = [1, 2, 3]

        self.assertEqual(_even_sample(items, 0), items)

    def test_sentences_split_from_one_parser_unit_get_unique_extraction_ids(self):
        documents = [{
            "doc_id": "report",
            "source": "report.pdf",
            "units": [{
                "unit_id": "parser-unit-1",
                "modality": "Text",
                "content": (
                    "Aeration rate increases froth depth during flotation. "
                    "The cleaning stage follows the roughing stage in the process."
                ),
            }],
        }]

        units = build_units(documents)

        self.assertEqual(len(units), 2)
        self.assertEqual(len({unit["unit_id"] for unit in units}), 2)
        self.assertTrue(all(unit["source_unit_id"] == "parser-unit-1" for unit in units))

    def test_demo_runs_optimization_two_stage_extraction_graph_and_evaluation(self):
        unit = {
            "unit_id": "u1",
            "doc_id": "report_a",
            "modality": "text",
            "text": "Aeration rate increases froth depth.",
            "image_url": None,
        }
        with TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.json"
            chunks_path.write_text("{}", encoding="utf-8")
            output_path = Path(tmp) / "triples.json"
            output_dir = Path(tmp) / "outputs"
            with (
                patch.object(config, "CHUNKS_JSON", chunks_path),
                patch.object(config, "TRIPLES_JSON", output_path),
                patch.object(config, "OUTPUT_DIR", output_dir),
                patch("main.select_demo_units", return_value=[unit]),
                patch("main._retriever", return_value="retriever"),
                patch("main._load_prompt", return_value="optimized template"),
                patch("prompting.prompt_optimizer.optimize",
                      return_value=(SimpleNamespace(version=2), [{}])) as optimize,
                patch("llm.refinement.TwoStageExtractor") as extractor_type,
                patch("main.cmd_build_kg") as build_graph,
                patch("main.cmd_evaluate") as evaluate,
            ):
                extractor_type.return_value.extract.return_value = {
                    "context": "RAG context",
                    "initial": [{"subject": "air", "relation": "increases"}],
                    "triples": [{"subject": "air", "relation": "increases"}],
                }
                args = Namespace(
                    replace_current=True,
                    source="report_a.pdf",
                    max_units=1,
                    eval_samples=2,
                )

                cmd_demo(args)

                optimize.assert_called_once_with(retriever="retriever")
                extractor_type.assert_called_once_with("optimized template", "retriever")
                extractor_type.return_value.extract.assert_called_once_with(unit["text"])
                self.assertTrue(output_path.exists())
                build_graph.assert_called_once_with(None)
                evaluate.assert_called_once()
                self.assertEqual(evaluate.call_args.args[0].max_samples, 2)


if __name__ == "__main__":
    unittest.main()
