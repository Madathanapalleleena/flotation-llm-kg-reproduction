import json
import unittest
from unittest.mock import Mock

from evaluation.semantic_metrics import semantic_assessment
from parser.mineru_parser import _pdf_text_rejection, _repeated_furniture


class ParserTextQualityTests(unittest.TestCase):
    def test_rejects_short_pdf_fragment(self):
        self.assertEqual(_pdf_text_rejection("Page 73"), "short fragment")

    def test_rejects_replacement_character_in_pdf_text(self):
        self.assertEqual(
            _pdf_text_rejection("The metallurgical pro\ufffdcess was measured in flotation tests."),
            "replacement/control characters",
        )

    def test_rejects_fragmented_pdf_text(self):
        self.assertEqual(
            _pdf_text_rejection("A b c d e f g h i j and flotation process."),
            "fragmented words",
        )

    def test_keeps_complete_prose(self):
        self.assertEqual(
            _pdf_text_rejection(
                "The flotation cell increases recovery when the aeration rate is adjusted."
            ),
            "",
        )

    def test_repeated_page_numbered_footer_is_detected(self):
        entries = [
            (page, "bottom", f"CSA Global Report No R161.2018 {page}")
            for page in range(4)
        ]
        self.assertIn(("bottom", "csa global report no r#.# #"), _repeated_furniture(entries, 4))

    def test_repeated_furniture_requires_multiple_pages(self):
        entries = [(page, "bottom", "Company report footer") for page in range(2)]
        self.assertFalse(_repeated_furniture(entries, 2))


class SemanticMetricValidationTests(unittest.TestCase):
    def test_judge_receives_only_triples_accepted_by_graph_validator(self):
        client = Mock()
        client.kind = "test"
        client.chat.return_value = '{"score": 0.8, "reason": "supported"}'
        units = [{
            "unit_id": "u1",
            "text": "Aeration rate increases froth depth.",
            "triples": [
                {
                    "subject": "aeration rate",
                    "subject_type": "Parameter",
                    "relation": "increases",
                    "object": "froth depth",
                    "object_type": "Parameter",
                },
                {
                    "subject": "aeration rate",
                    "subject_type": "Parameter",
                    "relation": "assessed",
                    "object": "froth depth",
                    "object_type": "Parameter",
                },
                {
                    "subject": "aeration rate",
                    "subject_type": "Organization",
                    "relation": "increases",
                    "object": "froth depth",
                    "object_type": "Parameter",
                },
            ],
        }]

        result = semantic_assessment(units, client=client, full=True)

        user_message = client.chat.call_args.args[0][1]["content"]
        judged_triples = json.loads(user_message.split("Triples: ", 1)[1])["triples"]
        self.assertEqual(len(judged_triples), 1)
        self.assertEqual(result["input_triples"], 3)
        self.assertEqual(result["validated_triples"], 1)
        self.assertEqual(result["rejected_triples"], 2)
        self.assertAlmostEqual(result["validation_rate"], 1 / 3)
        self.assertEqual(result["S2"], 0.8)


if __name__ == "__main__":
    unittest.main()
