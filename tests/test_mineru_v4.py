import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from parser import mineru_parser


class MineruV4MarkdownTests(unittest.TestCase):
    def test_strips_author_byline_and_page_url_from_mineru_output(self):
        markdown = (
            "<!-- page 1 of 1 -->\n"
            "Abdolrahim Foroutan<strong><sup>1</sup></strong>, "
            "Majid Abbas Zadeh HajiAbadi<strong><sup>1</sup></strong>, "
            "Yaser Kianinia & Mahdi Ghadiri\n\n"
            "www.nature.com/scientificreports\n\n"
            "Collector type and pulp pH play an important role in lead-zinc flotation."
        )

        parsed = mineru_parser._parse_mineru_v4_markdown(markdown, "sample")

        self.assertEqual(len(parsed["S_D"]), 1)
        self.assertEqual(parsed["S_D"][0],
                         "Collector type and pulp pH play an important role in lead-zinc flotation.")
        self.assertEqual(parsed["filtered_text_fragments"], 2)

    def test_strips_byline_merged_with_body_by_mineru_line_break(self):
        markdown = (
            "<!-- page 1 of 1 -->\n"
            "**Abdolrahim Foroutan**<strong><sup>1</sup></strong>, "
            "**Majid Abbas Zadeh HajiAbadi**<strong><sup>1</sup></strong>, "
            "**Yaser Kianinia**<strong><sup>1</sup></strong> & "
            "**Mahdi Ghadiri**<strong><sup>2,3</sup></strong><sup>\\</sup> "
            "**Collector type and pulp pH play an important role in lead-zinc flotation.**"
        )

        parsed = mineru_parser._parse_mineru_v4_markdown(markdown, "sample")

        self.assertEqual(parsed["S_D"], [
            "Collector type and pulp pH play an important role in lead-zinc flotation."
        ])

    def test_converts_markdown_text_tables_and_page_numbers(self):
        markdown = (
            "<!-- page 1 of 2 -->\n"
            "# Flotation Results\n\n"
            "The aeration rate increases froth depth during the flotation process.\n\n"
            "Table 1: Operating conditions\n\n"
            "| Parameter | Value |\n"
            "| --- | --- |\n"
            "| Aeration rate | 5 m3/min |\n"
            "<!-- page 2 of 2 -->\n"
            "The cleaning stage follows the roughing stage in the flotation circuit."
        )

        parsed = mineru_parser._parse_mineru_v4_markdown(markdown, "sample")

        self.assertEqual(parsed["parser"], "mineru-v4-flash")
        self.assertEqual(parsed["sections"], [{"title": "Flotation Results", "page": 0}])
        self.assertEqual(len(parsed["S_D"]), 2)
        self.assertEqual(parsed["units"][0]["position"], 0)
        row = next(unit for unit in parsed["units"] if unit["modality"] == "TableRow")
        self.assertEqual(row["position"], 0)
        self.assertIn("Parameter = Aeration rate; Value = 5 m3/min", row["content"])
        self.assertEqual(parsed["units"][-1]["position"], 1)
        self.assertEqual(parsed["I_D"], [])
        self.assertEqual(parsed["C_I"], [])

    def test_rejects_malformed_or_empty_content_as_error_at_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(config, "MINERU_OUT_DIR", Path(tmp)),
                patch.object(mineru_parser, "mineru_available", return_value="mineru"),
                patch.object(mineru_parser, "_is_mineru_v4", return_value=True),
                patch.object(mineru_parser, "_mineru_v4_parse_payload", return_value={"content": None}),
            ):
                with self.assertRaisesRegex(RuntimeError, "without returning Markdown"):
                    mineru_parser.parse_pdf_mineru(Path(tmp) / "input.pdf", "sample")


class MineruV4CliTests(unittest.TestCase):
    def test_identifies_v4_from_version_command(self):
        result = subprocess.CompletedProcess(
            args=["mineru", "--version"], returncode=0, stdout="MinerU version: 4.0.12\n", stderr=""
        )
        with patch.object(mineru_parser, "_run_mineru_command", return_value=result) as run:
            self.assertTrue(mineru_parser._is_mineru_v4("mineru"))
        run.assert_called_once_with(["mineru", "--version"], timeout=30)

    def test_starts_local_server_and_retries_when_cli_reports_server_missing(self):
        error = json.dumps({
            "error": {"code": "server_not_running", "message": "Run mineru server start."}
        })
        success = json.dumps({
            "parse": {"status": "done", "tier": "flash"},
            "content": {"format": "markdown", "content": "Flotation text"},
        })
        results = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout=error, stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Server started.", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=success, stderr=""),
        ]
        with patch.object(mineru_parser, "_run_mineru_command", side_effect=results) as run:
            payload = mineru_parser._mineru_v4_parse_payload("mineru", Path("sample.pdf"))

        self.assertEqual(payload["content"]["content"], "Flotation text")
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[1].args[0], ["mineru", "server", "start"])
        self.assertIn("--tier", run.call_args_list[0].args[0])
        self.assertIn("flash", run.call_args_list[0].args[0])
        self.assertIn("--json", run.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
