import tempfile
import unittest
from pathlib import Path

from main import build_units
from parser.mineru_parser import (_html_table_rows, _normalize_document, parse_csv, parse_html,
                                 parse_pdf_pymupdf)


class DocumentProvenanceTests(unittest.TestCase):
    def test_csv_rows_keep_source_row_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plant.csv"
            source.write_text(
                "Time,Process,Equipment,Operating status,Problem description\n"
                "10:00,Roughing,Cell 1,Running,No abnormalities\n",
                encoding="utf-8",
            )
            parsed = _normalize_document(parse_csv(source, "plant"), source, "plant", "csv")
            document = dict(parsed, doc_id="plant", source=source.name, modality="csv")

        units = build_units([document])

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["source"], "plant.csv")
        self.assertEqual(units[0]["row_number"], 1)
        self.assertIn('"Equipment": "Cell 1"', units[0]["evidence"])
        self.assertIsNone(units[0]["page_number"])

    def test_html_table_rows_keep_row_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.html"
            source.write_text(
                "<h1>Plant results</h1><p>Flotation recovery increases with stable process conditions.</p>"
                "<table><caption>Table 1 Operating data</caption><tr><th>Time</th><th>pH</th></tr>"
                "<tr><td>10:00</td><td>8.2</td></tr></table>",
                encoding="utf-8",
            )
            parsed = _normalize_document(parse_html(source, "report"), source, "report", "html")
            document = dict(parsed, doc_id="report", source=source.name, modality="html")

        units = build_units([document])
        row_units = [unit for unit in units if unit["modality"] == "record"]

        self.assertTrue(any(unit["modality"] == "text" for unit in units))
        self.assertEqual(len(row_units), 1)
        self.assertEqual(row_units[0]["row_number"], 1)
        self.assertIn('"pH": "8.2"', row_units[0]["evidence"])
        self.assertIsNone(row_units[0]["page_number"])

    def test_pdf_text_and_formula_keep_one_based_page_and_evidence(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "paper.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text((50, 80), "Flotation recovery increases when aeration rate is carefully adjusted.")
            page.insert_text((50, 120), "R = a + b (1)")
            pdf.save(source)
            pdf.close()

            with patch("parser.mineru_parser.PARSED_IMG_DIR", Path(directory) / "images"):
                parsed = _normalize_document(parse_pdf_pymupdf(source, "paper"), source, "paper", "pdf")
                document = dict(parsed, doc_id="paper", source=source.name, modality="pdf")

        units = build_units([document])

        self.assertTrue(any(unit["modality"] == "formula" for unit in units))
        self.assertTrue(all(unit["page_number"] == 1 for unit in units))
        self.assertTrue(all(unit["source"] == "paper.pdf" and unit["evidence"] for unit in units))

    def test_pdf_image_caption_is_a_provenance_preserving_extraction_unit(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "figures.pdf"
            image = Path(directory) / "figure.png"
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 150), 0)
            pixmap.clear_with(240)
            pixmap.save(image)
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_image(pymupdf.Rect(50, 50, 250, 200), filename=str(image))
            page.insert_text((50, 225), "Figure 1 Flotation cell overview")
            pdf.save(source)
            pdf.close()

            with patch("parser.mineru_parser.PARSED_IMG_DIR", Path(directory) / "parsed_images"):
                parsed = _normalize_document(parse_pdf_pymupdf(source, "figures"), source, "figures", "pdf")
                document = dict(parsed, doc_id="figures", source=source.name, modality="pdf")

        units = build_units([document])
        caption_units = [unit for unit in units if unit["modality"] == "caption"]

        self.assertEqual(len(caption_units), 1)
        self.assertIn("Figure 1", caption_units[0]["text"])
        self.assertEqual(caption_units[0]["page_number"], 1)
        self.assertTrue(caption_units[0]["image_url"].endswith(".png"))

    def test_mineru_html_table_markup_is_converted_to_rows(self):
        rows = _html_table_rows(
            "<table><tr><th>Time</th><th>Recovery</th></tr>"
            "<tr><td>10:00</td><td>82%</td></tr></table>"
        )

        self.assertEqual(rows, [["Time", "Recovery"], ["10:00", "82%"]])

    def test_duplicate_pdf_content_is_deduplicated_with_all_page_references(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "duplicate.pdf"
            parsed = _normalize_document({
                "parser": "test",
                "units": [
                    {"content": "The same flotation evidence occurs on this page.",
                     "modality": "Text", "position": 0},
                    {"content": "The same flotation evidence occurs on this page.",
                     "modality": "Text", "position": 4},
                ],
            }, source, "duplicate", "pdf")
            document = dict(parsed, doc_id="duplicate", source=source.name, modality="pdf")

        units = build_units([document])

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["page_number"], 1)
        self.assertEqual(units[0]["duplicate_page_numbers"], [1, 5])


if __name__ == "__main__":
    unittest.main()
