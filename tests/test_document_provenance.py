import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from main import build_units
from parser.mineru_parser import (_clean, _html_table_rows, _normalize_document, parse_csv,
                                 parse_html, parse_pdf_pymupdf, _image_panels)


class DocumentProvenanceTests(unittest.TestCase):
    def test_private_use_glyph_artifacts_are_removed_from_extracted_text(self):
        self.assertEqual(_clean("\ue01f e \ue01a otation process"), "The flotation process")
        self.assertEqual(_clean("the \ue02botation plant"), "the flotation plant")

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

    def test_separate_pdf_image_blocks_share_their_caption_and_figure(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "separate-images.pdf"
            left_image = Image.new("RGB", (200, 150), (30, 60, 90))
            right_image = Image.new("RGB", (200, 150), (90, 60, 30))
            left_bytes = BytesIO()
            right_bytes = BytesIO()
            left_image.save(left_bytes, format="PNG")
            right_image.save(right_bytes, format="PNG")

            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text(
                (50, 40),
                "Operating conditions and cell geometry influence flotation performance.",
            )
            page.insert_image(
                pymupdf.Rect(40, 50, 240, 200), stream=left_bytes.getvalue()
            )
            page.insert_image(
                pymupdf.Rect(260, 50, 460, 200), stream=right_bytes.getvalue()
            )
            page.insert_text((50, 225), "Figure 2 Separate flotation cell views.")
            pdf.save(source)
            pdf.close()

            with patch(
                "parser.mineru_parser.PARSED_IMG_DIR",
                Path(directory) / "parsed_images",
            ):
                parsed = _normalize_document(
                    parse_pdf_pymupdf(source, "separate-images"),
                    source,
                    "separate-images",
                    "pdf",
                )

        images = [unit for unit in parsed["units"] if unit.get("url")]
        captions = [
            unit for unit in parsed["units"]
            if unit.get("modality", "").casefold() == "caption"
        ]
        self.assertEqual(len(images), 2)
        self.assertEqual(len({unit["figure_id"] for unit in images}), 1)
        self.assertEqual(len({unit["caption_id"] for unit in images}), 1)
        self.assertEqual({unit["panel_index"] for unit in images}, {1, 2})
        self.assertTrue(all(unit["panel_count"] == 2 for unit in images))
        self.assertEqual(len(captions), 1)
        self.assertTrue(captions[0]["content"].startswith("Figure 2"))
        self.assertEqual(captions[0]["image_ids"], [unit["image_id"] for unit in images])
        self.assertEqual(captions[0]["image_urls"], [unit["url"] for unit in images])
        self.assertEqual(len(parsed["image_records"]), 2)
        self.assertTrue(all(record["caption"].startswith("Figure 2") for record in parsed["image_records"]))
        linked_text = [
            unit for unit in parsed["units"]
            if unit.get("modality", "").casefold() == "text"
            and unit.get("related_figure_ids")
        ]
        self.assertTrue(linked_text)
        self.assertTrue(all(
            unit["related_figure_ids"] == [images[0]["figure_id"]]
            for unit in linked_text
        ))

    def test_composite_pdf_figure_becomes_linked_panel_image_records(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "composite.pdf"
            image_bytes = BytesIO()
            composite = Image.new("RGB", (600, 400), "white")
            draw = ImageDraw.Draw(composite)
            draw.rectangle((0, 0, 293, 193), fill=(30, 60, 90))
            draw.rectangle((307, 0, 599, 193), fill=(90, 60, 30))
            draw.rectangle((0, 207, 293, 399), fill=(60, 90, 30))
            draw.rectangle((307, 207, 599, 399), fill=(30, 90, 90))
            composite.save(image_bytes, format="PNG")

            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text(
                (50, 70),
                "Operating conditions and cell geometry influence flotation performance.",
            )
            page.insert_image(pymupdf.Rect(40, 100, 555, 380), stream=image_bytes.getvalue())
            page.insert_text((50, 410), "Figure 1 Four flotation froth samples.")
            pdf.save(source)
            pdf.close()

            with patch("parser.mineru_parser.PARSED_IMG_DIR", Path(directory) / "parsed_images"):
                parsed = _normalize_document(
                    parse_pdf_pymupdf(source, "composite"), source, "composite", "pdf"
                )
                self.assertTrue(all(
                    (Path(directory) / "parsed_images" / unit["url"].rsplit("/", 1)[-1]).is_file()
                    for unit in parsed["units"] if unit.get("url")
                ))

        images = [unit for unit in parsed["units"] if unit.get("url")]
        image_records = parsed["image_records"]
        self.assertEqual(len(images), 4)
        self.assertEqual(len({unit["figure_id"] for unit in images}), 1)
        self.assertEqual(len({unit["image_id"] for unit in images}), 4)
        self.assertEqual({unit["panel_index"] for unit in images}, {1, 2, 3, 4})
        self.assertTrue(all(unit["panel_count"] == 4 for unit in images))
        caption_units = [
            unit for unit in parsed["units"]
            if unit.get("modality", "").casefold() == "caption"
        ]
        self.assertEqual(len(caption_units), 1)
        self.assertTrue(caption_units[0]["content"].startswith("Figure 1"))
        self.assertEqual(caption_units[0]["image_ids"], [unit["image_id"] for unit in images])
        self.assertEqual(len(caption_units[0]["image_urls"]), 4)
        self.assertEqual(len(image_records), 4)
        self.assertTrue(all(record["caption_id"] == images[0]["caption_id"] for record in image_records))
        related_text_ids = set(images[0]["related_text_ids"])
        self.assertTrue(related_text_ids)
        linked_text = [unit for unit in parsed["units"] if unit["unit_id"] in related_text_ids]
        self.assertTrue(any("Operating conditions" in unit["content"] for unit in linked_text))
        self.assertTrue(all(images[0]["figure_id"] in unit["related_figure_ids"] for unit in linked_text))
        self.assertTrue(all(record["related_text"] for record in image_records))
        self.assertTrue(all(record["page_number"] == 1 for record in image_records))
        extraction_units = build_units([dict(
            parsed, doc_id="composite", source=source.name, modality="pdf"
        )])
        extracted_captions = [unit for unit in extraction_units if unit["modality"] == "caption"]
        self.assertEqual(len(extracted_captions), 1)
        self.assertEqual(extracted_captions[0]["image_ids"], caption_units[0]["image_ids"])

    def test_sparse_vector_pdf_figure_stays_whole_and_includes_nearby_labels(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vector-figure.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.draw_rect(pymupdf.Rect(200, 140, 400, 340), width=2)
            page.draw_line((400, 170), (325, 205), width=2)
            page.draw_rect(pymupdf.Rect(220, 165, 250, 300), width=1)
            page.draw_rect(pymupdf.Rect(350, 165, 380, 300), width=1)
            page.insert_text((270, 130), "Back of Cell")
            page.insert_text((420, 165), "Mechanism")
            page.insert_text((132, 310), "Overflow Lip")
            page.insert_text((270, 330), "Front of Cell")
            page.insert_text((275, 365), "Figure 1 Cell mechanism")
            pdf.save(source)
            pdf.close()

            image_dir = Path(directory) / "parsed_images"
            with patch("parser.mineru_parser.PARSED_IMG_DIR", image_dir):
                parsed = parse_pdf_pymupdf(source, "vector-figure")

            image_units = [unit for unit in parsed["units"] if unit.get("url")]
            self.assertEqual(len(image_units), 1)
            image_path = image_dir / image_units[0]["url"].rsplit("/", 1)[-1]
            with Image.open(image_path) as image:
                self.assertGreater(image.width, 500)
                ink = image.convert("L").point(lambda pixel: 255 if pixel < 128 else 0)
                ink_bounds = ink.getbbox()
                self.assertIsNotNone(ink_bounds)
                self.assertLess(ink_bounds[0], 20)

    def test_panel_splitter_leaves_single_images_unchanged(self):
        image = Image.new("RGB", (300, 200), (25, 80, 120))
        stream = BytesIO()
        image.save(stream, format="PNG")
        panels = _image_panels(stream.getvalue())

        self.assertEqual(len(panels), 1)
        self.assertFalse(panels[0][1])

    def test_scan_only_pdf_uses_ocr_text_when_available(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scanned.pdf"
            image = Path(directory) / "scan.png"
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 150), 0)
            pixmap.clear_with(240)
            pixmap.save(image)
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(image))
            pdf.save(source)
            pdf.close()

            with patch("parser.mineru_parser.PARSED_IMG_DIR", Path(directory) / "parsed_images"), \
                    patch("parser.mineru_parser._ocr_page_text",
                          return_value="Flotation recovery increased after adjusting the aeration rate."):
                parsed = parse_pdf_pymupdf(source, "scanned")
                document = dict(
                    _normalize_document(parsed, source, "scanned", "pdf"),
                    doc_id="scanned", source=source.name, modality="pdf",
                )

        units = build_units([document])
        ocr_units = [unit for unit in units if unit.get("extraction_method") == "tesseract_ocr"]
        self.assertEqual(parsed["ocr_pages"], [1])
        self.assertEqual(parsed["ocr_unavailable_pages"], [])
        self.assertEqual(len(ocr_units), 1)
        self.assertEqual(ocr_units[0]["page_number"], 1)
        self.assertEqual(ocr_units[0]["source"], "scanned.pdf")

    def test_scan_only_pdf_reports_pages_when_ocr_is_unavailable(self):
        import pymupdf
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scanned.pdf"
            image = Path(directory) / "scan.png"
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 150), 0)
            pixmap.clear_with(240)
            pixmap.save(image)
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(image))
            pdf.save(source)
            pdf.close()

            with patch("parser.mineru_parser.PARSED_IMG_DIR", Path(directory) / "parsed_images"), \
                    patch("parser.mineru_parser._ocr_page_text", return_value=None):
                parsed = parse_pdf_pymupdf(source, "scanned")

        self.assertEqual(parsed["units"], [])
        self.assertEqual(parsed["ocr_unavailable_pages"], [1])

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
