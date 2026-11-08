import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.api import app


class ApiResultsTests(unittest.TestCase):
    def test_corpus_graph_includes_every_document_unit_image_and_candidate(self):
        documents = [
            {
                "doc_id": "paper",
                "source": "paper.pdf",
                "modality": "pdf",
                "units": [{
                    "unit_id": "paper:text-1",
                    "modality": "Text",
                    "content": "Flotation cell uses air bubbles.",
                    "page_number": 2,
                }],
                "image_records": [{
                    "image_id": "paper:figure-1",
                    "url": "http://127.0.0.1:5000/images/figure.png",
                    "caption": "Flotation cell",
                    "page_number": 2,
                }],
            },
            {"doc_id": "log", "source": "log.csv", "modality": "csv", "units": [], "image_records": []},
        ]
        units = [{
            "doc_id": "paper",
            "source": "paper.pdf",
            "initial": [{"subject": "Air", "subject_type": "Parameter", "relation": "affects",
                         "object": "Flotation cell", "object_type": "Equipment"}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            parsed_path = Path(directory) / "documents.json"
            triples_path = Path(directory) / "triples.json"
            parsed_path.write_text(json.dumps(documents), encoding="utf-8")
            triples_path.write_text(json.dumps(units), encoding="utf-8")
            with (
                patch("backend.api.load_graph", return_value={
                    "nodes": [{"id": "onto:FrothFlotation", "name": "Froth Flotation", "kind": "Ontology",
                               "type": "FlotationProcess"}],
                    "edges": [],
                }),
                patch("backend.api.config.PARSED_JSON", parsed_path),
                patch("backend.api.config.TRIPLES_JSON", triples_path),
            ):
                response = app.test_client().get("/api/graph?scope=corpus")

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        nodes = {node["id"]: node for node in result["nodes"]}
        self.assertEqual(result["documents"], 2)
        self.assertEqual(result["evidence_units"], 1)
        self.assertEqual(result["images"], 1)
        self.assertIn("doc:log", nodes)
        self.assertEqual(nodes["img:paper:figure-1"]["url"], "/images/figure.png")
        self.assertIn("candidate:Parameter:air", nodes)
        self.assertTrue(any(link.get("is_candidate") for link in result["links"]))
        self.assertTrue(any(link["type"] == "HAS_EVIDENCE" for link in result["links"]))

    def test_triples_endpoint_separates_final_and_stage_one_candidates(self):
        units = [
            {
                "unit_id": "unit-1",
                "source": "paper.pdf",
                "page_number": 3,
                "initial": [{"subject": "Ore", "relation": "feeds", "object": "Cell"}],
                "triples": [],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            triples_path = Path(directory) / "triples.json"
            triples_path.write_text(json.dumps(units), encoding="utf-8")
            with patch("backend.api.config.TRIPLES_JSON", triples_path):
                client = app.test_client()
                final = client.get("/api/triples?stage=final").get_json()
                candidates = client.get("/api/triples?stage=candidate").get_json()

        self.assertEqual(final["total"], 0)
        self.assertEqual(final["triples"], [])
        self.assertEqual(candidates["total"], 1)
        self.assertEqual(candidates["stage"], "candidate")
        self.assertEqual(candidates["triples"][0]["source"], "paper.pdf")
        self.assertEqual(candidates["triples"][0]["page_number"], 3)

    def test_images_endpoint_returns_caption_and_host_relative_local_url(self):
        documents = [{
            "source": "manual.pdf",
            "image_records": [{
                "image_id": "figure-1",
                "url": "http://127.0.0.1:5000/images/figure.png",
                "caption": "Flotation cell",
                "page_number": 2,
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            parsed_path = Path(directory) / "documents.json"
            parsed_path.write_text(json.dumps(documents), encoding="utf-8")
            with patch("backend.api.config.PARSED_JSON", parsed_path):
                response = app.test_client().get("/api/images")

        self.assertEqual(response.status_code, 200)
        image = response.get_json()["images"][0]
        self.assertEqual(image["url"], "/images/figure.png")
        self.assertEqual(image["caption"], "Flotation cell")
        self.assertEqual(image["source"], "manual.pdf")

    def test_invalid_pagination_is_reported(self):
        response = app.test_client().get("/api/images?limit=0")

        self.assertEqual(response.status_code, 400)
        self.assertIn("limit", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
