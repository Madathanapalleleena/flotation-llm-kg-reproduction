"""Eq. (18)-(21): representative node, HAS_ENTITIES ontology link, HAS_IMG image link."""
import unittest

from kg.ontology_mapper import assemble
from kg.representative_node import representative_node
from ontology.knowledge_tree import KnowledgeTree
from evaluation.graph_metrics import graph_consistency_assessment


def T(s, st, r, o, ot):
    return dict(subject=s, subject_type=st, relation=r, object=o, object_type=ot)


class KGAssembly(unittest.TestCase):
    triples = [T("pulp concentration", "Parameter", "increases", "recovery rate", "Product"),
               T("pulp concentration", "Parameter", "affects", "froth depth", "Parameter"),
               T("aeration rate", "Parameter", "increases", "froth depth", "Parameter")]

    def test_eq18_argmax_degree(self):
        rep = representative_node(self.triples)
        self.assertEqual(rep["name"], "pulp concentration")      # degree 2 (tie with froth depth -> earliest mention)
        self.assertEqual(rep["degree"], 2)

    def test_eq19_21_edges(self):
        unit = dict(unit_id="u0", doc_id="d", modality="caption", text="Fig. 1 pulp concentration ...",
                    image_url="http://127.0.0.1:5000/images/x.png", triples=self.triples)
        g, st = assemble([unit])
        types = {e["type"] for e in g["edges"]}
        self.assertIn("HAS_ENTITIES", types)                      # E_onto
        self.assertIn("HAS_IMG", types)                           # E_img
        self.assertIn("INCREASES", types)                         # E_triples
        img = [e for e in g["edges"] if e["type"] == "HAS_IMG"][0]
        self.assertTrue(img["source"].startswith("ent:Parameter:pulp concentration"))
        self.assertEqual(st["mapped"], 1)

    def test_caption_links_every_panel_of_a_split_figure(self):
        urls = [
            "http://127.0.0.1:5000/images/figure_panel_1.png",
            "http://127.0.0.1:5000/images/figure_panel_2.png",
        ]
        unit = dict(
            unit_id="figure-caption",
            doc_id="d",
            modality="caption",
            text="Figure 1. Flotation cell structure.",
            caption="Figure 1. Flotation cell structure.",
            image_url=urls[0],
            image_urls=urls,
            image_ids=["figure:panel-01", "figure:panel-02"],
            figure_id="figure",
            caption_id="figure:caption",
            related_text_ids=["d:text-001"],
            triples=self.triples,
        )

        graph, _ = assemble([unit])
        image_nodes = [node for node in graph["nodes"] if node["kind"] == "Image"]
        image_edges = [edge for edge in graph["edges"] if edge["type"] == "HAS_IMG"]

        self.assertEqual({node["url"] for node in image_nodes}, set(urls))
        self.assertEqual(len(image_edges), 2)
        self.assertTrue(all(node["figure_id"] == "figure" for node in image_nodes))
        self.assertTrue(all(node["caption_id"] == "figure:caption" for node in image_nodes))

    def test_caption_images_are_retained_without_extracted_triples(self):
        image_url = "http://127.0.0.1:5000/images/flotation-cell.png"
        unit = dict(
            unit_id="caption-without-triples",
            doc_id="d",
            modality="caption",
            text="Figure 1. Flotation cell overview.",
            caption="Figure 1. Flotation cell overview.",
            image_url=image_url,
            image_id="figure-1",
            triples=[],
        )

        graph, stats = assemble([unit])
        image_nodes = [node for node in graph["nodes"] if node["kind"] == "Image"]
        image_edges = [edge for edge in graph["edges"] if edge["type"] == "HAS_IMG"]

        self.assertEqual(len(image_nodes), 1)
        self.assertEqual(image_nodes[0]["url"], image_url)
        self.assertEqual(len(image_edges), 1)
        self.assertEqual(image_edges[0]["target"], image_nodes[0]["id"])
        self.assertEqual(image_edges[0]["source"], "onto:FlotationCell")
        self.assertEqual(stats["mapped"], 0)

    def test_ontology_has_literal_stage_equipment_type_path(self):
        kt = KnowledgeTree()
        edges = set(kt.edges)

        self.assertIn(("Roughing", "HAS_EQUIPMENT_DOWN", "Equipment"), edges)
        self.assertIn(("Equipment", "HAS_TYPE_DOWN", "FlotationCell"), edges)
        self.assertEqual(kt.nodes["Equipment"].attributes, ["EquipmentType"])
        self.assertIn("ProcessParameters", kt.nodes["FlotationCell"].attributes)

    def test_equipment_triple_maps_entity_to_specific_ontology_concept(self):
        unit = dict(
            unit_id="equipment-unit",
            doc_id="d",
            modality="text",
            text="Roughing uses a flotation cell.",
            triples=[T("roughing", "ProcessStage", "uses", "flotation cell", "Equipment")],
        )

        graph, _ = assemble([unit])
        edges = {(edge["source"], edge["type"], edge["target"]) for edge in graph["edges"]}

        self.assertIn(("onto:Roughing", "HAS_EQUIPMENT_DOWN", "onto:Equipment"), edges)
        self.assertIn(("onto:Equipment", "HAS_TYPE_DOWN", "onto:FlotationCell"), edges)
        self.assertIn(("onto:FlotationCell", "HAS_ENTITIES", "ent:Equipment:flotation cell"), edges)
        self.assertEqual(graph_consistency_assessment(graph)["n_issues"], 0)


if __name__ == "__main__":
    unittest.main()
