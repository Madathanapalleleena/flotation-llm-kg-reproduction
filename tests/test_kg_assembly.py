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
