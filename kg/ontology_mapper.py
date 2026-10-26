"""Step 13 - map validated triples onto the flotation knowledge tree and assemble the KG.   Paper Eq. (19)-(21).

    E       = E_triples  U  E_img  U  E_onto
    E_onto  = {(b, HAS_ENTITIES, v*) | b = M(v*)}
    E_img   = {(v*, HAS_IMG, u)  | v* in V_caption}

M(v*) maps the representative node to an ontology concept b by alias voting over the entity names of its subgraph
(+ the source sentence); if nothing matches, it falls back to the ontology node bound to the entity's type.
"""
import re
import json

from kg.representative_node import img_edges, representative_node
from kg.triple_validator import TripleValidator
from ontology.knowledge_tree import KnowledgeTree


def ent_id(name, typ):
    norm = re.sub(r"\s+", " ", name.lower()).strip()
    return f"ent:{typ}:{norm}"


class KGAssembler:
    def __init__(self, kt: KnowledgeTree = None):
        self.kt = kt or KnowledgeTree()
        self.validator = TripleValidator(self.kt)
        self.nodes, self.edges, self._seen_edges = {}, [], set()
        self._load_backbone()
        self.stats = dict(units=0, triples=0, rejected=0, mapped=0)

    # --------------------------------------------------------------- backbone
    def _load_backbone(self):
        for n in self.kt.nodes.values():
            self.nodes[f"onto:{n.id}"] = dict(id=f"onto:{n.id}", name=n.label, kind="Ontology", type=n.cls,
                                              level=n.level, attributes=n.attributes)
        for s, t, o in self.kt.edges:
            self._edge(f"onto:{s}", t, f"onto:{o}", kind="ontology")

    def _edge(self, s, typ, o, **props):
        key = (s, typ, o)
        if key in self._seen_edges:
            edge = next(item for item in self.edges if (item["source"], item["type"], item["target"]) == key)
            for field in ("evidence", "provenance"):
                current = edge.get(field, [])
                incoming = props.get(field, [])
                if not isinstance(current, list):
                    current = [current]
                if not isinstance(incoming, list):
                    incoming = [incoming]
                edge[field] = list(dict.fromkeys(current + incoming))
            return
        self._seen_edges.add(key)
        self.edges.append(dict(source=s, target=o, type=typ, **props))

    def _entity(self, name, typ):
        i = ent_id(name, typ)
        self.nodes.setdefault(i, dict(id=i, name=name, kind="Entity", type=typ))
        return i

    # ---------------------------------------------------------------- mapping
    def map_to_ontology(self, rep, triples, text=""):
        names = " ".join(f"{t['subject']} {t['object']}" for t in triples if rep["name"] in (t["subject"], t["object"]))
        nid, hits = self.kt.match_node(f"{rep['name']} {names}")
        if not hits:
            nid, hits = self.kt.match_node(text)
        if not hits:
            nid = self.kt.type_to_node.get(rep["type"], "FrothFlotation")
        return f"onto:{nid}"

    # ------------------------------------------------------------------- unit
    def add_unit(self, unit: dict):
        """unit: {unit_id, doc_id, modality, text, image_url?, triples}"""
        self.stats["units"] += 1
        triples, rejected = self.validator.validate(unit.get("triples", []), check_conflicts=True)
        self.stats["rejected"] += len(rejected)
        if not triples:
            return
        provenance = json.dumps({
            "source": unit.get("source"),
            "source_path": unit.get("source_path"),
            "page_number": unit.get("page_number"),
            "row_number": unit.get("row_number"),
            "duplicate_page_numbers": unit.get("duplicate_page_numbers", []),
            "duplicate_row_numbers": unit.get("duplicate_row_numbers", []),
            "section": unit.get("section"),
            "unit_id": unit.get("unit_id"),
            "evidence": unit.get("evidence", unit.get("text", "")),
        }, ensure_ascii=False, sort_keys=True)
        for t in triples:                                                    # E_triples
            s, o = self._entity(t["subject"], t["subject_type"]), self._entity(t["object"], t["object_type"])
            self._edge(s, t["relation"].upper(), o, kind="triple", text=unit["text"], doc=unit["doc_id"],
                       modality=unit["modality"], source_file=unit.get("source", unit["doc_id"]),
                       page_number=unit.get("page_number"), row_number=unit.get("row_number"),
                       duplicate_page_numbers=unit.get("duplicate_page_numbers", []),
                       duplicate_row_numbers=unit.get("duplicate_row_numbers", []),
                       section=unit.get("section", ""), unit_id=unit.get("unit_id"),
                       evidence=[unit.get("evidence", unit["text"])],
                       source_evidence=unit.get("source_evidence", unit.get("evidence", unit["text"])),
                       provenance=[provenance])
        self.stats["triples"] += len(triples)
        rep = representative_node(triples)                                   # Eq. (18)
        rid = ent_id(rep["name"], rep["type"])
        b = self.map_to_ontology(rep, triples, unit["text"])                 # M(v*)
        self._edge(b, "HAS_ENTITIES", rid, kind="onto_link", provenance=[provenance])  # Eq. (20)
        mapped_entities = set()
        for triple in triples:
            for role in ("subject", "object"):
                name, entity_type = triple[role], triple[f"{role}_type"]
                entity_id = ent_id(name, entity_type)
                if entity_id in mapped_entities:
                    continue
                mapped_entities.add(entity_id)
                concept_id = self.kt.map_entity(name, entity_type)
                self._edge(f"onto:{concept_id}", "HAS_ENTITIES", entity_id, kind="ontology_entity_link",
                           provenance=[provenance])
        self.stats["mapped"] += 1
        if unit["modality"] == "caption" and unit.get("image_url"):          # Eq. (21)
            u = unit["image_url"]
            iid = f"img:{u}"
            self.nodes.setdefault(iid, dict(id=iid, name=unit["text"][:60], kind="Image", type="Image", url=u))
            for _, typ, _ in img_edges(rep, u):
                self._edge(rid, typ, iid, kind="image_link", provenance=[provenance])

    def graph(self):
        return dict(nodes=list(self.nodes.values()), edges=self.edges)


def assemble(units, kt=None):
    a = KGAssembler(kt)
    for u in units:
        a.add_unit(u)
    return a.graph(), a.stats
