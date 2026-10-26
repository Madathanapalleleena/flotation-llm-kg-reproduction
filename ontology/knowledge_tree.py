"""Flotation knowledge tree  T = (V_T, E_T, A_T)   -- paper Eq. (4), (5), Fig. 5.

V_T = {V_root, V_stage, V_leaf}   nodes at three levels
E_T = {(v_i, v_j) | v_i, v_j in V, i != j}   bidirectional (…_UP / …_DOWN) edges
A_T = {A_root, A_stage, A_leaf}   attributes of each level
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class OntNode:
    id: str
    label: str
    level: str          # root | stage | leaf
    cls: str            # e.g. RoughingCategory
    aliases: list = field(default_factory=list)
    attributes: list = field(default_factory=list)


class KnowledgeTree:
    def __init__(self, path: Path = config.ONTOLOGY_JSON):
        self.raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.nodes = {n["id"]: OntNode(**n) for n in self.raw["nodes"]}
        self.edges = [(e["source"], e["type"], e["target"]) for e in self.raw["edges"]]
        self.entity_types = [t["name"] for t in self.raw["entity_types"]]
        self.relation_types = [t["name"] for t in self.raw["relation_types"]]
        self.entity_type_desc = {t["name"]: t["description"] for t in self.raw["entity_types"]}
        self.relation_type_desc = {t["name"]: t["description"] for t in self.raw["relation_types"]}
        self.relation_constraints = self.raw.get("relation_constraints", {})
        self.graph_schema = self.raw.get("graph_schema", {})
        self.core_stages = self.raw["core_stages"]
        self.standard_path = self.raw["standard_path"]
        self.type_to_node = self.raw["type_to_node"]

    # ---- the three node sets of Eq. (4)
    @property
    def V_root(self): return [n for n in self.nodes.values() if n.level == "root"]
    @property
    def V_stage(self): return [n for n in self.nodes.values() if n.level == "stage"]
    @property
    def V_leaf(self): return [n for n in self.nodes.values() if n.level == "leaf"]
    @property
    def A_T(self): return self.raw["attributes"]

    @property
    def L_optimal(self):
        """Length (in edges) of the standard raw-ore -> concentrate path."""
        return len(self.standard_path) - 1

    def alias_lexicon(self):
        """alias -> ontology node id (longest first)."""
        lex = {}
        for n in self.nodes.values():
            for a in n.aliases + [n.label]:
                lex[a.lower()] = n.id
        return dict(sorted(lex.items(), key=lambda kv: -len(kv[0])))

    def match_node(self, text: str):
        """Score ontology nodes against a text by alias hits; returns (node_id, hits)."""
        t = text.lower()
        best, best_hits = None, 0
        for n in self.nodes.values():
            hits = sum(1 for a in set(n.aliases + [n.label.lower()]) if a.lower() in t)
            if hits > best_hits:
                best, best_hits = n.id, hits
        return best, best_hits


if __name__ == "__main__":
    kt = KnowledgeTree()
    print(f"{len(kt.nodes)} nodes, {len(kt.edges)} edges, L_optimal={kt.L_optimal}")
    for lvl, ns in (("root", kt.V_root), ("stage", kt.V_stage), ("leaf", kt.V_leaf)):
        print(lvl, [n.cls for n in ns])
