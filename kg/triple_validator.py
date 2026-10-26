"""Triple validation (Step 13 input) -- schema check, normalisation, de-duplication, conflict detection."""
import re

from ontology.knowledge_tree import KnowledgeTree

_REL_SYN = {
    "increase": "increases", "raises": "increases", "raise": "increases", "improves": "increases", "improve": "increases",
    "enhances": "increases", "decrease": "decreases", "reduces": "decreases", "reduce": "decreases", "lowers": "decreases",
    "keep": "keeps", "maintains": "keeps", "maintain": "keeps", "affect": "affects", "influences": "affects",
    "cause": "causes", "leads_to": "causes", "results_in": "causes", "use": "uses", "adds": "uses",
    "produce": "produces", "yields": "produces", "follow": "follows", "has": "has_property", "have": "has_property",
}
CONTRA = [{"increases", "decreases"}, {"keeps", "increases"}, {"keeps", "decreases"}]


def _norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _norm_rel(r):
    r = re.sub(r"[\s\-]+", "_", _norm_text(r).lower())
    return _REL_SYN.get(r, r)


class TripleValidator:
    def __init__(self, kt: KnowledgeTree = None):
        self.kt = kt or KnowledgeTree()
        self.types = {t.lower(): t for t in self.kt.entity_types}
        self.types.update({"process stage": "ProcessStage", "stage": "ProcessStage", "raw ore": "RawOre", "ore": "RawOre"})
        self.rels = set(self.kt.relation_types)
        for relation, constraints in self.kt.relation_constraints.items():
            if relation not in self.rels:
                raise ValueError(f"Ontology constrains unknown relation {relation!r}")
            for field in ("subject_types", "object_types"):
                unknown = set(constraints.get(field, [])) - set(self.kt.entity_types)
                if unknown:
                    raise ValueError(f"Ontology relation {relation!r} has unknown {field}: {sorted(unknown)}")

    def normalize(self, t: dict):
        return dict(subject=_norm_text(t.get("subject")), subject_type=self.types.get(_norm_text(t.get("subject_type")).lower(),
                    _norm_text(t.get("subject_type"))), relation=_norm_rel(t.get("relation")),
                    object=_norm_text(t.get("object")), object_type=self.types.get(_norm_text(t.get("object_type")).lower(),
                    _norm_text(t.get("object_type"))))

    def check(self, t: dict):
        """Return reason string if invalid, else ''."""
        if not t["subject"] or not t["object"]:
            return "empty entity"
        if t["subject"].lower() == t["object"].lower():
            return "self loop"
        if t["relation"] not in self.rels:
            return f"unknown relation '{t['relation']}'"
        if t["subject_type"] not in self.kt.entity_types or t["object_type"] not in self.kt.entity_types:
            return "unknown entity type"
        constraints = self.kt.relation_constraints.get(t["relation"], {})
        if constraints.get("subject_types") and t["subject_type"] not in constraints["subject_types"]:
            return f"subject type '{t['subject_type']}' is not allowed for relation '{t['relation']}'"
        if constraints.get("object_types") and t["object_type"] not in constraints["object_types"]:
            return f"object type '{t['object_type']}' is not allowed for relation '{t['relation']}'"
        return ""

    def validate(self, triples, check_conflicts=False):
        ok, rejected, seen = [], [], set()
        for raw in triples or []:
            if not isinstance(raw, dict):
                rejected.append((raw, "not an object"))
                continue
            t = self.normalize(raw)
            why = self.check(t)
            key = (t["subject"].lower(), t["relation"], t["object"].lower())
            if why:
                rejected.append((t, why))
            elif key in seen:
                rejected.append((t, "duplicate"))
            else:
                seen.add(key)
                ok.append(t)
        if check_conflicts:
            conflict_free = []
            for triple in ok:
                conflicts = count_conflicts(conflict_free + [triple])
                if conflicts > count_conflicts(conflict_free):
                    rejected.append((triple, "contradicts an existing triple"))
                else:
                    conflict_free.append(triple)
            ok = conflict_free
        return ok, rejected


def count_conflicts(triples) -> int:
    """N_conflict: pairs of triples over the same entity pair whose relations are logically contradictory,
    plus pairs asserting `follows` in both directions."""
    idx = {}
    for t in triples:
        idx.setdefault((t["subject"].lower(), t["object"].lower()), set()).add(t["relation"])
    n = 0
    for (s, o), rels in idx.items():
        for c in CONTRA:
            if c <= rels:
                n += 1
        if "follows" in rels and "follows" in idx.get((o, s), set()) and s < o:
            n += 1
    return n
