"""Initial prompt generation -- paper Sec. III-B-1(a), Eq. (6), Fig. 6.

    P0 = {R0, C0, E0, O0, T0}
    R0 role definition | C0 processing principles | E0 / O0 few-shot input & output | T0 text to process
The template is constrained by the entity set V_T and the relationship set E_T of the flotation knowledge tree.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ontology.knowledge_tree import KnowledgeTree

# --- rule texts appended by the optimiser (Sec. III-B-1(c)); the offline MockLLM keys on these strings
RULE_DECOMPOSE = ("Split compound or complex sentences into simple clauses and extract the triples of every clause separately.")
RULE_CONSISTENCY = ("Keep triples logically consistent: never output contradictory relations for the same entity pair "
                    "and keep the direction of every relation grammatically correct (subject acts on object).")
RULE_COVERAGE = ("Extract every entity and every implicit or causal relationship; do not skip causes, ordering "
                 "(follows) or property (has_property) relations.")

ROLE = ("You are a Froth Flotation Domain Knowledge Engineer responsible for accurately extracting ternary "
        "information from various flotation process statements.")

DEFAULT_EXAMPLE = dict(
    input="Increasing the aeration rate increases the froth depth, and the cleaning stage uses MIBC.",
    output={"triples": [
        {"subject": "aeration rate", "subject_type": "Parameter", "relation": "increases",
         "object": "froth depth", "object_type": "Parameter"},
        {"subject": "cleaning", "subject_type": "ProcessStage", "relation": "uses",
         "object": "MIBC", "object_type": "Reagent"}]})


@dataclass
class PromptTemplate:
    role: str = ROLE
    principles: list = field(default_factory=list)       # C0
    examples: list = field(default_factory=lambda: [DEFAULT_EXAMPLE])   # (E0, O0)
    extra_rules: list = field(default_factory=list)      # added by the optimiser
    version: int = 0

    # ------------------------------------------------------------------ render
    def system_text(self):
        rules = self.principles + self.extra_rules
        return self.role + "\n\n### Processing principles\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))

    def user_text(self, question: str, context: str = ""):
        ex = "\n\n".join(f"Input example: {e['input']}\nSample output: {json.dumps(e['output'], ensure_ascii=False)}"
                         for e in self.examples)
        return (f"### Examples\n{ex}\n\n### Background knowledge\n{context or '(none)'}\n\n"
                f"### Pending text\n{question}")

    def messages(self, question: str, context: str = ""):
        return [{"role": "system", "content": self.system_text()},
                {"role": "user", "content": self.user_text(question, context)}]

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def clone(self, **kw):
        d = self.to_dict()
        d["principles"], d["examples"], d["extra_rules"] = list(d["principles"]), list(d["examples"]), list(d["extra_rules"])
        d.update(kw)
        return PromptTemplate(**d)


def build_base_prompt(kt: KnowledgeTree = None) -> PromptTemplate:
    kt = kt or KnowledgeTree()
    ents = "; ".join(f"{n} ({kt.entity_type_desc[n]})" for n in kt.entity_types)
    rels = "; ".join(f"{n} ({kt.relation_type_desc[n]})" for n in kt.relation_types)
    principles = [
        "Only extract knowledge related to the flotation process; if the input statement is not related to the "
        "flotation process or does not contain enough content to form a triple, return {\"triples\": []}.",
        f"Ontology label is assigned according to the flotation knowledge tree model. Noun-phrase entities are extracted and "
        f"categorised into the following {len(kt.entity_types)} types: {ents}.",
        f"Create inter-entity relationships categorised into the following {len(kt.relation_types)} types: {rels}.",
        "Triples must be logically clear and well connected.",
        'Strictly follow the JSON format of the sample output ({"triples": [{"subject","subject_type","relation","object","object_type"}]}); '
        "do not change the format and do not add any other characters.",
    ]
    return PromptTemplate(principles=principles)
