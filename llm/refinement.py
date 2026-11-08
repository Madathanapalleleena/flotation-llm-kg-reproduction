"""Stage 2 of the two-stage generation (Eq. 17, outer f_LLM): refine the initial triples with the large model.

The refinement prompt contains the ORIGINAL prompt (role, principles, examples, background) plus the stage-1
triples, and asks the model to correct labels / directions, drop unsupported triples and add missing ones.
"""
import json
import re

from llm.extraction import InitialExtractor, get_client, parse_triples
from kg.triple_validator import TripleValidator

REFINE_SYSTEM_TAG = "[REFINE]"

_RELATION_EVIDENCE = {
    "affects": (
        r"\b(?:affect\w*|influenc\w*|impact\w*|play\w*\s+(?:an?\s+)?"
        r"(?:important\s+|significant\s+|key\s+|critical\s+)?role\s+in|depend\w*\s+on)\b"
    ),
    "increases": r"\b(?:increas\w*|rais\w*|improv\w*|enhanc\w*|boost\w*|higher)\b",
    "decreases": r"\b(?:decreas\w*|reduc\w*|lower\w*|declin\w*|suppress\w*)\b",
    "keeps": r"\b(?:keep\w*|maintain\w*|hold\w*|remain\w*|stable)\b",
    "causes": r"\b(?:caus\w*|result\w*\s+in|lead\w*\s+to|induc\w*)\b",
    "uses": r"\b(?:use|uses|used|using|employ\w*|add\w*|dos\w*|utili[sz]\w*)\b",
    "produces": r"\b(?:produc\w*|yield\w*|generat\w*|obtain\w*)\b",
    "follows": r"\b(?:follow\w*|preced\w*|after|before|subsequent\w*|flow\w*\s+into)\b",
    "has_property": r"\b(?:has|have|is|are|was|were|with|contain\w*|consist\w*|located)\b",
}


def source_grounded_triples(triples, text):
    """Keep triples whose entities and relation have textual evidence in their source sentence."""
    normalized_text = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE)
    grounded = []
    for triple in triples:
        entities = (triple.get("subject", ""), triple.get("object", ""))
        normalized_entities = [
            re.sub(r"[^\w]+", " ", str(entity).casefold(), flags=re.UNICODE).strip()
            for entity in entities
        ]
        relation_pattern = _RELATION_EVIDENCE.get(str(triple.get("relation", "")).casefold())
        if not relation_pattern or not all(normalized_entities):
            continue
        subject_spans = [
            match.span()
            for match in re.finditer(
                rf"(?<!\w){re.escape(normalized_entities[0])}(?!\w)",
                normalized_text,
                flags=re.UNICODE,
            )
        ]
        object_spans = [
            match.span()
            for match in re.finditer(
                rf"(?<!\w){re.escape(normalized_entities[1])}(?!\w)",
                normalized_text,
                flags=re.UNICODE,
            )
        ]
        relation_spans = [
            match.span()
            for match in re.finditer(relation_pattern, normalized_text, flags=re.UNICODE)
        ]
        if any(
            subject_end <= relation_start <= relation_end <= object_start
            and object_end - subject_start <= 300
            for subject_start, subject_end in subject_spans
            for relation_start, relation_end in relation_spans
            for object_start, object_end in object_spans
        ):
            grounded.append(triple)
    return grounded


class Refiner:
    def __init__(self, template, client=None):
        self.template = template
        self.client = client or get_client(2)
        self.validator = TripleValidator()

    def refine(self, text: str, context: str, initial: list):
        initial, _ = self.validator.validate(initial, check_conflicts=True)
        sysm = (f"{REFINE_SYSTEM_TAG} {self.template.system_text()}\n\n"
                "### Refinement task\nYou are given initial triples produced by a lightweight model. Check each one against the "
                "pending text and the flotation ontology: fix wrong ontology labels or relation types, correct relation "
                "direction, delete triples that are not supported by the text, remove duplicates and contradictions, "
                "and add missing ones. The pending text is the only evidence: never add facts from background knowledge "
                "or the examples. Each subject and object must be explicitly named in the pending text. "
                "Return ONLY the final JSON in the same format.")
        user = (self.template.user_text(text, context) + "\n\nInitial triples:\n" +
                json.dumps({"triples": initial}, ensure_ascii=False) + "\n### end")
        raw = self.client.chat([{"role": "system", "content": sysm}, {"role": "user", "content": user}])
        refined = parse_triples(raw)
        validated, _ = self.validator.validate(refined, check_conflicts=True)
        result = validated if validated or not initial else initial
        return source_grounded_triples(result, text)


class TwoStageExtractor:
    """T_m = f_LLM(x_m, f_Fine(x_m, C_RAG))"""

    def __init__(self, template, retriever=None, stage1=None, stage2=None):
        self.stage1 = InitialExtractor(template, retriever, stage1)
        self.stage2 = Refiner(template, stage2)

    def extract(self, text: str):
        initial, ctx = self.stage1.extract(text)
        final = self.stage2.refine(text, ctx, initial)
        stage1_client = self.stage1.client
        stage2_client = self.stage2.client

        def model_name(client):
            for attribute in ("model_name", "name", "model"):
                value = getattr(client, attribute, None)
                if isinstance(value, str):
                    return value
            return type(client).__name__

        return dict(
            text=text,
            context=ctx,
            initial=initial,
            triples=final,
            stage1_backend=getattr(stage1_client, "kind", type(stage1_client).__name__),
            stage1_model=model_name(stage1_client),
            stage2_backend=getattr(stage2_client, "kind", type(stage2_client).__name__),
            stage2_model=model_name(stage2_client),
        )
