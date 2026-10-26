"""Stage 2 of the two-stage generation (Eq. 17, outer f_LLM): refine the initial triples with the large model.

The refinement prompt contains the ORIGINAL prompt (role, principles, examples, background) plus the stage-1
triples, and asks the model to correct labels / directions, drop unsupported triples and add missing ones.
"""
import json

from llm.extraction import InitialExtractor, get_client, parse_triples
from kg.triple_validator import TripleValidator

REFINE_SYSTEM_TAG = "[REFINE]"


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
                "and add missing ones. "
                "Return ONLY the final JSON in the same format.")
        user = (self.template.user_text(text, context) + "\n\nInitial triples:\n" +
                json.dumps({"triples": initial}, ensure_ascii=False) + "\n### end")
        raw = self.client.chat([{"role": "system", "content": sysm}, {"role": "user", "content": user}])
        refined = parse_triples(raw)
        validated, _ = self.validator.validate(refined, check_conflicts=True)
        return validated if validated or not initial else initial


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
