import unittest
from unittest.mock import Mock, patch

import config
from llm.extraction import OllamaLLM, get_client
from llm.refinement import TwoStageExtractor
from prompting.base_prompt import build_base_prompt


class FixedStageClient:
    kind = "test"

    def __init__(self, model):
        self.model = model
        self.messages = None

    def chat(self, messages, temperature=None):
        self.messages = messages
        return '{"triples": []}'


class OllamaClientTests(unittest.TestCase):
    def test_ollama_models_are_selected_independently_for_each_stage(self):
        with patch("llm.extraction.config.LLM_BACKEND", "ollama"), \
                patch("llm.extraction.config.STAGE1_MODEL", "initial-model"), \
                patch("llm.extraction.config.OLLAMA_STAGE2_MODEL", "refinement-model"):
            initial = get_client(1)
            refinement = get_client(2)

        self.assertEqual(initial.model, "initial-model")
        self.assertEqual(refinement.model, "refinement-model")

    def test_two_stage_extractor_calls_refinement_and_records_model_provenance(self):
        initial = FixedStageClient("initial-model")
        refinement = FixedStageClient("refinement-model")
        extractor = TwoStageExtractor(build_base_prompt(), stage1=initial, stage2=refinement)

        result = extractor.extract("Aeration rate increases froth depth.")

        self.assertIn("Initial triples:", refinement.messages[1]["content"])
        self.assertEqual(result["stage1_model"], "initial-model")
        self.assertEqual(result["stage2_model"], "refinement-model")
        self.assertEqual(result["stage1_backend"], "test")
        self.assertEqual(result["stage2_backend"], "test")

    @patch("llm.extraction.requests.post")
    def test_uses_native_chat_api_with_bounded_json_generation(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": '{"triples": []}'}}
        post.return_value = response

        client = OllamaLLM("qwen2.5:3b")
        result = client.chat([{"role": "user", "content": "Extract triples"}])

        self.assertEqual(result, '{"triples": []}')
        args, kwargs = post.call_args
        self.assertEqual(args[0], f"{config.OLLAMA_BASE_URL}/api/chat")
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:3b")
        self.assertFalse(kwargs["json"]["think"])
        self.assertEqual(kwargs["json"]["format"], "json")
        self.assertEqual(kwargs["json"]["options"]["num_predict"], config.OLLAMA_NUM_PREDICT)

    @patch("llm.extraction.requests.post")
    def test_timeout_has_actionable_error(self, post):
        import requests

        post.side_effect = requests.Timeout()
        with self.assertRaisesRegex(RuntimeError, "Use a smaller Ollama model"):
            OllamaLLM("deepseek-r1:8b").chat([])


if __name__ == "__main__":
    unittest.main()
