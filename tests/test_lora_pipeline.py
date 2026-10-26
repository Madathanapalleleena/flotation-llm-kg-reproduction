import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm.lora.training import (
    IGNORE_INDEX,
    maximum_sequence_length,
    render_messages,
    render_training_row,
    tokenize_training_row,
)


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": [ord(character) for character in text]}


class LoraPipelineTests(unittest.TestCase):
    def test_training_format_uses_paper_role_markers_and_response_only_loss(self):
        row = {"system": "Flotation expert", "user": "Extract triples", "assistant": '{"triples": []}'}
        prefix, full_text = render_training_row(row)

        self.assertIn("[SYSTEM]\nFlotation expert", full_text)
        self.assertIn("[USER]\nExtract triples", full_text)
        self.assertTrue(full_text.startswith(prefix))
        self.assertTrue(prefix.endswith("[ASSISTANT]\n"))

        encoded = tokenize_training_row(row, CharacterTokenizer(), max_length=512)

        prefix_length = len(prefix)
        self.assertTrue(all(value == IGNORE_INDEX for value in encoded["labels"][:prefix_length]))
        self.assertTrue(all(value != IGNORE_INDEX for value in encoded["labels"][prefix_length:]))

    def test_sequence_length_uses_paper_dynamic_bound(self):
        rows = [{"system": "x", "user": "y", "assistant": "z"}]

        self.assertEqual(maximum_sequence_length(rows, CharacterTokenizer()), 128)
        self.assertEqual(maximum_sequence_length(rows, CharacterTokenizer(), maximum=100), 128)

    def test_stage_one_auto_selects_trained_adapter_when_present(self):
        from llm.extraction import get_client

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "adapter_config.json").write_text("{}", encoding="utf-8")
            with patch("llm.extraction.config.STAGE1_BACKEND", "auto"), \
                    patch("llm.extraction.config.LORA_ADAPTER_DIR", Path(directory)), \
                    patch("llm.extraction.config.LLM_BACKEND", "ollama"), \
                    patch("llm.lora.model.LoRALocalLLM") as adapter:
                get_client(1)

        adapter.assert_called_once_with()

    def test_stage_two_keeps_independent_large_model_configuration(self):
        from llm.extraction import get_client

        with patch("llm.extraction.config.STAGE1_BACKEND", "default"), \
                patch("llm.extraction.config.LLM_BACKEND", "ollama"), \
                patch("llm.extraction.config.STAGE1_MODEL", "stage-one"), \
                patch("llm.extraction.config.OLLAMA_STAGE2_MODEL", "larger-stage-two"):
            self.assertEqual(get_client(1).model, "stage-one")
            self.assertEqual(get_client(2).model, "larger-stage-two")


if __name__ == "__main__":
    unittest.main()
