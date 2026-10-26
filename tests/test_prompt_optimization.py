import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompting.prompt_optimizer import optimize


class PromptOptimizationTests(unittest.TestCase):
    def test_optimizer_applies_feedback_and_records_prompt_versions(self):
        row = {
            "id": "sample-1",
            "user": "Aeration rate increases froth depth.",
            "gold": [{
                "subject": "aeration rate",
                "subject_type": "Parameter",
                "relation": "increases",
                "object": "froth depth",
                "object_type": "Parameter",
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch("prompting.prompt_optimizer.config.PROMPT_DIR", Path(directory)), \
                    patch("prompting.prompt_optimizer.config.OPT_MIN_ITERS", 1), \
                    patch("prompting.prompt_optimizer.config.OPT_STOP_F1", 0.9), \
                    patch("prompting.prompt_optimizer.config.OPT_STOP_CSIS", 0.8), \
                    patch("prompting.prompt_optimizer.run_extraction",
                          side_effect=[{"sample-1": []}, {"sample-1": row["gold"]}]) as run:
                best, history = optimize(rows=[row], max_iters=1, verbose=False)

            self.assertEqual([item["version"] for item in history], [0, 1])
            self.assertEqual(best.version, 1)
            saved = json.loads((Path(directory) / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_version"], 1)
            self.assertEqual(len(saved["history"]), 2)
            artifact = json.loads(next((Path(directory) / "optimization_runs").rglob("iteration_*.json"))
                                  .read_text(encoding="utf-8"))
            self.assertEqual(artifact["prompt"]["version"], 0)
            self.assertEqual(artifact["examples"][0]["generated"], [])
            self.assertEqual(artifact["updated_prompt"]["version"], 1)
            self.assertEqual([call.args[0].version for call in run.call_args_list], [0, 1])


if __name__ == "__main__":
    unittest.main()
