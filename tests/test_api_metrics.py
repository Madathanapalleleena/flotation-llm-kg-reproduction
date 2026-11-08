import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.api import app


class ApiMetricsTests(unittest.TestCase):
    def test_brief_metrics_keep_all_ui_values_and_only_required_structural_detail(self):
        metrics = {
            "evaluation_scope": "sampled 5 of 20 held-out examples",
            "extraction": {
                "Proposed_pipeline": {
                    "precision": 0.8, "recall": 0.7, "f1": 0.75, "c_sis": 0.9,
                    "per_sample": {"1": {"F1": 0.75}},
                },
            },
            "structural": {
                "I_s": 0.1, "C_s": 0.5, "B_s": 0.8, "Z_s": 0.6, "P_s": 1.0, "R_s": 0.7, "S1": 0.72,
                "detail": {"L_max": 4, "n_process_loops": 12},
            },
            "semantic": {"S2": 0.68, "samples": [{"score": 0.68}, {"score": 0.95}]},
            "S_final": 0.696,
        }
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with patch("backend.api.config.METRICS_JSON", metrics_path):
                response = app.test_client().get("/api/metrics?brief=1")

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(
            result["extraction"]["Proposed_pipeline"],
            {key: value for key, value in metrics["extraction"]["Proposed_pipeline"].items() if key != "per_sample"},
        )
        self.assertEqual(result["structural"]["detail"], {"L_max": 4})
        self.assertEqual(result["semantic"]["S2"], 0.68)
        self.assertNotIn("samples", result["semantic"])
        self.assertNotIn("per_sample", result["extraction"]["Proposed_pipeline"])
        self.assertEqual(
            result["semantic"]["score_distribution"],
            [
                {"range": "0.0-0.2", "count": 0},
                {"range": "0.2-0.4", "count": 0},
                {"range": "0.4-0.6", "count": 0},
                {"range": "0.6-0.8", "count": 1},
                {"range": "0.8-1.0", "count": 1},
            ],
        )
        self.assertEqual(result["S_final"], 0.696)


if __name__ == "__main__":
    unittest.main()
