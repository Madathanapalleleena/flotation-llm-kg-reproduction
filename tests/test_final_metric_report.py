import contextlib
import io
import unittest

from main import _print_final_metrics


class FinalMetricReportTests(unittest.TestCase):
    def test_prints_all_requested_metrics_and_distinguishes_graph_consistency(self):
        metrics = {
            "evaluation_scope": "sampled 5 of 20 held-out examples",
            "extraction": {
                "Proposed_pipeline": {
                    "precision": 0.81,
                    "recall": 0.72,
                    "f1": 0.76,
                    "c_sis": 0.95,
                },
            },
            "structural": {
                "I_s": 0.1,
                "C_s": 0.75,
                "B_s": 0.8,
                "Z_s": 0.6,
                "R_s": 0.7,
                "S1": 0.73,
                "detail": {"L_max": 3},
            },
            "semantic": {"S2": 0.68},
            "S_final": 0.70,
            "graph_consistency": {"consistency": 0.99, "n_issues": 1},
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_final_metrics(metrics)

        report = output.getvalue()
        for label in (
            "Precision: 0.8100",
            "Recall: 0.7200",
            "F1-score: 0.7600",
            "Consistency score (C_sis): 0.9500",
            "Isolated node ratio (I_s): 0.1000",
            "Process coverage rate (C_s): 0.7500",
            "Bidirectional relationship balance (B_s): 0.8000",
            "Process node knowledge coverage (Z_s): 0.6000",
            "Longest process path length (L_max): 3",
            "Process loop rationality (R_s): 0.7000",
            "Structural score (S1): 0.7300",
            "Semantic score (S2): 0.6800",
            "Final knowledge graph quality score (S_final): 0.7000",
            "Graph consistency check: 0.9900 (1 issues)",
            "Evaluation scope: sampled 5 of 20 held-out examples",
        ):
            with self.subTest(label=label):
                self.assertIn(label, report)


if __name__ == "__main__":
    unittest.main()
