"""Sanity tests tying the code to numbers printed in the paper.  Run:  python -m unittest tests.test_formulas -v"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from rag.fuzzy_membership import fuzzy_entropy, gaussian_membership, h_min_fuzzy, m_fuzzy, sim_fuzzy
from rag.term_weighting import TermWeighter, tokenize
from evaluation.extraction_metrics import prf, triplet_similarity
from evaluation.semantic_metrics import final_score, semantic_assessment
from rag.vector_store import storage_objective
from unittest.mock import patch


class FakeJudge:
    kind = "fake"
    def __init__(self, s): self.s = s
    def chat(self, messages, temperature=None): return '{"score": %s, "reason": "x"}' % self.s


class Formulas(unittest.TestCase):
    def test_eq8_gaussian_matches_paper_examples(self):
        self.assertAlmostEqual(gaussian_membership(0.8), 1.0)               # "slurry (1.0)"
        self.assertAlmostEqual(gaussian_membership(0.9), 0.8825, places=3)  # "bubble (0.882)"

    def test_eq7_weights_and_threshold(self):
        self.assertAlmostEqual(config.W_SUBJECT + config.W_OBJECT + config.W_RELATION + config.W_LABEL, 1.0)
        self.assertEqual(config.MATCH_THRESHOLD, 0.85)
        t = dict(subject="a b", subject_type="Parameter", relation="increases", object="c d", object_type="Product")
        self.assertAlmostEqual(triplet_similarity(t, t), 1.0)
        p, r, f = prf(3, 4, 6)
        self.assertAlmostEqual(f, 2 * p * r / (p + r))

    def test_eq28_reproduces_paper_S1(self):
        w = config.S1_WEIGHTS                       # paper values: Is=0, Cs=1, Bs=1, Zs=.9996, Ps=1, Rs=.5488
        s1 = w["I"] * 1 + w["C"] * 1 + w["B"] * 1 + w["Z"] * 0.9996 + w["P"] * 1 + w["R"] * 0.5488
        self.assertAlmostEqual(s1, 0.9774, places=4)

    def test_eq30_reproduces_paper_final(self):
        self.assertAlmostEqual(final_score(0.9774, 0.8809), 0.9195, places=4)

    def test_threshold_fuzzy(self):
        self.assertAlmostEqual(h_min_fuzzy(config.H_MIN), config.H_MIN * 1.2)      # exp(0) = 1
        self.assertLess(h_min_fuzzy(config.H_MIN + 5), config.H_MIN * 1.0001 + 0.001)  # far away -> ~H_min

    def test_eq9_reduces_to_shannon_when_membership_zero(self):
        class W:                                   # no terms; mu_h second branch ~ tiny for p far from 0.01
            terms = set()
            def membership(self, w): return 0.0
        toks = ["a", "b", "c", "d"]
        h = fuzzy_entropy(toks, W())
        shannon = 2.0
        mu = math.exp(-((0.25 - config.MUH_CENTER) ** 2) / (2 * config.MUH_SIGMA ** 2))
        self.assertAlmostEqual(h, shannon * (1 + config.ENTROPY_MU_COEF * 4 * mu / 4), places=6)

    @patch("rag.fuzzy_membership.mu_h", return_value=0.5)
    def test_eq9_counts_membership_for_each_token_occurrence(self, _):
        class W:
            terms = {"x", "y"}
        base = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
        self.assertAlmostEqual(fuzzy_entropy(["x", "x", "y"], W()), base * 1.15)

    def test_eq11_bounds(self):
        w = TermWeighter(["pulp concentration affects recovery rate", "xanthate is a collector", "froth depth matters"] * 5)
        self.assertGreaterEqual(m_fuzzy("pulp concentration", "pulp concentration affects recovery rate", w), 1.0)
        self.assertEqual(m_fuzzy("zzz", "pulp", w), 1.0)

    def test_eq11_applies_unscaled_cosine_and_eq8_membership(self):
        class W:
            terms = {"aeration"}
            def membership(self, _):
                return 0.75

        multiplier = m_fuzzy("aeration rate", "aeration", W())
        self.assertAlmostEqual(multiplier, 1.0 + (0.75 + 0.5) * 2.0)
        self.assertAlmostEqual(sim_fuzzy([1.0, 0.0], [0.8, 0.6], "aeration rate", "aeration", W()),
                               0.8 * multiplier)

    def test_eq12_divides_both_objective_terms_by_total_vector_count(self):
        result = storage_objective([0.0, 0.5], index_size_kb=10.0, n=4, lam=0.1)
        self.assertAlmostEqual(result, (1.0 + 0.25) / 4 + (0.1 * 10.0) / 4)

    def test_eq24_balance(self):
        n_down = n_up = 14
        self.assertAlmostEqual(1 - abs((n_down - n_up) / (n_down + n_up + 1)), 1.0)

    def test_eq29_constant_scores(self):
        units = [dict(text="x", triples=[{"a": 1}] * (1 + i % 3), unit_id=i) for i in range(30)]
        r = semantic_assessment(units, FakeJudge(0.9), full=True)
        self.assertAlmostEqual(r["S2_raw"], 0.9)                    # std = 0
        self.assertEqual(config.FIBONACCI, [1, 2, 3, 5, 8, 13])

    def test_eq29_is_reported_without_a_non_paper_clamp(self):
        class TwoScores:
            kind = "fake"
            def __init__(self):
                self.scores = iter([1.0, 0.0])
            def chat(self, messages, temperature=None):
                return '{"score": %s, "reason": "x"}' % next(self.scores)

        with patch("evaluation.semantic_metrics.config.SIGMA_S", 1.0):
            result = semantic_assessment(
                [dict(text="x", triples=[], unit_id=1), dict(text="y", triples=[], unit_id=2)],
                TwoScores(),
                full=True,
            )
        self.assertGreater(result["S2"], 1.0)
        self.assertEqual(result["S2"], result["S2_raw"])

    def test_dynamic_sampling_rate(self):
        units = [dict(text="x", triples=[{"a": 1}] * (1 + i % 5), unit_id=i) for i in range(250)]
        r = semantic_assessment(units, FakeJudge(0.8))
        self.assertEqual(r["n_sampled"], 20 + 20 + 10)              # batches of 100,100,50 at rho = 0.2


if __name__ == "__main__":
    unittest.main()
