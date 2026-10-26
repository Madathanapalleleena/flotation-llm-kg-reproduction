"""Sanity tests tying the code to numbers printed in the paper.  Run:  python -m unittest tests.test_formulas -v"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from rag.fuzzy_membership import fuzzy_entropy, gaussian_membership, h_min_fuzzy, m_fuzzy
from rag.term_weighting import TermWeighter, tokenize
from evaluation.extraction_metrics import prf, triplet_similarity
from evaluation.semantic_metrics import final_score, semantic_assessment


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

    def test_eq11_bounds(self):
        w = TermWeighter(["pulp concentration affects recovery rate", "xanthate is a collector", "froth depth matters"] * 5)
        self.assertGreaterEqual(m_fuzzy("pulp concentration", "pulp concentration affects recovery rate", w), 1.0)
        self.assertEqual(m_fuzzy("zzz", "pulp", w), 1.0)

    def test_eq24_balance(self):
        n_down = n_up = 14
        self.assertAlmostEqual(1 - abs((n_down - n_up) / (n_down + n_up + 1)), 1.0)

    def test_eq29_constant_scores(self):
        units = [dict(text="x", triples=[{"a": 1}] * (1 + i % 3), unit_id=i) for i in range(30)]
        r = semantic_assessment(units, FakeJudge(0.9), full=True)
        self.assertAlmostEqual(r["S2_raw"], 0.9)                    # std = 0
        self.assertEqual(config.FIBONACCI, [1, 2, 3, 5, 8, 13])

    def test_dynamic_sampling_rate(self):
        units = [dict(text="x", triples=[{"a": 1}] * (1 + i % 5), unit_id=i) for i in range(250)]
        r = semantic_assessment(units, FakeJudge(0.8))
        self.assertEqual(r["n_sampled"], 20 + 20 + 10)              # batches of 100,100,50 at rho = 0.2


if __name__ == "__main__":
    unittest.main()
