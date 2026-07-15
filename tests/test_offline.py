"""Offline unit tests — no API key, no network. Covers the deterministic core:
scoring math, retrieval, the lexical baseline, dataset cleaning, and aggregation.

Run:  python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import build_dataset  # noqa: E402
import config  # noqa: E402
import evaluator  # noqa: E402
from retrieval import BM25, Retriever  # noqa: E402


class TestCompositeScoring(unittest.TestCase):
    def test_bounds(self):
        dims = list(config.DIMENSIONS)
        self.assertEqual(evaluator.composite_from_scores({k: 5 for k in dims}), 1.0)
        self.assertEqual(evaluator.composite_from_scores({k: 1 for k in dims}), 0.0)
        self.assertEqual(evaluator.composite_from_scores({k: 3 for k in dims}), 0.5)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(config.WEIGHTS.values()), 1.0)

    def test_hard_flags_match_judge_implementation(self):
        self.assertEqual(set(config.HARD_FLAGS), set(evaluator._IMPLEMENTED_FLAGS))


class TestLexicalOverlap(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(evaluator.lexical_overlap("a b c d", "a b c d"), 1.0)

    def test_disjoint(self):
        self.assertEqual(evaluator.lexical_overlap("a b c", "x y z"), 0.0)

    def test_empty(self):
        self.assertEqual(evaluator.lexical_overlap("", "anything"), 0.0)

    def test_symmetry(self):
        a, b = "please refund my order now", "we will refund the order"
        self.assertAlmostEqual(
            evaluator.lexical_overlap(a, b), evaluator.lexical_overlap(b, a)
        )

    def test_case_and_punct_insensitive(self):
        self.assertEqual(evaluator.lexical_overlap("Hello, World!", "hello world"), 1.0)


class TestRetrieval(unittest.TestCase):
    KB = [
        {"id": "a", "subject": "refund request", "customer_message": "I want a refund for my broken blender"},
        {"id": "b", "subject": "delivery late", "customer_message": "my package has not arrived for two weeks"},
        {"id": "c", "subject": "login problem", "customer_message": "cannot log in to my account password reset fails"},
    ]

    def test_bm25_ranks_relevant_first(self):
        r = Retriever(self.KB)
        hits = r.retrieve("where is my package it never arrived", k=1)
        self.assertEqual(hits[0]["id"], "b")

    def test_exclude_id(self):
        r = Retriever(self.KB)
        hits = r.retrieve("refund for broken blender", k=3, exclude_id="a")
        self.assertNotIn("a", [h["id"] for h in hits])
        self.assertEqual(len(hits), 2)

    def test_empty_corpus_is_safe(self):
        self.assertEqual(Retriever([]).retrieve("anything", k=3), [])

    def test_bm25_empty_query(self):
        scores = BM25(["some document here"]).scores("")
        self.assertEqual(scores, [0.0])


class TestDatasetCleaning(unittest.TestCase):
    def test_clean_text_strips_handles_and_urls(self):
        out = build_dataset.clean_text("@AppleSupport my phone broke see https://x.co/abc please help")
        self.assertNotIn("@", out)
        self.assertNotIn("http", out)
        self.assertIn("my phone broke", out)

    def test_categorize(self):
        self.assertEqual(build_dataset.categorize("I was charged twice, need a refund"), "billing")
        self.assertEqual(build_dataset.categorize("my package never arrived"), "shipping")
        self.assertEqual(build_dataset.categorize("hello there friend"), "general")

    def test_acceptable_rejects_fragments(self):
        # too short / not a request
        self.assertFalse(build_dataset.acceptable("thanks!", "you're welcome, have a nice day friend"))
        # a real standalone request passes
        self.assertTrue(
            build_dataset.acceptable(
                "My order has not arrived and tracking shows nothing, can you help me please?",
                "Sorry about that — could you share your order number so we can investigate right away?",
            )
        )

    def test_subject_is_bounded(self):
        subj = build_dataset.make_subject("word " * 50, "general")
        self.assertLessEqual(len(subj.split()), 9)


class TestAggregation(unittest.TestCase):
    def _fake(self, composite, passed, flags=None, score=3):
        return {
            "composite": composite,
            "passed": passed,
            "scores": {k: score for k in config.DIMENSIONS},
            "flags": {f: bool(flags and f in flags) for f in config.HARD_FLAGS},
        }

    def test_aggregate_rates(self):
        results = [
            self._fake(0.9, True),
            self._fake(0.8, True),
            self._fake(0.3, False, flags=["hallucination_detected"]),
            self._fake(0.6, False),
        ]
        agg = evaluator.aggregate(results)
        self.assertEqual(agg["n"], 4)
        self.assertEqual(agg["pass_rate"], 0.5)
        self.assertEqual(agg["hard_flag_rate"]["hallucination_detected"], 0.25)
        self.assertEqual(sum(agg["composite_distribution"].values()), 4)
        self.assertAlmostEqual(agg["mean_composite"], 0.65)

    def test_aggregate_empty(self):
        self.assertEqual(evaluator.aggregate([]), {})


if __name__ == "__main__":
    unittest.main()
