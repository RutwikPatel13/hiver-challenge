"""Central configuration for the email suggested-response system.

Every knob that matters — models, scoring weights, thresholds, paths — lives here
so the pipeline is easy to reason about and cheap to re-tune.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
DATASET_PATH = DATA_DIR / "dataset.jsonl"

for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Everything defaults to Haiku to keep the whole run well under a dollar. The
# judge is deliberately model-agnostic: point MODEL_JUDGE at Sonnet/Opus for a
# production-grade grader without touching the evaluation logic. A *different*
# model is used for the reliability cross-check so the judge never grades its
# own family's style unchallenged.
MODEL_GENERATOR = os.environ.get("MODEL_GENERATOR", "claude-haiku-4-5")
MODEL_JUDGE = os.environ.get("MODEL_JUDGE", "claude-haiku-4-5")
MODEL_JUDGE_CROSSCHECK = os.environ.get("MODEL_JUDGE_CROSSCHECK", "claude-sonnet-5")

# Which model families accept a `temperature` parameter. Newer Sonnet/Opus/Fable
# reject sampling params entirely, so llm.py sends temperature only to models
# matching these prefixes — and warns loudly if a caller *requested* a non-default
# temperature that cannot be honored (that would silently corrupt experiments
# that rely on sampling variation, e.g. the judge-reliability repeats).
TEMPERATURE_CAPABLE_PREFIXES = ("claude-haiku",)

# ---------------------------------------------------------------------------
# Evaluation rubric
# ---------------------------------------------------------------------------
# Each dimension is scored 1-5 by the judge, normalized to 0-1, then combined
# into a weighted composite. Weights encode what actually makes a support reply
# "sendable": resolving the request and committing to the right thing matter
# most; tone matters but is secondary to correctness.
DIMENSIONS: dict[str, str] = {
    "intent_resolution": "Does the reply actually address and move to resolve what the customer asked?",
    "factual_correctness": "Does it commit to the right thing (agree vs. decline, right facts/dates) — consistent with the reference reply's substance?",
    "faithfulness": "Does it avoid inventing policies, prices, promises, or facts not supported by the email or reference?",
    "completeness": "Does it cover every distinct point/question the customer raised?",
    "tone": "Is the register appropriate for the situation (e.g. empathetic on a complaint, professional throughout)?",
}

WEIGHTS: dict[str, float] = {
    "intent_resolution": 0.28,
    "factual_correctness": 0.24,
    "faithfulness": 0.20,
    "completeness": 0.16,
    "tone": 0.12,
}
# hard runtime check (not `assert` — that is stripped under python -O)
if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
    raise ValueError("evaluation WEIGHTS must sum to 1")
if set(WEIGHTS) != set(DIMENSIONS):
    raise ValueError("WEIGHTS keys must match DIMENSIONS keys")

# A reply "passes" (is sendable) only if it clears the composite bar AND trips
# no hard flag. Hard flags encode disqualifiers that an average can otherwise
# mask: a hallucinated policy or a flipped commitment makes a reply unsendable
# no matter how good the prose is.
PASS_THRESHOLD = 0.70          # composite in [0,1]
HARD_FLAGS = ["hallucination_detected", "contradicts_reference_commitment"]

# ---------------------------------------------------------------------------
# Retrieval / generation
# ---------------------------------------------------------------------------
RETRIEVAL_K = 3                # number of past tickets shown to the generator
GEN_MAX_TOKENS = 400
JUDGE_MAX_TOKENS = 700

# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------
HF_DATASET = "MohammadOthman/mo-customer-support-tweets-945k"
HF_CONFIG = "default"
HF_SPLIT = "train"
RANDOM_SEED = 13
TARGET_TOTAL = 200           # clean pairs to keep after filtering (spec max: 50-200)
MIN_ACCEPTABLE = 50          # spec minimum — fail loudly if fetching can't reach it
TEST_FRACTION = 0.30         # held-out evaluation split
