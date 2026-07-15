"""The accuracy / evaluation system — the core of this project.

WHAT "ACCURATE" MEANS HERE
--------------------------
A suggested reply is accurate if a support agent could send it as-is: it resolves
the customer's request, commits to the right thing, matches the expected tone, and
invents nothing. That is deliberately NOT "matches the reply that was actually
sent" — exact/overlap match punishes correct answers for using different words,
and the sent reply is only one of many valid answers.

WHY A MULTI-DIMENSIONAL RUBRIC JUDGE
------------------------------------
"Sendable" decomposes into distinct failure modes that a single number blurs
together (see config.DIMENSIONS). We score each with an LLM-as-judge against an
explicit rubric, normalize, and combine into a weighted composite. Two HARD FLAGS
sit outside the average: a hallucinated specific or a flipped commitment makes a
reply unsendable regardless of how good the prose is — an average must not be able
to launder those away.

THE REFERENCE IS A REFERENCE, NOT A TEMPLATE
--------------------------------------------
We hand the judge the actually-sent reply, but instruct it to use it only to check
*substance* (did the candidate commit consistently, not contradict facts) and to
NOT penalize different wording or a different-but-valid resolution.

`lexical_overlap` is the naive surface-similarity baseline we contrast the judge
against in validation — it exists to be beaten, demonstrating why surface metrics
are insufficient.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import llm  # noqa: E402

# ---------------------------------------------------------------------------
# Naive baseline metric: ROUGE-L F1 (longest-common-subsequence overlap).
# Tokenization is shared with the BM25 retriever so the two never drift.
# ---------------------------------------------------------------------------
from retrieval import _tok as _tokens  # noqa: E402


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def lexical_overlap(candidate: str, reference: str) -> float:
    """ROUGE-L F1 in [0,1] between candidate and reference reply."""
    a, b = _tokens(candidate), _tokens(reference)
    if not a or not b:
        return 0.0
    l = _lcs(a, b)
    prec, rec = l / len(a), l / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# The rubric judge
# ---------------------------------------------------------------------------
_DIM_KEYS = list(config.DIMENSIONS.keys())

# The judge's prompt and schema implement exactly these hard flags. config.HARD_FLAGS
# must match — fail loudly at import rather than KeyError mid-run (or, worse, a new
# config flag being silently ignored by pass/fail logic).
_IMPLEMENTED_FLAGS = ("hallucination_detected", "contradicts_reference_commitment")
if set(config.HARD_FLAGS) != set(_IMPLEMENTED_FLAGS):
    raise RuntimeError(
        f"config.HARD_FLAGS {config.HARD_FLAGS} != flags implemented by the judge "
        f"{list(_IMPLEMENTED_FLAGS)} — update the judge prompt/schema in evaluator.py "
        f"together with config.HARD_FLAGS"
    )

JUDGE_SYSTEM = f"""You are a meticulous evaluator of customer-support email replies.
You are given the incoming email, a CANDIDATE reply to grade, and a REFERENCE
reply that a human agent actually sent.

Grade the CANDIDATE on each dimension using a 1-5 scale:
  5 = fully satisfies this dimension
  4 = minor shortfall
  3 = partially satisfies / notable gap
  2 = largely fails
  1 = fails entirely
Give a one-sentence reason for each score.

Dimensions:
{chr(10).join(f'- {k}: {v}' for k, v in config.DIMENSIONS.items())}

How to use the REFERENCE reply:
- It is ONE acceptable answer a human happened to send, NOT a gold template. A good
  reply may use different wording, a different structure, or a different but equally
  valid approach (e.g. troubleshooting vs. routing to the right team).
- Judge intent_resolution, completeness, tone, and faithfulness against the INCOMING
  EMAIL's needs — do NOT lower these just because the candidate's approach differs
  from the reference's.
- Use the reference ONLY for factual_correctness and the contradiction flag: does the
  candidate stay consistent with the facts and the commitment the reference reveals?

Two hard flags (set independently of the 1-5 scores):
- hallucination_detected: the candidate states a specific fact not supported by the
  email or reference — an invented order number, refund amount, date, or named policy,
  or promising something that was never offered. Generic empathy, general guidance, or
  asking for missing information is NOT hallucination. When unsure, set FALSE.
- contradicts_reference_commitment: set TRUE ONLY for a genuine polarity reversal on
  the customer's core yes/no decision — e.g. the reference GRANTS a refund but the
  candidate REFUSES it, or the reference says "yes, we can do X" and the candidate says
  "no, we cannot." A different approach, a different next step, asking for more detail,
  routing to another channel, or being more/less specific is NOT a contradiction — set
  FALSE. When unsure, set FALSE.

Be strict and specific. Return only the structured object."""

_SCORE_PROP = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: _SCORE_PROP for k in _DIM_KEYS},
        "hallucination_detected": {"type": "boolean"},
        "contradicts_reference_commitment": {"type": "boolean"},
        "overall_note": {"type": "string"},
    },
    "required": _DIM_KEYS + ["hallucination_detected", "contradicts_reference_commitment", "overall_note"],
    "additionalProperties": False,
}


def _normalize(score: int) -> float:
    return (score - 1) / 4.0


def composite_from_scores(scores: dict[str, int]) -> float:
    return round(sum(config.WEIGHTS[k] * _normalize(scores[k]) for k in _DIM_KEYS), 4)


def judge(email_text: str, candidate: str, reference: str | None, *,
          model: str | None = None, temperature: float = 0.0, salt: str = "") -> dict:
    """Score one candidate reply. Returns per-dimension scores, hard flags,
    composite, and a pass/fail verdict.

    `reference` may be None/empty for a brand-new email with no human answer — the
    judge then grades against the email alone (reference-free mode), scoring
    factual_correctness on internal consistency and leaving the contradiction flag
    off (there is nothing to contradict)."""
    if reference:
        ref_block = f"=== REFERENCE REPLY (a human-sent answer) ===\n{reference}"
    else:
        ref_block = (
            "=== REFERENCE REPLY ===\n(none available — judge the candidate against "
            "the incoming email alone; base factual_correctness on internal "
            "consistency and set contradicts_reference_commitment to false)"
        )
    user = (
        f"=== INCOMING EMAIL ===\n{email_text}\n\n"
        f"=== CANDIDATE REPLY (grade this) ===\n{candidate}\n\n"
        f"{ref_block}"
    )
    raw = llm.judge_json(JUDGE_SYSTEM, user, JUDGE_SCHEMA,
                         model=model, temperature=temperature, salt=salt)

    scores = {k: int(raw[k]["score"]) for k in _DIM_KEYS}
    reasons = {k: raw[k]["reason"] for k in _DIM_KEYS}
    flags = {f: bool(raw[f]) for f in config.HARD_FLAGS}
    if not reference:
        # with no reference there is nothing to contradict — enforce in code rather
        # than trusting the model to follow the prompt instruction
        flags["contradicts_reference_commitment"] = False
    composite = composite_from_scores(scores)
    passed = composite >= config.PASS_THRESHOLD and not any(flags.values())
    return {
        "scores": scores,
        "reasons": reasons,
        "flags": flags,
        "composite": composite,
        "passed": passed,
        "overall_note": raw.get("overall_note", ""),
        "judge_model": model or config.MODEL_JUDGE,
    }


# ---------------------------------------------------------------------------
# System-level aggregation
# ---------------------------------------------------------------------------
def aggregate(results: list[dict]) -> dict:
    """Roll per-response scores up to a system-level report. We report the
    distribution and a pass-rate, not just a mean — 'avg 0.78' hides whether a
    fifth of replies are unsendable."""
    n = len(results)
    if n == 0:
        return {}
    comps = [r["composite"] for r in results]
    per_dim = {
        k: round(sum(r["scores"][k] for r in results) / n, 3) for k in _DIM_KEYS
    }
    flag_rates = {
        f: round(sum(1 for r in results if r["flags"][f]) / n, 3) for f in config.HARD_FLAGS
    }
    buckets = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0, "0.85-1.0": 0}
    for c in comps:
        if c < 0.5:
            buckets["0.0-0.5"] += 1
        elif c < 0.7:
            buckets["0.5-0.7"] += 1
        elif c < 0.85:
            buckets["0.7-0.85"] += 1
        else:
            buckets["0.85-1.0"] += 1
    return {
        "n": n,
        "mean_composite": round(sum(comps) / n, 3),
        "median_composite": round(statistics.median(comps), 3),
        "pass_rate": round(sum(1 for r in results if r["passed"]) / n, 3),
        "per_dimension_mean_1to5": per_dim,
        "hard_flag_rate": flag_rates,
        "composite_distribution": buckets,
    }
