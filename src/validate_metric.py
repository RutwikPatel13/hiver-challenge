"""Validate that the accuracy metric reflects real quality, not just a number.

Three experiments (this is the heart of the "how do you know your metric works"
question):

1. PERTURBATION SENSITIVITY. Take good replies and break them one axis at a time
   (flip the commitment, inject a fabricated fact, make it rude, strip the actual
   help). A trustworthy metric must drop on the RIGHT dimension / trip the RIGHT
   hard flag for each break. This proves the metric is sensitive to real failure
   modes rather than reacting to noise — and needs no external human labels.

2. DISCRIMINATIVE POWER. The judge should rank a fitting reply above an unrelated
   one. We compare, per email, the system's own reply vs. a random other ticket's
   reply, and measure how often each metric ranks them correctly. We run the naive
   lexical-overlap baseline through the same test: the judge should separate them
   far better, demonstrating why surface overlap is insufficient.

3. JUDGE RELIABILITY. Re-judge the same replies multiple times (sampled) to measure
   score variance, and cross-check with a *different* model (Sonnet) to measure
   agreement. This quantifies how much of the score is signal vs. judge noise.

  python src/validate_metric.py
Outputs results/validation.json and a printed summary.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402
import evaluator  # noqa: E402
import generator  # noqa: E402
import llm  # noqa: E402

PERTURB_N = 12
DISCRIM_N = 20
RELIABILITY_N = 10
RELIABILITY_REPEATS = 3

# perturbation type -> (prompt, target dimension that should drop, hard flag that should trip)
PERTURBATIONS = {
    "flip_commitment": (
        "Rewrite this support reply so it takes the OPPOSITE position on the customer's "
        "request: if it offered or agreed to something, now refuse it; if it refused, now "
        "promise it. Keep the length and tone similar. Output only the rewritten reply.",
        "factual_correctness",
        "contradicts_reference_commitment",
    ),
    "hallucinate": (
        "Edit this support reply to insert ONE specific but fabricated detail — an invented "
        "order number, a made-up refund amount, or a fake named policy or date. Keep the rest "
        "unchanged. Output only the edited reply.",
        "faithfulness",
        "hallucination_detected",
    ),
    "make_rude": (
        "Rewrite this support reply to be curt, dismissive and unempathetic, while keeping the "
        "same factual content. Output only the rewritten reply.",
        "tone",
        None,
    ),
    "strip_help": (
        "Rewrite this support reply to remove any concrete help, resolution, or next step — "
        "leave only a generic pleasantry that does not actually address the request. Output "
        "only the rewritten reply.",
        "intent_resolution",
        None,
    ),
}


def _perturb(reply: str, prompt: str) -> str:
    return llm.generate(
        "You produce controlled test variants of support replies. Follow the instruction exactly.",
        f"{prompt}\n\n--- Reply ---\n{reply}",
        temperature=0.0,
    ).strip()


def experiment_perturbation() -> dict:
    test = dataio.split("test")[:PERTURB_N]
    per_type: dict[str, dict] = {}
    for ptype, (prompt, target_dim, target_flag) in PERTURBATIONS.items():
        comp_base, comp_pert = [], []
        dim_base, dim_pert = [], []
        flag_trips = 0
        for rec in test:
            email = dataio.render_email(rec)
            ref = rec["reference_reply"]
            base = evaluator.judge(email, ref, ref)  # good reply = the sent reply
            pert_reply = _perturb(ref, prompt)
            pert = evaluator.judge(email, pert_reply, ref)
            comp_base.append(base["composite"]); comp_pert.append(pert["composite"])
            dim_base.append(base["scores"][target_dim]); dim_pert.append(pert["scores"][target_dim])
            if target_flag and pert["flags"][target_flag]:
                flag_trips += 1
        per_type[ptype] = {
            "target_dimension": target_dim,
            "target_flag": target_flag,
            "mean_composite_good": round(statistics.mean(comp_base), 3),
            "mean_composite_broken": round(statistics.mean(comp_pert), 3),
            "mean_composite_drop": round(statistics.mean(comp_base) - statistics.mean(comp_pert), 3),
            f"mean_{target_dim}_good_1to5": round(statistics.mean(dim_base), 2),
            f"mean_{target_dim}_broken_1to5": round(statistics.mean(dim_pert), 2),
            "target_dimension_drop_1to5": round(statistics.mean(dim_base) - statistics.mean(dim_pert), 2),
            "hard_flag_trip_rate": round(flag_trips / len(test), 3) if target_flag else None,
        }
    return per_type


def _auc_paired(pos: list[float], neg: list[float]) -> float:
    """Fraction of paired (pos_i, neg_i) where pos ranks strictly above neg
    (ties count as 0.5). Chance = 0.5; perfect = 1.0."""
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in zip(pos, neg))
    return round(wins / len(pos), 3)


def experiment_discrimination() -> dict:
    test = dataio.split("test")[:DISCRIM_N]
    retriever = generator.build_retriever()
    # deterministic mismatch: pair each email with a different ticket's reply
    mism = test[1:] + test[:1]

    judge_pos, judge_neg, lex_pos, lex_neg = [], [], [], []
    for rec, other in zip(test, mism):
        email = dataio.render_email(rec)
        ref = rec["reference_reply"]
        gen = generator.generate_reply(email, retriever)["reply"]  # system's real reply
        wrong = other["reference_reply"]                            # unrelated reply

        # reference-free (production mode): a fitting reply vs an unrelated one,
        # both judged against the same incoming email
        judge_pos.append(evaluator.judge(email, gen, None)["composite"])
        judge_neg.append(evaluator.judge(email, wrong, None)["composite"])
        lex_pos.append(evaluator.lexical_overlap(gen, ref))
        lex_neg.append(evaluator.lexical_overlap(wrong, ref))

    return {
        "n_pairs": len(test),
        "rubric_judge": {
            "mean_fitting": round(statistics.mean(judge_pos), 3),
            "mean_mismatched": round(statistics.mean(judge_neg), 3),
            "separation_auc": _auc_paired(judge_pos, judge_neg),
        },
        "lexical_overlap_baseline": {
            "mean_fitting": round(statistics.mean(lex_pos), 3),
            "mean_mismatched": round(statistics.mean(lex_neg), 3),
            "separation_auc": _auc_paired(lex_pos, lex_neg),
        },
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 3) if dx and dy else 0.0


def experiment_reliability() -> dict:
    test = dataio.split("test")[:RELIABILITY_N]
    retriever = generator.build_retriever()
    mism = test[1:] + test[:1]  # each email paired with an unrelated reply

    within_stds, verdict_consistent = [], 0
    haiku_comp, sonnet_comp = [], []  # over a MIXED pool spanning the quality range
    verdict_agree, verdict_total = 0, 0
    for rec, other in zip(test, mism):
        email = dataio.render_email(rec)
        gen = generator.generate_reply(email, retriever)["reply"]
        wrong = other["reference_reply"]

        # (a) within-judge stability: repeat the same judgment N times at temp>0
        reps = [evaluator.judge(email, gen, None, temperature=0.7, salt=f"rel{k}")
                for k in range(RELIABILITY_REPEATS)]
        comps = [r["composite"] for r in reps]
        within_stds.append(statistics.pstdev(comps))
        if len({r["passed"] for r in reps}) == 1:
            verdict_consistent += 1

        # (b) cross-model agreement across the FULL quality range: a good (system)
        # reply AND a bad (mismatched) reply. Measuring only good replies gives a
        # near-constant sample where rank correlation is undefined/noisy.
        for cand in (gen, wrong):
            h = evaluator.judge(email, cand, None)
            s = evaluator.judge(email, cand, None, model=config.MODEL_JUDGE_CROSSCHECK)
            haiku_comp.append(h["composite"]); sonnet_comp.append(s["composite"])
            verdict_total += 1
            if h["passed"] == s["passed"]:
                verdict_agree += 1

    n = len(test)
    return {
        "n_items": n,
        "within_judge_composite_std_mean": round(statistics.mean(within_stds), 3),
        "within_judge_verdict_consistency": round(verdict_consistent / n, 3),
        "cross_model": {
            "judge_a": config.MODEL_JUDGE,
            "judge_b": config.MODEL_JUDGE_CROSSCHECK,
            "note": "over a mixed pool of good (system) + bad (mismatched) replies, so correlation reflects agreement across the full quality range",
            "n_points": verdict_total,
            "mean_abs_composite_diff": round(
                statistics.mean(abs(a - b) for a, b in zip(haiku_comp, sonnet_comp)), 3),
            "composite_correlation": _pearson(haiku_comp, sonnet_comp),
            "verdict_agreement": round(verdict_agree / verdict_total, 3),
        },
    }


def main() -> None:
    print("1/3 perturbation sensitivity ...")
    perturb = experiment_perturbation()
    print("2/3 discriminative power ...")
    discrim = experiment_discrimination()
    print("3/3 judge reliability ...")
    reliability = experiment_reliability()

    out = {"perturbation": perturb, "discrimination": discrim, "reliability": reliability}
    (config.RESULTS_DIR / "validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== VALIDATION SUMMARY ===")
    print(json.dumps(out, indent=2))
    print("\nsaved: results/validation.json")
    print(llm.usage_summary())


if __name__ == "__main__":
    main()
