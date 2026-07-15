"""Retrieval ablation: does grounding in past tickets actually help?

Generates a ZERO-SHOT reply (same generator prompt, no retrieved examples) for
every held-out test email and scores it with the same judge, in both modes:

  * reference-free   — production mode; measures how *sendable* the reply looks
  * reference-augmented — the sent human reply as ground truth; measures whether
    the reply commits to the right things (this is where grounding must show up,
    because §5 established the reference-free judge cannot see wrong-brand or
    wrong-commitment errors)

The RAG side comes from results/report.json (reference-free) and
results/blindspot.json (reference-augmented) — same emails, same judge.

Run AFTER report.py and blindspot.py:  python src/ablation.py
Output: results/ablation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402
import evaluator  # noqa: E402
import generator  # noqa: E402
import llm  # noqa: E402


def _zero_shot_reply(email_text: str) -> str:
    user = (
        f"=== New incoming email ===\n{email_text}\n\n"
        f"Write the reply body now."
    )
    return llm.generate(generator.SYSTEM, user).strip()


def main() -> None:
    report_path = config.RESULTS_DIR / "report.json"
    bs_path = config.RESULTS_DIR / "blindspot.json"
    if not (report_path.exists() and bs_path.exists()):
        sys.exit("run src/report.py and src/blindspot.py first")
    rag_free = {r["id"]: r for r in json.loads(report_path.read_text(encoding="utf-8"))["per_response"]}
    rag_aug = {r["id"]: r for r in json.loads(bs_path.read_text(encoding="utf-8"))["all_rows"]}

    test = dataio.split("test")
    zs_free, zs_aug = [], []
    for i, rec in enumerate(test, 1):
        email = dataio.render_email(rec)
        reply = _zero_shot_reply(email)
        free = evaluator.judge(email, reply, None)
        aug = evaluator.judge(email, reply, rec["reference_reply"])
        zs_free.append(free)
        zs_aug.append(aug)
        print(f"  [{i}/{len(test)}] {rec['id']} zero-shot free={free['composite']:.2f} "
              f"aug={aug['composite']:.2f}")

    n = len(test)
    rag_free_agg = {
        "mean_composite": round(sum(rag_free[r["id"]]["composite"] for r in test) / n, 3),
        "pass_rate": round(sum(rag_free[r["id"]]["passed"] for r in test) / n, 3),
    }
    rag_aug_agg = {
        "mean_composite": round(sum(rag_aug[r["id"]]["aug_composite"] for r in test) / n, 3),
        "pass_rate": round(sum(rag_aug[r["id"]]["aug_passed"] for r in test) / n, 3),
    }
    zs_free_agg = evaluator.aggregate(zs_free)
    zs_aug_agg = evaluator.aggregate(zs_aug)

    out = {
        "note": "RAG (k=3 few-shot from KB) vs zero-shot (no retrieved examples), same judge, "
                "both judging modes; reference-augmented is where grounding must show up (§5)",
        "n": n,
        "reference_free": {
            "rag": rag_free_agg,
            "zero_shot": {"mean_composite": zs_free_agg["mean_composite"],
                          "pass_rate": zs_free_agg["pass_rate"]},
        },
        "reference_augmented": {
            "rag": rag_aug_agg,
            "zero_shot": {"mean_composite": zs_aug_agg["mean_composite"],
                          "pass_rate": zs_aug_agg["pass_rate"]},
        },
        "zero_shot_detail": {"reference_free": zs_free_agg, "reference_augmented": zs_aug_agg},
    }
    (config.RESULTS_DIR / "ablation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: out[k] for k in ("reference_free", "reference_augmented")}, indent=2))
    print("saved: results/ablation.json")
    print(llm.usage_summary())


if __name__ == "__main__":
    main()
