"""Score the whole system: generate + judge every held-out test email, then roll
up to a system-level report (per-response scores + overall).

  python src/report.py            # full test split
  python src/report.py --limit 8  # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402
import evaluator  # noqa: E402
import generator  # noqa: E402
import llm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only score the first N test emails")
    args = ap.parse_args()

    test = dataio.split("test")
    if args.limit:
        test = test[: args.limit]
    retriever = generator.build_retriever()

    results = []
    human_baseline = []  # human-sent replies, judged reference-free, as an interpretability anchor
    for i, rec in enumerate(test, 1):
        email_text = dataio.render_email(rec)
        gen = generator.generate_reply(email_text, retriever)
        # Production mode: score the suggested reply against the incoming email alone
        # (in real use there is no human reply yet). The human reply is scored the
        # same reference-free way, so system-vs-human is apples-to-apples.
        res = evaluator.judge(email_text, gen["reply"], None)
        base = evaluator.judge(email_text, rec["reference_reply"], None)
        human_baseline.append(base)
        results.append(
            {
                "id": rec["id"],
                "category": rec["category"],
                "composite": res["composite"],
                "passed": res["passed"],
                "scores": res["scores"],
                "flags": res["flags"],
                "lexical_overlap": round(evaluator.lexical_overlap(gen["reply"], rec["reference_reply"]), 3),
                "human_reference_composite": base["composite"],
                "reply": gen["reply"],
                "reasons": res["reasons"],
            }
        )
        print(f"  [{i}/{len(test)}] {rec['id']} ({rec['category']}) "
              f"composite={res['composite']:.2f} {'PASS' if res['passed'] else 'FAIL'}"
              f"  (human ref {base['composite']:.2f})")

    overall = evaluator.aggregate(results)
    hb = evaluator.aggregate(human_baseline)
    overall["human_reference_baseline"] = {
        "note": "human-sent replies scored reference-free on the same rubric — a hard-data sanity anchor for the system score",
        "mean_composite": hb["mean_composite"],
        "median_composite": hb["median_composite"],
        "pass_rate": hb["pass_rate"],
    }
    out = {"overall": overall, "per_response": results}
    (config.RESULTS_DIR / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SYSTEM-LEVEL REPORT ===")
    print(json.dumps(overall, indent=2))
    print("\nsaved: results/report.json")
    print(llm.usage_summary())


if __name__ == "__main__":
    main()
