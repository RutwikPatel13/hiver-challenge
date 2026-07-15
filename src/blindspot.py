"""Quantify the reference-free metric's blind spot.

In production the judge scores a suggested reply with no ground truth (no human
reply exists yet). This script measures what that costs: it takes every system
reply already scored reference-free by report.py, re-judges it WITH the human
reference supplied, and compares verdicts.

A reply that passes reference-free but fails reference-augmented is a blind-spot
case: the reply looked sendable in isolation, but ground truth reveals a wrong
commitment, policy violation, or invented specifics. The rate is an UPPER bound —
some flips are the augmented judge penalizing valid divergence from the reference
— but spot checks (see README §5) show genuine catches dominate.

Run AFTER report.py:  python src/blindspot.py
Output: results/blindspot.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402
import evaluator  # noqa: E402
import llm  # noqa: E402


def main() -> None:
    report_path = config.RESULTS_DIR / "report.json"
    if not report_path.exists():
        sys.exit("results/report.json not found — run src/report.py first")
    recs = {r["id"]: r for r in dataio.load_dataset()}
    per_response = json.loads(report_path.read_text(encoding="utf-8"))["per_response"]

    rows = []
    for r in per_response:
        rec = recs[r["id"]]
        aug = evaluator.judge(dataio.render_email(rec), r["reply"], rec["reference_reply"])
        rows.append(
            {
                "id": r["id"],
                "free_composite": r["composite"],
                "free_passed": r["passed"],
                "aug_composite": aug["composite"],
                "aug_passed": aug["passed"],
                "aug_flags": {k: v for k, v in aug["flags"].items() if v},
            }
        )

    n = len(rows)
    missed = [x for x in rows if x["free_passed"] and not x["aug_passed"]]
    out = {
        "note": "reference-free vs reference-augmented verdicts on the same system replies; "
                "'blind_spot_rate' is an upper bound on what production scoring cannot see",
        "n": n,
        "free_pass_rate": round(sum(x["free_passed"] for x in rows) / n, 3),
        "aug_pass_rate": round(sum(x["aug_passed"] for x in rows) / n, 3),
        "pass_free_but_fail_augmented": len(missed),
        "blind_spot_rate": round(len(missed) / n, 3),
        "mean_abs_composite_diff": round(
            sum(abs(x["free_composite"] - x["aug_composite"]) for x in rows) / n, 3
        ),
        "missed_items": missed,
        "all_rows": rows,
    }
    (config.RESULTS_DIR / "blindspot.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "missed_items"}, indent=2))
    print("saved: results/blindspot.json")
    print(llm.usage_summary())


if __name__ == "__main__":
    main()
