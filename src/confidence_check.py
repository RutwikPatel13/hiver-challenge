"""Test a mitigation hypothesis: does BM25 retrieval confidence predict the
blind-spot flips measured by blindspot.py?

If low retrieval confidence predicted pass-free-but-fail-augmented flips, a simple
"defer low-confidence emails to a human" gate would close much of the blind spot.

Result (committed in README §5): it does NOT — AUC ≈ 0.43–0.48 (chance). The flips
are driven by brand/policy mismatch, which lexical similarity to *some* KB ticket
cannot capture: an email can match a telecom ticket strongly and still be answered
with the wrong brand's policy. The gate needs brand/domain grounding, not a
similarity threshold. Fully offline — no API calls.

Run AFTER blindspot.py:  python src/confidence_check.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402
from retrieval import Retriever  # noqa: E402


def main() -> None:
    bs_path = config.RESULTS_DIR / "blindspot.json"
    if not bs_path.exists():
        sys.exit("results/blindspot.json not found — run src/blindspot.py first")
    all_rows = json.loads(bs_path.read_text(encoding="utf-8"))["all_rows"]
    recs = {r["id"]: r for r in dataio.load_dataset()}
    ret = Retriever(dataio.split("kb"))

    rows = []
    for x in all_rows:
        email = dataio.render_email(recs[x["id"]])
        top = sorted(ret.bm25.scores(email), reverse=True)[:3]
        rows.append(
            {
                "id": x["id"],
                "top1": round(top[0], 3),
                "top3_mean": round(sum(top) / 3, 3),
                "flipped": x["free_passed"] and not x["aug_passed"],
            }
        )

    flips = [r for r in rows if r["flipped"]]
    ok = [r for r in rows if not r["flipped"]]

    def auc(key: str) -> float:
        wins = sum(
            1.0 if a[key] > b[key] else 0.5 if a[key] == b[key] else 0.0
            for a in ok for b in flips
        )
        return round(wins / (len(ok) * len(flips)), 3)

    deferral = {}
    srt = sorted(rows, key=lambda r: r["top1"])
    for frac in (0.2, 0.3, 0.5):
        k = int(len(rows) * frac)
        deferral[f"defer_lowest_{int(frac*100)}pct"] = {
            "items_deferred": k,
            "flips_caught": sum(1 for r in srt[:k] if r["flipped"]),
            "flips_total": len(flips),
        }

    out = {
        "note": "does BM25 retrieval confidence predict blind-spot flips? "
                "AUC ~0.5 = no signal; the naive confidence gate is insufficient",
        "n": len(rows),
        "n_flips": len(flips),
        "mean_top1_flipped": round(statistics.mean(r["top1"] for r in flips), 2),
        "mean_top1_ok": round(statistics.mean(r["top1"] for r in ok), 2),
        "auc_top1": auc("top1"),
        "auc_top3_mean": auc("top3_mean"),
        "deferral_curve": deferral,
    }
    (config.RESULTS_DIR / "confidence_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    print("saved: results/confidence_check.json")


if __name__ == "__main__":
    main()
