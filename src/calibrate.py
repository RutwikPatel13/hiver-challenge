"""Human calibration of the judge: label a blind sample of system replies and
measure judge-vs-human agreement.

Selection is stratified (all judge-FAILs + random judge-PASSes, 15 total) so the
sample carries signal in both classes; the labeler never sees scores, so labels
are blind. Kappa corrects for chance agreement under the resulting marginals.

Usage:
  python src/calibrate.py --make-sheet
      writes results/labeling_sheet.md — read it and decide, for each reply,
      "would I send this as-is?" (y/n)

  python src/calibrate.py --labels ynynyynnyyynyyn
      one y/n per item, in sheet order; computes agreement + Cohen's kappa vs the
      reference-free judge (and vs the reference-augmented judge if
      results/blindspot.json exists) and writes results/human_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402

N_ITEMS = 15


def _select() -> list[dict]:
    report = json.loads((config.RESULTS_DIR / "report.json").read_text(encoding="utf-8"))
    rows = report["per_response"]
    fails = [r for r in rows if not r["passed"]]
    passes = [r for r in rows if r["passed"]]
    rng = random.Random(config.RANDOM_SEED)
    picked = fails + rng.sample(passes, max(0, N_ITEMS - len(fails)))
    picked = picked[:N_ITEMS]
    rng.shuffle(picked)  # so fails aren't clustered at the top of the sheet
    return picked


def make_sheet() -> None:
    recs = {r["id"]: r for r in dataio.load_dataset()}
    picked = _select()
    lines = [
        "# Blind labeling sheet — would you SEND this reply as-is? (y/n)",
        "",
        "Judge scores are deliberately hidden. For each item, read the incoming",
        "email and the suggested reply, and decide as a support agent: send / not send.",
        "",
    ]
    for i, r in enumerate(picked, 1):
        rec = recs[r["id"]]
        lines += [
            f"## {i}.",
            "**Incoming email:**",
            "```",
            dataio.render_email(rec),
            "```",
            "**Suggested reply:**",
            "```",
            r["reply"],
            "```",
            "",
        ]
    out = config.RESULTS_DIR / "labeling_sheet.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    ids = [r["id"] for r in picked]
    (config.RESULTS_DIR / "labeling_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"wrote {out} ({len(picked)} items)")
    print("label with:  python src/calibrate.py --labels <15 chars of y/n in sheet order>")


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def score(labels: str) -> None:
    labels = labels.strip().lower().replace(" ", "").replace(",", "")
    ids = json.loads((config.RESULTS_DIR / "labeling_ids.json").read_text(encoding="utf-8"))
    if len(labels) != len(ids) or set(labels) - {"y", "n"}:
        sys.exit(f"need exactly {len(ids)} y/n characters (got {len(labels)})")
    human = [c == "y" for c in labels]

    report = {r["id"]: r for r in json.loads(
        (config.RESULTS_DIR / "report.json").read_text(encoding="utf-8"))["per_response"]}
    judge = [report[i]["passed"] for i in ids]
    comps = [report[i]["composite"] for i in ids]

    yes_comp = [c for c, h in zip(comps, human) if h]
    no_comp = [c for c, h in zip(comps, human) if not h]
    out = {
        "note": "human labeled 15 system replies blind (send as-is? y/n); sample stratified: "
                "all reference-free judge FAILs + random PASSes",
        "n": len(ids),
        "item_ids": ids,
        "human_labels_send": labels,
        "human_send_rate": round(sum(human) / len(human), 3),
        "judge_pass_rate_on_sample": round(sum(judge) / len(judge), 3),
        "agreement_with_judge": round(sum(h == j for h, j in zip(human, judge)) / len(ids), 3),
        "cohens_kappa_vs_judge": _kappa(human, judge),
        "mean_composite_human_send": round(sum(yes_comp) / len(yes_comp), 3) if yes_comp else None,
        "mean_composite_human_reject": round(sum(no_comp) / len(no_comp), 3) if no_comp else None,
        "disagreements": [
            {"id": i, "human_send": h, "judge_pass": j, "composite": c}
            for i, h, j, c in zip(ids, human, judge, comps) if h != j
        ],
    }

    bs_path = config.RESULTS_DIR / "blindspot.json"
    if bs_path.exists():
        bs = json.loads(bs_path.read_text(encoding="utf-8"))
        if "all_rows" in bs:  # written by blindspot.py; carries every aug verdict
            aug = {x["id"]: x["aug_passed"] for x in bs["all_rows"]}
            aug_verdicts = [aug[i] for i in ids]
            out["agreement_with_augmented_judge"] = round(
                sum(h == a for h, a in zip(human, aug_verdicts)) / len(ids), 3)
            out["cohens_kappa_vs_augmented_judge"] = _kappa(human, aug_verdicts)

    (config.RESULTS_DIR / "human_calibration.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("item_ids", "disagreements")}, indent=2))
    if out["disagreements"]:
        print("disagreements:", json.dumps(out["disagreements"], indent=2))
    print("saved: results/human_calibration.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-sheet", action="store_true")
    g.add_argument("--labels")
    args = ap.parse_args()
    make_sheet() if args.make_sheet else score(args.labels)
