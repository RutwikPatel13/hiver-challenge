"""End-to-end demo: incoming email -> suggested reply -> accuracy score.

Examples:
  python src/run.py --id t0003                       # use a held-out test email (has a reference)
  python src/run.py --email "Subject: ...\\n\\nHi, my order never arrived..."
  python src/run.py --email-file path/to/email.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio  # noqa: E402
import evaluator  # noqa: E402
import generator  # noqa: E402
import llm  # noqa: E402


def _fmt_score(res: dict) -> str:
    lines = [
        f"  composite: {res['composite']:.3f}   verdict: {'PASS (sendable)' if res['passed'] else 'FAIL (needs edit)'}",
    ]
    for k, s in res["scores"].items():
        lines.append(f"    {k:<22} {s}/5  — {res['reasons'][k]}")
    flags = [f for f, v in res["flags"].items() if v]
    lines.append(f"    hard flags: {', '.join(flags) if flags else 'none'}")
    if res.get("overall_note"):
        lines.append(f"    note: {res['overall_note']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", help="dataset record id (uses its reference reply)")
    g.add_argument("--email", help="raw incoming email text")
    g.add_argument("--email-file", help="path to a file with the incoming email")
    args = ap.parse_args()

    reference = None
    if args.id:
        rec = next((r for r in dataio.load_dataset() if r["id"] == args.id), None)
        if rec is None:
            sys.exit(f"no record with id {args.id}")
        email_text = dataio.render_email(rec)
        reference = rec["reference_reply"]
    elif args.email_file:
        email_text = Path(args.email_file).read_text().strip()
    else:
        email_text = args.email.replace("\\n", "\n")

    retriever = generator.build_retriever()
    # exclude the queried ticket from retrieval: if --id points at a KB record, its
    # own sent reply must not leak into its few-shot examples
    gen = generator.generate_reply(email_text, retriever, exclude_id=args.id)
    # Production mode: score against the incoming email alone (no human reply exists
    # yet for a live email). The reference below is shown only for context.
    res = evaluator.judge(email_text, gen["reply"], None)

    print("=" * 70)
    print("INCOMING EMAIL\n" + "-" * 70)
    print(email_text)
    print("\nSUGGESTED REPLY  (grounded in past tickets: "
          + ", ".join(gen["retrieved_ids"]) + ")\n" + "-" * 70)
    print(gen["reply"])
    print("\nACCURACY SCORE  (judge=" + res["judge_model"] + ")\n" + "-" * 70)
    print(_fmt_score(res))
    if reference:
        print("\nREFERENCE REPLY (what was actually sent)\n" + "-" * 70)
        print(reference)
        print(f"\n[baseline] lexical overlap vs reference (ROUGE-L F1): "
              f"{evaluator.lexical_overlap(gen['reply'], reference):.3f}")
    print("\n" + llm.usage_summary())


if __name__ == "__main__":
    main()
