"""Build the email/response dataset from REAL customer-support interactions.

Source: the "Customer Support on Twitter" (TWCS) corpus — genuine messages from
customers to brand support agents and the agents' real replies — accessed via the
paired Hugging Face mirror `MohammadOthman/mo-customer-support-tweets-945k`
(columns: `input` = customer message, `output` = agent reply).

What this script does (and, importantly, does NOT do):
  * Pulls real customer->agent pairs from several offsets across the 945k-row
    corpus (fixed offsets + seed => reproducible).
  * Cleans surface noise (@handles, URLs, whitespace) and filters for pairs that
    stand on their own (length/word-count bounds, English-ish, deduped).
  * Adds a light EMAIL ENVELOPE: a heuristic subject line and a category label.
  * Does NOT rewrite the message bodies. The incoming email body is the real
    customer message; the reference reply is the real agent response. Keeping the
    substance untouched is what makes this an honest "real data" dataset.

Run:  python src/build_dataset.py
Output: data/dataset.jsonl  and  a printed summary.
"""

from __future__ import annotations

import html
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROWS_URL = "https://datasets-server.huggingface.co/rows"

# Offsets spread across the corpus so we don't over-sample one brand/time window.
OFFSETS = [0, 5000, 20000, 40000, 80000, 120000, 180000, 250000, 330000, 400000,
           500000, 600000, 700000, 800000, 870000, 900000]
PAGE = 100

# Cues that a message is a self-contained support request (an inbox gets these as
# first-contact emails), used to filter out mid-thread conversational fragments.
REQUEST_CUES = [
    "can't", "cant", "cannot", "won't", "wont", "doesn't", "isn't", "unable",
    "not working", "no longer", "still not", "issue", "problem", "help", "how do",
    "how can", "how to", "why is", "why does", "when will", "refund", "cancel",
    "charged", "charge", "broken", "error", "need", "order", "delivery", "deliver",
    "missing", "wrong", "never received", "haven't received", "please", "trouble",
    "not received", "won’t", "can’t", "doesn’t", "isn’t",
]

# Heuristic category keywords. Order matters: first match wins.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("billing", ["charge", "charged", "refund", "bill", "payment", "invoice", "price", "overcharge", "money back"]),
    ("shipping", ["deliver", "delivery", "shipping", "shipment", "package", "parcel", "tracking", "arrive", "order status"]),
    ("cancellation", ["cancel", "unsubscribe", "close my account", "terminate"]),
    ("technical", ["error", "not working", "won't", "wont", "broken", "crash", "bug", "outage", "connection", "login", "log in", "password", "reset", "app", "website", "down"]),
    ("account", ["account", "profile", "settings", "email address", "phone number", "update my"]),
    ("complaint", ["worst", "terrible", "awful", "disappointed", "unacceptable", "rude", "angry", "frustrated", "horrible", "never again"]),
]

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = URL_RE.sub("", text)
    text = HANDLE_RE.sub("", text)          # drop @brand / @user handles
    text = text.replace("&amp;", "&")
    text = WS_RE.sub(" ", text).strip()
    # strip leading punctuation left over from removing a handle
    text = re.sub(r"^[\s,.:;\-]+", "", text)
    return text


def is_english_ish(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.9


def categorize(customer: str) -> str:
    low = customer.lower()
    for name, kws in CATEGORY_RULES:
        if any(kw in low for kw in kws):
            return name
    return "general"


def make_subject(customer: str, category: str) -> str:
    """Heuristic subject line: first clause of the message, trimmed to ~9 words."""
    first = re.split(r"[.!?\n]", customer, maxsplit=1)[0].strip()
    words = first.split()
    if len(words) < 3:
        words = customer.split()
    subject = " ".join(words[:9]).rstrip(",;:- ")
    subject = subject[:1].upper() + subject[1:] if subject else category.title()
    return subject or category.title()


def looks_like_request(customer: str) -> bool:
    """Bias toward messages that read as a standalone incoming support email
    rather than a mid-thread reply fragment."""
    low = customer.lower()
    if "?" in customer:
        return True
    return any(cue in low for cue in REQUEST_CUES)


def acceptable(customer: str, agent: str) -> bool:
    if not (is_english_ish(customer) and is_english_ish(agent)):
        return False
    if not (50 <= len(customer) <= 600 and 40 <= len(agent) <= 600):
        return False
    if len(customer.split()) < 10 or len(agent.split()) < 8:
        return False
    if not looks_like_request(customer):
        return False
    # drop near-content-free agent boilerplate that carries no resolution signal
    low = agent.lower()
    if low.startswith(("dm us", "please dm", "pls dm")) and len(agent.split()) < 14:
        return False
    return True


def fetch_page(offset: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "dataset": config.HF_DATASET,
            "config": config.HF_CONFIG,
            "split": config.HF_SPLIT,
            "offset": offset,
            "length": PAGE,
        }
    )
    url = f"{ROWS_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "hiver-challenge/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    return [r["row"] for r in payload.get("rows", [])]


def build() -> None:
    rng = random.Random(config.RANDOM_SEED)
    seen: set[str] = set()
    records: list[dict] = []

    for offset in OFFSETS:
        try:
            rows = fetch_page(offset)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: offset {offset} failed ({exc}); skipping", file=sys.stderr)
            continue
        for row in rows:
            customer = clean_text(row.get("input", ""))
            agent = clean_text(row.get("output", ""))
            if not acceptable(customer, agent):
                continue
            key = customer.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            category = categorize(customer)
            records.append(
                {
                    "source": f"twcs_via_hf:{config.HF_DATASET}#offset={offset}",
                    "category": category,
                    "subject": make_subject(customer, category),
                    "customer_message": customer,
                    "reference_reply": agent,
                }
            )
        print(f"  offset {offset}: kept {len(records)} total so far")

    # No early break: every offset contributes before truncation, so the kept
    # sample is genuinely spread across the corpus (early-exit here previously
    # skewed the dataset toward the first ~13% of the corpus).
    if len(records) < config.MIN_ACCEPTABLE:
        raise SystemExit(
            f"only {len(records)} acceptable pairs collected (need >= "
            f"{config.MIN_ACCEPTABLE}) — check network / HF datasets-server availability"
        )

    rng.shuffle(records)
    records = records[: config.TARGET_TOTAL]

    # deterministic split into KB (retrieval/few-shot pool) and held-out test set
    n_test = int(len(records) * config.TEST_FRACTION)
    for i, rec in enumerate(records):
        rec["id"] = f"t{i:04d}"
        rec["split"] = "test" if i < n_test else "kb"

    config.DATASET_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    # summary
    from collections import Counter

    cats = Counter(r["category"] for r in records)
    splits = Counter(r["split"] for r in records)
    print("\n=== dataset built ===")
    print(f"path: {config.DATASET_PATH}")
    print(f"total pairs: {len(records)}")
    print(f"splits: {dict(splits)}")
    print(f"categories: {dict(cats)}")


if __name__ == "__main__":
    build()
