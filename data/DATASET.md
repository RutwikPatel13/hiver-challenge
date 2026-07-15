# Dataset — provenance, construction, and honest limitations

## What it is
`dataset.jsonl` — 200 email/response pairs for a customer-support inbox (the spec's
maximum). Each record:

| field | meaning |
|---|---|
| `id` | stable id (`t0000`…) |
| `source` | origin, incl. the corpus offset the pair came from |
| `category` | heuristic label (billing, shipping, technical, account, cancellation, complaint, general) |
| `subject` | heuristic subject line (the email envelope) |
| `customer_message` | **the real customer message** — the incoming email body |
| `reference_reply` | **the real agent reply** — what was actually sent |
| `split` | `kb` (retrieval / few-shot pool, 140) or `test` (held-out evaluation, 60) |

## Where it comes from
The content is **real**. It is drawn from the **Customer Support on Twitter (TWCS)** corpus —
genuine messages from customers to brand support agents and the agents' real replies — via
the pre-paired Hugging Face mirror
[`MohammadOthman/mo-customer-support-tweets-945k`](https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k)
(`input` = customer, `output` = agent). We chose this over a synthetic set because Hiver is a
shared-inbox / support product, and this is real support-interaction data in that exact domain.

## How it was built (`src/build_dataset.py`, reproducible)
1. **Sample** real customer→agent pairs from 16 fixed offsets spread across the 945k-row
   corpus. All 16 offsets contribute candidates (740 in the committed build) *before* the
   shuffle-and-truncate to 200, so the kept sample genuinely spans the corpus. Fixed
   offsets + `RANDOM_SEED` ⇒ deterministic build.
2. **Clean surface noise** only: strip `@handles`, URLs, and collapse whitespace.
3. **Filter for standalone support emails**: English-ish, sensible length, ≥10 customer words,
   and a "looks like a first-contact request" check (contains a question mark or a request/issue
   cue) so we keep messages that read as incoming emails rather than mid-thread reply fragments.
   Deduped by normalized message.
4. **Add a light email envelope**: a heuristic `subject` line and a heuristic `category`.
5. **Split** deterministically into a KB pool and a held-out test set.

**What we did *not* do:** we did not rewrite the message bodies. The incoming email is the real
customer message; the reference reply is the real agent response. Preserving the substance is
what makes this an honest "real data" set rather than an LLM-paraphrased one.

## Why it's representative (and where it isn't)
Representative: real customer intents (billing disputes, delivery problems, technical faults,
cancellations, complaints), real emotional register (frustration, urgency, all-caps), and real
agent resolutions (answers, diagnostics, routing, apologies).

**Honest limitations:**
- **Origin is Twitter, not email.** Messages are short and occasionally assume prior context.
  The "email envelope" (subject line) is synthetic; we keep it minimal and clearly heuristic.
- **Terse reference replies.** Real Twitter-support replies are short and often just deflect or
  ask for info. On our rubric they frequently score *lower* than the model's fuller replies — a
  finding we report and discuss (it partly reflects the short-channel origin and partly a known
  verbosity lean in LLM judges). See the main README.
- **Heuristic categories/subjects.** Keyword-based, so imperfect; `general` is the catch-all.
- **Brand-support skew and English-only.** The corpus over-represents telecom/airline/retail
  brand handles, and we filter to English-ish text.
- **Category imbalance.** The heuristic labels skew toward `general` (~half) — the catch-all
  for messages whose intent keywords don't match; per-category conclusions are weak for the
  small categories (complaint, account, cancellation).
- **Small for training, adequate for evaluation.** 200 pairs (the spec's upper bound) is
  enough to exercise and validate the system honestly, not to train one. The KB pool of 140
  still cannot cover every product domain — retrieval misses on out-of-domain emails remain
  possible (see the t-blind-spot discussion in the main README).

## Reproducing
```
python src/build_dataset.py    # rewrites data/dataset.jsonl (no API key needed)
```
