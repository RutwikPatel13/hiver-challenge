# AI Email Suggested-Response System

Given an incoming support email: **suggest a reply** grounded in past tickets, and **measure how
good that reply is, and why**. The accuracy system is the centerpiece — designed to be defensible,
then *validated*, then *audited for its own blind spot*.

**Headline results** (all real and reproducible; Claude Haiku 4.5 throughout, every call disk-cached):

| | result |
|---|---|
| System quality (60 held-out real emails) | mean composite **0.83** · pass-rate **0.90** (production mode) |
| Metric catches broken replies | each of 4 targeted breaks drops the *right* dimension; commitment-flip flag trips **100%** |
| Metric separates good from bad | rubric judge **AUC 1.00** vs. lexical-overlap baseline **0.45** (≈chance) |
| Judge reliability | within-judge std **0.03** · cross-model (Haiku↔Sonnet) corr **0.93** |
| Metric's own blind spot — measured | pass-rate falls **0.90 → 0.42** when ground truth is added (§5) |

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env                          # paste your ANTHROPIC_API_KEY (only needed for uncached inputs)

./.venv/bin/python -m unittest discover tests # offline tests, no API key needed
./.venv/bin/python src/build_dataset.py       # rebuild the dataset (no API key needed)
./.venv/bin/python src/run.py --email-file examples/sample_email.txt   # end-to-end demo
./.venv/bin/python src/run.py --id t0035      # demo on a held-out test email
./.venv/bin/python src/report.py              # score the whole test set
./.venv/bin/python src/validate_metric.py     # validate the metric (3 experiments)
./.venv/bin/python src/blindspot.py           # measure the reference-free blind spot
./.venv/bin/python src/ablation.py            # retrieval ablation: RAG vs zero-shot
./.venv/bin/python src/calibrate.py --make-sheet   # blind human-calibration flow (§3d)
```

**The full LLM response cache is committed** (`cache/`, ~3 MB), so every command above reproduces
the committed numbers byte-for-byte at **$0, with no API key** — the client is only constructed on a
cache miss. A cache-less run from scratch (new emails, or after `rm -rf cache/`) needs a key and
costs **< $1**.

## 1. What "accurate" means

"Matches the reply that was actually sent" is a trap: overlap metrics (BLEU/ROUGE) punish correct
paraphrases; embedding similarity scores *"yes, we can refund"* ≈ *"no, we can't refund"*; and the
sent reply is one valid answer among many — imitating it measures mimicry, not quality. Our
definition: **a reply is accurate if a support agent could send it as-is** — it resolves the
request, commits to the right thing, matches the expected tone, and invents nothing.

## 2. The metric

Five dimensions, scored 1–5 by an LLM judge against an explicit rubric (`src/evaluator.py`),
normalized and weighted into a composite in [0,1]:

| dimension | weight | catches |
|---|---|---|
| intent_resolution | 0.28 | actually addresses & moves to resolve the request |
| factual_correctness | 0.24 | right commitment — agree vs. decline, right facts |
| faithfulness | 0.20 | no invented policies, prices, promises, channels |
| completeness | 0.16 | covers every point the customer raised |
| tone | 0.12 | right register (empathetic on complaints) |

**Two hard flags sit outside the average** — `hallucination_detected` and
`contradicts_reference_commitment` — because an average can launder away a disqualifier. **Pass =
composite ≥ 0.70 AND no flag.**

**Two judging modes.** *Reference-free* (production): a live email has no human reply yet, so the
judge scores against the email alone — this is the primary mode. *Reference-augmented* (offline):
on historical data the judge also sees the sent reply — used as a substance check, explicitly not a
template — which is what makes the validation below constructible.

**Reporting:** per-response scores + one-line justifications + flags; system-level distribution,
per-dimension means, and pass-rate (more product-meaningful than a mean).

## 3. Validating the metric (the core question: is it signal or noise?)

**3a. Perturbation sensitivity.** Break good replies one axis at a time; the *targeted* dimension
must drop and the *targeted* flag must trip (12 replies per break):

| break | composite | target dimension (1–5) | flag trip-rate |
|---|---|---|---|
| flip the commitment | 0.88 → 0.14 | factual 5.0 → 1.8 | contradiction **100%** |
| inject a hallucination | 0.88 → 0.15 | faithfulness 5.0 → 1.6 | hallucination **83%** |
| make it rude | 0.88 → 0.60 | tone 4.1 → 2.3 | — |
| strip the actual help | 0.88 → 0.44 | intent 4.3 → 1.1 | — |

The score moves for the *right reason* — no human labels required.

**3b. Discriminative power.** Rank each email's fitting reply against an unrelated ticket's reply
(20 pairs): rubric judge separates **0.84 vs 0.24, AUC 1.00**; the ROUGE-L baseline lands at
**0.12 vs 0.12, AUC 0.45** — at/below chance, because good replies paraphrase rather than echo.
Empirical proof that surface metrics are the wrong tool.

**3c. Reliability.** Re-judging at temperature 0.7: composite std **0.028**, verdict consistency
**90%**. Cross-model (Haiku vs Sonnet, over a mixed good+bad pool so correlation is meaningful):
corr **0.934**, mean |Δ| 0.16, verdict agreement 0.75 — disagreements cluster at the 0.70 boundary.

**3d. Human calibration (single rater, n=15, blind).** The author labeled 15 system replies blind —
"would I send this as-is?" — on a stratified sample (all 6 judge-FAILs + 9 random PASSes;
`src/calibrate.py`, artifacts in `results/`). Raw agreement **0.53**, Cohen's κ **−0.13** (unstable
at n=15 with skewed marginals: 93% human-send vs 60% judge-pass). The disagreement is *directional*,
which makes it informative: all six judge-FAILs were human-send, and five of them sit at composite
0.54–0.68 — just under the 0.70 bar. **Re-scoring at a 0.55 bar lifts agreement to 0.80 without
touching the metric**, so the gap is mostly threshold conservatism, not broken ranking. The one
human-reject (a reply re-asking for info the customer had already given) passed the judge at 0.75 —
a genuine judge miss worth a rubric tweak. Honest caveats: one rater, small n, and single raters
skew lenient; the follow-up is multi-rater labels, then tuning the pass bar on them (§10).
**Why 0.70 stays for now:** the two error types are not symmetric in a draft-suggestion product — a
false FAIL costs an agent a quick edit, a false PASS risks a bad send — so we keep the conservative
bar until multi-rater labels justify moving it.

## 4. System results (60 held-out emails, production mode)

```
mean 0.831 · median 0.85 · pass-rate 0.90 (54/60)
dimensions (1–5): intent 3.87 · factual 4.83 · faithfulness 4.83 · completeness 3.42 · tone 4.72
flags: hallucination 1/60 · contradiction 0/60
human-reply baseline (same rubric): mean 0.487 · pass-rate 0.13
```

- **Statistical honesty:** a 0.90 pass-rate on n=60 carries a wide interval (Wilson 95% CI ≈
  0.80–0.95) — read it as "roughly nine in ten," not a third significant digit.
- **Real weak spot: completeness.** 5 of 6 failures are two-part questions where the reply drops one
  part; the 6th is a context-free angry email where the hallucination flag correctly fired.
  Per-dimension scoring makes this diagnosis possible — a blended number would say "0.83, fine".
- **The system out-scores the human replies (0.83 vs 0.49)** — real Twitter-support replies are terse
  and often deflect; partly legitimate, partly judge verbosity-lean (§5).
- **0.90 is the reference-free number.** With ground truth it falls to 0.42 — see §5.

**Closing the loop — a measured negative result.** We used the diagnosis above to test the obvious
fix: one added generator instruction (*"Covers EVERY distinct question or request in the email.
Before writing, silently enumerate them; a two-part email needs both parts answered, even
briefly."*), then re-ran the full report. Outcome: completeness **3.42 → 3.50** (barely moved),
pass-rate **0.90 → 0.85** — within the n=60 noise band. Per-item, the variant fixed only 1 of the 6
diagnosed failures, left 5 unchanged, and 4 previously-passing replies flipped to FAIL (one's
completeness *dropped* 4 → 2). Takeaway: an instruction-level nudge doesn't fix coverage —
resampling noise dominates — so the fix needs structure: extract the distinct asks first, then
draft against that list (§10). The variant's LLM calls are in the committed cache; re-adding that
line and re-running `src/report.py` reproduces the experiment at $0.

## 5. Honest limitations — including a measured blind spot

Rather than assert the production metric's limits, we **measured them** (`src/blindspot.py`):
every system reply was re-judged *with* the human reference, and verdicts compared.

- **Reference-free can't catch confidently-wrong replies: up to 29/60 (48%) flip to FAIL** when
  ground truth is added (pass 0.90 → 0.42, mean |Δ| 0.22). Spot-checks show genuine catches, not
  judge strictness: asking a customer to post an order number publicly where brand policy forbids
  it; "you've contacted the wrong company" to the right inbox; console menu paths invented for a
  streaming app. Root cause: **the generator doesn't know which brand it speaks for** beyond 3
  retrieved examples — and without ground truth the judge can't know either. 48% is an upper bound,
  but the direction is unambiguous; §4's near-ceiling faithfulness partly reflects this blind spot.
  **Production fixes:** (1) a brand/policy profile in the generator — the primary fix; (2) defer
  low-confidence emails to a human — but we *tested* the naive version (`src/confidence_check.py`):
  BM25 retrieval confidence does **not** predict the flips (AUC 0.43–0.48, ≈chance) — an email can
  match a ticket strongly and still be answered with the wrong brand's policy, so the gate must be
  brand-aware, i.e. fix (1); (3) reference-augmented audits continuously — in a real inbox the sent
  reply eventually exists, making this stronger check free retrospective labeling.
- **Verbosity lean.** LLM judges favor fuller answers; appropriately-terse replies score low (hence
  the human baseline). Mitigated by rubric anchors + forced justifications, not eliminated.
- **The judge is the ceiling.** Model-swappable (`MODEL_JUDGE`), cross-checked with Sonnet, but a
  systematically biased judge biases scores; run-to-run variance quantified in §3c. A specific
  instance: generator and judge share a model family (Haiku), so **self-preference bias** — LLM
  judges favoring their own family's outputs — likely inflates the system-vs-human gap in §4; the
  cross-family Sonnet check (corr 0.93) bounds it but does not eliminate it.
- **Dataset is Twitter-origin and single-turn**: short, sometimes low-context, no thread history
  (real inboxes are multi-turn conversations); subject lines are synthetic.

## 6. Dataset — real data, honestly documented

200 pairs (spec max), 140 KB / 60 test, from the **Customer Support on Twitter** corpus (real
customer messages + real agent replies, HF mirror `MohammadOthman/mo-customer-support-tweets-945k`),
sampled from 16 offsets spanning the full 945k rows, surface-cleaned and filtered to standalone
requests — **message bodies are never rewritten**. Full provenance and limitations:
[`data/DATASET.md`](data/DATASET.md).

*Why 200 and not more?* The KB **is** the graded dataset — indexing more while documenting 200 would
be dishonest. And volume from this corpus dilutes: only ~46% of raw rows survive quality filtering,
and the corpus's own replies score 0.49 on our rubric — few-shot copies what it retrieves. The test
split stops at 60 to keep a full fresh run under $1. In production the KB would be the customer's
full ticket history, quality-weighted (§10).

## 7. Response generator

**RAG + few-shot, no training**: pure-Python BM25 retrieves the 3 most similar past tickets; the
generator drafts a reply in the same house style, instructed to answer answerable questions directly
and invent nothing. *Why not fine-tuning?* No training-scale data, slower to iterate, hides the
grounding. *Why BM25 over embeddings?* Support tickets share salient surface vocabulary; zero
downloads, fully reproducible, and swappable behind `retrieve()` — §5 shows retrieval is the
highest-leverage upgrade target.

**Measured, not assumed: the retrieval ablation** (`src/ablation.py`). Zero-shot (no retrieved
examples) vs. RAG on all 60 test emails, same judge, both judging modes:

| | RAG (k=3) | zero-shot |
|---|---|---|
| reference-free: mean · pass | 0.831 · 0.90 | 0.853 · 0.88 |
| reference-augmented: mean · pass | 0.625 · 0.42 | 0.656 · 0.48 |

**Grounding as implemented adds no measurable quality** (differences are within the n=60 noise
band) — an honest null that coheres with two other findings: the KB's own replies score 0.49 on
the rubric (§6 — imitating them isn't valuable), and retrieval is brand-blind (§5 — a wrong-brand
example can inject wrong-brand substance). The conclusion is not "drop retrieval"; it's that
retrieval must earn its place via quality-weighted, brand-filtered KBs (§10) — and this ablation
is now the harness that will tell us whether any such change actually helps.

**How this maps to a production shared-inbox product.** The two halves of this challenge are the two
halves of a real support-AI suite: the generator is a reply-drafting copilot grounded in past
conversations; the judge is an automated QA layer scoring replies against an explicit rubric.
Production systems typically measure suggestion quality by *agent adoption* (did the agent insert or
copy the draft) — a strong live signal our metric complements rather than duplicates: rubric scoring
works *before* adoption data exists, and catches what adoption can't (a mediocre draft an agent
lazily sends still counts as "adopted"). Conversely, adoption events and eventually-sent human
replies are exactly the ground truth the §5 reference-augmented audit consumes. Every suggestion
here is scored, flagged, and treated as an editable starting point for a human — never an
auto-send.

## 8. Repository layout

```
src/build_dataset.py    fetch real TWCS pairs -> clean/filter/split -> data/dataset.jsonl
src/config.py           models, rubric weights, thresholds — all knobs in one place
src/llm.py              cached Anthropic client (.env loader, atomic disk cache, cost tracking)
src/retrieval.py        pure-Python BM25 retriever
src/generator.py        RAG + few-shot response generator
src/evaluator.py        rubric judge + hard flags + lexical baseline + aggregation   <- core
src/run.py              end-to-end demo: email -> reply -> score
src/report.py           whole-system scoring        -> results/report.json
src/validate_metric.py  the three validation experiments -> results/validation.json
src/blindspot.py        reference-free blind-spot audit  -> results/blindspot.json
src/ablation.py         retrieval ablation (RAG vs zero-shot) -> results/ablation.json
src/calibrate.py        blind human-calibration study    -> results/human_calibration.json
src/confidence_check.py test naive confidence gating (offline)  -> results/confidence_check.json
tests/                  offline unit tests (no API key)
data/DATASET.md         dataset provenance & limitations
results/                committed evidence (report, validation, blindspot)
```

## 9. How I used AI tools

- **Built with Claude Code (Claude Opus)**: designed the accuracy framework in dialogue, wrote the
  modules, and iterated on findings — smoke tests exposed an over-firing hard flag, an over-cautious
  generator prompt, and a flawed reliability measurement (correlation on a no-variance sample), each
  diagnosed and fixed.
- **A structured multi-angle code review** (/code-review, high effort) surfaced 14 real issues —
  few-shot leakage in the demo path, a sampling bug skewing the dataset toward the first 13% of the
  corpus, crash/portability edges — all fixed, dataset rebuilt afterwards.
- **Haiku 4.5** is generator + judge (cost); **Sonnet 5** is the independent cross-check judge. Key
  design calls (two judging modes, perturbation validation, hard flags outside the average) emerged
  from that dialogue and are documented here for scrutiny.

## 10. Future improvements

- Close the §5 blind spot: brand/policy profile for the generator plus brand-aware deferral (naive
  BM25-score gating tested and rejected — §5), and continuous reference-augmented audits once sent
  replies exist. Root cause is self-inflicted: `build_dataset.py` strips `@brandhandle` before the
  brand is recorded — keeping it as a `brand` field is a small builder change that unlocks
  brand-filtered retrieval.
- Scale the KB to a customer's full ticket history with quality-weighted retrieval (score past
  replies with this very rubric; prefer high scorers).
- Fix completeness structurally: a prompt-level nudge measurably failed (§4), so extract the
  email's distinct asks as a list first, then draft and self-check the reply against that list.
- Extend §3d's single-rater calibration to multiple raters and tune the pass threshold on those
  labels (0.70 looks conservative); length-controlled rubric anchors to debias verbosity; embedding
  or hybrid retrieval, measured with this same metric.
