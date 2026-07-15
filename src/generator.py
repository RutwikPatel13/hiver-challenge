"""RAG + few-shot response generator.

Approach and trade-offs (see README for the full discussion):
  * We retrieve the k most similar *past resolved tickets* from the KB pool and
    show them to the model as worked examples, then ask it to draft a reply to
    the new email in the same house style.
  * Grounding in real past replies gives the model the brand's tone and the kinds
    of resolutions that are actually offered, without any training/fine-tuning.
  * We explicitly instruct the model NOT to invent specifics (order numbers,
    policies, prices) — the single biggest failure mode for support drafting — and
    to ask for missing info instead. The evaluator then checks whether it obeyed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataio  # noqa: E402
import llm  # noqa: E402
from retrieval import Retriever  # noqa: E402

SYSTEM = """You are a customer-support agent drafting a reply to an incoming email.
You will be shown a few past tickets (email + the reply that was actually sent) to
match the team's tone and the kinds of resolutions offered, then the new email.

Write a reply that:
- Directly addresses every point the customer raised and moves toward a resolution.
  If the customer asks a question you can reasonably answer, ANSWER it — do not
  reflexively ask for more information.
- Requests missing information only when it is genuinely required to proceed, and
  even then still gives whatever help or clear next step you can right now.
- Matches a warm, professional support tone; is empathetic on complaints.
- Stays grounded in the incoming email and the example resolutions. Do NOT invent
  order numbers, account details, prices, dates, policies, or support channels that
  are not present.
- Is concise — a few sentences, not a bulleted interrogation. Output only the reply
  body: no subject line, no "Draft:" preamble."""


def _format_examples(examples: list[dict]) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"--- Past ticket {i} ---\n"
            f"Email: {dataio.render_email(ex)}\n"
            f"Sent reply: {ex['reference_reply']}"
        )
    return "\n\n".join(blocks)


def generate_reply(email_text: str, retriever: Retriever, *,
                   k: int = config.RETRIEVAL_K, model: str | None = None,
                   exclude_id: str | None = None) -> dict:
    examples = retriever.retrieve(email_text, k, exclude_id=exclude_id)
    user = (
        f"{_format_examples(examples)}\n\n"
        f"=== New incoming email ===\n{email_text}\n\n"
        f"Write the reply body now."
    )
    reply = llm.generate(SYSTEM, user, model=model).strip()
    return {"reply": reply, "retrieved_ids": [e["id"] for e in examples]}


def build_retriever() -> Retriever:
    return Retriever(dataio.split("kb"))
