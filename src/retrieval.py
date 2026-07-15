"""BM25 retrieval over the KB pool of past tickets.

Why lexical BM25 rather than neural embeddings: support tickets share a lot of
salient surface vocabulary (product names, error strings, "refund", "cancel"),
BM25 handles that well, it needs zero model downloads or extra dependencies, and
it is fully reproducible. It's the sensible default for a small, on-topic KB; the
retriever is isolated behind `retrieve()` so it can be swapped later.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [_tok(d) for d in docs]
        self.N = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.freqs = [Counter(d) for d in self.corpus]
        df: Counter = Counter()
        for d in self.corpus:
            df.update(set(d))
        # BM25 idf with the +1 smoothing that keeps values non-negative
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _tok(query)
        out = [0.0] * self.N
        for i in range(self.N):
            f = self.freqs[i]
            denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            s = 0.0
            for t in q:
                if t not in f:
                    continue
                s += self.idf.get(t, 0.0) * (f[t] * (self.k1 + 1)) / (f[t] + denom_norm)
            out[i] = s
        return out


class Retriever:
    def __init__(self, kb_records: list[dict]):
        self.records = kb_records
        # index on the customer message + subject — that's what a new email matches
        self.bm25 = BM25([f"{r['subject']} {r['customer_message']}" for r in kb_records])

    def retrieve(self, email_text: str, k: int, exclude_id: str | None = None) -> list[dict]:
        """Top-k most similar past tickets. `exclude_id` drops a specific ticket —
        used when the query email IS a KB ticket, so its own sent reply can never
        leak into its few-shot examples."""
        scores = self.bm25.scores(email_text)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if exclude_id is not None and self.records[i]["id"] == exclude_id:
                continue
            out.append(self.records[i])
            if len(out) == k:
                break
        return out
