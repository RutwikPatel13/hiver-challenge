"""Thin, cached wrapper around the Anthropic Messages API.

Design goals:
  * Zero-cost re-runs. Every call is keyed by a hash of its full request and
    cached to disk, so iterating on downstream code never re-bills the API.
  * Honest cost accounting. Running token spend is tracked and printable.
  * Structured output. `judge_json` constrains the model to a JSON schema so the
    evaluator never has to regex model prose.
  * Per-model quirks handled in one place (sampling params, thinking).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# .env loading (no external dependency)
# ---------------------------------------------------------------------------
def _load_env() -> None:
    env_path = config.ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):  # tolerate shell-style .env files
            line = line[len("export "):]
        key, _, val = line.partition("=")
        val = val.strip()
        # tolerate quoted values (KEY="..." / KEY='...'), a common dotenv format
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


_load_env()

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) — sticker prices, used for the running estimate.
# ---------------------------------------------------------------------------
_PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}

_usage = {"input": 0, "output": 0, "cost": 0.0, "cached_hits": 0, "api_calls": 0}
_warned_temp_drop: set[str] = set()

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Copy .env.example to .env and paste your key."
            )
        _client = anthropic.Anthropic()
    return _client


def _cache_path(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def _hash_request(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _account(model: str, usage) -> None:
    pin, pout = _PRICES.get(model, (1.0, 5.0))
    _usage["input"] += usage.input_tokens
    _usage["output"] += usage.output_tokens
    _usage["cost"] += usage.input_tokens / 1e6 * pin + usage.output_tokens / 1e6 * pout


def _call(model: str, system: str, messages: list[dict], max_tokens: int,
          schema: dict | None, temperature: float | None, salt: str) -> Any:
    """Core cached call. Returns dict (if schema) or str."""
    req = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
        "schema": schema,
        "temperature": temperature,
        "salt": salt,
    }
    key = _hash_request(req)
    cpath = _cache_path(key)
    if cpath.exists():
        try:
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            _usage["cached_hits"] += 1
            return cached["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            # corrupt cache entry (e.g. a killed run mid-write) — treat as a miss
            # and overwrite below rather than crashing every future run
            cpath.unlink(missing_ok=True)

    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if schema is not None:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    # Per-model quirks. Only some families accept sampling params (see config);
    # Sonnet 5 rejects them and runs thinking-on by default, so we disable thinking
    # to keep the cross-check cheap and stable.
    temp_capable = model.startswith(config.TEMPERATURE_CAPABLE_PREFIXES)
    if temp_capable and temperature is not None:
        kwargs["temperature"] = temperature
    elif temperature not in (None, 0.0) and model not in _warned_temp_drop:
        # a caller explicitly wanted sampling variation but this model can't do it —
        # say so, or experiments relying on that variation silently measure nothing
        _warned_temp_drop.add(model)
        print(f"[llm] warning: temperature={temperature} requested but {model} does not "
              f"accept sampling params — sending without it", file=sys.stderr)
    if model == "claude-sonnet-5":
        kwargs["thinking"] = {"type": "disabled"}

    resp = client.messages.create(**kwargs)
    _account(model, resp.usage)
    _usage["api_calls"] += 1

    text = next((b.text for b in resp.content if b.type == "text"), "")
    if schema is not None and not text.strip():
        # refusal / truncation: don't let json.loads('') blow up a long run with a
        # cryptic error, and never cache the failure
        raise RuntimeError(
            f"model returned no text for a structured request "
            f"(model={model}, stop_reason={resp.stop_reason}) — cannot parse JSON"
        )
    value: Any = json.loads(text) if schema is not None else text
    # atomic write: a killed run must never leave a truncated cache entry
    tmp = cpath.with_suffix(".tmp")
    tmp.write_text(json.dumps({"value": value}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cpath)
    return value


def generate(system: str, user: str, *, model: str | None = None,
             max_tokens: int | None = None, temperature: float = 0.3) -> str:
    """Free-text completion (used by the response generator)."""
    return _call(
        model=model or config.MODEL_GENERATOR,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens or config.GEN_MAX_TOKENS,
        schema=None,
        temperature=temperature,
        salt="",
    )


def judge_json(system: str, user: str, schema: dict, *, model: str | None = None,
               max_tokens: int | None = None, temperature: float = 0.0,
               salt: str = "") -> dict:
    """Schema-constrained completion (used by the evaluator). `salt` lets the
    reliability test force N independent samples past the cache."""
    return _call(
        model=model or config.MODEL_JUDGE,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens or config.JUDGE_MAX_TOKENS,
        schema=schema,
        temperature=temperature,
        salt=salt,
    )


def usage_summary() -> str:
    return (
        f"api_calls={_usage['api_calls']} cache_hits={_usage['cached_hits']} "
        f"tokens_in={_usage['input']} tokens_out={_usage['output']} "
        f"est_cost=${_usage['cost']:.4f}"
    )
