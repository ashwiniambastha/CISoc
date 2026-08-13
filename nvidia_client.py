"""
Single call path for the whole project.

batch_runner.py and dashboard.py both go through call_model() here, so there is
one rate limiter, one retry policy, and one place where a row is constructed.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import unicodedata
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

import config

# --------------------------------------------------------------------------
# Prompt bank
# --------------------------------------------------------------------------


def load_prompt_bank(path=None) -> Dict[str, Any]:
    """Load the bank and hash the file bytes (Appendix C.7.5).

    The hash is written into every log record. Elicitations made under
    different hashes are never pooled.
    """
    path = path or config.PROMPT_BANK_PATH
    raw = path.read_bytes()
    bank = json.loads(raw.decode("utf-8"))
    prompts = bank["prompts"]

    # Store and transmit NFC; verify rather than silently recompose.
    for p in prompts:
        if unicodedata.normalize("NFC", p["prompt_text"]) != p["prompt_text"]:
            raise ValueError(
                f"{p['prompt_id']} is not in Unicode NFC. Fix the bank file "
                "rather than normalising at call time."
            )

    ids = [p["prompt_id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prompt_id in bank.")

    bank["sha256"] = hashlib.sha256(raw).hexdigest()
    return bank


# --------------------------------------------------------------------------
# Rate limiting (spec section 5)
# --------------------------------------------------------------------------


class RateLimiter:
    """Rolling 60-second window, one instance per model.

    Lock-guarded, because the dashboard fires both models concurrently and the
    batch runner may be running in another thread.
    """

    def __init__(self, rpm_limit: int):
        self.rpm_limit = rpm_limit
        self.timestamps: deque = deque()
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        with self._lock:
            now = time.time()
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.rpm_limit:
                sleep_for = 60 - (now - self.timestamps[0]) + 0.5
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.time()
                while self.timestamps and now - self.timestamps[0] > 60:
                    self.timestamps.popleft()
            self.timestamps.append(time.time())


_LIMITERS: Dict[str, RateLimiter] = {
    name: RateLimiter(cfg["rpm_limit"]) for name, cfg in config.MODELS.items()
}

_CALL_COUNT = 0
_PAUSE_LOCK = threading.Lock()


def _flat_safety_pause(verbose: bool = True) -> None:
    """Layer 2: pause after every N calls regardless of RPM maths.

    The lock is held across the sleep on purpose. The cap is an account-level
    protection, not a per-model one, so when it trips every thread waits.
    """
    global _CALL_COUNT
    with _PAUSE_LOCK:
        _CALL_COUNT += 1
        if _CALL_COUNT % config.CALLS_BEFORE_PAUSE == 0:
            if verbose:
                print(
                    f"  [pause] {_CALL_COUNT} calls made, "
                    f"sleeping {config.PAUSE_SECONDS}s"
                )
            time.sleep(config.PAUSE_SECONDS)


def calls_made() -> int:
    return _CALL_COUNT


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

_CLIENT: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            base_url=config.BASE_URL,
            api_key=config.get_api_key(),
            timeout=config.REQUEST_TIMEOUT_SECONDS,
            max_retries=0,  # retries are ours, so every attempt gets logged
        )
    return _CLIENT


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blank_row() -> Dict[str, Any]:
    return {col: "" for col in config.RECORD_FIELDS}


def apply_frame(prompt_text: str, language: str) -> tuple:
    """Attach the response-length frame, if one is configured.

    Returns (prompt_sent, frame_id, frame_text). The frame is always in the
    prompt's own language and always a suffix, so placement and language never
    vary across the arms being compared.
    """
    if not config.USE_BREVITY_FRAME:
        return prompt_text, "none", ""

    frame = config.BREVITY_FRAMES.get(language)
    if not frame:
        # A prompt with no language tag (a custom dashboard prompt) gets no
        # frame rather than an English one silently applied to Hindi text.
        return prompt_text, "none", ""

    return (
        prompt_text + config.BREVITY_FRAME_SEPARATOR + frame,
        config.BREVITY_FRAME_ID,
        frame,
    )


# --------------------------------------------------------------------------
# The one call
# --------------------------------------------------------------------------


def call_model(
    model_name: str,
    prompt_text: str,
    condition: str = "default",
    replicate_index: int = 1,
    source: str = "batch",
    prompt_meta: Optional[Dict[str, Any]] = None,
    temperature_override: Optional[float] = None,
    bank_sha256: str = "",
    bank_version: str = "",
    order_seed: Any = "",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Issue one elicitation. Returns one row per attempt, in order.

    One call = one fresh session: no system prompt, no preamble, no history.
    The prompt text is the entire user message.

    condition:
      "default"      -> temperature and top_p left unset, provider default applies
      "determinism"  -> temperature=0, top_p=1, fixed seed where supported
      "manual"       -> dashboard runs with an explicit temperature_override

    On transport failure the call is retried once (spec 4.5). Both attempts are
    written; the retry never overwrites the failure it followed.
    """
    if model_name not in config.MODELS:
        raise KeyError(f"Unknown model_name {model_name!r}")

    model_cfg = config.MODELS[model_name]
    model_id = model_cfg["model_id"]
    meta = prompt_meta or {}

    prompt_sent, frame_id, frame_text = apply_frame(
        prompt_text, meta.get("language", "")
    )

    # Still one user message and nothing else: no system prompt, no preamble,
    # no prior turn. The frame, where present, is inside this string and is
    # logged verbatim in frame_text.
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_sent}],
        "max_tokens": config.MAX_TOKENS,
    }
    temperature_logged: Any = "unset"
    top_p_logged: Any = "unset"

    if condition == "determinism":
        kwargs["temperature"] = config.DETERMINISM_TEMPERATURE
        kwargs["top_p"] = config.DETERMINISM_TOP_P
        kwargs["seed"] = config.DETERMINISM_SEED
        temperature_logged = config.DETERMINISM_TEMPERATURE
        top_p_logged = config.DETERMINISM_TOP_P
    elif temperature_override is not None:
        kwargs["temperature"] = float(temperature_override)
        temperature_logged = float(temperature_override)
    # else: "default" -- nothing set, provider default applies.

    base_row = _blank_row()
    base_row.update(
        {
            "timestamp": _now(),
            "source": source,
            "prompt_id": meta.get("prompt_id", "custom"),
            "family": meta.get("family", ""),
            "arm": meta.get("arm", ""),
            "form": meta.get("form", ""),
            "paraphrase": meta.get("paraphrase", ""),
            "language": meta.get("language", ""),
            "model_name": model_name,
            "model_id_sent": model_id,
            "condition": condition,
            "frame_id": frame_id,
            "frame_text": frame_text,
            "replicate_index": replicate_index,
            "temperature": temperature_logged,
            "top_p": top_p_logged,
            "max_tokens": config.MAX_TOKENS,
            "order_seed": order_seed,
            "bank_version": bank_version,
            "bank_sha256": bank_sha256,
            "prompt_text": prompt_text,
            "prompt_sent": prompt_sent,
        }
    )

    client = get_client()
    rows: List[Dict[str, Any]] = []

    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        row = dict(base_row)
        row["record_id"] = str(uuid.uuid4())
        row["attempt"] = attempt
        row["timestamp"] = _now()

        _LIMITERS[model_name].wait_if_needed()
        started = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.time() - started

            choice = resp.choices[0]
            text = choice.message.content or ""
            # Some reasoning models return the answer alongside a trace; keep
            # the trace verbatim in its own field rather than discarding it.
            reasoning = getattr(choice.message, "reasoning_content", None)
            if not text and reasoning:
                text = reasoning

            finish = getattr(choice, "finish_reason", "") or ""
            usage = getattr(resp, "usage", None)

            row.update(
                {
                    "model_id_returned": getattr(resp, "model", "") or "",
                    "status": "success",
                    "finish_reason": finish,
                    "truncated": finish == "length",
                    "response_text": text,
                    "response_chars": len(text),
                    "prompt_tokens": getattr(usage, "prompt_tokens", "") if usage else "",
                    "completion_tokens": getattr(usage, "completion_tokens", "") if usage else "",
                    "total_tokens": getattr(usage, "total_tokens", "") if usage else "",
                    "latency_seconds": round(elapsed, 3),
                }
            )
            rows.append(row)
            _flat_safety_pause(verbose)
            return rows

        except Exception as exc:  # transport, rate-limit, provider error
            elapsed = time.time() - started
            err = f"{type(exc).__name__}: {exc}"
            row.update(
                {
                    "status": "failed",
                    "finish_reason": "error",
                    "truncated": False,
                    "response_text": err,
                    "response_chars": len(err),
                    "latency_seconds": round(elapsed, 3),
                }
            )
            rows.append(row)
            _flat_safety_pause(verbose)

            # An endpoint that rejects `seed` should not cost us the datum.
            if "seed" in err.lower() and "seed" in kwargs:
                kwargs.pop("seed")
                if verbose:
                    print("  [warn] endpoint rejected `seed`; retrying without it")

            if attempt < config.MAX_ATTEMPTS:
                if verbose:
                    print(f"  [retry] {meta.get('prompt_id', 'custom')} / {model_name}: {err[:120]}")
                time.sleep(config.RETRY_BACKOFF_SECONDS)
            else:
                if verbose:
                    print(f"  [failed] {meta.get('prompt_id', 'custom')} / {model_name}: {err[:120]}")

    return rows


def elicitation_plan() -> List[Dict[str, Any]]:
    """The 9 elicitations for one prompt x one model: 8 default + 1 at temp 0."""
    plan = [
        {"condition": "default", "replicate_index": i}
        for i in range(1, config.DEFAULT_REPLICATES + 1)
    ]
    plan += [
        {"condition": "determinism", "replicate_index": i}
        for i in range(1, config.DETERMINISM_REPLICATES + 1)
    ]
    return plan


if __name__ == "__main__":
    # Smoke test: one call per model, printed not logged.
    bank = load_prompt_bank()
    print(f"bank {bank['bank_version']}  {len(bank['prompts'])} prompts  "
          f"sha256={bank['sha256'][:16]}...")
    probe = bank["prompts"][12]  # K-K1a-EN
    for name in config.MODELS:
        print(f"\n--- {name} ---")
        for r in call_model(
            name,
            probe["prompt_text"],
            condition="default",
            prompt_meta=probe,
            source="smoke_test",
            bank_sha256=bank["sha256"],
            bank_version=bank["bank_version"],
        ):
            print(f"[{r['status']}] model_id_returned={r['model_id_returned']} "
                  f"tokens={r['total_tokens']} finish={r['finish_reason']} "
                  f"{r['latency_seconds']}s")
            print(r["response_text"][:500])
