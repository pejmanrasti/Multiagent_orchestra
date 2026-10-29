"""
llm.py — the one place that talks to the LLM provider (OpenRouter by default).

MODELS ARE NEVER CHOSEN HERE. Each agent role has its own model, set only by
configure_models(), which the interactive selection (model_selection.py) calls
after the user has picked and confirmed a model for every role. Calling an agent
before that raises an error; there is no default and no automatic fallback.

    roles:  "strategist"  (Model Strategist)
            "coder"       (Coder Agent)
            "analyst"     (Error Analysis Agent)

API key: read from the environment at call time:
    export OPENROUTER_API_KEY=sk-or-v1-...      (LLM_API_KEY also accepted)
If it isn't there, the selection dialogue asks for it (hidden input) and hands it to
set_session_key(): it then lives only in this process's memory, is never written
to disk or into os.environ, and is stripped from the training subprocesses.

Paid models: a model whose OpenRouter price is not zero can only be configured
when it is explicitly listed in `approved_paid` (the user confirmed it), and every
call re-checks that, so no paid request can happen without approval.

Reliability:
  - exponential backoff with jitter (honours Retry-After) for rate limits (429),
    timeouts, 5xx and empty provider responses; no retry for auth/credit/bad-model
    errors, which are reported as-is;
  - call_json() validates against the Pydantic schema and re-asks on malformed JSON;
  - every call is logged in USAGE: role, requested + served model, latency, retries,
    token counts (written to the run folder by loop.py).
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

ROLES = ("strategist", "coder", "analyst")
ROLE_LABELS = {"strategist": "Model Strategist", "coder": "Coder Agent",
               "analyst": "Error Analysis Agent"}

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
KEY_ENV_VARS = ("OPENROUTER_API_KEY", "LLM_API_KEY")

MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8000"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))      # transient-error retries
BACKOFF_BASE = float(os.environ.get("LLM_BACKOFF_BASE", "2"))  # seconds
BACKOFF_MAX = float(os.environ.get("LLM_BACKOFF_MAX", "60"))
JSON_ATTEMPTS = 3
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_session_key: Optional[str] = None    # key typed by the user (memory only)
_models: Dict[str, str] = {}          # role -> model id (set only by configure_models)
_free: set = set()                    # model ids known to be free
_paid_approved: set = set()           # paid model ids the user explicitly approved
USAGE: List[dict] = []                # one row per call (see _call)
_sleep = time.sleep                   # patched in tests


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def api_key() -> Optional[str]:
    for var in KEY_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return _session_key


def set_session_key(key: Optional[str]) -> None:
    """Use a key the user typed in, for this process only."""
    global _session_key
    _session_key = (key or "").strip() or None


def configure_models(models: Dict[str, str], *, free_models: Iterable[str],
                     approved_paid: Iterable[str] = ()) -> None:
    """Assign the user's chosen model to every role. Raises if any role is missing
    or a non-free model has not been explicitly approved."""
    missing = [r for r in ROLES if not (models.get(r) or "").strip()]
    if missing:
        raise ValueError(f"No model selected for: {', '.join(ROLE_LABELS[r] for r in missing)}")
    free, approved = set(free_models), set(approved_paid)
    unapproved = sorted({m for m in models.values() if m not in free and m not in approved})
    if unapproved:
        raise PermissionError(f"Paid model(s) not approved by the user: {', '.join(unapproved)}")
    _models.clear()
    _models.update({r: models[r].strip() for r in ROLES})
    _free.clear(); _free.update(free)
    _paid_approved.clear(); _paid_approved.update(approved)


def clear_models() -> None:
    _models.clear(); _free.clear(); _paid_approved.clear()


def selected_models() -> Dict[str, str]:
    return dict(_models)


def models_configured() -> bool:
    return all(r in _models for r in ROLES)


def model_for(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown agent role {role!r}")
    if role not in _models:
        raise RuntimeError(f"No LLM selected for the {ROLE_LABELS[role]}. Models are never "
                           f"chosen automatically; run the model selection first.")
    return _models[role]


def _check_allowed(model: str) -> None:
    if model not in _free and model not in _paid_approved:
        raise PermissionError(f"'{model}' is not a free model and was not approved for paid "
                              f"calls; refusing to call it.")


# --------------------------------------------------------------------------- #
# provider call (one attempt) — the only provider-specific code
# --------------------------------------------------------------------------- #
class EmptyResponse(RuntimeError):
    """Provider answered without content (common on busy free endpoints)."""


def _client():
    from openai import OpenAI  # lazy import: only needed once a call is made
    key = api_key()
    if not key:
        raise RuntimeError("No API key found. Set OPENROUTER_API_KEY in your environment or "
                           "enter it when asked.")
    headers = {"HTTP-Referer": "https://localhost", "X-Title": "ml-agent"}
    # max_retries=0: retries/backoff are handled (and logged) by _call below.
    return OpenAI(base_url=BASE_URL, api_key=key, default_headers=headers, max_retries=0)


def _complete(model: str, system: str, user: str) -> Tuple[str, dict]:
    """One request. Returns (text, info) with token usage and the served model."""
    resp = _client().chat.completions.create(
        model=model, max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    choices = getattr(resp, "choices", None)
    if not choices or not choices[0].message.content:
        err = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
        raise EmptyResponse(f"empty response from provider{f' ({err})' if err else ''}")
    u = getattr(resp, "usage", None)
    info = {"served_model": getattr(resp, "model", None),
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None)}
    return choices[0].message.content, info


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, EmptyResponse):
        return True
    status = getattr(e, "status_code", None)
    if status is not None:
        return status in RETRY_STATUS
    return type(e).__name__ in ("APIConnectionError", "APITimeoutError", "Timeout",
                                "ConnectionError")


def _retry_delay(attempt: int, e: Exception) -> float:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        ra = float(headers.get("retry-after"))
        if ra >= 0:
            return min(ra, BACKOFF_MAX)
    except (TypeError, ValueError):
        pass
    return min(BACKOFF_MAX, BACKOFF_BASE * 2 ** attempt) * (0.5 + random.random() / 2)


def _call(role: str, model: str, system: str, user: str) -> str:
    """Call `model` with backoff on transient errors; log usage and latency."""
    _check_allowed(model)
    row = {"role": role, "model": model, "ok": False, "attempts": 0, "retries": 0,
           "latency_s": None, "total_time_s": None, "served_model": None,
           "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
           "error": "", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t_start = time.time()
    try:
        for attempt in range(MAX_RETRIES + 1):
            row["attempts"] = attempt + 1
            t0 = time.time()
            try:
                text, info = _complete(model, system, user)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                if not _is_retryable(e) or attempt == MAX_RETRIES:
                    raise RuntimeError(f"{ROLE_LABELS.get(role, role)} call to '{model}' failed: "
                                       f"{row['error']}") from e
                row["retries"] += 1
                _sleep(_retry_delay(attempt, e))
                continue
            row.update(info, ok=True, error="", latency_s=round(time.time() - t0, 3))
            served = info.get("served_model")
            if served and served.split(":")[0] != model.split(":")[0]:
                row["note"] = f"provider served '{served}' for requested '{model}'"
            return text
    finally:
        row["total_time_s"] = round(time.time() - t_start, 3)
        USAGE.append(row)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# public entry points used by the agents
# --------------------------------------------------------------------------- #
def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        for lang in ("python", "py", "json"):
            if text.startswith(lang):
                text = text[len(lang):]
    return text.strip("` \n")


def call_text(role: str, system: str, user: str) -> str:
    return _strip_code_fence(_call(role, model_for(role), system, user))


def call_json(role: str, system: str, user: str, schema: Type[T]) -> T:
    """Ask for JSON matching `schema`; re-ask with the validation error if malformed."""
    fields = json.dumps(schema.model_json_schema().get("properties", {}), indent=2)
    prompt = f"{user}\n\nReply with ONLY a JSON object matching these fields:\n{fields}"
    problem = ""
    for _ in range(JSON_ATTEMPTS):
        raw = _strip_code_fence(_call(role, model_for(role), system, prompt))
        s, e = raw.find("{"), raw.rfind("}")
        try:
            if s < 0 or e <= s:
                raise json.JSONDecodeError("no JSON object found", raw, 0)
            return schema.model_validate(json.loads(raw[s:e + 1]))
        except json.JSONDecodeError as err:
            problem = f"not valid JSON ({err.msg})"
        except ValidationError as err:
            problem = "schema violations: " + "; ".join(
                f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in err.errors()[:5])
        USAGE[-1]["json_error"] = problem
        prompt += f"\n\nYour previous reply was rejected ({problem}). Return ONLY the JSON object."
    raise RuntimeError(f"{ROLE_LABELS[role]} ('{model_for(role)}') did not return valid "
                       f"{schema.__name__} JSON after {JSON_ATTEMPTS} attempts: {problem}")


def test_model(model: str) -> Tuple[bool, str]:
    """One tiny request to check the key + model work (counts as a logged call)."""
    try:
        text = _call("selection_check", model, "You are a connection test.",
                     "Reply with the single word: OK")
        return True, text.strip()[:60]
    except Exception as e:
        return False, str(e)[:240]


# --------------------------------------------------------------------------- #
# usage tracking
# --------------------------------------------------------------------------- #
def reset_usage() -> None:
    USAGE.clear()


def usage_summary(rows: Optional[List[dict]] = None) -> Dict[str, dict]:
    """Per role: calls, failures, retries, tokens, latency."""
    out: Dict[str, dict] = {}
    for r in (USAGE if rows is None else rows):
        s = out.setdefault(r["role"], {"model": r["model"], "calls": 0, "failed": 0, "retries": 0,
                                       "prompt_tokens": 0, "completion_tokens": 0,
                                       "total_tokens": 0, "latency_s": []})
        s["calls"] += 1
        s["failed"] += 0 if r["ok"] else 1
        s["retries"] += r["retries"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            s[k] += r.get(k) or 0
        if r["latency_s"] is not None:
            s["latency_s"].append(r["latency_s"])
    for s in out.values():
        lat = s.pop("latency_s")
        s["mean_latency_s"] = round(sum(lat) / len(lat), 3) if lat else None
        s["max_latency_s"] = max(lat) if lat else None
    return out
