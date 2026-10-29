"""
model_selection.py — the user picks the LLM for each agent. Nothing here chooses
a model on the user's behalf:

  * the live OpenRouter catalogue is fetched and the FREE models are listed
    (name + exact identifier); `search <text>` lists any model, free or paid;
  * the user types an identifier (or a list number) for each of the three agents;
    empty input is never turned into a default;
  * every identifier is validated against the catalogue (must exist and produce text);
  * paid models and router aliases (which let OpenRouter pick the model) are flagged
    and need an explicit "yes" before they can be used;
  * each distinct model gets one tiny test call, and a failure sends the user back to
    choose (never an automatic substitute);
  * if no API key is in the environment, the user is asked for it (hidden input);
    it is checked with OpenRouter and kept in memory only, never saved;
  * the confirmed choice can be saved (models only, never the API key) and offered
    again next time, but it is only reused after the user confirms it.
"""
from __future__ import annotations

import getpass
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import llm
from llm import ROLE_LABELS, ROLES

CONFIG_PATH = Path.home() / ".ml_agent_config.json"
SHORT = {"strategist": "Strategist", "coder": "Coder", "analyst": "Error Analyst"}


@dataclass
class ModelInfo:
    id: str
    name: str
    prompt_price: float          # USD per token
    completion_price: float
    context_length: Optional[int]
    text_output: bool

    @property
    def free(self) -> bool:
        return self.prompt_price == 0 and self.completion_price == 0

    @property
    def is_router(self) -> bool:
        """Aliases where OpenRouter, not the user, decides which model answers."""
        return self.id.startswith("openrouter/") or self.id.startswith("~")

    def price_line(self) -> str:
        if self.free:
            return "free"
        return (f"${self.prompt_price * 1e6:.2f} / 1M input tokens, "
                f"${self.completion_price * 1e6:.2f} / 1M output tokens")


# --------------------------------------------------------------------------- #
# catalogue
# --------------------------------------------------------------------------- #
def parse_catalog(payload: dict) -> Dict[str, ModelInfo]:
    out = {}
    for m in payload.get("data", []):
        p = m.get("pricing") or {}
        try:
            prompt, completion = float(p.get("prompt", "nan")), float(p.get("completion", "nan"))
        except (TypeError, ValueError):
            prompt = completion = float("nan")
        if prompt != prompt or completion != completion or prompt < 0 or completion < 0:
            prompt = completion = float("inf")     # unknown/variable price: treat as paid
        mods = ((m.get("architecture") or {}).get("output_modalities")) or ["text"]
        out[m["id"]] = ModelInfo(m["id"], m.get("name") or m["id"], prompt, completion,
                                 m.get("context_length"), "text" in mods)
    return out


def fetch_catalog(timeout: float = 20) -> Dict[str, ModelInfo]:
    """Current model list from the provider's /models endpoint (no key needed)."""
    req = urllib.request.Request(f"{llm.BASE_URL.rstrip('/')}/models",
                                 headers={"User-Agent": "ml-agent"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_catalog(json.loads(r.read().decode("utf-8")))


def free_models(catalog: Dict[str, ModelInfo]) -> List[ModelInfo]:
    return sorted((m for m in catalog.values() if m.free and m.text_output and not m.is_router),
                  key=lambda m: m.name.lower())


def validate_model(model_id: str, catalog: Dict[str, ModelInfo]) -> str:
    """'' if usable, else the reason it is rejected."""
    if not model_id:
        return "no model entered"
    if model_id != model_id.strip() or " " in model_id:
        return "identifiers contain no spaces"
    m = catalog.get(model_id)
    if m is None:
        return f"'{model_id}' is not in the current OpenRouter model list"
    if not m.text_output:
        return f"'{model_id}' does not produce text output"
    return ""


# --------------------------------------------------------------------------- #
# saved configuration (models only — the API key stays in the environment)
# --------------------------------------------------------------------------- #
def load_saved(path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    path = path or CONFIG_PATH
    try:
        models = json.loads(path.read_text(encoding="utf-8")).get("agent_models") or {}
    except Exception:
        return None
    return models if all(models.get(r) for r in ROLES) else None


def save(models: Dict[str, str], path: Optional[Path] = None) -> None:
    """Writes ONLY the model choices (this also drops any API key an older version saved)."""
    path = path or CONFIG_PATH
    try:
        path.write_text(json.dumps({"agent_models": {r: models[r] for r in ROLES}}, indent=2), encoding="utf-8")
        path.chmod(0o600)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# interactive selection
# --------------------------------------------------------------------------- #
Ask = Callable[[str], str]
Say = Callable[[str], None]


def check_api_key(key: str, timeout: float = 15) -> Tuple[Optional[bool], str]:
    """Ask the provider whether the key is valid: (True|False|None if unknown, message)."""
    req = urllib.request.Request(f"{llm.BASE_URL.rstrip('/')}/key", headers={
        "Authorization": f"Bearer {key}", "User-Agent": "ml-agent"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True, "ok"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, "OpenRouter does not recognise this key"
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)[:160]


def _hidden_input(prompt: str) -> str:
    return getpass.getpass(f"\n{prompt}\n> ")


def ensure_api_key(ask: Ask, say: Say, ask_secret: Optional[Ask] = None,
                   key_checker: Optional[Callable[[str], Tuple[Optional[bool], str]]] = None) -> bool:
    """Make sure a key is available: from the environment, or typed in by the user."""
    if llm.api_key():
        return True
    ask_secret = ask_secret or _hidden_input
    key_checker = key_checker or check_api_key
    say("\nNo OpenRouter API key was found in your environment (OPENROUTER_API_KEY).")
    say("You can paste it here. Typing is hidden, and the key is kept in memory for this "
        "session only: it is not saved to disk and not passed to the generated code.")
    say("Get a key at https://openrouter.ai/keys. To skip this next time, set it in your "
        "shell: export OPENROUTER_API_KEY=sk-or-v1-...")
    while True:
        key = ask_secret("Paste your OpenRouter API key (press Enter to cancel):").strip()
        if not key:
            return False
        say("Checking the key with OpenRouter...")
        ok, msg = key_checker(key)
        if ok:
            llm.set_session_key(key)
            say("Key accepted (kept in memory for this session only).")
            return True
        if ok is None:
            say(f"I couldn't verify the key right now ({msg}).")
            if _yes_no(ask, say, "Use it anyway? (y/n)"):
                llm.set_session_key(key)
                return True
        else:
            say(f"That key was rejected: {msg}.")
        if not _yes_no(ask, say, "Try another key? (y/n)"):
            return False


def _yes_no(ask: Ask, say: Say, question: str) -> bool:
    """Explicit y/n; anything else (including Enter) is asked again."""
    while True:
        a = ask(question).strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        say("Please answer y or n.")


def _show_free(say: Say, free: List[ModelInfo]) -> None:
    say("Available free OpenRouter models:")
    if not free:
        say("  (none listed right now — type any identifier, or `search <text>`)")
    for i, m in enumerate(free, 1):
        say(f"  {i:>2}. {m.name} ({m.id})")
    say("Type a number from this list or any exact OpenRouter identifier. "
        "`search <text>` lists all matching models (free and paid); `list` shows this list again.")


def _search(say: Say, catalog: Dict[str, ModelInfo], text: str) -> None:
    text = text.lower()
    hits = sorted((m for m in catalog.values()
                   if text in m.id.lower() or text in m.name.lower()), key=lambda m: m.id)
    if not hits:
        say(f"  no models match '{text}'")
    for m in hits[:40]:
        tag = "FREE" if m.free else "paid"
        say(f"  [{tag}] {m.name} ({m.id})")
    if len(hits) > 40:
        say(f"  ... {len(hits) - 40} more; refine the search")


def _ask_model(role: str, ask: Ask, say: Say, catalog: Dict[str, ModelInfo],
               free: List[ModelInfo]) -> str:
    while True:
        raw = ask(f"{ROLE_LABELS[role]}: Which LLM model would you like to use?").strip()
        if raw.lower() == "list":
            _show_free(say, free); continue
        if raw.lower().startswith("search "):
            _search(say, catalog, raw[7:].strip()); continue
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(free):
                say(f"  -> {free[n - 1].id}")
                return free[n - 1].id
            say(f"  {n} is not a number from the list."); continue
        problem = validate_model(raw, catalog)
        if problem:
            say(f"  Rejected: {problem}. Please enter a valid identifier.")
            continue
        return raw


def _summary(say: Say, models: Dict[str, str], catalog: Dict[str, ModelInfo]) -> None:
    say("Selected configuration:")
    say("-" * 50)
    for r in ROLES:
        m = catalog.get(models[r])
        tag = "" if m is None else ("  (free)" if m.free else "  (PAID)")
        say(f"{SHORT[r] + ':':<16} {models[r]}{tag}")
    say("-" * 50)


def _confirm_special(ask: Ask, say: Say, models: Dict[str, str],
                     catalog: Dict[str, ModelInfo]) -> Optional[set]:
    """Paid models / routers need an explicit 'yes'. Returns approved paid ids, or None."""
    chosen = sorted(set(models.values()))
    paid = [catalog[m] for m in chosen if not catalog[m].free]
    routers = [catalog[m] for m in chosen if catalog[m].is_router]
    if routers:
        say("\nNote: these identifiers are routers/aliases. OpenRouter decides which "
            "underlying model answers, and it may change between calls:")
        for m in routers:
            say(f"  - {m.id}")
    if paid:
        say("\nWARNING: these models are NOT free. Using them will make paid API calls "
            "billed to your OpenRouter account:")
        for m in paid:
            say(f"  - {m.id}: {m.price_line()}")
    if not paid and not routers:
        return set()
    a = ask("Type 'yes' to allow this (anything else goes back to model selection)").strip().lower()
    return {m.id for m in paid} if a == "yes" else None


def select_models(ask: Ask, say: Say, *,
                  catalog_fetcher: Optional[Callable[[], Dict[str, ModelInfo]]] = None,
                  config_path: Optional[Path] = None, test_calls: bool = True,
                  ask_secret: Optional[Ask] = None) -> Optional[Dict[str, str]]:
    """Run the whole selection. Returns {role: model} after the user confirmed
    'Start optimization?', with llm configured; None if the user stops."""
    if not ensure_api_key(ask, say, ask_secret):
        say("No API key, so the agents can't run.")
        return None

    catalog_fetcher = catalog_fetcher or fetch_catalog
    while True:
        try:
            catalog = catalog_fetcher()
            break
        except Exception as e:
            say(f"Couldn't fetch the OpenRouter model list ({str(e)[:160]}). Models can't be "
                f"validated without it.")
            if not _yes_no(ask, say, "Try again? (y/n)"):
                return None
    free = free_models(catalog)

    say("\nSelect LLM models for your agents:")
    models: Optional[Dict[str, str]] = None
    saved = load_saved(config_path)
    if saved:
        say("You used this configuration last time:")
        for r in ROLES:
            say(f"  {SHORT[r] + ':':<16} {saved[r]}")
        bad = {r: p for r in ROLES if (p := validate_model(saved[r], catalog))}
        if bad:
            say("It can't be reused as is: " + "; ".join(bad.values()) + ".")
        elif _yes_no(ask, say, "Reuse this configuration? (y/n)"):
            models = dict(saved)

    while True:
        if models is None:
            _show_free(say, free)
            models = {r: _ask_model(r, ask, say, catalog, free) for r in ROLES}
        _summary(say, models, catalog)
        approved = _confirm_special(ask, say, models, catalog)
        if approved is None:
            say("Okay, let's choose again.")
            models = None
            continue
        llm.configure_models(models, free_models=[m.id for m in catalog.values() if m.free],
                             approved_paid=approved)
        if test_calls:
            failed = []
            for m in sorted(set(models.values())):
                say(f"Checking {m} ...")
                ok, msg = llm.test_model(m)
                say(f"  {'ok' if ok else 'FAILED: ' + msg}")
                if not ok:
                    failed.append(m)
            if failed:
                say("These models did not respond, so I won't use them. Nothing is switched "
                    "automatically (a free model may be busy; try later or pick another).")
                llm.clear_models()
                if not _yes_no(ask, say, "Choose different models? (y/n)"):
                    return None
                models = None
                continue
        if _yes_no(ask, say, "Start optimization? (y/n)"):
            save(models, config_path)
            return models
        llm.clear_models()
        if not _yes_no(ask, say, "Choose different models? (y/n)"):
            return None
        models = None
