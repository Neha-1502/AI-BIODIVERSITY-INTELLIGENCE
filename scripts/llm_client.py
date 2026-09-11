"""
llm_client.py

Phase 3 - swappable LLM backend. Reads which provider to use from environment
variables so you can switch between Claude, GPT-4, Gemini, or Groq without
touching any other code.

Setup:
  1. Copy .env.example to .env
  2. Fill in your API key(s)
  3. Set LLM_PROVIDER to "anthropic", "openai", "gemini", or "groq"

Groq is free (no credit card): sign up at https://console.groq.com,
create a key at https://console.groq.com/keys, set GROQ_API_KEY in .env
and LLM_PROVIDER=groq. Its API is OpenAI-compatible, so no new SDK is
required.

Usage from other modules:
    from llm_client import chat
    reply_text = chat(system_prompt="...", user_prompt="...", json_mode=True)
"""

import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str | None = None) -> str | None:
    """Read config from env, then Streamlit Cloud secrets (st.secrets)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


PROVIDER = (_secret("LLM_PROVIDER", "gemini") or "gemini").lower()
# Optional: a second provider to try automatically if PROVIDER is rate-limited
# or otherwise unavailable after MAX_RETRIES. Leave unset to disable fallback.
# e.g. LLM_PROVIDER=groq / FALLBACK_LLM_PROVIDER=gemini
FALLBACK_PROVIDER = (_secret("FALLBACK_LLM_PROVIDER", "") or "").lower() or None
ANTHROPIC_MODEL = _secret("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = _secret("OPENAI_MODEL", "gpt-4o")
GROQ_MODEL = _secret("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_RETRIES = 2  # total attempts = MAX_RETRIES + 1
BASE_BACKOFF_SECONDS = 3


class LLMUnavailableError(Exception):
    """Raised when the configured LLM provider fails after retries (e.g. a
    persistent rate limit / quota exhaustion). Callers should catch this
    specifically to show the user a clear, honest message or fall back to
    a non-LLM path, rather than letting a raw provider traceback surface."""


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "resource_exhausted", "rate limit", "quota"))


def _parse_retry_delay_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of a provider-suggested retry delay,
    e.g. Gemini's "'retryDelay': '22s'" or "Please retry in 22.9s"."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if not match:
        match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None



def _chat_anthropic(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))

    effective_system = system_prompt
    if json_mode:
        effective_system += (
            "\n\nIMPORTANT: Respond with ONLY valid JSON. No preamble, no "
            "markdown code fences, no explanation before or after the JSON."
        )

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=effective_system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def _chat_openai(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_secret("OPENAI_API_KEY"))

    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        system_prompt += "\n\nRespond with ONLY valid JSON."

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
        **kwargs,
    )
    return response.choices[0].message.content

def _chat_gemini(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_secret("GEMINI_API_KEY"))

    config_kwargs = {"system_instruction": system_prompt}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text


def _chat_groq(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    """Groq's API is OpenAI-compatible, so this reuses the openai SDK
    pointed at Groq's endpoint instead of adding a new dependency.
    Free tier: no credit card required, sign up at console.groq.com."""
    from openai import OpenAI

    client = OpenAI(
        api_key=_secret("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        system_prompt += "\n\nRespond with ONLY valid JSON."

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
        **kwargs,
    )
    return response.choices[0].message.content

def _dispatch(provider: str, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    if provider == "anthropic":
        return _chat_anthropic(system_prompt, user_prompt, json_mode)
    elif provider == "openai":
        return _chat_openai(system_prompt, user_prompt, json_mode)
    elif provider == "gemini":
        return _chat_gemini(system_prompt, user_prompt, json_mode)
    elif provider == "groq":
        return _chat_groq(system_prompt, user_prompt, json_mode)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic', 'openai', 'gemini', or 'groq'.")


def _chat_with_provider(provider: str, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    """Retries a single provider up to MAX_RETRIES times on rate-limit errors,
    honoring any provider-suggested retry delay. Raises LLMUnavailableError
    (chained from the last exception) if every attempt against this provider
    fails, or immediately for non-rate-limit errors (bad key, bad request,
    etc.) since retrying those wastes time without changing the outcome."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _dispatch(provider, system_prompt, user_prompt, json_mode)
        except Exception as e:
            last_exc = e
            if attempt >= MAX_RETRIES or not _is_rate_limit_error(e):
                raise LLMUnavailableError(
                    f"LLM provider '{provider}' failed after {attempt + 1} attempt(s): {e}"
                ) from e
            delay = _parse_retry_delay_seconds(e) or (BASE_BACKOFF_SECONDS * (2 ** attempt))
            print(f"[warning] {provider} call failed ({e}); retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES + 1})")
            time.sleep(delay)
    # Unreachable in practice (the loop always returns or raises above), but
    # guards against silently falling through if that ever changes.
    raise LLMUnavailableError(f"LLM provider '{provider}' failed after {MAX_RETRIES + 1} attempts: {last_exc}")


def chat(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Main entry point. Returns the raw text response from whichever provider
    is configured. If json_mode=True, strips markdown code fences as a safety
    net in case the model wraps its JSON output anyway.

    Tries PROVIDER first (with its own retry/backoff for rate limits). If
    PROVIDER is exhausted (e.g. free-tier quota hit) and FALLBACK_LLM_PROVIDER
    is set in the environment to a different provider, automatically retries
    the whole request against the fallback before giving up. This is what
    keeps the app answering with real LLM synthesis instead of dropping to
    the raw-structured-match fallback in reasoning.py every time one
    provider's free tier is temporarily rate-limited.

    Raises LLMUnavailableError only if every configured provider failed, so
    callers can distinguish "the model said nothing useful" from "the model
    was unreachable" instead of both looking like an empty result."""
    providers = [PROVIDER]
    if FALLBACK_PROVIDER and FALLBACK_PROVIDER != PROVIDER:
        providers.append(FALLBACK_PROVIDER)

    errors = []
    raw = None
    for i, provider in enumerate(providers):
        try:
            raw = _chat_with_provider(provider, system_prompt, user_prompt, json_mode)
            break
        except LLMUnavailableError as e:
            errors.append(str(e))
            if i < len(providers) - 1:
                print(f"[warning] '{provider}' unavailable, falling back to '{providers[i + 1]}'")
    else:
        raise LLMUnavailableError(
            "All configured LLM providers failed: " + " | ".join(errors)
        )

    if json_mode:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
    return raw


def chat_json(system_prompt: str, user_prompt: str) -> dict:
    """Convenience wrapper: calls chat() in json_mode and parses the result.
    Raises json.JSONDecodeError if the model didn't return valid JSON --
    catch this in the calling code and retry or surface a friendly error."""
    raw = chat(system_prompt, user_prompt, json_mode=True)
    return json.loads(raw)


if __name__ == "__main__":
    # Quick manual test -- requires a real API key in .env
    print(f"Testing with provider: {PROVIDER}")
    result = chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'connection successful' and nothing else.",
    )
    print(f"Response: {result}")