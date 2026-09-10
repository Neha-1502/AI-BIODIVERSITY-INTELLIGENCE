"""
llm_client.py

Phase 3 - swappable LLM backend. Reads which provider to use from environment
variables so you can switch between Claude and GPT-4 without touching any
other code.

Setup:
  1. Copy .env.example to .env
  2. Fill in your API key(s)
  3. Set LLM_PROVIDER to "anthropic" or "openai"

Usage from other modules:
    from llm_client import chat
    reply_text = chat(system_prompt="...", user_prompt="...", json_mode=True)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def _chat_anthropic(system_prompt: str, user_prompt: str, json_mode: bool) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    config_kwargs = {"system_instruction": system_prompt}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text

def chat(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Main entry point. Returns the raw text response from whichever provider
    is configured. If json_mode=True, strips markdown code fences as a safety
    net in case the model wraps its JSON output anyway."""
    if PROVIDER == "anthropic":
        raw = _chat_anthropic(system_prompt, user_prompt, json_mode)
    elif PROVIDER == "openai":
        raw = _chat_openai(system_prompt, user_prompt, json_mode)
    elif PROVIDER == "gemini":
        raw = _chat_gemini(system_prompt, user_prompt, json_mode)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{PROVIDER}'. Use 'anthropic', 'openai', or 'gemini'.")

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