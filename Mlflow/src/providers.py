import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal
from dotenv import load_dotenv

from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

ProviderName = Literal["openai", "anthropic", "google", "groq"]


@dataclass
class ModelConfig:
    provider: ProviderName
    model_id: str


def _get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package is not installed") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def _get_anthropic_client():
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("anthropic package is not installed") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=api_key)


def _get_google_client():
    try:
        import google.genai as genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package is not installed") from exc

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
    genai.configure(api_key=api_key)
    return genai


def _get_groq_client():
    try:
        from groq import Groq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("groq package is not installed") from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return Groq(api_key=api_key)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3))
def call_model(
    provider: ProviderName,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 500,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    """
    Unified chat-completions style interface across providers.

    Returns a dict with:
      - text: str (model response)
      - latency_ms: float
      - raw: Any (raw provider response, JSON-serialisable where possible)
    """
    start = time.time()

    if provider == "openai":
        client = _get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        text = resp.choices[0].message.content or ""
        
        # OpenAI responses are pydantic models (model_dump) in recent SDKs
        raw = getattr(resp, "model_dump", lambda: resp)()

    elif provider == "anthropic":
        client = _get_anthropic_client()
        # Anthropic uses a different message shape
        system = ""
        user_parts: List[Dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            elif m["role"] == "user":
                user_parts.append({"type": "text", "text": m["content"]})

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or None,
            messages=[{"role": "user", "content": user_parts}],
        )
        # Concatenate all text blocks
        text = "".join(
            getattr(block, "text", "") for block in resp.content  # type: ignore[attr-defined]
        )
        # Newer Anthropic SDK uses pydantic-style model_dump
        raw = getattr(resp, "model_dump", lambda: str(resp))()

    elif provider == "google":
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

        client = genai.Client(api_key=api_key)

        # Merge system + user messages into a single prompt
        system_text = "\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        user_text = "\n".join(
            m["content"] for m in messages if m["role"] == "user"
        )
        full_prompt = (system_text + "\n\n" + user_text).strip()

        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": top_p,
            },
        )
        text = resp.text or ""
        raw = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)

    elif provider == "groq":
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        text = resp.choices[0].message.content or ""
        # Groq response objects are simple dataclasses; __dict__ is fine
        raw = getattr(resp, "__dict__", str(resp))
        
    else:
        raise ValueError(f"Unknown provider: {provider}")

    latency_ms = (time.time() - start) * 1000.0
    return {
        "text": text,
        "latency_ms": latency_ms,
        "raw": raw,
    }

