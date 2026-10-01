"""
LLM Configuration and Provider Factory.

Supports:
- Google Gemini (langchain-google-genai)
- OpenAI (langchain-openai)
- Mock / Deterministic mode for zero-cost execution and testing
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_configured_llm(provider: Optional[str] = None, model: Optional[str] = None):
    """
    Returns a configured LangChain Chat model, or None if mock mode is requested.

    Args:
        provider: 'google', 'openai', 'mock', or None (reads LLM_PROVIDER env).
        model: Optional model name override.

    Returns:
        A BaseChatModel instance if configured with valid credentials, or None for mock mode.
    """
    resolved_provider = (
        provider or os.getenv("LLM_PROVIDER", "mock")
    ).strip().lower()

    if resolved_provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = model or os.getenv("MODEL_NAME", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )

    elif resolved_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        from langchain_openai import ChatOpenAI

        model_name = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.2,
        )

    # Default to mock/deterministic mode
    return None
