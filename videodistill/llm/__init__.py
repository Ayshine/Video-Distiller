"""LLM provider abstraction.

Pipeline code must talk to models ONLY through the ``LLMClient`` protocol in
:mod:`videodistill.llm.base`. Never import openai/anthropic SDKs from a stage —
that indirection is what lets us swap OpenAI → Anthropic → Bedrock later without
touching stage logic.
"""

from videodistill.llm.base import LLMClient, LLMMessage
from videodistill.llm.cache import CachedLLMClient

__all__ = ["LLMClient", "LLMMessage", "CachedLLMClient"]
