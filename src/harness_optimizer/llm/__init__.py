"""Block B - LLM gateway: providers, cache, retries, cost accounting, tracing."""

from harness_optimizer.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    RetryableLLMError,
    Usage,
)
from harness_optimizer.llm.cache import MemoryCache, NullCache, SQLiteCache
from harness_optimizer.llm.gateway import CostLedger, LLMGateway, RoleStats
from harness_optimizer.llm.mock import MockLLM
from harness_optimizer.llm.ollama_client import OllamaClient

__all__ = [
    "CostLedger",
    "LLMClient",
    "LLMError",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "MemoryCache",
    "Message",
    "MockLLM",
    "NullCache",
    "OllamaClient",
    "RetryableLLMError",
    "RoleStats",
    "SQLiteCache",
    "Usage",
]
