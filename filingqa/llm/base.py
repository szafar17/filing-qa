"""Chat-LLM interface: one method that returns schema-constrained JSON plus usage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMOutputError(RuntimeError):
    """The model returned something that is not valid JSON for the requested schema."""


@dataclass
class LLMResult:
    data: dict[str, Any]
    raw: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached_input_tokens: int = 0


class LLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], schema_name: str
    ) -> LLMResult: ...
