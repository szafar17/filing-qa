"""OpenAI Chat Completions with Structured Outputs (json_schema, strict)."""

from __future__ import annotations

import json
import time
from typing import Any

from filingqa.llm.base import LLM, LLMOutputError, LLMResult
from filingqa.openai_common import make_client, retry_after_seconds, retryable_errors
from filingqa.retry import retry_call


class OpenAIChatLLM(LLM):
    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        temperature: float | None = 0.0,
        max_output_tokens: int = 700,
        timeout_s: float = 30.0,
        max_retries: int = 4,
        client: Any = None,
        sleep: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client
        self._sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client(timeout=self.timeout_s)
        return self._client

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], schema_name: str) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            "max_completion_tokens": self.max_output_tokens,
        }
        # Reasoning models (gpt-5*, o-series) reject a temperature parameter; set it to null
        # in config for those.
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        extra = {"sleep": self._sleep} if self._sleep else {}
        t0 = time.perf_counter()
        resp = retry_call(
            lambda: self.client.chat.completions.create(**kwargs),
            retry_on=retryable_errors(),
            max_attempts=self.max_retries,
            base_delay=0.5,
            max_delay=20.0,
            retry_after=retry_after_seconds,
            **extra,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        refusal = getattr(choice.message, "refusal", None)
        content = choice.message.content or ""
        if refusal:
            raise LLMOutputError(f"model refused to produce output: {refusal}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"model output is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMOutputError("model output is not a JSON object")
        return LLMResult(
            data=data,
            raw=content,
            model=getattr(resp, "model", None) or self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cached_input_tokens=cached,
        )
