from __future__ import annotations

from filingqa.config import LLMConfig
from filingqa.llm.base import LLM, LLMOutputError, LLMResult


def make_llm(cfg: LLMConfig) -> LLM:
    if cfg.provider == "fake":
        from filingqa.llm.fake import FakeLLM

        return FakeLLM(model=cfg.model)
    if cfg.provider == "openai":
        from filingqa.llm.openai_llm import OpenAIChatLLM

        return OpenAIChatLLM(
            cfg.model,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
    raise ValueError(f"unknown llm provider {cfg.provider}")


__all__ = ["LLM", "LLMOutputError", "LLMResult", "make_llm"]
