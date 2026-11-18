"""Shared OpenAI client helpers: which errors are retryable and Retry-After parsing."""

from __future__ import annotations

from filingqa.retry import parse_retry_after


def retryable_errors() -> tuple[type[BaseException], ...]:
    import openai

    return (
        openai.RateLimitError,
        openai.APIConnectionError,  # includes APITimeoutError
        openai.InternalServerError,
    )


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return parse_retry_after(headers.get("retry-after"))


def make_client(timeout: float = 30.0):
    """OpenAI client with SDK retries disabled: our retry_call does it, visibly logged."""
    from openai import OpenAI

    return OpenAI(max_retries=0, timeout=timeout)
