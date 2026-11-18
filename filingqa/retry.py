"""Retry with exponential backoff and jitter, used for OpenAI and SEC calls."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def backoff_delay(attempt: int, base: float, max_delay: float, jitter: float) -> float:
    """Delay before retry number ``attempt`` (1-based): base * 2^(attempt-1), capped, +/- jitter."""
    delay = min(max_delay, base * (2 ** (attempt - 1)))
    if jitter:
        delay *= 1 + random.uniform(-jitter, jitter)
    return max(0.0, delay)


def retry_call(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    max_attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    jitter: float = 0.25,
    retry_after: Callable[[BaseException], float | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``fn`` until it succeeds, retrying only on ``retry_on`` exceptions.

    ``retry_after`` may extract a server-provided wait (e.g. a Retry-After header);
    when it returns a value, that wait is used instead of the computed backoff.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except retry_on as exc:
            if attempt >= max_attempts:
                raise
            delay = None
            if retry_after is not None:
                delay = retry_after(exc)
            if delay is None:
                delay = backoff_delay(attempt, base_delay, max_delay, jitter)
            delay = min(delay, max_delay)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            else:
                log.warning(
                    "retrying after error",
                    extra={
                        "data": {
                            "event": "retry",
                            "attempt": attempt,
                            "delay_s": round(delay, 3),
                            "error": type(exc).__name__,
                        }
                    },
                )
            sleep(delay)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
