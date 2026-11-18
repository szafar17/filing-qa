"""Per-request trace: stage latencies, token usage, and cost, logged as one JSON record."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    stages_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, Usage] = field(default_factory=dict)  # keyed by purpose: embed, answer, ...
    meta: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_stage(name, (time.perf_counter() - t0) * 1000)

    def add_stage(self, name: str, ms: float) -> None:
        self.stages_ms[name] = round(self.stages_ms.get(name, 0.0) + ms, 3)

    def add_usage(
        self,
        purpose: str,
        *,
        model: str,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        u = self.usage.setdefault(purpose, Usage(model=model, provider=provider))
        u.input_tokens += int(input_tokens or 0)
        u.output_tokens += int(output_tokens or 0)
        u.cost_usd += float(cost_usd or 0.0)
        u.calls += 1

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 3)

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stages_ms": dict(self.stages_ms),
            "total_ms": self.total_ms,
            "tokens": {
                k: {"model": u.model, "input": u.input_tokens, "output": u.output_tokens}
                for k, u in self.usage.items()
            },
            "cost_usd": round(self.cost_usd, 8),
            **self.meta,
        }
