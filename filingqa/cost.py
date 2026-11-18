"""Token price table and cost arithmetic.

Cost of one call = input_tokens * input_price + output_tokens * output_price, with
prices in USD per 1M tokens from config/prices.yaml. Local and fake models cost 0.
Unknown models are reported (not silently priced at 0) so a $/query figure is
never quoted for a model that is missing from the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

FREE_PROVIDERS = {"hashing", "sentence-transformers", "fake", "lexical", "cross-encoder", "none"}


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float = 0.0
    cached_input: float | None = None


@dataclass
class PriceTable:
    models: dict[str, ModelPrice] = field(default_factory=dict)
    as_of: str | None = None
    source: str | None = None
    unpriced: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: str | Path) -> PriceTable:
        data = yaml.safe_load(Path(path).read_text()) or {}
        models = {
            name: ModelPrice(
                input=float(p.get("input", 0.0)),
                output=float(p.get("output", 0.0)),
                cached_input=(float(p["cached_input"]) if "cached_input" in p else None),
            )
            for name, p in (data.get("models") or {}).items()
        }
        return cls(models=models, as_of=data.get("as_of"), source=data.get("source"))

    def cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        *,
        provider: str | None = None,
        cached_input_tokens: int = 0,
    ) -> float:
        if provider in FREE_PROVIDERS:
            return 0.0
        price = self.models.get(model)
        if price is None:
            # Try the base name for dated snapshots, e.g. gpt-4.1-mini-2025-04-14.
            for name, p in self.models.items():
                if model.startswith(name + "-"):
                    price = p
                    break
        if price is None:
            self.unpriced.add(model)
            return 0.0
        uncached = max(input_tokens - cached_input_tokens, 0)
        cached_rate = price.cached_input if price.cached_input is not None else price.input
        return (
            uncached * price.input + cached_input_tokens * cached_rate + output_tokens * price.output
        ) / 1_000_000
