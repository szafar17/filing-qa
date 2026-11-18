"""OpenAI adapters against fake client objects (no network, no API key)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import httpx
import numpy as np
import openai
import pytest

from filingqa.embeddings.openai_embedder import OpenAIEmbedder
from filingqa.llm.base import LLMOutputError
from filingqa.llm.openai_llm import OpenAIChatLLM


def rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.openai.com/v1/x"))
    return openai.RateLimitError("rate limited", response=resp, body=None)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def chat_response(content: str, refusal=None, cached=0):
    usage = NS(prompt_tokens=1200, completion_tokens=150, prompt_tokens_details=NS(cached_tokens=cached))
    return NS(
        model="gpt-4.1-mini-2025-04-14",
        usage=usage,
        choices=[NS(message=NS(content=content, refusal=refusal))],
    )


def make_llm(outcomes, **kw):
    comp = FakeCompletions(outcomes)
    client = NS(chat=NS(completions=comp))
    sleeps: list[float] = []
    llm = OpenAIChatLLM("gpt-4.1-mini", client=client, sleep=sleeps.append, **kw)
    return llm, comp, sleeps


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
}


def test_chat_request_uses_strict_json_schema():
    llm, comp, _ = make_llm([chat_response(json.dumps({"a": "x"}), cached=200)])
    res = llm.complete_json(system="sys", user="usr", schema=SCHEMA, schema_name="answer")
    kw = comp.kwargs[0]
    assert kw["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": SCHEMA, "strict": True},
    }
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["max_completion_tokens"] == 700 and kw["temperature"] == 0.0
    assert res.data == {"a": "x"} and res.input_tokens == 1200 and res.output_tokens == 150
    assert res.cached_input_tokens == 200 and res.model.startswith("gpt-4.1-mini")


def test_temperature_omitted_when_none():
    llm, comp, _ = make_llm([chat_response('{"a": "x"}')], temperature=None)
    llm.complete_json(system="s", user="u", schema=SCHEMA, schema_name="answer")
    assert "temperature" not in comp.kwargs[0]


def test_retries_rate_limit_with_backoff_then_succeeds():
    llm, comp, sleeps = make_llm([rate_limit_error("2"), rate_limit_error(), chat_response('{"a": "y"}')])
    assert llm.complete_json(system="s", user="u", schema=SCHEMA, schema_name="n").data == {"a": "y"}
    assert len(comp.kwargs) == 3
    assert sleeps[0] == 2.0  # Retry-After honoured
    assert 0.5 <= sleeps[1] <= 1.5  # then exponential backoff with jitter


def test_gives_up_after_max_retries():
    llm, _, _ = make_llm([rate_limit_error()] * 3, max_retries=3)
    with pytest.raises(openai.RateLimitError):
        llm.complete_json(system="s", user="u", schema=SCHEMA, schema_name="n")


def test_non_retryable_errors_propagate_immediately():
    resp = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/x"))
    llm, comp, _ = make_llm([openai.AuthenticationError("bad key", response=resp, body=None)])
    with pytest.raises(openai.AuthenticationError):
        llm.complete_json(system="s", user="u", schema=SCHEMA, schema_name="n")
    assert len(comp.kwargs) == 1


def test_invalid_json_and_refusals_raise_output_error():
    llm, _, _ = make_llm([chat_response("not json")])
    with pytest.raises(LLMOutputError):
        llm.complete_json(system="s", user="u", schema=SCHEMA, schema_name="n")
    llm, _, _ = make_llm([chat_response("", refusal="I can't help with that")])
    with pytest.raises(LLMOutputError, match="refused"):
        llm.complete_json(system="s", user="u", schema=SCHEMA, schema_name="n")


class FakeEmbeddings:
    def __init__(self, dim=4, fail_first=False):
        self.dim = dim
        self.calls = []
        self.fail_first = fail_first

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise rate_limit_error("1")
        inputs = kwargs["input"]
        # return out of order to check that we sort by index
        data = [NS(index=i, embedding=[float(i + 1)] * self.dim) for i in range(len(inputs))][::-1]
        return NS(data=data, usage=NS(prompt_tokens=10 * len(inputs), total_tokens=10 * len(inputs)))


def test_openai_embedder_batches_orders_and_normalizes():
    fake = FakeEmbeddings(dim=4, fail_first=True)
    emb = OpenAIEmbedder(
        "text-embedding-3-small", dimensions=4, batch_size=2, client=NS(embeddings=fake), sleep=lambda s: None
    )
    assert emb.model == "text-embedding-3-small@4" and emb.dim == 4
    out = emb.embed_documents(["a", "b", "c"])
    assert out.vectors.shape == (3, 4) and out.tokens == 30
    assert np.allclose(np.linalg.norm(out.vectors, axis=1), 1.0)
    # 1 failed + 2 batches (2 + 1 inputs); dimensions passed through
    assert [len(c["input"]) for c in fake.calls] == [2, 2, 1]
    assert all(c["dimensions"] == 4 for c in fake.calls)


def test_openai_embedder_default_dims_and_unknown_model():
    assert OpenAIEmbedder("text-embedding-3-small", client=object()).dim == 1536
    with pytest.raises(ValueError):
        OpenAIEmbedder("mystery-embedder", client=object())
