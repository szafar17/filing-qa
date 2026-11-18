"""Configuration: YAML file + environment overrides, validated with pydantic.

Resolution order (later wins):
  1. defaults in the models below
  2. the YAML file (``--config`` flag, else $FILINGQA_CONFIG, else config/default.yaml)
  3. environment variables: FILINGQA_DATABASE_URL
Secrets (OPENAI_API_KEY, SEC_USER_AGENT) are only ever read from the environment,
which may be populated from a local, gitignored ``.env`` file.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = "config/default.yaml"

RetrievalMode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]
ALL_MODES: tuple[str, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank")


class DatabaseConfig(BaseModel):
    url: str = "postgresql://filingqa:filingqa@localhost:5433/filingqa"


class IngestConfig(BaseModel):
    companies_file: str = "config/companies.yaml"
    forms: list[str] = Field(default_factory=lambda: ["10-K", "10-Q"])
    start_date: date = date(2023, 7, 1)
    end_date: date = date(2024, 12, 31)
    include_amendments: bool = False
    raw_dir: str = "data/raw"
    max_requests_per_second: float = 8.0
    max_workers: int = 4
    max_filings: int | None = None

    @field_validator("max_requests_per_second")
    @classmethod
    def _sec_limit(cls, v: float) -> float:
        # SEC fair-access policy: at most 10 requests per second per client.
        if not 0 < v <= 10:
            raise ValueError("max_requests_per_second must be in (0, 10] (SEC fair-access limit)")
        return v


class ChunkingConfig(BaseModel):
    target_chars: int = 1600
    max_chars: int = 2400
    overlap_chars: int = 200
    min_chars: int = 200


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "sentence-transformers", "hashing"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    batch_size: int = 128
    query_prefix: str = ""
    document_prefix: str = ""
    max_retries: int = 5


class LLMConfig(BaseModel):
    provider: Literal["openai", "fake"] = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float | None = 0.0
    max_output_tokens: int = 700
    timeout_s: float = 30.0
    max_retries: int = 4


class RerankerConfig(BaseModel):
    provider: Literal["cross-encoder", "lexical", "none"] = "cross-encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_length: int = 512


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = "hybrid_rerank"
    top_k: int = 5
    candidates: int = 50
    rerank_candidates: int = 30
    rrf_k: int = 60
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    hnsw_ef_search: int = 100


class RefusalConfig(BaseModel):
    min_rerank_score: float = 0.10
    min_vector_similarity: float = 0.25
    min_bm25_score: float | None = None
    require_citations: bool = True


class JudgeConfig(LLMConfig):
    max_output_tokens: int = 1200


class EvalConfig(BaseModel):
    golden_path: str = "eval/golden.jsonl"
    thresholds_path: str = "eval/thresholds.json"
    results_dir: str = "results"
    modes: list[RetrievalMode] = Field(default_factory=lambda: list(ALL_MODES))
    answer_modes: list[RetrievalMode] = Field(default_factory=lambda: ["hybrid_rerank"])
    ks: list[int] = Field(default_factory=lambda: [1, 5, 10])
    min_verified: int = 100
    relevance_overlap: float = 0.5
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    refusal: RefusalConfig = Field(default_factory=RefusalConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    prices_file: str = "config/prices.yaml"
    config_path: str | None = None


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; never overrides variables already set."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | os.PathLike[str] | None = None, *, use_env: bool = True) -> AppConfig:
    if use_env:
        load_dotenv()
    cfg_path = path or os.environ.get("FILINGQA_CONFIG") or DEFAULT_CONFIG_PATH
    data: dict = {}
    p = Path(cfg_path)
    if p.is_file():
        data = yaml.safe_load(p.read_text()) or {}
    elif path is not None:
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    cfg = AppConfig.model_validate(data)
    cfg.config_path = str(cfg_path) if p.is_file() else None
    if use_env and os.environ.get("FILINGQA_DATABASE_URL"):
        cfg.database.url = os.environ["FILINGQA_DATABASE_URL"]
    return cfg
