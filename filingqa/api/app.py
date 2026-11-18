"""FastAPI service.

    uvicorn filingqa.api.app:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \\
         -d '{"question": "What was Apple'"'"'s net sales in fiscal 2023?"}'
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from filingqa.answer.schema import AskRequest, AskResponse
from filingqa.answer.service import AnswerService
from filingqa.logging_utils import setup_logging
from filingqa.store.repository import SearchFilters


def create_app(service_factory: Callable[[], tuple[AnswerService, Any]] | None = None) -> FastAPI:
    """``service_factory`` returns (service, closeable); defaults to building from config."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        if service_factory is not None:
            service, closeable = service_factory()
        else:
            from filingqa.components import build_components
            from filingqa.config import load_config

            comps = build_components(load_config(), pooled=True)
            service, closeable = comps.service, comps.conn
        app.state.service = service
        try:
            yield
        finally:
            if closeable is not None and hasattr(closeable, "close"):
                closeable.close()

    app = FastAPI(title="Filing Q&A", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        service: AnswerService = app.state.service
        filters = None
        if req.filters is not None:
            filters = SearchFilters(
                tickers=req.filters.tickers,
                form_types=req.filters.form_types,
                fiscal_years=req.filters.fiscal_years,
            )
        try:
            outcome = await run_in_threadpool(
                service.ask, req.question, mode=req.mode, top_k=req.top_k, filters=filters
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return outcome.response

    return app


app = create_app()
