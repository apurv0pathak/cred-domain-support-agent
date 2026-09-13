"""
FastAPI wrapper for the frozen Cred Domain Support Agent.

Phase 3 Tasks 11-12:
- Exposes the existing Phase 2 agent through HTTP.
- Uses Pydantic request/response models.
- Writes exactly one masked JSONL log entry per API request.
- Does not replace or modify the Phase 2 agent.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from agent.graph import run_query
from agent.schema import AgentResponse
from observability.logging import create_trace_id, log_request


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cred Domain Support Agent",
    version="3.0.0",
    description="FastAPI wrapper around the frozen Phase 2 support agent.",
)


# ---------------------------------------------------------------------------
# Pydantic API models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    route: str
    escalation_score: float | None
    thread_id: str
    record_id: str | None
    grounded: bool | None


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = ConfigDict(extra="forbid")

    status: str
    service: str


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next: Any,
):
    """Log exactly one masked JSONL entry for every HTTP request."""

    trace_id = create_trace_id()
    start_time = time.perf_counter()

    request_text: str | None = None
    route: str | None = None
    grounded: bool | None = None
    status_code = 500
    error: str | None = None

    if request.method == "POST" and request.url.path == "/ask":
        try:
            body = await request.json()

            if isinstance(body, dict):
                query = body.get("query")

                if isinstance(query, str):
                    request_text = query

        except Exception:
            request_text = None

    try:
        response = await call_next(request)

        status_code = response.status_code

        return response

    except Exception as exc:
        error = type(exc).__name__
        raise

    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        log_request(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            request_text=request_text,
            status_code=status_code,
            duration_ms=duration_ms,
            route=route,
            grounded=grounded,
            error=error,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic service health information."""

    return HealthResponse(
        status="ok",
        service="cred-domain-support-agent",
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Run a query through the existing frozen Phase 2 agent.

    The Phase 2 graph remains responsible for:
    - input guardrails
    - memory
    - intent classification
    - RAG/status routing
    - structured response generation
    - persistence
    """

    result = run_query(
        query=request.query,
        thread_id=request.thread_id,
    )

    structured_response = result["structured_response"]

    # Validate the frozen Phase 2 structured response before
    # returning it through the API.
    agent_response = AgentResponse.model_validate(structured_response)

    return AskResponse(
        answer=agent_response.answer,
        route=agent_response.route,
        escalation_score=agent_response.escalation_score,
        thread_id=agent_response.thread_id,
        record_id=agent_response.record_id,
        grounded=agent_response.grounded,
    )