"""Structured response schema for the Cred support agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentResponse(BaseModel):
    """Required structured output for every agent response."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    route: Literal["rag", "status"]
    escalation_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    thread_id: str = Field(min_length=1)
    record_id: str | None = None
    grounded: bool | None = None


def validate_agent_response(
    response: dict,
) -> AgentResponse:
    """Validate and return a structured agent response."""

    return AgentResponse.model_validate(response)