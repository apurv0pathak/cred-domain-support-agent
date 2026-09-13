"""
Structured JSONL request logging for the Cred Domain Support Agent.

Phase 3 Task 12:
- Exactly one JSONL record per API request.
- Request text is masked with the frozen Phase 2 mask_pii().
- No raw fixed-format PII is written to the log.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.guardrails import mask_pii


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIRECTORY = Path("transcripts/logs")
LOG_FILE = LOG_DIRECTORY / "api_requests.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_trace_id() -> str:
    """Create a unique trace ID for one API request."""

    return str(uuid.uuid4())


def log_request(
    *,
    trace_id: str,
    method: str,
    path: str,
    request_text: str | None,
    status_code: int,
    duration_ms: float,
    route: str | None = None,
    grounded: bool | None = None,
    error: str | None = None,
) -> None:
    """Write exactly one structured JSONL record for an API request."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    masked_request_text = (
        mask_pii(request_text)
        if request_text is not None
        else None
    )

    entry: dict[str, Any] = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "masked_request_text": masked_request_text,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
        "route": route,
        "grounded": grounded,
        "error": error,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")