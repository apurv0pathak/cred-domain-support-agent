"""Simple deterministic JSON conversation memory for the Cred support agent.

Memory is persisted locally and separated by thread ID.

Storage format:

    transcripts/memory/<thread_id>.json

Each file contains:

    {
        "thread_id": "...",
        "messages": [
            {
                "role": "user",
                "content": "..."
            },
            {
                "role": "assistant",
                "content": "..."
            }
        ]
    }

No external service, database, API key, or network access is required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MEMORY_DIRECTORY = Path("transcripts") / "memory"

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_thread_id(thread_id: str) -> None:
    """Reject unsafe thread IDs before using them as filenames."""

    if not thread_id:
        raise ValueError("thread_id must not be empty.")

    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ValueError(
            "thread_id may contain only letters, numbers, "
            "underscore, hyphen, and period."
        )


def _memory_path(thread_id: str) -> Path:
    """Return the JSON storage path for a thread."""

    _validate_thread_id(thread_id)
    return MEMORY_DIRECTORY / f"{thread_id}.json"


def load_history(thread_id: str) -> list[dict[str, str]]:
    """Load conversation history for a thread.

    A new thread has no history and therefore returns an empty list.
    """

    path = _memory_path(thread_id)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("thread_id") != thread_id:
        raise ValueError(
            f"Memory file contains a different thread ID: {path}"
        )

    history = payload.get("messages", [])

    if not isinstance(history, list):
        raise ValueError(f"Invalid messages list in {path}")

    return history


def save_history(
    thread_id: str,
    messages: list[dict[str, str]],
) -> None:
    """Persist complete conversation history for a thread."""

    path = _memory_path(thread_id)
    MEMORY_DIRECTORY.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "messages": messages,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_message(
    thread_id: str,
    role: str,
    content: str,
) -> list[dict[str, str]]:
    """Append one message and persist the updated history."""

    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'.")

    history = load_history(thread_id)

    history.append(
        {
            "role": role,
            "content": content,
        }
    )

    save_history(thread_id, history)

    return history