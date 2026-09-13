"""Input and output guardrails for the Cred support agent.

Phase 2 guardrails:

1. PII masking for PAN, Aadhaar, and bank-account-shaped strings.
2. Prompt-injection detection and refusal.
3. RAG groundedness enforcement using the frozen Phase 1
   generation result.

Names and income values are intentionally NOT masked because they
are explicitly out of scope for this project's PII requirement.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# PAN:
# Five uppercase letters + four digits + one uppercase letter.
#
# Example:
#     ABCDE1234F

PAN_PATTERN = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
)


# Aadhaar:
# For this deterministic demonstration we recognize the standard grouped
# representation:
#
#     1234 5678 9012
#     1234-5678-9012
#
# A compact 12-digit value is deliberately NOT classified automatically,
# because a compact 12-digit number is ambiguous with a bank-account number.

AADHAAR_GROUPED_PATTERN = re.compile(
    r"(?<![\d-])"
    r"\d{4}[\s-]\d{4}[\s-]\d{4}"
    r"(?![\d-])"
)


# Bank account:
# Recognize a digit sequence of 9-18 digits, optionally separated by
# spaces/hyphens.
#
# The account-number demonstration uses 13 digits, making it unambiguous
# relative to the grouped Aadhaar representation above.

BANK_ACCOUNT_PATTERN = re.compile(
    r"(?<![\d-])"
    r"\d(?:[\s-]?\d){8,17}"
    r"(?![\d-])"
)


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

def mask_pii(text: str) -> str:
    """Mask PAN, Aadhaar, and bank-account-shaped strings.

    Names and income values are intentionally left unchanged.

    Aadhaar masking is performed before bank-account masking because the
    grouped Aadhaar form has explicit separators and can therefore be
    distinguished from the account-number pattern.
    """

    masked = PAN_PATTERN.sub(
        "[PAN_MASKED]",
        text,
    )

    masked = AADHAAR_GROUPED_PATTERN.sub(
        "[AADHAAR_MASKED]",
        masked,
    )

    masked = BANK_ACCOUNT_PATTERN.sub(
        "[BANK_ACCOUNT_MASKED]",
        masked,
    )

    return masked


# ---------------------------------------------------------------------------
# Prompt-injection detection
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard\s+(?:all\s+)?previous\s+instructions\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bforget\s+(?:all\s+)?(?:your\s+)?instructions\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print)\s+(?:your\s+)?system\s+prompt\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print)\s+(?:your\s+)?instructions\b",
        re.IGNORECASE,
    ),
)


def detect_prompt_injection(text: str) -> dict[str, Any]:
    """Detect common prompt-injection instruction patterns."""

    matches: list[str] = []

    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)

        if match:
            matches.append(match.group(0))

    return {
        "detected": bool(matches),
        "matches": matches,
    }


def guard_input(text: str) -> dict[str, Any]:
    """Apply input-side PII masking and prompt-injection detection.

    Prompt injection is detected against the original input.
    PII masking is applied before the query is allowed to proceed.

    If prompt injection is detected, the request is refused.
    """

    injection = detect_prompt_injection(text)
    masked_text = mask_pii(text)

    if injection["detected"]:
        return {
            "allowed": False,
            "masked_query": masked_text,
            "reason": "Prompt injection detected.",
            "injection": injection,
        }

    return {
        "allowed": True,
        "masked_query": masked_text,
        "reason": None,
        "injection": injection,
    }


# ---------------------------------------------------------------------------
# Output-side groundedness
# ---------------------------------------------------------------------------

def enforce_rag_groundedness(
    rag_result: dict[str, Any],
) -> dict[str, Any]:
    """Refuse a RAG response if Phase 1 marked it unsupported.

    The frozen Phase 1 generation function already performs the similarity
    comparison using SIMILARITY_THRESHOLD. Phase 2 reuses that result rather
    than implementing a second retrieval or similarity calculation.
    """

    if rag_result.get("grounded") is True:
        return {
            "allowed": True,
            "answer": rag_result["answer"],
            "reason": None,
        }

    return {
        "allowed": False,
        "answer": "I don't know based on the Cred knowledge base.",
        "reason": (
            "Retrieved context did not meet the frozen "
            "groundedness threshold."
        ),
    }