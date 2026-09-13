"""
Deterministic grounded generation for the Cred support agent.

This module intentionally does NOT call an LLM.

The generation process is:

1. Retrieve the top-k knowledge-base chunks.
2. Inspect the best similarity score.
3. If it is below the calibrated threshold, return an "I don't know"
   response.
4. Otherwise, assemble an answer only from the retrieved context.

The threshold was calibrated from the Phase 1 similarity measurements.
"""

from __future__ import annotations

import re
from typing import Any

from rag.retrieval import retrieve


# ---------------------------------------------------------------------------
# Frozen Phase 1 calibration
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Public generation function
# ---------------------------------------------------------------------------

def generate_grounded_answer(
    query: str,
    strategy: str = "sentence",
    top_k: int = 3,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Generate a deterministic answer grounded only in retrieved KB context.

    Parameters
    ----------
    query:
        User's knowledge-base question.

    strategy:
        Either "fixed" or "sentence".

    top_k:
        Number of chunks to retrieve.

    threshold:
        Minimum top-1 similarity required to answer.

    Returns
    -------
    dict
        Contains:
          - query
          - strategy
          - threshold
          - answer
          - grounded
          - retrieved_chunks
    """

    results = retrieve(
        query=query,
        strategy=strategy,
        top_k=top_k,
    )

    if not results:
        return {
            "query": query,
            "strategy": strategy,
            "threshold": threshold,
            "answer": (
                "I don't know based on the Cred knowledge base."
            ),
            "grounded": False,
            "retrieved_chunks": [],
        }

    best_similarity = results[0]["similarity"]

    # Calibrated fallback gate.
    if best_similarity < threshold:
        return {
            "query": query,
            "strategy": strategy,
            "threshold": threshold,
            "answer": (
                "I don't know based on the Cred knowledge base."
            ),
            "grounded": False,
            "retrieved_chunks": results,
        }

    answer = _assemble_grounded_answer(
        query=query,
        results=results,
    )

    return {
        "query": query,
        "strategy": strategy,
        "threshold": threshold,
        "answer": answer,
        "grounded": True,
        "retrieved_chunks": results,
    }


def _assemble_grounded_answer(
    query: str,
    results: list[dict[str, Any]],
) -> str:
    """
    Build a deterministic extractive answer from the best retrieved chunk.

    Only the top-1 retrieved chunk is used for answer construction.
    This prevents unrelated but moderately similar documents from
    contributing sentences to the final answer.

    No new factual information is generated.
    """

    if not results:
        return "I don't know based on the Cred knowledge base."

    best_result = results[0]

    sentences = _split_sentences(
        best_result["text"]
    )

    if not sentences:
        return best_result["text"].strip()

    query_terms = _query_terms(query)

    scored_sentences: list[tuple[int, int, str]] = []

    for sentence_index, sentence in enumerate(sentences):
        sentence_terms = _query_terms(sentence)

        overlap = len(
            query_terms & sentence_terms
        )

        scored_sentences.append(
            (
                overlap,
                -sentence_index,
                sentence,
            )
        )

    scored_sentences.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    selected: list[str] = []

    for overlap, _, sentence in scored_sentences:
        if overlap > 0:
            selected.append(sentence)

        if len(selected) >= 3:
            break

    # If lexical matching finds nothing, return the complete top-1 chunk.
    if not selected:
        return best_result["text"].strip()

    return " ".join(selected)


def _split_sentences(text: str) -> list[str]:
    """Deterministically split text into sentences."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]


def _query_terms(text: str) -> set[str]:
    """
    Extract simple normalized content words.

    Common stop words are removed so that terms such as "what", "is",
    and "the" do not dominate sentence selection.
    """

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "what",
        "which",
        "with",
    }

    words = re.findall(
        r"[a-zA-Z]+",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in stop_words and len(word) > 1
    }