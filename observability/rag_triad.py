"""
Phase 3 RAG Triad evaluation.

Evaluates the frozen Phase 2 Cred Domain Support Agent using:

1. Context Relevance
2. Groundedness
3. Answer Relevance

The evaluation judge runs in deterministic MOCK_LLM mode.
No production agent code is modified by this module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from agent.graph import run_query
from rag.retrieval import retrieve

from eval.part3_eval_dataset import (
    EVALUATION_DATASET,
    EvaluationQuery,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"

RAG_STRATEGY = "sentence"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.05

FALLBACK_ANSWER = "I don't know based on the Cred knowledge base."


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriadScore:
    query_id: int
    query: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    route: str
    grounded: bool


# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """
You are the evaluation judge for the Cred Domain Support Agent.

Evaluate one user query, its retrieved context, and the generated answer.

Score three dimensions independently from 0.0 to 1.0:

1. Context Relevance
   Does the retrieved context contain information that can help answer
   the user's query?

2. Groundedness
   Is the generated answer supported by the retrieved context?
   Do not reward information that is absent from the context.

3. Answer Relevance
   Does the generated answer directly address the user's query?
   A correct refusal such as "I don't know based on the Cred knowledge
   base." is appropriate when the available context does not support
   an answer.

Return deterministic scores.

This judge is operating in MOCK_LLM mode.
No network calls or external model calls are permitted.
""".strip()


# ---------------------------------------------------------------------------
# Deterministic MOCK_LLM judge
# ---------------------------------------------------------------------------

class MockLLMJudge:
    """
    Deterministic evaluation-only mock judge.

    This class represents the LLM-as-judge interface required for Phase 3,
    while MOCK_LLM keeps the entire evaluation offline and reproducible.
    """

    def __init__(self) -> None:
        if not MOCK_LLM:
            raise RuntimeError(
                "Phase 3 evaluation requires MOCK_LLM=true."
            )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    @staticmethod
    def _meaningful_query_tokens(query: str) -> set[str]:
        stopwords = {
            "what",
            "are",
            "the",
            "for",
            "how",
            "can",
            "is",
            "my",
            "me",
            "tell",
            "about",
            "there",
            "any",
            "does",
            "do",
            "an",
            "a",
            "to",
            "of",
            "and",
            "in",
            "on",
            "it",
            "your",
            "you",
        }

        return {
            token
            for token in MockLLMJudge._tokens(query)
            if token not in stopwords
        }

    def judge(
        self,
        *,
        query: str,
        context: str,
        answer: str,
        grounded_signal: bool,
    ) -> dict[str, float]:
        """
        Evaluate query/context/answer using deterministic MOCK_LLM rules.

        The prompt above defines the judging criteria. The implementation
        below provides the deterministic mock behavior for those criteria.
        """

        context_relevance = self._context_relevance(
            query=query,
            context=context,
        )

        groundedness = self._groundedness(
            answer=answer,
            context=context,
            grounded_signal=grounded_signal,
        )

        answer_relevance = self._answer_relevance(
            query=query,
            answer=answer,
            context=context,
        )

        return {
            "context_relevance": context_relevance,
            "groundedness": groundedness,
            "answer_relevance": answer_relevance,
        }

    def _context_relevance(
        self,
        *,
        query: str,
        context: str,
    ) -> float:
        if not context.strip():
            return 0.0

        query_tokens = self._meaningful_query_tokens(query)
        context_tokens = self._tokens(context)

        if not query_tokens:
            return 0.0

        overlap = query_tokens & context_tokens
        ratio = len(overlap) / len(query_tokens)

        if ratio >= 0.75:
            return 1.0
        if ratio >= 0.50:
            return 0.75
        if ratio >= 0.25:
            return 0.50
        if ratio > 0:
            return 0.25

        return 0.0

    def _groundedness(
        self,
        *,
        answer: str,
        context: str,
        grounded_signal: bool,
    ) -> float:
        if not answer.strip():
            return 0.0

        if answer.strip() == FALLBACK_ANSWER:
            return 1.0 if not context.strip() else 0.0

        if not grounded_signal:
            return 0.0

        if not context.strip():
            return 0.0

        answer_tokens = self._tokens(answer)
        context_tokens = self._tokens(context)

        if not answer_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        ratio = len(overlap) / len(answer_tokens)

        if ratio >= 0.50:
            return 1.0
        if ratio >= 0.25:
            return 0.75
        if ratio > 0:
            return 0.50

        return 0.0

    def _answer_relevance(
        self,
        *,
        query: str,
        answer: str,
        context: str,
    ) -> float:
        if not answer.strip():
            return 0.0

        if answer.strip() == FALLBACK_ANSWER:
            # The refusal is relevant when the retrieved context does not
            # provide useful support for the question.
            if not context.strip():
                return 1.0

            return 0.50

        query_tokens = self._meaningful_query_tokens(query)
        answer_tokens = self._tokens(answer)

        if not query_tokens:
            return 0.0

        overlap = query_tokens & answer_tokens
        ratio = len(overlap) / len(query_tokens)

        if ratio >= 0.75:
            return 1.0
        if ratio >= 0.50:
            return 0.75
        if ratio >= 0.25:
            return 0.50
        if ratio > 0:
            return 0.25

        return 0.0


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def _extract_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Combine retrieved chunk text into one evaluation context."""

    parts: list[str] = []

    for chunk in retrieved_chunks:
        text = chunk.get("text")

        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


def _retrieve_context(
    query: str,
) -> list[dict[str, Any]]:
    """Retrieve context using the frozen sentence strategy."""

    return retrieve(
        query=query,
        strategy=RAG_STRATEGY,
        top_k=TOP_K,
    )


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    item: EvaluationQuery,
    judge: MockLLMJudge,
) -> TriadScore:
    """Evaluate one query through the frozen agent."""

    retrieved_chunks = _retrieve_context(item.query)
    context = _extract_context(retrieved_chunks)

    result = run_query(
        query=item.query,
        thread_id=f"phase3-eval-{item.query_id:02d}",
    )

    structured_response = result["structured_response"]

    answer = structured_response["answer"]
    route = structured_response["route"]
    grounded = bool(structured_response.get("grounded"))

    judged_scores = judge.judge(
        query=item.query,
        context=context,
        answer=answer,
        grounded_signal=grounded,
    )

    return TriadScore(
        query_id=item.query_id,
        query=item.query,
        context_relevance=judged_scores["context_relevance"],
        groundedness=judged_scores["groundedness"],
        answer_relevance=judged_scores["answer_relevance"],
        route=route,
        grounded=grounded,
    )


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset() -> list[TriadScore]:
    """Evaluate exactly the required 15-query dataset."""

    validate_dataset()

    if len(EVALUATION_DATASET) != 15:
        raise AssertionError(
            "Phase 3 RAG triad must evaluate exactly 15 queries."
        )

    judge = MockLLMJudge()

    scores: list[TriadScore] = []

    for item in EVALUATION_DATASET:
        scores.append(
            evaluate_query(
                item=item,
                judge=judge,
            )
        )

    return scores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def print_evaluation(
    scores: list[TriadScore],
) -> None:
    """Print per-query scores and aggregate averages."""

    print("Phase 3 RAG Triad Evaluation")
    print("=" * 30)
    print(f"MOCK_LLM: {MOCK_LLM}")
    print("Judge: deterministic MockLLMJudge")
    print(f"Strategy: {RAG_STRATEGY}")
    print(f"Top-k: {TOP_K}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Queries evaluated: {len(scores)}")
    print()

    print(
        "ID | Context Relevance | Groundedness | Answer Relevance | Route"
    )
    print("-" * 72)

    for score in scores:
        print(
            f"{score.query_id:02d} | "
            f"{score.context_relevance:17.2f} | "
            f"{score.groundedness:12.2f} | "
            f"{score.answer_relevance:16.2f} | "
            f"{score.route}"
        )

    context_relevance_avg = _average(
        [score.context_relevance for score in scores]
    )

    groundedness_avg = _average(
        [score.groundedness for score in scores]
    )

    answer_relevance_avg = _average(
        [score.answer_relevance for score in scores]
    )

    print()
    print("Overall Averages")
    print("----------------")
    print(f"Context Relevance: {context_relevance_avg:.2f}")
    print(f"Groundedness:      {groundedness_avg:.2f}")
    print(f"Answer Relevance:  {answer_relevance_avg:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    scores = evaluate_dataset()
    print_evaluation(scores)


if __name__ == "__main__":
    main()