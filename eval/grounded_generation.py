"""
Phase 1 grounded-generation demonstration.

Uses the five in-scope queries selected for the retrieval evaluation,
plus one deliberately out-of-scope query.

Run:
    python -m eval.grounded_generation_demo
"""

from __future__ import annotations

from rag.generation import generate_grounded_answer
from rag.generation import SIMILARITY_THRESHOLD

QUERIES = [
    "What documents are required for KYC?",
    "How is EMI calculated for a loan?",
    "What factors affect a credit score?",
    "What is the process for closing an account?",
    "What are the rules for prepayment penalties?",
    "What is the weather in Delhi today?",
]


def main() -> None:
    print("=== GROUNDED GENERATION DEMO ===")
    print("Strategy: sentence")
    print("Top-k: 3")
    f"Similarity threshold: {SIMILARITY_THRESHOLD:.2f}"

    for query in QUERIES:
        result = generate_grounded_answer(
            query=query,
            strategy="sentence",
            top_k=3,
        )

        print("\n" + "=" * 70)
        print(f"QUERY: {result['query']}")
        print(f"GROUNDED: {result['grounded']}")
        print(f"THRESHOLD: {result['threshold']:.2f}")

        if result["retrieved_chunks"]:
            best = result["retrieved_chunks"][0]

            print(
                f"TOP-1 SIMILARITY: "
                f"{best['similarity']:.6f}"
            )

            print(
                f"TOP-1 DOCUMENT: "
                f"{best['document_id']}"
            )

        print(f"ANSWER: {result['answer']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()