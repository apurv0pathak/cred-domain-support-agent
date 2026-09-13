"""
Measure top-1 and top-3 retrieval similarity for threshold calibration.

This script deliberately does NOT choose a similarity threshold.

Run it after the two ChromaDB collections have been built.
"""

from __future__ import annotations

from rag.retrieval import retrieve


IN_SCOPE_QUERIES = [
    "What documents are required for KYC?",
    "How is EMI calculated for a loan?",
    "What factors affect a credit score?",
    "What is the process for closing an account?",
    "What are the rules for prepayment penalties?",
]

OUT_OF_SCOPE_QUERIES = [
    "What is the weather in Delhi today?",
    "Who won the latest cricket match?",
]


def measure_query(query: str) -> None:
    print(f"\nQUERY: {query}")

    for strategy in ("fixed", "sentence"):
        results = retrieve(
            query=query,
            strategy=strategy,
            top_k=3,
        )

        print(f"\n  STRATEGY: {strategy}")

        for rank, result in enumerate(results, start=1):
            print(
                f"    rank={rank} "
                f"similarity={result['similarity']:.6f} "
                f"document={result['document_id']} "
                f"chunk={result['chunk_id']}"
            )


def main() -> None:
    print("=== IN-SCOPE QUERIES ===")

    for query in IN_SCOPE_QUERIES:
        measure_query(query)

    print("\n=== OUT-OF-SCOPE QUERIES ===")

    for query in OUT_OF_SCOPE_QUERIES:
        measure_query(query)


if __name__ == "__main__":
    main()