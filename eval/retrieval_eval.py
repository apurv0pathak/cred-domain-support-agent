"""
Phase 1 retrieval evaluation.

Evaluates the five grounded-generation queries at the parent-document level.

Metrics:
    Precision@3 = relevant retrieved documents / 3
    Recall@3    = relevant retrieved documents / total relevant documents

Chunks are mapped back to their parent document_id and deduplicated before
the metrics are calculated.

The same five queries used in grounded generation are intentionally reused.
"""

from __future__ import annotations

from rag.retrieval import retrieve


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

EVALUATION_QUERIES = [
    {
        "query": "What documents are required for KYC?",
        "relevant_documents": {
            "04_kyc_requirements",
        },
    },
    {
        "query": "How is EMI calculated for a loan?",
        "relevant_documents": {
            "02_emi_calculation",
        },
    },
    {
        "query": "What factors affect a credit score?",
        "relevant_documents": {
            "10_credit_score_factors",
        },
    },
    {
        "query": "What is the process for closing an account?",
        "relevant_documents": {
            "06_account_closure",
        },
    },
    {
        "query": "What are the rules for prepayment penalties?",
        "relevant_documents": {
            "08_prepayment_penalties",
        },
    },
]


STRATEGIES = [
    "fixed",
    "sentence",
]


def evaluate_query(
    query: str,
    relevant_documents: set[str],
    strategy: str,
    top_k: int = 3,
) -> dict:
    """
    Evaluate one query for one chunking strategy.
    """

    results = retrieve(
        query=query,
        strategy=strategy,
        top_k=top_k,
    )

    # Map retrieved chunks to parent documents and deduplicate.
    retrieved_documents: list[str] = []

    for result in results:
        document_id = result["document_id"]

        if document_id not in retrieved_documents:
            retrieved_documents.append(document_id)

    retrieved_document_set = set(retrieved_documents)

    relevant_retrieved = (
        retrieved_document_set & relevant_documents
    )

    precision = len(relevant_retrieved) / top_k

    recall = (
        len(relevant_retrieved)
        / len(relevant_documents)
    )

    return {
        "query": query,
        "strategy": strategy,
        "relevant_documents": relevant_documents,
        "retrieved_documents": retrieved_documents,
        "relevant_retrieved": relevant_retrieved,
        "precision_at_3": precision,
        "recall_at_3": recall,
        "raw_results": results,
    }


def print_evaluation(result: dict) -> None:
    """Print visible arithmetic for one evaluated query."""

    print("\n" + "=" * 72)

    print(f"QUERY: {result['query']}")
    print(f"STRATEGY: {result['strategy']}")

    print(
        "RELEVANT DOCUMENTS: "
        f"{sorted(result['relevant_documents'])}"
    )

    print(
        "RETRIEVED DOCUMENTS (deduplicated): "
        f"{result['retrieved_documents']}"
    )

    print(
        "RELEVANT RETRIEVED: "
        f"{sorted(result['relevant_retrieved'])}"
    )

    relevant_count = len(
        result["relevant_retrieved"]
    )

    relevant_total = len(
        result["relevant_documents"]
    )

    print(
        f"Precision@3 = "
        f"{relevant_count} / 3 = "
        f"{result['precision_at_3']:.4f}"
    )

    print(
        f"Recall@3 = "
        f"{relevant_count} / {relevant_total} = "
        f"{result['recall_at_3']:.4f}"
    )

    print("Retrieved chunk details:")

    for rank, retrieved in enumerate(
        result["raw_results"],
        start=1,
    ):
        print(
            f"  {rank}. "
            f"document={retrieved['document_id']} "
            f"similarity={retrieved['similarity']:.6f} "
            f"chunk={retrieved['chunk_id']}"
        )


def main() -> None:
    all_results: list[dict] = []

    print("=== PHASE 1 RETRIEVAL EVALUATION ===")
    print("Top-k: 3")
    print("Metric level: parent document")
    print("Duplicate parent documents: deduplicated")

    for strategy in STRATEGIES:
        print("\n" + "#" * 72)
        print(f"STRATEGY: {strategy}")
        print("#" * 72)

        strategy_results: list[dict] = []

        for item in EVALUATION_QUERIES:
            result = evaluate_query(
                query=item["query"],
                relevant_documents=item[
                    "relevant_documents"
                ],
                strategy=strategy,
                top_k=3,
            )

            strategy_results.append(result)
            all_results.append(result)

            print_evaluation(result)

        mean_precision = sum(
            result["precision_at_3"]
            for result in strategy_results
        ) / len(strategy_results)

        mean_recall = sum(
            result["recall_at_3"]
            for result in strategy_results
        ) / len(strategy_results)

        print("\n" + "-" * 72)
        print(f"{strategy.upper()} SUMMARY")

        print(
            "Mean Precision@3 = "
            f"{mean_precision:.4f}"
        )

        print(
            "Mean Recall@3 = "
            f"{mean_recall:.4f}"
        )

    # -----------------------------------------------------------------------
    # Final comparison
    # -----------------------------------------------------------------------

    fixed_results = [
        result
        for result in all_results
        if result["strategy"] == "fixed"
    ]

    sentence_results = [
        result
        for result in all_results
        if result["strategy"] == "sentence"
    ]

    fixed_precision = sum(
        result["precision_at_3"]
        for result in fixed_results
    ) / len(fixed_results)

    fixed_recall = sum(
        result["recall_at_3"]
        for result in fixed_results
    ) / len(fixed_results)

    sentence_precision = sum(
        result["precision_at_3"]
        for result in sentence_results
    ) / len(sentence_results)

    sentence_recall = sum(
        result["recall_at_3"]
        for result in sentence_results
    ) / len(sentence_results)

    print("\n" + "=" * 72)
    print("=== FINAL STRATEGY COMPARISON ===")
    print("=" * 72)

    print(
        f"Fixed-size:   "
        f"Precision@3={fixed_precision:.4f}, "
        f"Recall@3={fixed_recall:.4f}"
    )

    print(
        f"Sentence:     "
        f"Precision@3={sentence_precision:.4f}, "
        f"Recall@3={sentence_recall:.4f}"
    )


if __name__ == "__main__":
    main()