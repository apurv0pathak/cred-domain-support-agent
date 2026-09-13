"""
Phase 3 evaluation dataset for the Cred Domain Support Agent.

Task 13:
- Exactly 15 evaluation queries.
- Covers all 12 knowledge-base topics.
- Includes out-of-scope / edge-case queries.
- Provides deterministic expected metadata for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: int
    query: str
    topic: str
    expected_route: str
    expected_grounded: bool


EVALUATION_DATASET: tuple[EvaluationQuery, ...] = (
    EvaluationQuery(
        query_id=1,
        query="What are the eligibility requirements for a loan?",
        topic="loan eligibility",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=2,
        query="How is EMI calculated for a loan?",
        topic="EMI calculation",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=3,
        query="What fees are charged for a credit card?",
        topic="credit card fees",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=4,
        query="What documents are required for KYC?",
        topic="KYC requirements",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=5,
        query="How are fraud disputes resolved?",
        topic="fraud dispute resolution",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=6,
        query="How can I close my account?",
        topic="account closure",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=7,
        query="What are the interest rate slabs?",
        topic="interest rate slabs",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=8,
        query="Are there any penalties for prepayment?",
        topic="prepayment penalties",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=9,
        query="What is the minimum balance requirement?",
        topic="minimum balance",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=10,
        query="What factors affect my credit score?",
        topic="credit score factors",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=11,
        query="What are the rules for a joint account?",
        topic="joint account rules",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=12,
        query="Can an NRI open an account?",
        topic="NRI account eligibility",
        expected_route="rag",
        expected_grounded=True,
    ),
    EvaluationQuery(
        query_id=13,
        query="What is the weather in Delhi today?",
        topic="out of scope",
        expected_route="rag",
        expected_grounded=False,
    ),
    EvaluationQuery(
        query_id=14,
        query="Tell me a joke about banking.",
        topic="out of scope",
        expected_route="rag",
        expected_grounded=False,
    ),
    EvaluationQuery(
        query_id=15,
        query="Can you tell me the current status of application LA-1001?",
        topic="application status edge case",
        expected_route="status",
        expected_grounded=False,
    ),
)


def validate_dataset() -> None:
    """Validate the fixed Phase 3 evaluation dataset."""

    assert len(EVALUATION_DATASET) == 15, (
        f"Expected exactly 15 queries, found {len(EVALUATION_DATASET)}."
    )

    query_ids = [item.query_id for item in EVALUATION_DATASET]
    assert query_ids == list(range(1, 16)), (
        "Query IDs must be exactly 1 through 15."
    )

    kb_topics = {
        "loan eligibility",
        "EMI calculation",
        "credit card fees",
        "KYC requirements",
        "fraud dispute resolution",
        "account closure",
        "interest rate slabs",
        "prepayment penalties",
        "minimum balance",
        "credit score factors",
        "joint account rules",
        "NRI account eligibility",
    }

    covered_topics = {
        item.topic
        for item in EVALUATION_DATASET
        if item.topic in kb_topics
    }

    assert covered_topics == kb_topics, (
        "The evaluation dataset must cover all 12 knowledge-base topics."
    )

    out_of_scope_count = sum(
        item.topic == "out of scope"
        for item in EVALUATION_DATASET
    )

    assert out_of_scope_count >= 2, (
        "The evaluation dataset must contain at least two "
        "out-of-scope / edge-case queries."
    )


def print_dataset_summary() -> None:
    """Print a deterministic summary for transcript capture."""

    validate_dataset()

    print("Phase 3 Evaluation Dataset")
    print("=" * 28)
    print(f"Total queries: {len(EVALUATION_DATASET)}")
    print("Knowledge-base topics covered: 12")
    print(f"Out-of-scope queries: {sum(item.topic == 'out of scope' for item in EVALUATION_DATASET)}")
    print("Dataset validation: PASS")
    print()

    for item in EVALUATION_DATASET:
        print(
            f"{item.query_id:02d}. "
            f"[{item.topic}] "
            f"{item.query}"
        )


if __name__ == "__main__":
    print_dataset_summary()