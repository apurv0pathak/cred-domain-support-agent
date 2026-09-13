
"""
Deterministic mock loan-application dataset for the Cred support agent.

This module intentionally contains no network calls, API keys, or LLM usage.

The generator:
- uses a deterministic seed;
- generates records randomly;
- validates category/status coverage;
- validates the fraud-review percentage;
- regenerates with another deterministic seed if necessary;
- never hand-edits individual records to force constraints.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TypedDict


# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------

class LoanApplication(TypedDict):
    record_id: str
    category: str
    status: str
    loan_amount_inr: int
    days_since_created: int
    flagged_for_fraud_review: bool


# ---------------------------------------------------------------------------
# Deterministic configuration
# ---------------------------------------------------------------------------

INITIAL_SEED = 42
NUM_RECORDS = 50

CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
]


# Realistic loan amount ranges by category.
LOAN_AMOUNT_RANGES = {
    "Personal Loan": (50_000, 1_500_000),
    "Home Loan": (2_000_000, 15_000_000),
    "Auto Loan": (300_000, 3_000_000),
    "Education Loan": (100_000, 3_000_000),
    "Business Loan": (500_000, 10_000_000),
}


# Status probabilities.
STATUS_WEIGHTS = {
    "Submitted": 0.15,
    "Under Review": 0.20,
    "Approved": 0.25,
    "Rejected": 0.15,
    "Disbursed": 0.25,
}


# The generation probability is deliberately inside the required
# 10%-30% final-result interval.
FRAUD_REVIEW_PROBABILITY = 0.20


# Required final fraud-review percentage.
MIN_FRAUD_PERCENTAGE = 10.0
MAX_FRAUD_PERCENTAGE = 30.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_applications(seed: int) -> list[LoanApplication]:
    """
    Generate one deterministic candidate dataset from the supplied seed.

    This function does not modify individual records after generation.
    """

    rng = random.Random(seed)

    applications: list[LoanApplication] = []

    for index in range(1, NUM_RECORDS + 1):
        category = rng.choice(CATEGORIES)

        status = rng.choices(
            population=STATUSES,
            weights=[
                STATUS_WEIGHTS[status]
                for status in STATUSES
            ],
            k=1,
        )[0]

        minimum, maximum = LOAN_AMOUNT_RANGES[category]

        raw_amount = rng.randint(
            minimum,
            maximum,
        )

        # Round to the nearest ₹5,000 to keep the mock amounts realistic.
        loan_amount = round(
            raw_amount / 5_000
        ) * 5_000

        days_since_created = rng.randint(
            0,
            30,
        )

        flagged_for_fraud_review = (
            rng.random() < FRAUD_REVIEW_PROBABILITY
        )

        applications.append(
            {
                "record_id": f"LA-{index:04d}",
                "category": category,
                "status": status,
                "loan_amount_inr": loan_amount,
                "days_since_created": days_since_created,
                "flagged_for_fraud_review": (
                    flagged_for_fraud_review
                ),
            }
        )

    return applications


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dataset(
    applications: list[LoanApplication],
) -> tuple[bool, str]:
    """
    Validate all Phase 1 dataset constraints.

    Returns:
        (True, reason) when valid.
        (False, reason) when invalid.

    No records are modified by this function.
    """

    if len(applications) < 40:
        return (
            False,
            f"record count is {len(applications)}, below 40",
        )

    category_counts = Counter(
        application["category"]
        for application in applications
    )

    for category in CATEGORIES:
        count = category_counts[category]

        if count < 3:
            return (
                False,
                f"category '{category}' has only {count} records",
            )

    status_counts = Counter(
        application["status"]
        for application in applications
    )

    for status in STATUSES:
        count = status_counts[status]

        if count < 1:
            return (
                False,
                f"status '{status}' has no records",
            )

    for application in applications:
        if not (
            0 <= application["days_since_created"] <= 30
        ):
            return (
                False,
                (
                    f"{application['record_id']} has invalid "
                    f"days_since_created="
                    f"{application['days_since_created']}"
                ),
            )

        if not isinstance(
            application["flagged_for_fraud_review"],
            bool,
        ):
            return (
                False,
                (
                    f"{application['record_id']} has a non-boolean "
                    "fraud-review flag"
                ),
            )

    fraud_count = sum(
        application["flagged_for_fraud_review"]
        for application in applications
    )

    fraud_percentage = (
        fraud_count
        / len(applications)
        * 100
    )

    if not (
        MIN_FRAUD_PERCENTAGE
        <= fraud_percentage
        <= MAX_FRAUD_PERCENTAGE
    ):
        return (
            False,
            (
                f"fraud-review percentage is "
                f"{fraud_percentage:.2f}%"
            ),
        )

    return True, "all constraints satisfied"


def _generate_valid_dataset() -> tuple[
    list[LoanApplication],
    int,
    int,
]:
    """
    Generate datasets using deterministic seed progression until a
    candidate satisfies every Phase 1 constraint.

    A new seed is used for each attempt. No record is manually changed.
    """

    seed = INITIAL_SEED
    attempts = 0

    while True:
        attempts += 1

        applications = _generate_applications(
            seed=seed
        )

        valid, reason = _validate_dataset(
            applications
        )

        if valid:
            return applications, seed, attempts

        seed += 1


# Public deterministic dataset consumed by later phases.
LOAN_APPLICATIONS, DATASET_SEED, GENERATION_ATTEMPTS = (
    _generate_valid_dataset()
)


# ---------------------------------------------------------------------------
# Direct execution / verification output
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    """Print dataset statistics for manual verification."""

    category_counts = Counter(
        application["category"]
        for application in LOAN_APPLICATIONS
    )

    status_counts = Counter(
        application["status"]
        for application in LOAN_APPLICATIONS
    )

    fraud_count = sum(
        application["flagged_for_fraud_review"]
        for application in LOAN_APPLICATIONS
    )

    fraud_percentage = (
        fraud_count
        / len(LOAN_APPLICATIONS)
        * 100
    )

    print("=== CRED LOAN APPLICATION DATASET ===")
    print(f"Dataset seed: {DATASET_SEED}")
    print(f"Generation attempts: {GENERATION_ATTEMPTS}")
    print(f"Total records: {len(LOAN_APPLICATIONS)}")

    print("\nCategory counts:")

    for category in CATEGORIES:
        print(
            f"  {category}: "
            f"{category_counts[category]}"
        )

    print("\nStatus counts:")

    for status in STATUSES:
        print(
            f"  {status}: "
            f"{status_counts[status]}"
        )

    print(
        "\nFraud-review flags: "
        f"{fraud_count}/{len(LOAN_APPLICATIONS)} "
        f"({fraud_percentage:.2f}%)"
    )


if __name__ == "__main__":
    _print_summary()
