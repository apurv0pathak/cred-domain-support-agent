"""Deterministic tools used by the Cred support agent.

Phase 2 must use the frozen Phase 1 dataset rather than generating or
maintaining a second loan-application dataset.
"""

from __future__ import annotations

from dataset import LOAN_APPLICATIONS


# ---------------------------------------------------------------------------
# Escalation-score design
# ---------------------------------------------------------------------------
#
# days_since_created is contractually in the range 0..30.
#
# Newer applications receive a larger recency signal:
#
#     recency = 1 - (days_since_created / 30)
#
# The final escalation score is:
#
#     escalation_score =
#         0.55 * fraud_review_signal
#         + 0.45 * recency
#
# where fraud_review_signal is 1.0 when flagged_for_fraud_review is True,
# otherwise 0.0.
#
# Fraud review therefore has the larger individual weight, while recency
# still contributes continuously instead of being reduced to a boolean.
#
# The escalation threshold is calibrated from the actual deterministic
# dataset. We calculate the 25th percentile of days_since_created, which
# represents the boundary of the newest quartile of applications.
#
#     p25_days = percentile(days_since_created, 25)
#     p25_recency = 1 - (p25_days / 30)
#     threshold = 0.45 * p25_recency
#
# This means an unflagged application around the newest-quartile boundary
# can itself warrant escalation, while a fraud-review flag provides an
# additional strong signal.


FRAUD_WEIGHT = 0.55
RECENCY_WEIGHT = 0.45
MAX_DAYS_SINCE_CREATED = 30


def _percentile(values: list[int], percentile: float) -> float:
    """Calculate a percentile using linear interpolation.

    This implementation avoids adding NumPy as a dependency solely for
    percentile calculation.

    The method is equivalent to linear interpolation at rank:
        (n - 1) * percentile / 100
    """
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty dataset.")

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]

    return lower_value + fraction * (upper_value - lower_value)


def _dataset_days() -> list[int]:
    """Return days_since_created values from the canonical Phase 1 dataset."""
    return [
        application["days_since_created"]
        for application in LOAN_APPLICATIONS
    ]


DAYS_SINCE_CREATED_P25 = _percentile(_dataset_days(), 25)

P25_RECENCY_SIGNAL = (
    1.0 - (DAYS_SINCE_CREATED_P25 / MAX_DAYS_SINCE_CREATED)
)

ESCALATION_THRESHOLD = RECENCY_WEIGHT * P25_RECENCY_SIGNAL


def _calculate_escalation_score(
    flagged_for_fraud_review: bool,
    days_since_created: int,
) -> float:
    """Return the designed escalation score in the closed interval [0, 1]."""

    recency_signal = 1.0 - (
        days_since_created / MAX_DAYS_SINCE_CREATED
    )

    fraud_signal = 1.0 if flagged_for_fraud_review else 0.0

    score = (
        FRAUD_WEIGHT * fraud_signal
        + RECENCY_WEIGHT * recency_signal
    )

    # Defensive clamping keeps the public contract within [0, 1] even if
    # malformed data somehow reaches this helper.
    return max(0.0, min(1.0, score))


def check_loan_application_status(record_id: str) -> dict:
    """Look up a loan application and return status/escalation information.

    Args:
        record_id:
            Canonical record ID from the Phase 1 LOAN_APPLICATIONS dataset.

    Returns:
        For a known record:
            {
                "found": True,
                "record_id": str,
                "status": str,
                "loan_amount_inr": int,
                "escalation_score": float,
                "escalation_threshold": float,
                "should_escalate": bool,
            }

        For an unknown record:
            {
                "found": False,
                "record_id": str,
                "error": str,
            }

    The function never creates or modifies application data.
    """

    normalized_record_id = record_id.strip()

    for application in LOAN_APPLICATIONS:
        if application["record_id"] != normalized_record_id:
            continue

        score = _calculate_escalation_score(
            flagged_for_fraud_review=application[
                "flagged_for_fraud_review"
            ],
            days_since_created=application["days_since_created"],
        )

        return {
            "found": True,
            "record_id": application["record_id"],
            "status": application["status"],
            "loan_amount_inr": application["loan_amount_inr"],
            "escalation_score": round(score, 6),
            "escalation_threshold": round(ESCALATION_THRESHOLD, 6),
            "should_escalate": score >= ESCALATION_THRESHOLD,
        }

    return {
        "found": False,
        "record_id": normalized_record_id,
        "error": "Loan application record not found.",
    }