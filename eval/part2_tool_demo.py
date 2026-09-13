"""Task A demonstration for the Phase 2 status-lookup tool.

The demo selects examples from the actual deterministic dataset instead of
hard-coding convenient record IDs.
"""

from dataset import LOAN_APPLICATIONS

from agent.tools import (
    DAYS_SINCE_CREATED_P25,
    ESCALATION_THRESHOLD,
    FRAUD_WEIGHT,
    RECENCY_WEIGHT,
    check_loan_application_status,
)


def print_case(label: str, application: dict) -> None:
    result = check_loan_application_status(application["record_id"])

    print(f"\n--- {label} ---")
    print(f"record_id: {application['record_id']}")
    print(
        "flagged_for_fraud_review:",
        application["flagged_for_fraud_review"],
    )
    print("days_since_created:", application["days_since_created"])
    print("status:", result["status"])
    print("loan_amount_inr:", result["loan_amount_inr"])
    print("escalation_score:", result["escalation_score"])
    print("escalation_threshold:", result["escalation_threshold"])
    print("should_escalate:", result["should_escalate"])


def main() -> None:
    print("PHASE 2 — TASK A: STATUS TOOL DEMO")
    print("=" * 50)

    print("\nEscalation formula:")
    print(
        "score = 0.55 * fraud_signal "
        "+ 0.45 * (1 - days_since_created / 30)"
    )

    print(f"\nFraud weight: {FRAUD_WEIGHT}")
    print(f"Recency weight: {RECENCY_WEIGHT}")
    print(
        "Actual dataset 25th percentile of days_since_created:",
        DAYS_SINCE_CREATED_P25,
    )
    print(
        "Dataset-calibrated escalation threshold:",
        round(ESCALATION_THRESHOLD, 6),
    )

    evaluated = []

    for application in LOAN_APPLICATIONS:
        result = check_loan_application_status(application["record_id"])
        evaluated.append((application, result))

    # We deliberately find examples from the real dataset so the demo is
    # evidence of actual behavior rather than hand-picked assumptions.
    flagged_above = next(
        (
            application
            for application, result in evaluated
            if application["flagged_for_fraud_review"]
            and result["should_escalate"]
        ),
        None,
    )

    unflagged_above = next(
        (
            application
            for application, result in evaluated
            if not application["flagged_for_fraud_review"]
            and result["should_escalate"]
        ),
        None,
    )

    unflagged_below = next(
        (
            application
            for application, result in evaluated
            if not application["flagged_for_fraud_review"]
            and not result["should_escalate"]
        ),
        None,
    )

    if flagged_above is None:
        raise RuntimeError(
            "Dataset did not contain a flagged above-threshold example."
        )

    if unflagged_above is None:
        raise RuntimeError(
            "Dataset did not contain an unflagged above-threshold example."
        )

    if unflagged_below is None:
        raise RuntimeError(
            "Dataset did not contain an unflagged below-threshold example."
        )

    print_case(
        "FLAGGED / ABOVE THRESHOLD",
        flagged_above,
    )

    print_case(
        "UNFLAGGED / ABOVE THRESHOLD",
        unflagged_above,
    )

    print_case(
        "UNFLAGGED / BELOW THRESHOLD",
        unflagged_below,
    )

    print("\n--- UNKNOWN RECORD ---")
    unknown = check_loan_application_status(
        "CRED-NOT-A-REAL-RECORD"
    )
    print(unknown)


if __name__ == "__main__":
    main()