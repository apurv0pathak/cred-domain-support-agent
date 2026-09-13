"""Task C + D demonstration: memory plus structured-output validation."""

from pathlib import Path

from agent.graph import run_query
from agent.schema import validate_agent_response


THREAD_ID = "phase2_multiturn_demo"


def validate_and_print(result: dict, label: str) -> None:
    print(f"\n--- {label} STRUCTURED OUTPUT VALIDATION ---")

    try:
        validated = validate_agent_response(
            result["structured_response"]
        )
        print("VALIDATION: PASS")
        print(f"STRUCTURED_RESPONSE: {validated.model_dump()}")
    except Exception as exc:
        print("VALIDATION: FAIL")
        print(f"ERROR: {exc}")


def main() -> None:
    memory_file = (
        Path("transcripts")
        / "memory"
        / f"{THREAD_ID}.json"
    )

    if memory_file.exists():
        memory_file.unlink()

    print("PHASE 2 — TASK C/D: MULTI-TURN MEMORY + SCHEMA")
    print("=" * 60)

    first_query = "What is the status of loan application LA-0001?"

    first = run_query(
        query=first_query,
        thread_id=THREAD_ID,
    )

    print("\n--- TURN 1 ---")
    print(f"QUERY: {first_query}")
    print(f"ROUTE: {first['route']}")
    print(f"TRACE: {' -> '.join(first['trace'])}")
    print(f"ANSWER: {first['answer']}")

    validate_and_print(first, "TURN 1")

    second_query = "What is its current status?"

    second = run_query(
        query=second_query,
        thread_id=THREAD_ID,
    )

    print("\n--- TURN 2 ---")
    print(f"QUERY: {second_query}")
    print(f"ROUTE: {second['route']}")
    print(f"REUSED_RECORD_ID: {second['record_id']}")
    print(f"TRACE: {' -> '.join(second['trace'])}")
    print(f"ANSWER: {second['answer']}")

    validate_and_print(second, "TURN 2")

    print("\n--- PERSISTED HISTORY ---")
    for message in second["history"]:
        print(f"{message['role']}: {message['content']}")


if __name__ == "__main__":
    main()