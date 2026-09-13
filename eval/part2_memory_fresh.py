"""Task C + D demonstration: fresh thread plus schema validation."""

from pathlib import Path

from agent.graph import run_query
from agent.schema import validate_agent_response


OLD_THREAD_ID = "phase2_multiturn_demo"
FRESH_THREAD_ID = "phase2_fresh_demo"
FRESH_MEMORY_FILE = (
    Path("transcripts")
    / "memory"
    / f"{FRESH_THREAD_ID}.json"
)

def main() -> None:
    old_memory = (
        Path("transcripts")
        / "memory"
        / f"{OLD_THREAD_ID}.json"
    )

    if not old_memory.exists():
        raise RuntimeError(
            "Run phase2_memory_multiturn first so the old thread exists."
        )

    if FRESH_MEMORY_FILE.exists():
        FRESH_MEMORY_FILE.unlink()
        
    print("PHASE 2 — TASK C/D: FRESH THREAD + SCHEMA")
    print("=" * 60)

    query = "What is its current status?"

    result = run_query(
        query=query,
        thread_id=FRESH_THREAD_ID,
    )

    print(f"QUERY: {query}")
    print(f"THREAD_ID: {FRESH_THREAD_ID}")
    print("INITIAL_HISTORY_LENGTH: 0")
    print(f"ROUTE: {result['route']}")
    print(f"REUSED_RECORD_ID: {result['record_id']}")
    print(f"TRACE: {' -> '.join(result['trace'])}")
    print(f"ANSWER: {result['answer']}")

    print("\n--- STRUCTURED OUTPUT VALIDATION ---")

    try:
        validated = validate_agent_response(
            result["structured_response"]
        )
        print("VALIDATION: PASS")
        print(f"STRUCTURED_RESPONSE: {validated.model_dump()}")
    except Exception as exc:
        print("VALIDATION: FAIL")
        print(f"ERROR: {exc}")

    print("\n--- HISTORY AFTER THIS TURN ---")
    for message in result["history"]:
        print(f"{message['role']}: {message['content']}")


if __name__ == "__main__":
    main()