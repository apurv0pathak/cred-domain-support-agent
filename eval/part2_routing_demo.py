"""Task B demonstration: prove both LangGraph routes execute."""

from agent.graph import run_query


def demonstrate(label: str, query: str) -> None:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(f"QUERY: {query}")

    result = run_query(query)

    print(f"ROUTE: {result['route']}")
    print(f"TRACE: {' -> '.join(result['trace'])}")
    print(f"ANSWER: {result['answer']}")

    if result["escalation_score"] is not None:
        print(f"ESCALATION_SCORE: {result['escalation_score']}")


def main() -> None:
    print("PHASE 2 — TASK B: LANGGRAPH ROUTING DEMO")

    demonstrate(
        "RAG ROUTE",
        "What documents are required for KYC?",
    )

    demonstrate(
        "STATUS TOOL ROUTE",
        "What is the status of loan application LA-0001?",
    )


if __name__ == "__main__":
    main()