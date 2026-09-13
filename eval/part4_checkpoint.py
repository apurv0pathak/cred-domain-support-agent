"""
Phase 4 Task 15 — SQLite checkpointing demonstration.

Demonstrates:

1. A LangGraph run executing at least two nodes.
2. Deliberate interruption after the second node.
3. Persistence of state in checkpoints.sqlite.
4. Resumption using the SAME thread_id.
5. Previously completed nodes are not re-executed.

The frozen Phase 2 agent is not modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


CHECKPOINT_DB = "checkpoints.sqlite"
THREAD_ID = "phase4-checkpoint-demo"


class CheckpointState(TypedDict):
    query: str
    node_a_result: str
    node_b_result: str
    node_c_result: str
    node_d_result: str
    execution_log: list[str]


def node_a(state: CheckpointState) -> dict:
    """First node — records execution explicitly."""
    print("EXECUTING node_a")
    print("  node_a was NOT loaded from checkpoint.")

    execution_log = list(state.get("execution_log", []))
    execution_log.append("node_a executed")

    return {
        "node_a_result": "Node A completed.",
        "execution_log": execution_log,
    }


def node_b(state: CheckpointState) -> dict:
    """Second node — records execution explicitly."""
    print("EXECUTING node_b")
    print("  node_b was NOT loaded from checkpoint.")

    execution_log = list(state.get("execution_log", []))
    execution_log.append("node_b executed")

    return {
        "node_b_result": "Node B completed.",
        "execution_log": execution_log,
    }


def node_c(state: CheckpointState) -> dict:
    """Third node — only runs after checkpoint resume."""
    print("EXECUTING node_c")

    execution_log = list(state.get("execution_log", []))
    execution_log.append("node_c executed")

    return {
        "node_c_result": "Node C completed.",
        "execution_log": execution_log,
    }


def node_d(state: CheckpointState) -> dict:
    """Fourth node — completes the resumed execution."""
    print("EXECUTING node_d")

    execution_log = list(state.get("execution_log", []))
    execution_log.append("node_d executed")

    return {
        "node_d_result": "Node D completed.",
        "execution_log": execution_log,
    }


def build_graph(checkpointer: SqliteSaver):
    """Build the checkpointed demonstration graph."""

    builder = StateGraph(CheckpointState)

    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.add_node("node_d", node_d)

    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "node_c")
    builder.add_edge("node_c", "node_d")
    builder.add_edge("node_d", END)

    # Deliberately stop immediately after node_b.
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["node_b"],
    )


def print_state(label: str, state: object) -> None:
    """Print a checkpoint state in a readable form."""

    print()
    print(label)
    print("-" * len(label))

    if hasattr(state, "values"):
        values = state.values
        next_nodes = state.next

        print(f"next nodes: {next_nodes}")
        print(f"node_a_result: {values.get('node_a_result')}")
        print(f"node_b_result: {values.get('node_b_result')}")
        print(f"node_c_result: {values.get('node_c_result')}")
        print(f"node_d_result: {values.get('node_d_result')}")
        print(f"execution_log: {values.get('execution_log')}")


def main() -> None:
    checkpoint_path = Path(CHECKPOINT_DB)

    # Remove an old demo database so every transcript starts from a
    # deterministic clean checkpoint state.
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Phase 4 Task 15 — SQLite Checkpointing")
    print("=" * 42)
    print(f"Checkpoint database: {CHECKPOINT_DB}")
    print(f"Thread ID: {THREAD_ID}")
    print()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)

        config = {
            "configurable": {
                "thread_id": THREAD_ID,
            }
        }

        # ---------------------------------------------------------------
        # First execution: node_a -> node_b -> INTERRUPT
        # ---------------------------------------------------------------

        print("FIRST EXECUTION")
        print("----------------")

        first_result = graph.invoke(
            {
                "query": "checkpoint demonstration",
                "node_a_result": "",
                "node_b_result": "",
                "node_c_result": "",
                "node_d_result": "",
                "execution_log": [],
            },
            config,
        )

        print()
        print("First execution returned at the configured breakpoint.")

        first_state = graph.get_state(config)
        print_state(
            "Checkpointed state after first execution",
            first_state,
        )

        print()
        print(f"Checkpoint database exists: {checkpoint_path.exists()}")

        # ---------------------------------------------------------------
        # Resume SAME thread
        # ---------------------------------------------------------------

        print()
        print("RESUMING SAME THREAD")
        print("--------------------")
        print(f"Using thread ID: {THREAD_ID}")
        print("Passing None so LangGraph resumes from the checkpoint.")

        resumed_result = graph.invoke(
            None,
            config,
        )

        print()
        print("Resumed execution completed.")

        final_state = graph.get_state(config)
        print_state(
            "Final state after resume",
            final_state,
        )

        # ---------------------------------------------------------------
        # Explicit verification
        # ---------------------------------------------------------------

        execution_log = final_state.values["execution_log"]

        print()
        print("CHECKPOINT VERIFICATION")
        print("-----------------------")
        print(f"Execution log: {execution_log}")

        node_a_count = execution_log.count("node_a executed")
        node_b_count = execution_log.count("node_b executed")
        node_c_count = execution_log.count("node_c executed")
        node_d_count = execution_log.count("node_d executed")

        print(f"node_a execution count: {node_a_count}")
        print(f"node_b execution count: {node_b_count}")
        print(f"node_c execution count: {node_c_count}")
        print(f"node_d execution count: {node_d_count}")

        assert node_a_count == 1, (
            "node_a should execute exactly once."
        )

        assert node_b_count == 1, (
            "node_b should execute exactly once."
        )

        assert node_c_count == 1, (
            "node_c should execute exactly once after resume."
        )

        assert node_d_count == 1, (
            "node_d should execute exactly once after resume."
        )

        assert final_state.values["node_a_result"] == (
            "Node A completed."
        )

        assert final_state.values["node_b_result"] == (
            "Node B completed."
        )

        assert final_state.values["node_c_result"] == (
            "Node C completed."
        )

        assert final_state.values["node_d_result"] == (
            "Node D completed."
        )

        assert final_state.next == ()

        print()
        print("Checkpoint verification: PASS")
        print("node_a and node_b were completed before the pause.")
        print("node_a and node_b were NOT re-executed after resume.")
        print("node_c and node_d executed only after resume.")
        print("Same thread_id was used for both executions.")


if __name__ == "__main__":
    main()