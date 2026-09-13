"""LangGraph orchestration for the Cred Domain Support Agent.

Phase 2 graph:

    START
      |
      v
    load_memory
      |
      v
    classify_intent
      |
      +--------------------+
      |                    |
      v                    v
    rag_answer       status_lookup
      |                    |
      +---------+----------+
                |
                v
         format_response
                |
                v
          persist_memory
                |
                v
               END

The graph remains deterministic under MOCK_LLM.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.memory import save_history, load_history
from agent.tools import check_loan_application_status
from rag.generation import generate_grounded_answer
from agent.schema import validate_agent_response
from agent.guardrails import (
    enforce_rag_groundedness,
    guard_input,
)

RECORD_ID_PATTERN = re.compile(r"\bLA-\d{4}\b")

STATUS_INTENT_TERMS = {
    "status",
    "application status",
    "loan status",
    "application update",
    "loan application",
    "application",
    "approved",
    "approval",
    "rejected",
    "submitted",
    "disbursed",
    "under review",
}


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    thread_id: str
    query: str
    history: list[dict[str, str]]
    route: str
    record_id: str | None
    tool_result: dict[str, Any] | None
    rag_result: dict[str, Any] | None
    answer: str
    escalation_score: float | None
    trace: list[str]
    guardrail_result: dict[str, Any] | None
    input_blocked: bool
    rag_grounded: bool
    rag_guardrail_allowed: bool


def _append_trace(
    state: AgentState,
    node_name: str,
) -> list[str]:
    """Return a new trace containing the current node."""

    return [
        *state.get("trace", []),
        node_name,
    ]


# ---------------------------------------------------------------------------
# Node 1: Load persisted history
# ---------------------------------------------------------------------------

def load_memory(state: AgentState) -> AgentState:
    """Load the existing history for this thread ID."""

    history = load_history(state["thread_id"])

    return {
        **state,
        "history": history,
        "trace": _append_trace(state, "load_memory"),
    }


# ---------------------------------------------------------------------------
# Node 2: Intent classification
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    """Classify the query deterministically as RAG or status lookup.

    A status route requires both a valid LA-#### record ID and status/
    application language.

    Memory is used here for follow-up questions. If the current message
    does not contain its own record ID but the previous user message
    referenced one, that record ID can be carried into the status route.

    Example:

        Turn 1: "What is the status of LA-0001?"
        Turn 2: "What is its current status?"

    The second turn can therefore use the first turn's context.
    """

    query = state["query"]
    lowered = query.lower()

    current_record_match = RECORD_ID_PATTERN.search(query)

    if current_record_match:
        record_id = current_record_match.group(0)
    else:
        record_id = None

        # Search backwards through earlier user messages for the most
        # recently mentioned application ID.
        for message in reversed(state.get("history", [])):
            if message.get("role") != "user":
                continue

            previous_match = RECORD_ID_PATTERN.search(
                message.get("content", "")
            )

            if previous_match:
                record_id = previous_match.group(0)
                break

    has_status_intent = any(
        term in lowered
        for term in STATUS_INTENT_TERMS
    )

    # Explicit application references are strong evidence of status intent.
    # Follow-up language such as "its status" can use the previous record ID.
    has_follow_up_reference = any(
        phrase in lowered
        for phrase in (
            "its status",
            "their status",
            "that application",
            "that loan",
            "this application",
            "this loan",
        )
    )

    if record_id and (has_status_intent or has_follow_up_reference):
        route = "status"
    else:
        route = "rag"
        record_id = None

    return {
        **state,
        "route": route,
        "record_id": record_id,
        "trace": _append_trace(state, "classify_intent"),
    }


# ---------------------------------------------------------------------------
# Node 3A: Frozen Phase 1 RAG generation
# ---------------------------------------------------------------------------

def rag_answer(state: AgentState) -> AgentState:
    """Call the frozen Phase 1 grounded-generation interface."""

    result = generate_grounded_answer(
        query=state["query"],
        strategy="sentence",
        top_k=3,
    )

    grounding = enforce_rag_groundedness(result)

    return {
        **state,
        "rag_result": result,
        "answer": grounding["answer"],
        "rag_grounded": result.get("grounded", False),
        "rag_guardrail_allowed": grounding["allowed"],
        "trace": _append_trace(state, "rag_answer"),
    }


# ---------------------------------------------------------------------------
# Node 3B: Phase 2 status lookup
# ---------------------------------------------------------------------------

def status_lookup(state: AgentState) -> AgentState:
    """Call the Phase 2 loan-application status tool."""

    record_id = state.get("record_id")

    if not record_id:
        result = {
            "found": False,
            "record_id": None,
            "error": "No loan-application record ID was provided.",
        }
    else:
        result = check_loan_application_status(record_id)

    if result["found"]:
        answer = (
            f"Loan application {result['record_id']} has status "
            f"{result['status']}. "
            f"The loan amount is ₹{result['loan_amount_inr']:,}. "
            f"Escalation score: {result['escalation_score']:.3f} "
            f"(threshold: {result['escalation_threshold']:.3f})."
        )
        escalation_score = result["escalation_score"]
    else:
        answer = result["error"]
        escalation_score = None

    return {
        **state,
        "tool_result": result,
        "answer": answer,
        "escalation_score": escalation_score,
        "trace": _append_trace(state, "status_lookup"),
    }


# ---------------------------------------------------------------------------
# Node 4: Common response formatting
# ---------------------------------------------------------------------------

def format_response(state: AgentState) -> AgentState:
    """Finalize the response before persistence."""

    if state.get("input_blocked"):
        answer = (
            "I can't process that request because it contains "
            "an instruction that conflicts with the support agent's "
            "operating rules."
        )

        return {
            **state,
            "answer": answer,
            "route": "rag",
            "escalation_score": None,
            "trace": _append_trace(state, "format_response"),
        }

    return {
        **state,
        "trace": _append_trace(state, "format_response"),
    }


# ---------------------------------------------------------------------------
# Node 5: Persist conversation
# ---------------------------------------------------------------------------

def persist_memory(state: AgentState) -> AgentState:
    """Persist both the current user message and assistant response."""

    history = load_history(state["thread_id"])

    history.append(
        {
            "role": "user",
            "content": state["query"],
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": state["answer"],
        }
    )

    # append_message is deliberately not used here because both messages
    # are persisted as one atomic history update.
    from agent.memory import save_history

    save_history(
        state["thread_id"],
        history,
    )

    return {
        **state,
        "history": history,
        "trace": _append_trace(state, "persist_memory"),
    }

def route_after_input_guard(
    state: AgentState,
) -> Literal["blocked", "allowed"]:
    """Stop malicious input before it reaches tools or RAG."""

    if state["input_blocked"]:
        return "blocked"

    return "allowed"

# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_classification(
    state: AgentState,
) -> Literal["rag_answer", "status_lookup"]:
    """Select the next node based on deterministic intent classification."""

    if state["route"] == "status":
        return "status_lookup"

    return "rag_answer"

def guard_input_node(state: AgentState) -> AgentState:
    """Apply input-side PII masking and injection detection."""

    result = guard_input(state["query"])

    return {
        **state,
        "guardrail_result": result,
        "query": result["masked_query"],
        "input_blocked": not result["allowed"],
        "trace": _append_trace(state, "guard_input"),
    }

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Construct and compile the Cred support-agent graph."""

    builder = StateGraph(AgentState)

    builder.add_node("guard_input", guard_input_node)
    builder.add_node("load_memory", load_memory)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("rag_answer", rag_answer)
    builder.add_node("status_lookup", status_lookup)
    builder.add_node("format_response", format_response)
    builder.add_node("persist_memory", persist_memory)

    builder.add_edge(START, "guard_input")

    builder.add_conditional_edges(
        "guard_input",
        route_after_input_guard,
        {
            "blocked": "format_response",
            "allowed": "load_memory",
        },
    )

    builder.add_edge("load_memory", "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "rag_answer": "rag_answer",
            "status_lookup": "status_lookup",
        },
    )

    builder.add_edge("rag_answer", "format_response")
    builder.add_edge("status_lookup", "format_response")

    builder.add_edge("format_response", "persist_memory")
    builder.add_edge("persist_memory", END)

    return builder.compile()


graph = build_graph()


def run_query(
    query: str,
    thread_id: str,
) -> AgentState:
    """Run one query through the guarded support-agent graph."""

    initial_state: AgentState = {
        "thread_id": thread_id,
        "query": query,
        "history": [],
        "record_id": None,
        "tool_result": None,
        "rag_result": None,
        "answer": "",
        "escalation_score": None,
        "trace": [],
        "guardrail_result": None,
        "input_blocked": False,
        "rag_grounded": False,
        "rag_guardrail_allowed": True,
    }

    result = graph.invoke(initial_state)

    rag_result = result.get("rag_result")

    structured_response = {
        "answer": result["answer"],
        "route": result.get("route", "rag"),
        "escalation_score": result.get("escalation_score"),
        "thread_id": result["thread_id"],
        "record_id": result.get("record_id"),
        "grounded": (
            rag_result.get("grounded")
            if rag_result is not None
            else None
        ),
    }

    validated = validate_agent_response(
        structured_response
    )

    return {
        **result,
        "structured_response": validated.model_dump(),
    }