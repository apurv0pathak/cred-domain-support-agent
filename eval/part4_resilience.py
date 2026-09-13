"""
Phase 4 Task 16 — Timeouts and retries demonstration.

Demonstrates:

1. Exponential-backoff retry recovery from transient failures.
2. Per-node timeout handling.
3. Global graph timeout handling.

The frozen Phase 2 agent is not modified.
All failures are deterministic and simulated locally.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRY_MAX_ATTEMPTS = 4
RETRY_INITIAL_INTERVAL = 0.05
RETRY_MAX_INTERVAL = 0.20
RETRY_JITTER = 0.0

NODE_TIMEOUT_SECONDS = 0.20
NODE_SIMULATED_DELAY_SECONDS = 0.60

GLOBAL_TIMEOUT_SECONDS = 0.50
GLOBAL_SIMULATED_DELAY_SECONDS = 1.00


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for deterministic exponential-backoff retries."""

    max_attempts: int
    initial_interval: float
    max_interval: float
    jitter: float


RETRY_POLICY = RetryPolicy(
    max_attempts=RETRY_MAX_ATTEMPTS,
    initial_interval=RETRY_INITIAL_INTERVAL,
    max_interval=RETRY_MAX_INTERVAL,
    jitter=RETRY_JITTER,
)


class TransientFailure(Exception):
    """Raised when the simulated node experiences a transient failure."""


# ---------------------------------------------------------------------------
# Transient node
# ---------------------------------------------------------------------------

class TransientNode:
    """
    Deterministic node that fails the first two calls and then succeeds.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self) -> str:
        self.call_count += 1

        print(f"Transient node attempt {self.call_count}")

        if self.call_count <= 2:
            raise TransientFailure(
                f"Simulated transient failure on attempt "
                f"{self.call_count}"
            )

        return "Transient node succeeded."


async def run_with_retry(
    operation: Callable[[], object],
    policy: RetryPolicy,
) -> object:
    """
    Execute an async operation with exponential-backoff retries.

    Delay progression:

        initial_interval
        initial_interval * 2
        ...

    capped at max_interval.

    Jitter is explicitly configurable and set to zero for deterministic
    transcript output.
    """

    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()

        except Exception as exc:
            last_error = exc

            if attempt >= policy.max_attempts:
                raise

            backoff = min(
                policy.initial_interval * (2 ** (attempt - 1)),
                policy.max_interval,
            )

            # Deterministic jitter.
            jitter = policy.jitter

            delay = backoff + jitter

            print(
                f"  retrying after {delay:.2f}s "
                f"(attempt {attempt + 1}/{policy.max_attempts})"
            )

            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Per-node timeout
# ---------------------------------------------------------------------------

async def slow_node() -> str:
    """
    Simulated node that exceeds the configured per-node timeout.
    """

    print(
        f"Slow node sleeping for "
        f"{NODE_SIMULATED_DELAY_SECONDS:.2f}s"
    )

    await asyncio.sleep(NODE_SIMULATED_DELAY_SECONDS)

    return "Slow node completed."


async def run_with_node_timeout() -> None:
    """Execute a node with an explicit per-node timeout."""

    try:
        await asyncio.wait_for(
            slow_node(),
            timeout=NODE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        print(
            "Per-node timeout fired cleanly after "
            f"{NODE_TIMEOUT_SECONDS:.2f}s."
        )
        raise


# ---------------------------------------------------------------------------
# Global graph timeout
# ---------------------------------------------------------------------------

async def simulated_graph_run() -> str:
    """
    Simulated whole-graph execution that exceeds the global timeout.
    """

    print(
        f"Simulated graph work sleeping for "
        f"{GLOBAL_SIMULATED_DELAY_SECONDS:.2f}s"
    )

    await asyncio.sleep(GLOBAL_SIMULATED_DELAY_SECONDS)

    return "Graph completed."


async def run_with_global_timeout() -> None:
    """Execute the complete simulated run under a global timeout."""

    try:
        await asyncio.wait_for(
            simulated_graph_run(),
            timeout=GLOBAL_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        print(
            "Global graph timeout fired cleanly after "
            f"{GLOBAL_TIMEOUT_SECONDS:.2f}s."
        )
        raise


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

async def demonstrate_retry() -> None:
    print("RETRY DEMONSTRATION")
    print("-------------------")

    transient_node = TransientNode()

    result = await run_with_retry(
        transient_node,
        RETRY_POLICY,
    )

    print(f"Result: {result}")
    print(f"Total attempts: {transient_node.call_count}")

    assert transient_node.call_count == 3
    assert result == "Transient node succeeded."

    print("Retry verification: PASS")
    print(
        "The transient node failed twice and recovered on the "
        "third attempt."
    )
    print(
        "Retry policy: "
        f"max_attempts={RETRY_POLICY.max_attempts}, "
        f"initial_interval={RETRY_POLICY.initial_interval:.2f}s, "
        f"max_interval={RETRY_POLICY.max_interval:.2f}s, "
        f"jitter={RETRY_POLICY.jitter:.2f}s"
    )


async def demonstrate_node_timeout() -> None:
    print()
    print("PER-NODE TIMEOUT DEMONSTRATION")
    print("------------------------------")

    print(
        f"Configured node timeout: "
        f"{NODE_TIMEOUT_SECONDS:.2f}s"
    )
    print(
        f"Simulated node duration: "
        f"{NODE_SIMULATED_DELAY_SECONDS:.2f}s"
    )

    started = time.perf_counter()

    try:
        await run_with_node_timeout()

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started

        print(f"Observed elapsed time: {elapsed:.2f}s")
        print("Per-node timeout verification: PASS")
        print(
            "The simulated node was cancelled by the per-node "
            "timeout instead of hanging."
        )


async def demonstrate_global_timeout() -> None:
    print()
    print("GLOBAL GRAPH TIMEOUT DEMONSTRATION")
    print("----------------------------------")

    print(
        f"Configured global timeout: "
        f"{GLOBAL_TIMEOUT_SECONDS:.2f}s"
    )
    print(
        f"Simulated total duration: "
        f"{GLOBAL_SIMULATED_DELAY_SECONDS:.2f}s"
    )

    started = time.perf_counter()

    try:
        await run_with_global_timeout()

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started

        print(f"Observed elapsed time: {elapsed:.2f}s")
        print("Global timeout verification: PASS")
        print(
            "The simulated whole-graph run was cancelled by the "
            "global timeout."
        )


async def main() -> None:
    print("Phase 4 Task 16 — Timeouts and Retries")
    print("=" * 42)
    print()

    await demonstrate_retry()
    await demonstrate_node_timeout()
    await demonstrate_global_timeout()

    print()
    print("ALL TASK 16 VERIFICATIONS: PASS")


if __name__ == "__main__":
    asyncio.run(main())