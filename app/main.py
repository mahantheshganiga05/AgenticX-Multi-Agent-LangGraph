"""
Interactive CLI & Demonstration for the AgenticX Multi-Agent LangGraph Workflow.

Demonstrates:
- Planner decomposing task into steps
- Worker generating initial output
- Reviewer approving or rejecting with feedback
- Revision loops and two-rejection terminal failure prevention
- Real-time step-by-step execution visualization via graph.stream()
"""

import argparse
import sys
from typing import Dict, Any

from app.graph import create_workflow_graph
from app.llm import get_configured_llm
from app.state import create_initial_state


DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def print_banner():
    """Prints the application banner."""
    print("\n" + DIVIDER)
    print("   AGENTICX LABS - PROJECT 2: MULTI-AGENT WORKFLOW WITH LANGGRAPH")
    print("   Architecture: Planner -> Worker -> Reviewer (Conditional Revision)")
    print(DIVIDER + "\n")


def display_planner_step(data: Dict[str, Any]):
    print(f"\n[NODE: PLANNER] Formulation Complete")
    print(SUB_DIVIDER)
    plan = data.get("plan", [])
    if plan:
        print("Generated Execution Steps:")
        for idx, step in enumerate(plan, 1):
            print(f"  {idx}. {step}")
    if data.get("error"):
        print(f"  [Notice]: {data.get('error')}")


def display_worker_step(data: Dict[str, Any]):
    print(f"\n[NODE: WORKER] Task Execution Complete")
    print(SUB_DIVIDER)
    worker_res = data.get("worker_result", "")
    # Print preview or full content
    lines = worker_res.strip().splitlines()
    preview = "\n".join(lines[:8])
    print(preview)
    if len(lines) > 8:
        print(f"  ... [{len(lines) - 8} more lines omitted for conciseness]")


def display_reviewer_step(data: Dict[str, Any]):
    status = data.get("review_status", "UNKNOWN")
    revisions = data.get("revision_count", 0)
    feedback = data.get("review_feedback", "")

    status_tag = f"[APPROVED]" if status == "APPROVED" else f"[REJECTED]"
    print(f"\n[NODE: REVIEWER] Evaluation: {status_tag}")
    print(SUB_DIVIDER)
    print(f"Review Status:   {status}")
    print(f"Revision Count:  {revisions} / 2 max")
    print(f"Review Feedback: {feedback}")


def display_failure_step(data: Dict[str, Any]):
    print(f"\n[NODE: FAILURE] Terminal Failure Encountered")
    print(SUB_DIVIDER)
    print(f"Failure Reason: {data.get('error')}")


def run_workflow(
    task: str,
    policy: str = "auto",
    provider: str = "mock",
    model: str = None,
):
    """Executes the graph with real-time streaming display of every node."""
    print_banner()
    print(f"Target Task:      '{task}'")
    print(f"Selected Policy:  {policy}")
    print(f"LLM Provider:     {provider}")
    print(DIVIDER)

    # Initialize LLM if not mock
    llm = None
    if provider.lower() != "mock":
        llm = get_configured_llm(provider=provider, model=model)
        if not llm:
            print(
                f"[WARNING] Could not initialize credentials for provider '{provider}'. "
                f"Falling back to deterministic mock mode."
            )

    # Compile the LangGraph workflow
    graph = create_workflow_graph(llm=llm, review_policy=policy)
    initial_state = create_initial_state(task)

    print("\n>>> Initiating Multi-Agent StateGraph Execution...\n")

    # Stream graph updates step by step
    final_state = initial_state.copy()
    for event in graph.stream(initial_state):
        for node_name, node_output in event.items():
            final_state.update(node_output)

            if node_name == "planner":
                display_planner_step(node_output)
            elif node_name == "worker":
                display_worker_step(node_output)
            elif node_name == "reviewer":
                display_reviewer_step(node_output)
            elif node_name == "failure":
                display_failure_step(node_output)

    # Final summary display
    print("\n" + DIVIDER)
    print("                 FINAL WORKFLOW SUMMARY")
    print(DIVIDER)
    print(f"Final Status:     {final_state.get('review_status')}")
    print(f"Total Revisions:  {final_state.get('revision_count', 0)}")

    if final_state.get("final_result"):
        print("\n[SUCCESS] Final Approved Result:")
        print(SUB_DIVIDER)
        print(final_state["final_result"])
    else:
        print("\n[FAILURE] Workflow Terminated Unsuccessfully:")
        print(SUB_DIVIDER)
        print(final_state.get("error", "Unknown workflow error."))

    print(DIVIDER + "\n")
    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="AgenticX Multi-Agent LangGraph Workflow CLI"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Explain the main applications of generative AI in healthcare.",
        help="Task for the multi-agent system to execute.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["auto", "always_approve", "reject_once", "always_reject"],
        default="reject_once",
        help="Review policy in deterministic/mock mode: 'reject_once' demonstrates the full revision cycle.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mock", "google", "openai"],
        default="mock",
        help="LLM provider: 'mock' (default, no API key needed), 'google', or 'openai'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name (e.g. gemini-2.0-flash, gpt-4o-mini).",
    )

    args = parser.parse_args()

    try:
        run_workflow(
            task=args.task,
            policy=args.policy,
            provider=args.provider,
            model=args.model,
        )
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
