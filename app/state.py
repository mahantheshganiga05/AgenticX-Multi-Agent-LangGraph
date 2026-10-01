"""
Shared State Schema for the Multi-Agent LangGraph Workflow.

This module defines the explicit typed dictionary that tracks data flow and
decisions across all specialized nodes: Planner, Worker, Reviewer, and Failure.
"""

from typing import List, Literal, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """
    Explicit state schema shared across all nodes in the workflow.

    Fields:
        task (str): The original user request / problem statement.
        plan (List[str]): Structured execution steps formulated by the Planner node.
        worker_result (str): Output produced or revised by the Worker node.
        review_status (Literal["APPROVED", "REJECTED"]): Outcome decision from Reviewer node.
        review_feedback (str): Actionable feedback or explanation from the Reviewer.
        revision_count (int): Counter tracking the number of revisions attempted.
        final_result (Optional[str]): Approved final output upon successful workflow completion.
        error (Optional[str]): Failure description if workflow terminates unsuccessfully.
    """

    task: str
    plan: List[str]
    worker_result: str
    review_status: Literal["APPROVED", "REJECTED"]
    review_feedback: str
    revision_count: int
    final_result: Optional[str]
    error: Optional[str]


def create_initial_state(task: str) -> WorkflowState:
    """
    Constructs a pristine initial state for the LangGraph workflow.

    Args:
        task: The user-supplied task string.

    Returns:
        WorkflowState initialized with default values.
    """
    if not task or not task.strip():
        raise ValueError("Task cannot be empty or blank.")

    return {
        "task": task.strip(),
        "plan": [],
        "worker_result": "",
        "review_status": "REJECTED",
        "review_feedback": "",
        "revision_count": 0,
        "final_result": None,
        "error": None,
    }
