"""
Conditional Routing Logic for the Multi-Agent LangGraph Workflow.

Directs workflow execution based on the Reviewer's status and the number
of revisions performed:
- APPROVED -> END
- REJECTED and revision_count < 2 -> WORKER (revision cycle)
- REJECTED and revision_count >= 2 -> FAILURE (terminal failure, no infinite loop)
"""

from typing import Literal
from app.state import WorkflowState


def route_after_review(state: WorkflowState) -> Literal["END", "WORKER", "FAILURE"]:
    """
    Evaluates Reviewer outcome and revision counter to determine the next graph node.

    Args:
        state: The current WorkflowState dictionary.

    Returns:
        One of the literal branch names: "END", "WORKER", or "FAILURE".

    Raises:
        ValueError: If review_status is missing or unrecognized.
    """
    review_status = state.get("review_status")
    revision_count = state.get("revision_count", 0)

    if review_status == "APPROVED":
        return "END"

    elif review_status == "REJECTED":
        if revision_count < 2:
            return "WORKER"
        else:
            return "FAILURE"

    else:
        raise ValueError(
            f"Invalid or missing review_status: '{review_status}'. "
            f"Expected 'APPROVED' or 'REJECTED'."
        )
