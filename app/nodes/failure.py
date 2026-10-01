"""
Failure Node Implementation.

Executes when the workflow reaches its terminal failure state, such as when
the Reviewer Agent rejects the work twice. This guarantees the workflow halts
gracefully and documents the failure mode without infinite loops.
"""

from typing import Any, Dict
from app.state import WorkflowState


def failure_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Handles terminal workflow failure.

    Records a clear error message in the shared state and ensures final_result is None.
    """
    revisions = state.get("revision_count", 0)
    last_feedback = state.get("review_feedback", "Quality requirements not met.")

    error_message = (
        f"Workflow terminated after {revisions} reviewer rejection(s). "
        f"Last reviewer feedback: {last_feedback}"
    )

    return {
        "error": error_message,
        "final_result": None,
    }
