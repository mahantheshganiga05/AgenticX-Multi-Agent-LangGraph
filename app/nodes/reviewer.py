"""
Reviewer Node Implementation.

The Reviewer Agent evaluates whether the Worker's output satisfies the original task
and plan with respect to relevance, completeness, clarity, and quality.

Outputs:
- review_status: "APPROVED" or "REJECTED"
- review_feedback: Explanation of the decision and actionable feedback if rejected.
- revision_count: Incremented upon rejection to track the revision cycle.
- final_result: Populated with worker_result upon approval.
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import WorkflowState


def parse_llm_review(raw_text: str) -> Tuple[Literal["APPROVED", "REJECTED"], str]:
    """Parses LLM output into review_status and review_feedback."""
    upper = raw_text.upper()
    feedback = raw_text.strip()

    if "STATUS: APPROVED" in upper or "STATUS:APPROVED" in upper:
        status: Literal["APPROVED", "REJECTED"] = "APPROVED"
    elif "STATUS: REJECTED" in upper or "STATUS:REJECTED" in upper:
        status = "REJECTED"
    elif "APPROVED" in upper and "REJECTED" not in upper:
        status = "APPROVED"
    else:
        status = "REJECTED"

    # Extract feedback if formatted as FEEDBACK: ...
    if "FEEDBACK:" in upper:
        idx = upper.find("FEEDBACK:")
        feedback = raw_text[idx + len("FEEDBACK:") :].strip()

    return status, feedback


def evaluate_mock_policy(
    state: WorkflowState,
    policy: Union[str, Callable[[WorkflowState], Tuple[Literal["APPROVED", "REJECTED"], str]]],
) -> Tuple[Literal["APPROVED", "REJECTED"], str]:
    """Evaluates the state using a deterministic policy (useful for testing and demo)."""
    if callable(policy):
        return policy(state)

    revision_count = state.get("revision_count", 0)
    worker_result = state.get("worker_result", "")

    if policy == "always_approve":
        return "APPROVED", "Approved: Output thoroughly satisfies all criteria."

    elif policy == "always_reject":
        return (
            "REJECTED",
            "Critique: Output fails quality standards. Needs deeper depth and concrete real-world case studies.",
        )

    elif policy == "reject_once":
        if revision_count == 0:
            return (
                "REJECTED",
                "Critique: Initial output lacks sufficient real-world examples and rigorous practical details.",
            )
        else:
            return (
                "APPROVED",
                "Approved: Revised output successfully incorporated real-world examples and addresses all previous critiques.",
            )

    # Default 'auto' heuristic for mock mode:
    # If the output contains "(Revised)" or has already undergone revision, approve it.
    if "(Revised)" in worker_result or revision_count > 0:
        return (
            "APPROVED",
            "Approved: Revised submission satisfies completeness and depth requirements.",
        )
    else:
        return (
            "REJECTED",
            "Critique: Output is too brief. Please expand with concrete examples, practical implementation notes, and deeper analysis.",
        )


def create_reviewer_node(
    llm: Optional[Any] = None,
    policy: Union[str, Callable[[WorkflowState], Tuple[Literal["APPROVED", "REJECTED"], str]]] = "auto",
) -> Callable[[WorkflowState], Dict[str, Any]]:
    """
    Factory creating the Reviewer node.

    Args:
        llm: Optional LangChain chat model.
        policy: Deterministic policy when llm is None ('auto', 'always_approve', 'always_reject', 'reject_once', or custom callable).

    Returns:
        Node function updating WorkflowState with review_status, review_feedback,
        revision_count (incremented on rejection), and final_result (on approval).
    """

    def reviewer_node(state: WorkflowState) -> Dict[str, Any]:
        task = state.get("task", "")
        plan = state.get("plan", [])
        worker_result = state.get("worker_result", "")
        current_revisions = state.get("revision_count", 0)

        if not worker_result or not worker_result.strip():
            # Immediate rejection if worker output is empty
            new_revisions = current_revisions + 1
            return {
                "review_status": "REJECTED",
                "review_feedback": "Review error: Worker result was empty.",
                "revision_count": new_revisions,
                "final_result": None,
            }

        if llm is None:
            status, feedback = evaluate_mock_policy(state, policy)
        else:
            system_prompt = (
                "You are a specialized Reviewer Agent in a multi-agent system. "
                "Your role is to rigorously evaluate whether the Worker's result satisfies "
                "the user's task and the planned steps.\n"
                "Evaluate for: relevance, completeness, clarity, accuracy, and depth.\n\n"
                "You MUST format your output exactly as:\n"
                "STATUS: [APPROVED or REJECTED]\n"
                "FEEDBACK: [Clear, constructive explanation of your decision and specific areas to improve if rejected]\n"
            )
            user_content = (
                f"Original Task: {task}\n\n"
                f"Plan:\n" + "\n".join(f"- {s}" for s in plan) + "\n\n"
                f"Worker Result to Evaluate:\n{worker_result}\n\n"
                f"Revision Count So Far: {current_revisions}"
            )

            try:
                response = llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_content),
                    ]
                )
                raw_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                status, feedback = parse_llm_review(raw_text)
            except Exception as e:
                # Fallback to mock policy if LLM call fails
                status, feedback = evaluate_mock_policy(state, policy)
                feedback += f" (Note: LLM fallback triggered due to: {str(e)})"

        if status == "APPROVED":
            return {
                "review_status": "APPROVED",
                "review_feedback": feedback,
                "revision_count": current_revisions,
                "final_result": worker_result,
            }
        else:
            new_revisions = current_revisions + 1
            return {
                "review_status": "REJECTED",
                "review_feedback": feedback,
                "revision_count": new_revisions,
                "final_result": None,
            }

    return reviewer_node
