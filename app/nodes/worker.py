"""
Worker Node Implementation.

The Worker Agent is responsible for executing the steps laid out by the Planner.
If the Reviewer previously rejected a draft, the Worker inspects the reviewer_feedback
and generates an updated, revised result that specifically addresses the critiques.
"""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import WorkflowState


def default_mock_worker_execution(
    task: str,
    plan: List[str],
    feedback: str = "",
    is_revision: bool = False,
) -> str:
    """Generates structured content in mock mode, simulating initial or revised work."""
    plan_summary = "\n".join(f"  - Step {i+1}: {step}" for i, step in enumerate(plan))

    if is_revision and feedback:
        return (
            f"# Comprehensive Execution: {task} (Revised)\n\n"
            f"### Plan Executed:\n{plan_summary}\n\n"
            f"### Revisions Applied Based on Reviewer Feedback:\n"
            f"> Incorporating critique: {feedback}\n\n"
            f"### Detailed Analysis & Practical Implementation:\n"
            f"1. **Core Foundations**: Fully explored the fundamental principles of '{task}', "
            f"addressing previous gaps in depth and context.\n"
            f"2. **Specific Applications & Real-World Case Studies**: Concrete examples include "
            f"predictive modeling, automated workflows, diagnostic imaging, and operational optimization.\n"
            f"3. **Synthesis & Validation**: Every point specified in the structured plan has been "
            f"rigorously addressed and verified for completeness, clarity, and relevance."
        )

    return (
        f"# Initial Execution: {task}\n\n"
        f"### Plan Executed:\n{plan_summary}\n\n"
        f"### Summary of Findings:\n"
        f"Comprehensive exploration addressing the user request for '{task}'. "
        f"Initial implementation covering foundational concepts, key methodologies, "
        f"and structured domain highlights."
    )


def create_worker_node(
    llm: Optional[Any] = None,
) -> Callable[[WorkflowState], Dict[str, Any]]:
    """
    Factory creating the Worker node.

    Args:
        llm: Optional LangChain chat model. If None, deterministic execution is used.

    Returns:
        Node function updating WorkflowState with worker_result.
    """

    def worker_node(state: WorkflowState) -> Dict[str, Any]:
        task = state.get("task", "")
        plan = state.get("plan", [])
        feedback = state.get("review_feedback", "")
        current_result = state.get("worker_result", "")
        revision_count = state.get("revision_count", 0)

        is_revision = bool(feedback and revision_count > 0)

        if llm is None:
            result = default_mock_worker_execution(
                task=task,
                plan=plan,
                feedback=feedback,
                is_revision=is_revision,
            )
            return {"worker_result": result}

        if is_revision:
            system_prompt = (
                "You are a specialized Worker Agent in a multi-agent system. "
                "Your previous submission was REJECTED by the Reviewer Agent. "
                "You MUST carefully inspect the review feedback, address all critiques, "
                "and produce a significantly improved, complete, and high-quality final result."
            )
            user_content = (
                f"Original Task: {task}\n\n"
                f"Execution Plan:\n" + "\n".join(f"- {s}" for s in plan) + "\n\n"
                f"Previous Output:\n{current_result}\n\n"
                f"Reviewer Feedback to Address:\n{feedback}\n\n"
                f"Please produce the fully revised, complete response."
            )
        else:
            system_prompt = (
                "You are a specialized Worker Agent in a multi-agent system. "
                "Your task is to execute the structured plan provided by the Planner "
                "and deliver a thorough, clear, high-quality response satisfying the user task."
            )
            user_content = (
                f"Task: {task}\n\n"
                f"Execution Plan:\n" + "\n".join(f"- {s}" for s in plan) + "\n\n"
                f"Execute the plan and generate the full, detailed answer."
            )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )
            raw_content = (
                response.content if hasattr(response, "content") else str(response)
            )
            return {"worker_result": raw_content}
        except Exception as e:
            fallback = default_mock_worker_execution(
                task=task,
                plan=plan,
                feedback=feedback,
                is_revision=is_revision,
            )
            return {
                "worker_result": fallback,
                "error": f"Worker LLM notice: {str(e)} (fallback execution applied)",
            }

    return worker_node
