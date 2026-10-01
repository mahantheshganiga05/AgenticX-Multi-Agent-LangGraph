"""
Planner Node Implementation.

The Planner Agent is responsible for decomposing the user task into clear,
actionable, structured steps. It does NOT generate the final answer directly;
its sole responsibility is strategic planning for the Worker.
"""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from app.state import WorkflowState


def default_mock_plan(task: str) -> List[str]:
    """Generates structured, task-specific steps in mock/deterministic mode."""
    cleaned = task.strip().rstrip(".")
    return [
        f"Analyze core requirements and domain scope for: {cleaned}",
        "Detail key concepts, mechanisms, and architectural foundations",
        "Provide concrete, real-world examples and practical applications",
        "Synthesize comprehensive explanations into a rigorous, polished response",
    ]


def parse_plan_text(text: str) -> List[str]:
    """Parses raw LLM text into a clean list of steps."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    steps = []
    for line in lines:
        # Strip common list indicators like "1.", "1)", "-", "*"
        cleaned = line
        if cleaned and cleaned[0].isdigit():
            cleaned = cleaned.lstrip("0123456789.-) ")
        elif cleaned.startswith(("-", "*", "•")):
            cleaned = cleaned.lstrip("-*• ")
        cleaned = cleaned.strip()
        if cleaned:
            steps.append(cleaned)
    return steps or [text.strip()]


def create_planner_node(
    llm: Optional[Any] = None,
) -> Callable[[WorkflowState], Dict[str, Any]]:
    """
    Factory creating the Planner node.

    Args:
        llm: Optional LangChain chat model. If None, deterministic planning is used.

    Returns:
        Node function updating WorkflowState with the structured plan.
    """

    def planner_node(state: WorkflowState) -> Dict[str, Any]:
        task = state.get("task", "").strip()
        if not task:
            return {
                "plan": [],
                "error": "Planner error: Task is empty or missing.",
            }

        if llm is None:
            plan = default_mock_plan(task)
            return {"plan": plan}

        system_prompt = (
            "You are a specialized Planner Agent in a multi-agent system. "
            "Your sole objective is to analyze the user's task and decompose it into "
            "3 to 5 clear, sequential, actionable steps for a Worker Agent. "
            "CRITICAL: Do NOT execute or answer the task yourself. Only output the steps, "
            "one per line, numbered 1 to N."
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Task: {task}"),
                ]
            )
            raw_content = (
                response.content if hasattr(response, "content") else str(response)
            )
            plan = parse_plan_text(raw_content)
            return {"plan": plan}
        except Exception as e:
            # Fallback to deterministic plan if LLM call fails
            plan = default_mock_plan(task)
            return {
                "plan": plan,
                "error": f"Planner LLM notice: {str(e)} (fallback plan applied)",
            }

    return planner_node
