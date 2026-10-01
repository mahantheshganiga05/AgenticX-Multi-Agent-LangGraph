"""
LangGraph Multi-Agent Workflow Definition.

Assembles Planner, Worker, Reviewer, and Failure nodes into a state graph
with explicit conditional routing.
"""

from typing import Any, Callable, Optional, Union
from langgraph.graph import END, START, StateGraph

from app.nodes.failure import failure_node
from app.nodes.planner import create_planner_node
from app.nodes.reviewer import create_reviewer_node
from app.nodes.worker import create_worker_node
from app.routing.review_router import route_after_review
from app.state import WorkflowState


def create_workflow_graph(
    llm: Optional[Any] = None,
    planner_node_fn: Optional[Callable] = None,
    worker_node_fn: Optional[Callable] = None,
    reviewer_node_fn: Optional[Callable] = None,
    review_policy: Union[str, Callable] = "auto",
):
    """
    Constructs and compiles the multi-agent LangGraph workflow.

    Workflow topology:
        START -> planner -> worker -> reviewer
                                         |
                                         +--> [APPROVED] --> END
                                         +--> [REJECTED & revision_count < 2] --> worker
                                         +--> [REJECTED & revision_count >= 2] --> failure --> END

    Args:
        llm: Optional LangChain chat model.
        planner_node_fn: Optional custom planner node override (for testing).
        worker_node_fn: Optional custom worker node override (for testing).
        reviewer_node_fn: Optional custom reviewer node override (for testing).
        review_policy: Review evaluation policy for mock mode ('auto', 'always_approve',
                       'reject_once', 'always_reject', or custom callable).

    Returns:
        Compiled StateGraph application.
    """
    builder = StateGraph(WorkflowState)

    # 1. Resolve and register nodes
    planner = planner_node_fn or create_planner_node(llm=llm)
    worker = worker_node_fn or create_worker_node(llm=llm)
    reviewer = reviewer_node_fn or create_reviewer_node(llm=llm, policy=review_policy)

    builder.add_node("planner", planner)
    builder.add_node("worker", worker)
    builder.add_node("reviewer", reviewer)
    builder.add_node("failure", failure_node)

    # 2. Add static transitions
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", "reviewer")

    # 3. Add conditional edge from reviewer
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "END": END,
            "WORKER": "worker",
            "FAILURE": "failure",
        },
    )

    # 4. Terminal transition from failure to END
    builder.add_edge("failure", END)

    return builder.compile()
