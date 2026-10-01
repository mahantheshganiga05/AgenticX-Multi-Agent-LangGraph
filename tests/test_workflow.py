"""
Automated Pytest Suite for Multi-Agent LangGraph Workflow.

Verifies:
- Test 1: Immediate approval on the first attempt (Planner -> Worker -> Reviewer -> APPROVED -> END)
- Test 2: One rejection, Worker revision with feedback, then approval (revision_count == 1, final_result success)
- Test 3: Two rejections, routing to FAILURE node, terminating cleanly (revision_count == 2, error recorded)
- Test 4: Infinite loop prevention guarantee
- Test 5: Router unit tests for all boundary conditions
- Test 6: Node isolation: Planner outputs plan, not final answer
- Test 7: Worker reads and addresses review feedback on revision
- Test 8: Validation of input task
"""

import pytest
from app.graph import create_workflow_graph
from app.routing.review_router import route_after_review
from app.state import WorkflowState, create_initial_state


class TestMultiAgentWorkflow:
    """Test suite covering all mandatory evaluation criteria."""

    def test_scenario_1_immediate_approval(self):
        """
        TEST 1:
        Reviewer approves on the first attempt.
        Expected:
            Planner -> Worker -> Reviewer -> APPROVED -> END
            revision_count == 0
            final_result is populated and successful.
            error is None.
        """
        graph = create_workflow_graph(review_policy="always_approve")
        initial_state = create_initial_state("Explain applications of generative AI in healthcare.")

        final_state = graph.invoke(initial_state)

        # Assertions
        assert final_state["review_status"] == "APPROVED"
        assert final_state["revision_count"] == 0
        assert final_state["final_result"] is not None
        assert len(final_state["final_result"]) > 0
        assert final_state["error"] is None
        assert len(final_state["plan"]) >= 3

    def test_scenario_2_one_rejection_then_approval(self):
        """
        TEST 2:
        Reviewer rejects once and then approves the revised submission.
        Expected:
            Planner -> Worker -> Reviewer -> REJECTED -> Worker (revision) -> Reviewer -> APPROVED -> END
        Verify:
            revision_count == 1
            final_result is successful.
            review_status == 'APPROVED'.
        """
        graph = create_workflow_graph(review_policy="reject_once")
        initial_state = create_initial_state("Design a fault-tolerant microservice architecture.")

        final_state = graph.invoke(initial_state)

        # Assertions
        assert final_state["review_status"] == "APPROVED"
        assert final_state["revision_count"] == 1
        assert final_state["final_result"] is not None
        assert final_state["error"] is None
        assert "(Revised)" in final_state["final_result"] or "Revisions Applied" in final_state["final_result"]

    def test_scenario_3_two_rejections_lead_to_failure(self):
        """
        TEST 3:
        Reviewer rejects twice.
        Expected:
            Planner -> Worker -> Reviewer -> REJECTED (count=1)
                    -> Worker revision -> Reviewer -> REJECTED (count=2)
                    -> FAILURE -> END
        Verify:
            revision_count == 2
            review_status == 'REJECTED'
            workflow ends in failure with error populated.
            final_result is None.
        """
        graph = create_workflow_graph(review_policy="always_reject")
        initial_state = create_initial_state("Write an overview of quantum computing algorithms.")

        final_state = graph.invoke(initial_state)

        # Assertions
        assert final_state["review_status"] == "REJECTED"
        assert final_state["revision_count"] == 2
        assert final_state["final_result"] is None
        assert final_state["error"] is not None
        assert "Workflow terminated after 2 reviewer rejection(s)" in final_state["error"]

    def test_scenario_4_infinite_loop_prevention(self):
        """
        TEST 4:
        Verify that there is no infinite loop.
        The graph must strictly terminate even under stubborn reviewer rejections.
        """
        execution_trace = []

        def tracking_reviewer(state: WorkflowState):
            count = state.get("revision_count", 0)
            execution_trace.append(f"review_attempt_{count + 1}")
            return {
                "review_status": "REJECTED",
                "review_feedback": f"Rejection #{count + 1}: Insufficient rigor.",
                "revision_count": count + 1,
                "final_result": None,
            }

        graph = create_workflow_graph(reviewer_node_fn=tracking_reviewer)
        initial_state = create_initial_state("Test infinite loop termination guard.")

        final_state = graph.invoke(initial_state)

        # Exactly 2 review attempts occurred before transitioning to failure and terminating
        assert execution_trace == ["review_attempt_1", "review_attempt_2"]
        assert final_state["revision_count"] == 2
        assert final_state["error"] is not None
        assert final_state["final_result"] is None

    def test_route_after_review_logic(self):
        """
        TEST 5:
        Unit tests for conditional router route_after_review.
        """
        # 1. Approved -> END
        assert route_after_review({"review_status": "APPROVED", "revision_count": 0}) == "END"
        assert route_after_review({"review_status": "APPROVED", "revision_count": 1}) == "END"

        # 2. Rejected with revision available (< 2) -> WORKER
        assert route_after_review({"review_status": "REJECTED", "revision_count": 0}) == "WORKER"
        assert route_after_review({"review_status": "REJECTED", "revision_count": 1}) == "WORKER"

        # 3. Rejected twice (>= 2) -> FAILURE
        assert route_after_review({"review_status": "REJECTED", "revision_count": 2}) == "FAILURE"
        assert route_after_review({"review_status": "REJECTED", "revision_count": 3}) == "FAILURE"

        # 4. Invalid or missing status -> ValueError
        with pytest.raises(ValueError):
            route_after_review({"review_status": "UNKNOWN", "revision_count": 0})

        with pytest.raises(ValueError):
            route_after_review({"revision_count": 0})

    def test_planner_does_not_produce_final_answer(self):
        """
        TEST 6:
        Planner generates a structured plan list, not the final answer.
        """
        from app.nodes.planner import create_planner_node

        planner = create_planner_node(llm=None)
        state = create_initial_state("Analyze zero-knowledge proofs in blockchain.")
        output = planner(state)

        assert "plan" in output
        assert isinstance(output["plan"], list)
        assert len(output["plan"]) >= 3
        # Ensure it didn't write worker_result or final_result
        assert "worker_result" not in output
        assert "final_result" not in output

    def test_worker_incorporates_reviewer_feedback(self):
        """
        TEST 7:
        Worker explicitly consumes reviewer feedback during a revision step.
        """
        from app.nodes.worker import create_worker_node

        worker = create_worker_node(llm=None)
        state = create_initial_state("Build an API gateway.")
        state["plan"] = ["Step 1", "Step 2"]
        state["review_feedback"] = "Please add circuit breaker pattern details."
        state["revision_count"] = 1

        output = worker(state)
        result = output["worker_result"]

        assert "circuit breaker pattern details" in result
        assert "Revised" in result

    def test_empty_task_rejection(self):
        """
        TEST 8:
        Verify initial state rejects empty tasks.
        """
        with pytest.raises(ValueError, match="Task cannot be empty"):
            create_initial_state("")

        with pytest.raises(ValueError, match="Task cannot be empty"):
            create_initial_state("   ")
