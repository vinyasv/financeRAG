"""Audit-transcript regression tests for the DAG executor.

Covers three intertwined bugs that previously broke the calculator's
"every number traceable to its source" guarantee:

1. The executor pre-substituted {step.field} references for calculator
   steps, erasing the operand bindings the calculator's own resolver
   builds.
2. sql_query returned a bare scalar for 1x1 results, so {step.value}
   could not navigate into a non-dict context value.
3. The executor's reference resolver silently substituted "0" when a
   reference resolved to None, fabricating numbers downstream.

All tests use the sync def + asyncio.run(...) pattern from
tests/test_executor.py. No pytest-asyncio.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add repo root to path so `import src...` works when pytest is invoked
# from anywhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.executor import DAGExecutor
from src.models import (
    CalculationTranscript,
    ExecutionPlan,
    ToolCall,
    ToolName,
)
from src.tools.base import Tool


class _StubTool(Tool):
    """Minimal Tool stub that returns a canned result and records its inputs.

    We can't reuse MagicMock here because Tool.run() is the real wrapper that
    calls .execute() and produces ToolResult, and we want to exercise that
    wrapper end-to-end.
    """

    def __init__(self, name: ToolName, return_value: Any):
        self.name = name
        self.description = "stub"
        self._return_value = return_value
        self.calls: list[tuple[str, str]] = []  # list of (step_id, input_str)

    async def execute(
        self, input_str: str, context: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append(("__execute__", input_str))
        return self._return_value

    async def run(
        self, step_id: str, input_str: str, context: dict[str, Any] | None = None
    ):
        # Record what the executor actually handed in (post-substitution for
        # non-calculator tools, raw for calculator tools).
        self.calls.append((step_id, input_str))
        return await super().run(step_id, input_str, context)


def _build_plan(query: str, steps: list[ToolCall]) -> ExecutionPlan:
    return ExecutionPlan(query=query, reasoning="audit transcript test", steps=steps)


def test_calculator_bindings_populated_from_sql_scalar():
    """A calculator step that consumes a 1x1 SQL result must produce a
    CalculationTranscript whose `bindings` list is non-empty and points back
    to the SQL step via `source_step`. This is the smoke test for fixes #1
    and #2 working together.
    """
    sql_stub = _StubTool(
        ToolName.SQL_QUERY,
        return_value={"value": 100, "column": "revenue"},
    )

    executor = DAGExecutor(sql_query=sql_stub)

    plan = _build_plan(
        query="double the revenue",
        steps=[
            ToolCall(
                id="step_1",
                tool=ToolName.SQL_QUERY,
                input="get revenue",
                depends_on=[],
                description="fetch revenue",
            ),
            ToolCall(
                id="step_2",
                tool=ToolName.CALCULATOR,
                input="{step_1.value} * 2",
                depends_on=["step_1"],
                description="double it",
            ),
        ],
    )

    results = asyncio.run(executor.execute(plan))

    assert results["step_1"].success, results["step_1"].error
    assert results["step_2"].success, results["step_2"].error

    transcript = results["step_2"].result
    assert isinstance(transcript, CalculationTranscript)
    assert transcript.result == 200.0

    # The headline assertion: bindings are populated and traceable.
    assert transcript.bindings, "Calculator transcript bindings must not be empty"
    assert any(
        b.source_step == "step_1" for b in transcript.bindings
    ), f"At least one binding must reference step_1; got {transcript.bindings!r}"

    # And the executor must have handed the calculator the RAW expression
    # (with the {step_1.value} placeholder still intact) so its own resolver
    # could capture provenance. If the executor pre-substituted it, the
    # calculator's bindings would have been empty.
    # The first .calls entry is the run() invocation.
    cal_run_call = results["step_2"].result  # already verified above
    # Re-verify by checking the resolved expression contains the substituted
    # numeric value (proof the calculator did the substitution itself).
    assert "100" in transcript.resolved_expression
    assert transcript.original_expression == "{step_1.value} * 2"


def test_calculator_fails_loudly_on_unresolved_reference():
    """When a SQL step returns the empty-result shape and a calculator step
    references a field that does not exist on it, the calculator step must
    surface a clear failure rather than silently producing 0.

    This guards fix #3 (executor.py:_resolve_references hard-fail) AND fix
    #1 (the calculator path bypasses _resolve_references) -- because the
    calculator's own resolver also raises on missing fields, the failure
    surfaces either way.
    """
    sql_stub = _StubTool(
        ToolName.SQL_QUERY,
        return_value={"columns": [], "rows": [], "message": "No results found"},
    )

    executor = DAGExecutor(sql_query=sql_stub)

    plan = _build_plan(
        query="double a missing value",
        steps=[
            ToolCall(
                id="step_1",
                tool=ToolName.SQL_QUERY,
                input="get something nonexistent",
                depends_on=[],
                description="fetch nothing",
            ),
            ToolCall(
                id="step_2",
                tool=ToolName.CALCULATOR,
                input="{step_1.value} * 2",
                depends_on=["step_1"],
                description="multiply nothing",
            ),
        ],
    )

    results = asyncio.run(executor.execute(plan))

    assert results["step_1"].success
    assert results["step_2"].success is False, (
        "Calculator must NOT silently succeed with a fabricated value"
    )
    assert results["step_2"].error is not None
    error_text = results["step_2"].error.lower()
    assert (
        "step_1" in error_text
        or "none" in error_text
        or "resolved to none" in error_text
    ), f"Error must mention the failed reference; got: {results['step_2'].error!r}"


def test_executor_resolves_references_for_non_calculator_steps():
    """Regression guard for fix #1: the executor must STILL substitute
    {step.field} references for non-calculator tools. Calculator is the only
    special case.
    """
    sql_stub = _StubTool(
        ToolName.SQL_QUERY,
        return_value={"value": 42, "column": "amount"},
    )
    vector_stub = _StubTool(
        ToolName.VECTOR_SEARCH,
        return_value=[{"content": "stub chunk", "score": 0.9}],
    )

    executor = DAGExecutor(sql_query=sql_stub, vector_search=vector_stub)

    plan = _build_plan(
        query="search using sql output",
        steps=[
            ToolCall(
                id="step_1",
                tool=ToolName.SQL_QUERY,
                input="get an amount",
                depends_on=[],
                description="fetch amount",
            ),
            ToolCall(
                id="step_2",
                tool=ToolName.VECTOR_SEARCH,
                input="search for {step_1.value}",
                depends_on=["step_1"],
                description="search using the amount",
            ),
        ],
    )

    results = asyncio.run(executor.execute(plan))

    assert results["step_1"].success, results["step_1"].error
    assert results["step_2"].success, results["step_2"].error

    # The executor must have substituted {step_1.value} -> "42" before
    # invoking vector_search.run(). Look at the recorded calls.
    run_inputs = [
        input_str for (step_id, input_str) in vector_stub.calls if step_id == "step_2"
    ]
    assert run_inputs, "vector_search.run() should have been invoked for step_2"
    assert run_inputs[0] == "search for 42", (
        "Executor must substitute references for non-calculator tools; "
        f"got {run_inputs[0]!r}"
    )
