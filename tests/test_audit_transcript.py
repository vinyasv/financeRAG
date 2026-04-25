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
from typing import Any
from unittest.mock import AsyncMock

from src.agent.executor import DAGExecutor
from src.models import (
    CalculationTranscript,
    ExecutionPlan,
    ToolCall,
    ToolName,
)
from src.storage.sqlite_store import SQLiteStore
from src.tools.base import Tool
from src.tools.sql_query import SQLQueryTool


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

    # The headline assertion: bindings are populated and traceable. We pin the
    # full shape so a regression that broke OperandBinding fields (e.g. always
    # zero resolved_value) gets caught instead of a vacuous source_step pass.
    assert len(transcript.bindings) == 1
    binding = transcript.bindings[0]
    assert binding.source_step == "step_1"
    assert binding.reference == "{step_1.value}"
    assert binding.resolved_value == 100.0

    # The calculator must have done the substitution itself (proof that the
    # executor passed RAW {step_1.value} through without pre-substituting).
    assert transcript.original_expression == "{step_1.value} * 2"
    assert "100" in transcript.resolved_expression


def test_calculator_fails_loudly_on_unresolved_reference():
    """When a SQL step returns the empty-result shape and a calculator step
    references a field that does not exist on it, the calculator step must
    surface a clear failure rather than silently producing 0.

    This is a regression guard for the calculator's own resolver. The path
    where the *executor's* `_resolve_references` raises on None is covered
    separately in test_executor_raises_on_none_reference_for_non_calculator_tool.
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
        or "not found" in error_text
    ), f"Error must mention the failed reference; got: {results['step_2'].error!r}"


def test_executor_raises_on_none_reference_for_non_calculator_tool():
    """Targeted regression for the executor's own _resolve_references: when a
    non-calculator tool's input references a step field that resolves to None,
    the executor must raise ValueError (caught by Tool.run as success=False)
    instead of silently substituting "0".

    This is the path that the *executor* fix patches (calculator steps bypass
    _resolve_references entirely per fix #1, so they cannot exercise this code
    path). Without this test, fix #3 has no coverage in the executor itself.
    """
    sql_stub = _StubTool(
        ToolName.SQL_QUERY,
        # 1x1 with NULL value -- new shape from fix #2.
        return_value={"value": None, "column": "missing_total"},
    )
    vector_stub = _StubTool(
        ToolName.VECTOR_SEARCH,
        return_value=[{"content": "stub", "score": 0.5}],
    )

    executor = DAGExecutor(sql_query=sql_stub, vector_search=vector_stub)

    plan = _build_plan(
        query="search for a missing number",
        steps=[
            ToolCall(
                id="step_1",
                tool=ToolName.SQL_QUERY,
                input="get missing total",
                depends_on=[],
                description="fetch missing total",
            ),
            ToolCall(
                id="step_2",
                tool=ToolName.VECTOR_SEARCH,
                input="search for amount {step_1.value}",
                depends_on=["step_1"],
                description="search using missing value",
            ),
        ],
    )

    results = asyncio.run(executor.execute(plan))

    assert results["step_1"].success
    assert results["step_2"].success is False, (
        "Non-calculator tool with a None-resolving reference must NOT silently "
        "execute with a fabricated '0' substituted in its input"
    )
    assert results["step_2"].error is not None
    error_text = results["step_2"].error.lower()
    assert "resolved to none" in error_text or "none" in error_text, (
        f"Error must come from the executor's None-reference guard; "
        f"got: {results['step_2'].error!r}"
    )

    # And the vector_search tool must NOT have been invoked at all -- the
    # exception fires before .run() reaches .execute().
    execute_calls = [
        c for (step_id, c) in vector_stub.calls if step_id == "__execute__"
    ]
    assert not execute_calls, (
        "vector_search.execute() must not have been called; got "
        f"{execute_calls!r}"
    )


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


def test_sql_query_tool_returns_dict_shape_for_1x1_result(tmp_path, monkeypatch):
    """Targeted regression for fix #2 against the REAL SQLQueryTool.

    Previously a 1x1 result returned a bare scalar; now it must return
    {"value": <scalar>, "column": <colname>} so downstream {step.value}
    references resolve correctly. This guards against someone reverting
    src/tools/sql_query.py:99-104 to the old scalar shortcut -- the stub-based
    test above could not catch that.
    """
    store = SQLiteStore(db_path=tmp_path / "scalar.db")
    tool = SQLQueryTool(sqlite_store=store)

    # Bypass NL->SQL generation by pinning the SQL the tool will execute.
    monkeypatch.setattr(
        tool, "_generate_sql", AsyncMock(return_value="SELECT 100 AS revenue")
    )

    result = asyncio.run(tool.execute("get the revenue"))

    assert isinstance(result, dict), (
        f"1x1 SQL result must be a dict, not a bare scalar; got {type(result).__name__}: {result!r}"
    )
    assert result == {"value": 100, "column": "revenue"}
