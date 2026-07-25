"""Regression tests for strict reference resolution in the DAG executor.

`DAGExecutor._resolve_references` used to navigate nested fields with
`value.get(part, value)`. On a MISS that returns the PARENT dict, which was
then str()'d straight into the tool input:

    "... WHERE x = {step_1.net_income}" with {"step_1": {"revenue": 100}}
    -> "... WHERE x = {'revenue': 100}"

That is the same class of bug as the None case the P0.4 fix already guards:
a fabricated value reaching a tool. The non-dict branch was equally lax --
it `break`-ed and substituted whatever partial value it had reached.

These tests pin the strict behaviour while preserving the deliberate
pass-through for an UNKNOWN step_id (a plan input may contain literal braces).

Sync def + asyncio.run(...) pattern from tests/test_executor.py. No
pytest-asyncio.
"""

import asyncio
from typing import Any

import pytest

from src.agent.executor import DAGExecutor
from src.models import ExecutionPlan, ToolCall, ToolName
from src.tools.base import Tool


class _StubTool(Tool):
    """Minimal Tool that records the input the executor handed it."""

    def __init__(self, name: ToolName, return_value: Any):
        self.name = name
        self.description = "stub"
        self._return_value = return_value
        self.execute_inputs: list[str] = []

    async def execute(
        self, input_str: str, context: dict[str, Any] | None = None
    ) -> Any:
        self.execute_inputs.append(input_str)
        return self._return_value


def _build_plan(query: str, steps: list[ToolCall]) -> ExecutionPlan:
    return ExecutionPlan(query=query, reasoning="strict reference test", steps=steps)


# --- (a) the exact probe case: missing field must raise, not stringify a dict ---

def test_missing_field_on_known_step_raises():
    """The probe case. Must raise instead of substituting "{'revenue': 100, ...}"."""
    executor = DAGExecutor()
    context = {"step_1": {"revenue": 100, "year": 2024}}

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references(
            "SELECT * FROM t WHERE x = {step_1.net_income}", context
        )

    message = str(exc_info.value)
    assert "net_income" in message, f"Error must name the field; got {message!r}"
    assert "step_1" in message, f"Error must name the step; got {message!r}"


def test_missing_field_never_substitutes_parent_dict():
    """Belt-and-braces: no code path may leak the parent dict's repr."""
    executor = DAGExecutor()
    context = {"step_1": {"revenue": 100, "year": 2024}}

    try:
        resolved = executor._resolve_references("x = {step_1.net_income}", context)
    except ValueError:
        return  # correct behaviour

    pytest.fail(
        f"Missing field silently resolved to {resolved!r} instead of raising"
    )


def test_missing_nested_field_raises():
    """A miss on the SECOND hop of the path must raise too."""
    executor = DAGExecutor()
    context = {"step_1": {"totals": {"revenue": 100}}}

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references("{step_1.totals.net_income}", context)

    assert "net_income" in str(exc_info.value)


def test_non_navigable_value_raises_instead_of_breaking():
    """The old `else: break` substituted the partially-navigated value."""
    executor = DAGExecutor()
    context = {"step_1": 100}

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references("x = {step_1.value}", context)

    message = str(exc_info.value)
    assert "value" in message, f"Error must name the field; got {message!r}"


# --- (b) successful navigation still works ---

def test_successful_nested_navigation_still_resolves():
    executor = DAGExecutor()
    context = {"step_1": {"revenue": 100, "year": 2024}}

    assert executor._resolve_references("{step_1.revenue}", context) == "100"
    assert (
        executor._resolve_references("rev={step_1.revenue} yr={step_1.year}", context)
        == "rev=100 yr=2024"
    )


def test_deep_nested_navigation_still_resolves():
    executor = DAGExecutor()
    context = {"step_1": {"totals": {"revenue": 100}}}

    assert executor._resolve_references("{step_1.totals.revenue}", context) == "100"


def test_bare_step_reference_still_resolves():
    executor = DAGExecutor()

    assert executor._resolve_references("{step_1} * 2", {"step_1": 42}) == "42 * 2"


def test_public_data_attribute_navigation_still_resolves():
    """A genuine, public, non-callable data attribute is still navigable.

    This is the SAFE half of attribute access -- pydantic models (ToolResult,
    CalculationTranscript) are legitimately referenced this way. The unsafe
    half is pinned by the callable/private tests below.
    """
    class _Holder:
        revenue = 100

    executor = DAGExecutor()

    assert executor._resolve_references("{step_1.revenue}", {"step_1": _Holder()}) == "100"


# --- (b2) the attribute branch must not fabricate ---
# Regression for the round-1 gap: `elif hasattr(value, part)` ran BEFORE the
# missing-field raise, so getattr returned bound methods that were str()'d
# straight into the tool input. Both cases below are reachable in production --
# vector_search returns a bare list (src/tools/vector_search.py), and `count` /
# `title` are entirely plausible field names for an LLM to emit.

def test_bound_method_on_list_result_raises_not_stringified():
    executor = DAGExecutor()
    context = {"step_1": [{"content": "a"}]}

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references(
            "SELECT * FROM t WHERE x = {step_1.count}", context
        )

    message = str(exc_info.value)
    assert "count" in message
    assert "method" in message.lower(), (
        f"Error should explain it is a method, not data; got {message!r}"
    )


def test_bound_method_on_str_result_raises_not_stringified():
    executor = DAGExecutor()

    with pytest.raises(ValueError):
        executor._resolve_references("find {step_1.title}", {"step_1": "Acme 10-K"})


def test_private_attribute_access_raises():
    """`{step_1.__class__}` must not resolve to "<class 'str'>"."""
    executor = DAGExecutor()

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references("{step_1.__class__}", {"step_1": "x"})

    assert "private" in str(exc_info.value).lower()


def test_navigated_path_landing_on_container_raises():
    """A surviving list/dict is the parent-dict bug wearing a different hat."""
    executor = DAGExecutor()
    context = {"step_1": {"rows": [{"a": 1}, {"a": 2}]}}

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references("SELECT {step_1.rows}", context)

    assert "list" in str(exc_info.value)


# --- (c) unknown step_id still passes through untouched ---

def test_unknown_step_id_passes_through_untouched():
    executor = DAGExecutor()

    assert executor._resolve_references("{step_9.revenue}", {}) == "{step_9.revenue}"
    assert (
        executor._resolve_references("{step_9.revenue}", {"step_1": {"revenue": 1}})
        == "{step_9.revenue}"
    )


def test_literal_braces_pass_through_untouched():
    """A plan input may legitimately contain brace-delimited text."""
    executor = DAGExecutor()

    assert executor._resolve_references('{"a": 1}', {}) == '{"a": 1}'


# --- (d) end-to-end through DAGExecutor.execute ---

def test_missing_field_surfaces_as_failed_tool_result():
    """Routed through execute(), a missing field must yield a failed
    ToolResult with a useful error -- not a crash, and not a corrupted query
    string handed to the tool.
    """
    sql_stub = _StubTool(
        ToolName.SQL_QUERY,
        return_value={"revenue": 100, "year": 2024},
    )
    vector_stub = _StubTool(
        ToolName.VECTOR_SEARCH,
        return_value=[{"content": "stub", "score": 0.5}],
    )

    executor = DAGExecutor(sql_query=sql_stub, vector_search=vector_stub)

    plan = _build_plan(
        query="search using a field that does not exist",
        steps=[
            ToolCall(
                id="step_1",
                tool=ToolName.SQL_QUERY,
                input="get the revenue row",
                depends_on=[],
                description="fetch row",
            ),
            ToolCall(
                id="step_2",
                tool=ToolName.VECTOR_SEARCH,
                input="net income was {step_1.net_income}",
                depends_on=["step_1"],
                description="search using a missing field",
            ),
        ],
    )

    results = asyncio.run(executor.execute(plan))

    assert results["step_1"].success, results["step_1"].error
    assert results["step_2"].success is False, (
        "A missing reference field must NOT silently succeed with the parent "
        "dict stringified into the tool input"
    )
    assert results["step_2"].error is not None
    error_text = results["step_2"].error
    assert "net_income" in error_text and "step_1" in error_text, (
        f"Error must identify the field and step; got {error_text!r}"
    )

    # The downstream tool must never have run with a corrupted input.
    assert not vector_stub.execute_inputs, (
        f"vector_search.execute() must not have been called; got "
        f"{vector_stub.execute_inputs!r}"
    )


def test_successful_reference_still_reaches_tool_end_to_end():
    """The strict check must not break the happy path through execute()."""
    sql_stub = _StubTool(
        ToolName.SQL_QUERY,
        return_value={"value": 42, "column": "amount"},
    )
    vector_stub = _StubTool(
        ToolName.VECTOR_SEARCH,
        return_value=[{"content": "stub", "score": 0.9}],
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

    assert results["step_2"].success, results["step_2"].error
    assert vector_stub.execute_inputs == ["search for 42"]


def test_none_reference_still_raises_end_to_end():
    """The pre-existing P0.4 None guard must remain intact."""
    executor = DAGExecutor()
    context = {"step_1": {"value": None}}

    with pytest.raises(ValueError) as exc_info:
        executor._resolve_references("x = {step_1.value}", context)

    assert "resolved to None" in str(exc_info.value)


# --- (e) the verbose production path ---
# src/rag_agent.py uses ExecutionMonitor.execute_with_monitoring whenever
# verbose=True, which substitutes a DIFFERENT step_runner (_timed_execute).
# Nothing else in this file exercises a raising step through that runner, so
# the strictness above was unproven on the path real verbose queries take.

def test_missing_field_fails_cleanly_through_execution_monitor():
    from src.agent.executor import ExecutionMonitor

    source = _StubTool(ToolName.SQL_QUERY, {"revenue": 100, "year": 2024})
    downstream = _StubTool(ToolName.VECTOR_SEARCH, ["unused"])

    executor = DAGExecutor(sql_query=source, vector_search=downstream)
    plan = _build_plan(
        "verbose strict-ref",
        [
            ToolCall(
                id="step_1",
                tool=ToolName.SQL_QUERY,
                input="revenue",
                depends_on=[],
                description="source",
            ),
            ToolCall(
                id="step_2",
                tool=ToolName.VECTOR_SEARCH,
                input="lookup {step_1.net_income}",
                depends_on=["step_1"],
                description="references a field that does not exist",
            ),
        ],
    )

    monitor = ExecutionMonitor()
    results, timing = asyncio.run(monitor.execute_with_monitoring(executor, plan))

    assert results["step_1"].success is True
    assert results["step_2"].success is False, (
        "A missing field must surface as a failed ToolResult under the monitor too"
    )
    assert "net_income" in (results["step_2"].error or "")
    assert downstream.execute_inputs == [], (
        "The downstream tool must never be called with a fabricated input"
    )
    assert timing["step_count"] == 2
