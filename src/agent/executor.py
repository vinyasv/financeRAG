"""DAG executor with parallel execution."""

import asyncio
import logging
import re
import time
from typing import Any

from ..models import CalculationTranscript, ExecutionPlan, ToolCall, ToolName, ToolResult
from ..tools.base import Tool
from ..tools.calculator import CalculatorTool
from ..tools.get_document import GetDocumentTool
from ..tools.sql_query import SQLQueryTool
from ..tools.vector_search import VectorSearchTool

logger = logging.getLogger(__name__)


class DAGExecutor:
    """
    Executes execution plans with maximum parallelism.
    
    Features:
    - Parallel execution of independent steps
    - Automatic dependency resolution
    - Reference substitution between steps
    - Error handling and partial results
    """
    
    def __init__(
        self,
        calculator: CalculatorTool | None = None,
        sql_query: SQLQueryTool | None = None,
        vector_search: VectorSearchTool | None = None,
        get_document: GetDocumentTool | None = None
    ):
        """
        Initialize executor with tools.
        
        Args:
            calculator: Calculator tool instance
            sql_query: SQL query tool instance
            vector_search: Vector search tool instance
            get_document: Get document tool instance
        """
        self.tools: dict[ToolName, Tool] = {}
        
        # Calculator has no dependencies, can safely default
        self.tools[ToolName.CALCULATOR] = calculator or CalculatorTool()
        
        # Other tools require dependencies - only register if provided
        if sql_query:
            self.tools[ToolName.SQL_QUERY] = sql_query
        if vector_search:
            self.tools[ToolName.VECTOR_SEARCH] = vector_search
        if get_document:
            self.tools[ToolName.GET_DOCUMENT] = get_document
    
    async def execute(self, plan: ExecutionPlan) -> dict[str, ToolResult]:
        """
        Execute a plan with maximum parallelism.
        
        Args:
            plan: The execution plan to run
            
        Returns:
            Dict mapping step IDs to their results
        """
        results, _ = await self._run_layers(plan)
        return results
    
    async def _execute_step(
        self,
        step: ToolCall,
        context: dict[str, Any]
    ) -> ToolResult:
        """Execute a single step."""
        tool = self.tools.get(step.tool)

        if not tool:
            return ToolResult(
                step_id=step.id,
                tool=step.tool,
                success=False,
                result=None,
                error=f"Unknown tool: {step.tool}"
            )

        # The calculator has its own reference resolver that captures operand
        # bindings (provenance) for the audit transcript. Pre-substituting
        # references here would erase those bindings, so pass raw input through
        # for calculator steps and let it resolve references itself.
        if step.tool == ToolName.CALCULATOR:
            resolved_input = step.input
        else:
            resolved_input = self._resolve_references(step.input, context)

        # Execute the tool
        return await tool.run(step.id, resolved_input, context)

    async def _run_layers(
        self,
        plan: ExecutionPlan,
        step_runner=None,
        on_step_finished=None,
        on_layer_finished=None,
    ) -> tuple[dict[str, ToolResult], dict[str, Any]]:
        """Execute plan layers and optionally emit callbacks for monitoring."""
        results: dict[str, ToolResult] = {}
        result_values: dict[str, Any] = {}
        layers = plan.get_execution_layers()
        runner = step_runner or self._execute_step

        for layer_index, layer in enumerate(layers):
            layer_start = time.perf_counter()
            layer_tasks = [runner(step, result_values) for step in layer]
            layer_outputs = await asyncio.gather(*layer_tasks, return_exceptions=True)
            layer_elapsed = time.perf_counter() - layer_start

            for step, raw_output in zip(layer, layer_outputs):
                result, metadata = self._normalize_step_output(step, raw_output)
                results[step.id] = result
                self._store_result_value(result_values, step.id, result)

                if on_step_finished:
                    on_step_finished(step, result, metadata)

            if on_layer_finished:
                on_layer_finished(layer_index, layer, layer_elapsed)

        return results, result_values

    def _normalize_step_output(
        self,
        step: ToolCall,
        raw_output: ToolResult | tuple[ToolResult, float] | Exception,
    ) -> tuple[ToolResult, dict[str, Any]]:
        """Normalize outputs returned by the step runner."""
        if isinstance(raw_output, Exception):
            return (
                ToolResult(
                    step_id=step.id,
                    tool=step.tool,
                    success=False,
                    result=None,
                    error=str(raw_output),
                ),
                {},
            )

        if isinstance(raw_output, tuple):
            result, elapsed = raw_output
            return result, {"elapsed_s": elapsed}

        return raw_output, {}

    def _store_result_value(
        self,
        result_values: dict[str, Any],
        step_id: str,
        result: ToolResult,
    ) -> None:
        """Store raw result values used for later reference resolution."""
        if isinstance(result.result, CalculationTranscript):
            result_values[step_id] = result.result.result
        else:
            result_values[step_id] = result.result
    
    def _resolve_references(self, input_str: str, context: dict[str, Any]) -> str:
        """
        Resolve references to previous step results.
        
        References are in the format {step_id} or {step_id.field}
        """
        def replace_ref(match: re.Match) -> str:
            ref = match.group(1)
            parts = ref.split(".")
            
            step_id = parts[0]
            if step_id not in context:
                return match.group(0)  # Keep original if not found
            
            value = context[step_id]
            
            # Navigate nested fields
            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part, value)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    break
            
            if value is not None:
                return str(value)
            else:
                # Fail loudly: silently substituting "0" produces fabricated
                # numbers in downstream calculations and breaks the audit
                # contract. The Tool.run wrapper catches this and turns it
                # into a structured ToolResult(success=False, error=...).
                raise ValueError(f"Reference {ref} resolved to None")
        
        return re.sub(r'\{([^}]+)\}', replace_ref, input_str)


class ExecutionMonitor:
    """
    Monitor execution progress and timing.
    
    Useful for debugging and performance analysis.
    """
    
    def __init__(self):
        self.step_timings: dict[str, float] = {}
        self.layer_timings: list[float] = []
        self.total_time: float = 0
    
    async def execute_with_monitoring(
        self,
        executor: DAGExecutor,
        plan: ExecutionPlan
    ) -> tuple[dict[str, ToolResult], dict[str, Any]]:
        """
        Execute plan with timing information.
        
        Returns:
            Tuple of (results, timing_info)
        """
        start_time = time.perf_counter()

        self.step_timings = {}
        self.layer_timings = []

        results, _ = await executor._run_layers(
            plan,
            step_runner=lambda step, context: self._timed_execute(executor, step, context),
            on_step_finished=self._record_step_timing,
            on_layer_finished=self._record_layer_timing,
        )

        self.total_time = time.perf_counter() - start_time
        
        timing_info = {
            "total_time_ms": self.total_time * 1000,
            "layer_times_ms": [t * 1000 for t in self.layer_timings],
            "step_times_ms": {k: v * 1000 for k, v in self.step_timings.items()},
            "layer_count": len(self.layer_timings),
            "step_count": len(plan.steps)
        }
        
        return results, timing_info

    def _record_step_timing(self, step: ToolCall, _: ToolResult, metadata: dict[str, Any]) -> None:
        """Record timing emitted by the executor step runner."""
        elapsed = metadata.get("elapsed_s")
        if elapsed is not None:
            self.step_timings[step.id] = elapsed

    def _record_layer_timing(self, _: int, __: list[ToolCall], elapsed_s: float) -> None:
        """Record layer execution timing."""
        self.layer_timings.append(elapsed_s)
    
    async def _timed_execute(
        self,
        executor: DAGExecutor,
        step: ToolCall,
        context: dict[str, Any]
    ) -> tuple[ToolResult, float]:
        """Execute a step and return timing."""
        start = time.perf_counter()
        result = await executor._execute_step(step, context)
        elapsed = time.perf_counter() - start
        return result, elapsed
