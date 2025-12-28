"""DAG executor with parallel execution."""

import asyncio
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

from ..models import ExecutionPlan, ToolCall, ToolResult, ToolName, CalculationTranscript
from ..tools.base import Tool
from ..tools.calculator import CalculatorTool
from ..tools.sql_query import SQLQueryTool
from ..tools.vector_search import VectorSearchTool
from ..tools.get_document import GetDocumentTool


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
        results: dict[str, ToolResult] = {}
        result_values: dict[str, Any] = {}  # Raw values for reference resolution
        completed: set[str] = set()
        
        # Get execution layers (groups of steps that can run in parallel)
        layers = plan.get_execution_layers()
        
        for layer in layers:
            # Execute all steps in this layer in parallel
            layer_tasks = [
                self._execute_step(step, result_values)
                for step in layer
            ]
            
            layer_results = await asyncio.gather(*layer_tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(layer, layer_results):
                if isinstance(result, Exception):
                    # Handle exceptions
                    results[step.id] = ToolResult(
                        step_id=step.id,
                        tool=step.tool,
                        success=False,
                        result=None,
                        error=str(result)
                    )
                    result_values[step.id] = None
                else:
                    results[step.id] = result
                    # For CalculationTranscript, store the numeric result for reference resolution
                    # This allows dependent steps to use {step_id} and get the computed number
                    if isinstance(result.result, CalculationTranscript):
                        result_values[step.id] = result.result.result
                    else:
                        result_values[step.id] = result.result
                
                completed.add(step.id)
        
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
        
        # Resolve references in input
        resolved_input = self._resolve_references(step.input, context)
        
        # Execute the tool
        return await tool.run(step.id, resolved_input, context)
    
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
                logger.warning(f"Reference resolution: {ref} resolved to None, using '0'")
                return "0"
        
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
        
        results: dict[str, ToolResult] = {}
        result_values: dict[str, Any] = {}
        
        layers = plan.get_execution_layers()
        
        for layer_idx, layer in enumerate(layers):
            layer_start = time.perf_counter()
            
            # Execute layer
            layer_tasks = [
                self._timed_execute(executor, step, result_values)
                for step in layer
            ]
            
            layer_results = await asyncio.gather(*layer_tasks)
            
            layer_time = time.perf_counter() - layer_start
            self.layer_timings.append(layer_time)
            
            # Process results
            for step, (result, step_time) in zip(layer, layer_results):
                self.step_timings[step.id] = step_time
                results[step.id] = result
                # For CalculationTranscript, store the numeric result for reference resolution
                if isinstance(result.result, CalculationTranscript):
                    result_values[step.id] = result.result.result
                else:
                    result_values[step.id] = result.result
        
        self.total_time = time.perf_counter() - start_time
        
        timing_info = {
            "total_time_ms": self.total_time * 1000,
            "layer_times_ms": [t * 1000 for t in self.layer_timings],
            "step_times_ms": {k: v * 1000 for k, v in self.step_timings.items()},
            "layer_count": len(layers),
            "step_count": len(plan.steps)
        }
        
        return results, timing_info
    
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

