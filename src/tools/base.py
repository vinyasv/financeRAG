"""Base class for all tools."""

from abc import ABC, abstractmethod
from typing import Any
import time

from ..models import ToolName, ToolResult


class Tool(ABC):
    """Base class for all tools in the agent toolkit."""
    
    name: ToolName
    description: str
    
    @abstractmethod
    async def execute(self, input_str: str, context: dict[str, Any] | None = None) -> Any:
        """
        Execute the tool with the given input.
        
        Args:
            input_str: The input string for the tool
            context: Optional context with previous results for reference resolution
            
        Returns:
            The tool's result (type varies by tool)
        """
        pass
    
    async def run(self, step_id: str, input_str: str, context: dict[str, Any] | None = None) -> ToolResult:
        """
        Run the tool and wrap result in ToolResult.
        
        Args:
            step_id: ID of the execution step
            input_str: Input for the tool
            context: Optional context for reference resolution
            
        Returns:
            ToolResult with success/failure status
        """
        start_time = time.perf_counter()
        
        try:
            result = await self.execute(input_str, context)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolResult(
                step_id=step_id,
                tool=self.name,
                success=True,
                result=result,
                error=None,
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolResult(
                step_id=step_id,
                tool=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time
            )

