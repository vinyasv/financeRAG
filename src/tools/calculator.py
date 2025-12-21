"""Safe calculator tool for arithmetic operations."""

import ast
import operator
import re
from typing import Any

from .base import Tool
from ..models import ToolName


class CalculatorTool(Tool):
    """
    Safe calculator that evaluates math expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /, **
    - Parentheses for grouping
    - References to previous step results: {step_1}, {step_1.revenue}
    """
    
    name = ToolName.CALCULATOR
    description = "Evaluate mathematical expressions safely. Can reference previous step results."
    
    # Allowed operators for safe evaluation
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Reference pattern: {step_id} or {step_id.field} or {step_id.field.subfield}
    REFERENCE_PATTERN = re.compile(r'\{([^}]+)\}')
    
    async def execute(self, input_str: str, context: dict[str, Any] | None = None) -> float:
        """
        Evaluate a math expression.
        
        Args:
            input_str: Math expression, may contain references like {step_1.revenue}
            context: Dict mapping step IDs to their results
            
        Returns:
            The computed result as a float
        """
        # Resolve references first
        resolved = self._resolve_references(input_str, context or {})
        
        # Parse and evaluate
        return self._safe_eval(resolved)
    
    def _resolve_references(self, expression: str, context: dict[str, Any]) -> str:
        """Replace {step_id.field} references with actual values."""
        
        def replace_ref(match: re.Match) -> str:
            ref = match.group(1)  # e.g., "step_1.revenue" or "step_3"
            parts = ref.split(".")
            
            # Get the step result
            step_id = parts[0]
            if step_id not in context:
                raise ValueError(f"Reference to unknown step: {step_id}")
            
            value = context[step_id]
            
            # Navigate nested fields
            for part in parts[1:]:
                if isinstance(value, dict):
                    if part not in value:
                        raise ValueError(f"Field '{part}' not found in {step_id}")
                    value = value[part]
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    raise ValueError(f"Cannot access '{part}' on {type(value)}")
            
            # Convert to number
            if isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, str):
                # Try to parse as number
                try:
                    return str(float(value.replace(",", "").replace("$", "")))
                except ValueError:
                    raise ValueError(f"Cannot convert '{value}' to number")
            else:
                raise ValueError(f"Cannot use {type(value)} in calculation")
        
        return self.REFERENCE_PATTERN.sub(replace_ref, expression)
    
    def _safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a math expression using AST parsing.
        
        This avoids the security risks of eval() by only allowing
        specific, safe mathematical operations.
        """
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {expression}") from e
    
    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate an AST node."""
        
        # Numeric constant
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        
        
        # Binary operation: left op right
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            
            op_func = self.SAFE_OPERATORS[op_type]
            
            # Handle division by zero
            if op_type == ast.Div and right == 0:
                raise ValueError("Division by zero")
            
            return op_func(left, right)
        
        # Unary operation: -x or +x
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            
            return self.SAFE_OPERATORS[op_type](operand)
        
        # Parenthesized expression
        elif isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# Convenience function for direct use
async def calculate(expression: str, context: dict[str, Any] | None = None) -> float:
    """
    Evaluate a math expression.
    
    Examples:
        >>> await calculate("2 + 3 * 4")
        14.0
        >>> await calculate("{step_1.revenue} * 0.7", {"step_1": {"revenue": 4200000}})
        2940000.0
    """
    tool = CalculatorTool()
    return await tool.execute(expression, context)

