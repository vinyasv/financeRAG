"""Safe calculator tool for arithmetic operations."""

import ast
import operator
import re
from typing import Any

from .base import Tool
from ..models import ToolName, CalculationTranscript, OperandBinding, QueryRefusal, RefusalReason


class ComparabilityError(Exception):
    """Raised when operands are not comparable for a calculation."""
    
    def __init__(self, message: str, what_was_found: str, what_is_missing: list[str]):
        super().__init__(message)
        self.message = message
        self.what_was_found = what_was_found
        self.what_is_missing = what_is_missing
    
    def to_refusal(self) -> QueryRefusal:
        """Convert to a QueryRefusal for structured handling."""
        return QueryRefusal(
            reason=RefusalReason.INCOMPARABLE_METRICS,
            explanation=self.message,
            what_was_found=self.what_was_found,
            what_is_missing=self.what_is_missing,
            suggested_alternatives=[
                "Query each metric separately",
                "Verify the metrics use the same accounting standard",
                "Check if the time periods are aligned"
            ]
        )


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
    
    async def execute(self, input_str: str, context: dict[str, Any] | None = None) -> CalculationTranscript | QueryRefusal:
        """
        Evaluate a math expression with full audit transparency.
        
        Args:
            input_str: Math expression, may contain references like {step_1.revenue}
            context: Dict mapping step IDs to their results
            
        Returns:
            CalculationTranscript with bindings, resolved expression, and result,
            OR QueryRefusal if comparability checks fail
        """
        ctx = context or {}
        
        try:
            # Resolve references and collect bindings
            resolved, bindings = self._resolve_references_with_bindings(input_str, ctx)
            
            # Parse and evaluate
            result = self._safe_eval(resolved)
            
            # Build the transcript
            return CalculationTranscript(
                original_expression=input_str,
                bindings=bindings,
                resolved_expression=resolved,
                result=result,
                formula_description=self._infer_formula_description(input_str)
            )
        except ComparabilityError as e:
            # Convert to structured refusal
            return e.to_refusal()
    
    def _infer_formula_description(self, expression: str) -> str | None:
        """Infer a human-readable description from the expression."""
        expr_lower = expression.lower()
        
        # Common financial calculations
        if " - " in expression and ("growth" in expr_lower or "change" in expr_lower):
            return "Change/Growth Calculation"
        elif " / " in expression:
            if "margin" in expr_lower:
                return "Margin Calculation"
            elif "ratio" in expr_lower:
                return "Ratio Calculation"
            return "Division"
        elif " * " in expression:
            if "%" in expression or "100" in expression:
                return "Percentage Calculation"
            return "Multiplication"
        elif " - " in expression:
            return "Difference"
        elif " + " in expression:
            return "Sum"
        return None
    
    def _resolve_references_with_bindings(
        self, expression: str, context: dict[str, Any]
    ) -> tuple[str, list[OperandBinding]]:
        """
        Replace {step_id.field} references with actual values and track bindings.
        
        Returns:
            Tuple of (resolved_expression, list_of_bindings)
        """
        bindings: list[OperandBinding] = []
        
        def replace_ref(match: re.Match) -> str:
            ref_full = match.group(0)  # e.g., "{step_1.revenue}"
            ref = match.group(1)       # e.g., "step_1.revenue"
            parts = ref.split(".")
            
            # Get the step result
            step_id = parts[0]
            if step_id not in context:
                raise ValueError(f"Reference to unknown step: {step_id}")
            
            value = context[step_id]
            field_path = parts[1:] if len(parts) > 1 else []
            
            # Build source description from context
            source_desc = self._build_source_description(step_id, field_path, context)
            
            # Navigate nested fields
            for part in field_path:
                if isinstance(value, dict):
                    if part not in value:
                        raise ValueError(f"Field '{part}' not found in {step_id}")
                    value = value[part]
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    raise ValueError(f"Cannot access '{part}' on {type(value)}")
            
            # Convert to number
            numeric_value: float
            if isinstance(value, (int, float)):
                numeric_value = float(value)
            elif isinstance(value, str):
                try:
                    numeric_value = float(value.replace(",", "").replace("$", ""))
                except ValueError:
                    raise ValueError(f"Cannot convert '{value}' to number")
            else:
                raise ValueError(f"Cannot use {type(value)} in calculation")
            
            # Record the binding
            bindings.append(OperandBinding(
                reference=ref_full,
                resolved_value=numeric_value,
                source_step=step_id,
                source_description=source_desc
            ))
            
            return str(numeric_value)
        
        resolved = self.REFERENCE_PATTERN.sub(replace_ref, expression)
        return resolved, bindings
    
    def _build_source_description(
        self, step_id: str, field_path: list[str], context: dict[str, Any]
    ) -> str:
        """Build a human-readable description of where a value came from."""
        step_result = context.get(step_id, {})
        
        # Try to get tool type from context metadata
        tool_type = "unknown"
        if isinstance(step_result, dict):
            if "columns" in step_result:
                tool_type = "SQL query"
            elif "content" in step_result:
                tool_type = "document"
        
        if field_path:
            field_name = ".".join(field_path)
            return f"{tool_type}: {field_name} from {step_id}"
        else:
            return f"{tool_type}: result from {step_id}"
    
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
    
    # Maximum depth for expression evaluation (DoS protection)
    MAX_AST_DEPTH = 50
    
    def _eval_node(self, node: ast.AST, depth: int = 0) -> float:
        """
        Recursively evaluate an AST node with depth limiting.
        
        Security: Limits recursion depth to prevent stack overflow from
        maliciously deeply nested expressions.
        """
        if depth > self.MAX_AST_DEPTH:
            raise ValueError(f"Expression too deeply nested (max {self.MAX_AST_DEPTH} levels)")
        
        # Numeric constant
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        
        
        # Binary operation: left op right
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, depth + 1)
            right = self._eval_node(node.right, depth + 1)
            
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
            operand = self._eval_node(node.operand, depth + 1)
            
            op_type = type(node.op)
            if op_type not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            
            return self.SAFE_OPERATORS[op_type](operand)
        
        # Parenthesized expression
        elif isinstance(node, ast.Expression):
            return self._eval_node(node.body, depth + 1)
        
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# Convenience function for direct use
async def calculate(expression: str, context: dict[str, Any] | None = None) -> CalculationTranscript | QueryRefusal:
    """
    Evaluate a math expression with full audit transparency.
    
    Examples:
        >>> result = await calculate("2 + 3 * 4")
        >>> result.result
        14.0
        >>> result = await calculate("{step_1.revenue} * 0.7", {"step_1": {"revenue": 4200000}})
        >>> result.result
        2940000.0
        >>> print(result.format_for_display())  # Shows full calculation transcript
        
    Returns:
        CalculationTranscript on success, or QueryRefusal if comparability checks fail.
    """
    tool = CalculatorTool()
    return await tool.execute(expression, context)

