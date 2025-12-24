"""Tests for the calculator tool."""

import pytest
import asyncio
from src.tools.calculator import CalculatorTool, calculate
from src.models import CalculationTranscript, OperandBinding


class TestCalculatorTool:
    """Test the calculator tool."""
    
    @pytest.fixture
    def calculator(self):
        return CalculatorTool()
    
    def test_basic_addition(self, calculator):
        result = asyncio.run(calculator.execute("2 + 3"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 5.0
        assert result.formula_description == "Sum"
    
    def test_basic_subtraction(self, calculator):
        result = asyncio.run(calculator.execute("10 - 4"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 6.0
        assert result.formula_description == "Difference"
    
    def test_multiplication(self, calculator):
        result = asyncio.run(calculator.execute("6 * 7"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 42.0
    
    def test_division(self, calculator):
        result = asyncio.run(calculator.execute("20 / 4"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 5.0
    
    def test_complex_expression(self, calculator):
        result = asyncio.run(calculator.execute("(10 + 5) * 2 - 3"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 27.0
    
    def test_large_numbers(self, calculator):
        result = asyncio.run(calculator.execute("4200000 * 0.7"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 2940000.0
    
    def test_percentage_calculation(self, calculator):
        result = asyncio.run(calculator.execute("(4800000 - 3200000) / 3200000 * 100"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 50.0
    
    def test_reference_resolution(self, calculator):
        context = {
            "step_1": {"revenue": 4200000, "costs": 2800000}
        }
        result = asyncio.run(calculator.execute("{step_1.revenue} * 0.7", context))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 2940000.0
        # Verify bindings are tracked
        assert len(result.bindings) == 1
        assert result.bindings[0].reference == "{step_1.revenue}"
        assert result.bindings[0].resolved_value == 4200000.0
    
    def test_multiple_references(self, calculator):
        context = {
            "step_1": {"revenue": 4200000},
            "step_2": {"costs": 2800000}
        }
        result = asyncio.run(calculator.execute("{step_1.revenue} - {step_2.costs}", context))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 1400000.0
        assert len(result.bindings) == 2
    
    def test_nested_reference(self, calculator):
        context = {
            "step_1": 1000,
            "step_2": 500
        }
        result = asyncio.run(calculator.execute("{step_1} + {step_2}", context))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 1500.0
    
    def test_division_by_zero(self, calculator):
        with pytest.raises(ValueError, match="Division by zero"):
            asyncio.run(calculator.execute("10 / 0"))
    
    def test_invalid_expression(self, calculator):
        with pytest.raises(ValueError):
            asyncio.run(calculator.execute("2 +"))
    
    def test_unknown_reference(self, calculator):
        with pytest.raises(ValueError, match="unknown step"):
            asyncio.run(calculator.execute("{unknown_step} + 1", {}))
    
    def test_transcript_format_for_display(self, calculator):
        """Test that format_for_display produces readable output."""
        context = {"step_1": {"revenue": 145600000000}}
        result = asyncio.run(calculator.execute("{step_1.revenue} * 0.7", context))
        display = result.format_for_display()
        assert "Calculation" in display
        assert "$145.60B" in display or "145.60" in display  # Should format as B


class TestCalculateFunction:
    """Test the convenience calculate function."""
    
    def test_simple_calculation(self):
        result = asyncio.run(calculate("2 + 2"))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 4.0
    
    def test_with_context(self):
        context = {"step_1": 100}
        result = asyncio.run(calculate("{step_1} * 2", context))
        assert isinstance(result, CalculationTranscript)
        assert result.result == 200.0


class TestOperandBinding:
    """Test operand binding tracking."""
    
    def test_bindings_capture_source(self):
        """Verify bindings capture source information."""
        calculator = CalculatorTool()
        context = {
            "step_1": {"columns": ["revenue"], "revenue": 1000000}  # SQL-like result
        }
        result = asyncio.run(calculator.execute("{step_1.revenue} + 500", context))
        
        assert len(result.bindings) == 1
        binding = result.bindings[0]
        assert binding.source_step == "step_1"
        assert "SQL" in binding.source_description or "revenue" in binding.source_description
