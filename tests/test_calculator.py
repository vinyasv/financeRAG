"""Tests for the calculator tool."""

import pytest
import asyncio
from src.tools.calculator import CalculatorTool, calculate


class TestCalculatorTool:
    """Test the calculator tool."""
    
    @pytest.fixture
    def calculator(self):
        return CalculatorTool()
    
    def test_basic_addition(self, calculator):
        result = asyncio.run(calculator.execute("2 + 3"))
        assert result == 5.0
    
    def test_basic_subtraction(self, calculator):
        result = asyncio.run(calculator.execute("10 - 4"))
        assert result == 6.0
    
    def test_multiplication(self, calculator):
        result = asyncio.run(calculator.execute("6 * 7"))
        assert result == 42.0
    
    def test_division(self, calculator):
        result = asyncio.run(calculator.execute("20 / 4"))
        assert result == 5.0
    
    def test_complex_expression(self, calculator):
        result = asyncio.run(calculator.execute("(10 + 5) * 2 - 3"))
        assert result == 27.0
    
    def test_large_numbers(self, calculator):
        result = asyncio.run(calculator.execute("4200000 * 0.7"))
        assert result == 2940000.0
    
    def test_percentage_calculation(self, calculator):
        result = asyncio.run(calculator.execute("(4800000 - 3200000) / 3200000 * 100"))
        assert result == 50.0
    
    def test_reference_resolution(self, calculator):
        context = {
            "step_1": {"revenue": 4200000, "costs": 2800000}
        }
        result = asyncio.run(calculator.execute("{step_1.revenue} * 0.7", context))
        assert result == 2940000.0
    
    def test_multiple_references(self, calculator):
        context = {
            "step_1": {"revenue": 4200000},
            "step_2": {"costs": 2800000}
        }
        result = asyncio.run(calculator.execute("{step_1.revenue} - {step_2.costs}", context))
        assert result == 1400000.0
    
    def test_nested_reference(self, calculator):
        context = {
            "step_1": 1000,
            "step_2": 500
        }
        result = asyncio.run(calculator.execute("{step_1} + {step_2}", context))
        assert result == 1500.0
    
    def test_division_by_zero(self, calculator):
        with pytest.raises(ValueError, match="Division by zero"):
            asyncio.run(calculator.execute("10 / 0"))
    
    def test_invalid_expression(self, calculator):
        with pytest.raises(ValueError):
            asyncio.run(calculator.execute("2 +"))
    
    def test_unknown_reference(self, calculator):
        with pytest.raises(ValueError, match="unknown step"):
            asyncio.run(calculator.execute("{unknown_step} + 1", {}))


class TestCalculateFunction:
    """Test the convenience calculate function."""
    
    def test_simple_calculation(self):
        result = asyncio.run(calculate("2 + 2"))
        assert result == 4.0
    
    def test_with_context(self):
        context = {"step_1": 100}
        result = asyncio.run(calculate("{step_1} * 2", context))
        assert result == 200.0

