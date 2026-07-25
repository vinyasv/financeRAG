"""Regression tests for CalculationTranscript number formatting.

_format_number used to hardcode a "$" prefix for any |value| >= 1000, so a
12,500-person headcount rendered as "$12.50K". Because format_for_display()
is fed straight into the synthesis prompt (see synthesizer._format_result),
that invented unit reached the LLM as fact. The transcript carries no unit
metadata, so it must not assert one.
"""

import asyncio
import math

from src.models import CalculationTranscript, OperandBinding

_fmt = CalculationTranscript._format_number


def _transcript(**overrides) -> CalculationTranscript:
    kwargs = {
        "original_expression": "{step_1.headcount} - {step_2.headcount}",
        "bindings": [
            OperandBinding(
                reference="{step_1.headcount}",
                resolved_value=12_500.0,
                source_step="step_1",
                source_description="SQL: headcount from acme_hr",
            ),
            OperandBinding(
                reference="{step_2.revenue}",
                resolved_value=145_600_000_000.0,
                source_step="step_2",
                source_description="SQL: revenue from acme_financials",
            ),
        ],
        "resolved_expression": "12500 - 145600000000",
        "result": -145_599_987_500.0,
        "formula_description": "Headcount minus revenue (deliberately unitless)",
    }
    kwargs.update(overrides)
    return CalculationTranscript(**kwargs)


def test_format_number_never_emits_currency_symbol():
    """No magnitude bucket may prefix a currency symbol."""
    values = [
        0.0,
        1.0,
        -1.0,
        999.99,
        -999.99,
        1_000.0,
        12_500.0,
        -12_500.0,
        250_000.0,
        1_500_000.0,
        -1_500_000.0,
        145_600_000_000.0,
        -145_600_000_000.0,
        1_234_500_000_000.0,
    ]
    for value in values:
        formatted = _fmt(value)
        assert "$" not in formatted, f"{value!r} formatted as {formatted!r}"


def test_format_number_magnitude_scaling_and_precision():
    """Magnitude suffixes and 2dp precision survive the currency removal."""
    assert _fmt(0.0) == "0.00"
    assert _fmt(1.5) == "1.50"
    assert _fmt(999.99) == "999.99"
    assert _fmt(1_000.0) == "1.00K"
    assert _fmt(1_500.0) == "1.50K"
    assert _fmt(12_500.0) == "12.50K"
    assert _fmt(250_000.0) == "250.00K"
    assert _fmt(999_999.0) == "1,000.00K"
    assert _fmt(1_000_000.0) == "1.00M"
    assert _fmt(1_500_000.0) == "1.50M"
    assert _fmt(1_000_000_000.0) == "1.00B"
    assert _fmt(145.6e9) == "145.60B"
    assert _fmt(1_234_500_000_000.0) == "1,234.50B"


def test_format_number_negatives_keep_sign_and_suffix():
    """abs() only picks the bucket; the signed value is what gets divided."""
    assert _fmt(-1.5) == "-1.50"
    assert _fmt(-999.99) == "-999.99"
    assert _fmt(-12_500.0) == "-12.50K"
    assert _fmt(-1_500_000.0) == "-1.50M"
    assert _fmt(-145.6e9) == "-145.60B"


def test_format_for_display_contains_no_currency_symbol():
    """The audit block handed to the synthesis prompt must stay unit-agnostic."""
    display = _transcript().format_for_display()
    assert "$" not in display, display
    assert "12.50K" in display
    assert "145.60B" in display
    assert "-145.60B" in display  # the result line


def test_format_for_display_keeps_structure():
    """Bindings, resolved expression and result all still appear."""
    display = _transcript().format_for_display()
    assert "**Calculation:**" in display
    assert "{step_1.headcount}" in display
    assert "SQL: headcount from acme_hr" in display
    assert "`12500 - 145600000000`" in display
    assert "**Result:**" in display


def test_format_for_display_without_bindings():
    """A binding-free transcript still renders a clean, symbol-free result."""
    transcript = CalculationTranscript(
        original_expression="12500 * 2",
        bindings=[],
        resolved_expression="12500 * 2",
        result=25_000.0,
        formula_description=None,
    )
    display = transcript.format_for_display()
    assert "$" not in display
    assert "25.00K" in display


# =============================================================================
# Round-2: non-finite and signed-zero values.
#
# Reachable via float overflow in the calculator -- Mult and Pow are permitted
# (exponent up to MAX_EXPONENT=100) and the division-by-zero guard does not
# catch 1e308 * 10. "infB" in an audit trail is worse than useless.
# =============================================================================

def test_non_finite_values_do_not_get_a_magnitude_suffix():
    fmt = CalculationTranscript._format_number

    assert fmt(float("inf")) == "undefined (+inf)"
    assert fmt(float("-inf")) == "undefined (-inf)"
    assert fmt(float("nan")) == "undefined (nan)"

    for value in (float("inf"), float("-inf"), float("nan")):
        rendered = fmt(value)
        assert not rendered.endswith(("B", "M", "K")), (
            f"{value} rendered as a magnitude: {rendered!r}"
        )
        assert "$" not in rendered


def test_signed_zero_normalizes():
    assert CalculationTranscript._format_number(-0.0) == "0.00"


def test_overflow_from_the_calculator_renders_readably():
    """End-to-end: the calculator can actually produce inf."""
    from src.tools.calculator import CalculatorTool

    transcript = asyncio.run(CalculatorTool().execute("1e308 * 10"))

    assert math.isinf(transcript.result)
    assert "undefined (+inf)" in transcript.format_for_display()
