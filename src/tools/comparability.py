"""Field comparability checking for audit transparency.

This module provides functions to check whether fields from different tables
or time periods can be meaningfully compared or computed together.
"""

from ..models import FieldDefinition, ComparabilityResult, AccountingStandard, QueryRefusal, RefusalReason


def check_field_comparability(field_a: FieldDefinition, field_b: FieldDefinition) -> ComparabilityResult:
    """
    Check if two fields can be meaningfully compared.
    
    This is a convenience wrapper around ComparabilityResult.check_comparability.
    
    Args:
        field_a: First field definition
        field_b: Second field definition
        
    Returns:
        ComparabilityResult with analysis of whether comparison is valid
    """
    return ComparabilityResult.check_comparability(field_a, field_b)


def create_comparability_refusal(
    result: ComparabilityResult,
    field_a: FieldDefinition,
    field_b: FieldDefinition
) -> QueryRefusal:
    """
    Create a QueryRefusal from a failed comparability check.
    
    Args:
        result: The failed ComparabilityResult
        field_a: First field that was being compared
        field_b: Second field that was being compared
        
    Returns:
        QueryRefusal with detailed explanation of why comparison failed
    """
    # Determine the refusal reason based on differences
    reason = RefusalReason.INCOMPARABLE_METRICS
    for diff in result.differences:
        if "accounting standard" in diff.lower():
            reason = RefusalReason.DEFINITION_MISMATCH
            break
        elif "currency" in diff.lower():
            reason = RefusalReason.INCOMPARABLE_METRICS
            break
        elif "segment" in diff.lower():
            reason = RefusalReason.DEFINITION_MISMATCH
            break
    
    # Build explanation
    explanation_parts = [
        f"Cannot directly compare '{field_a.field_name}' with '{field_b.field_name}'.",
        "",
        "**Issues identified:**"
    ]
    for diff in result.differences:
        explanation_parts.append(f"  • {diff}")
    
    if result.warnings:
        explanation_parts.append("")
        explanation_parts.append("**Additional concerns:**")
        for warning in result.warnings:
            explanation_parts.append(f"  • {warning}")
    
    # Build what was found
    found_parts = []
    if field_a.fiscal_period:
        found_parts.append(f"{field_a.field_name} ({field_a.fiscal_period})")
    else:
        found_parts.append(field_a.field_name)
    if field_b.fiscal_period:
        found_parts.append(f"{field_b.field_name} ({field_b.fiscal_period})")
    else:
        found_parts.append(field_b.field_name)
    what_was_found = f"Retrieved both metrics: {', '.join(found_parts)}"
    
    # Build what is missing
    what_is_missing = [
        "Comparable metric definitions (same accounting standard, currency, segment scope)",
    ]
    if result.differences:
        what_is_missing.append(f"Resolution of: {'; '.join(result.differences)}")
    
    # Suggested alternatives
    alternatives = [
        f"Query each metric separately to understand their individual values",
        "Verify that both metrics use the same accounting methodology",
    ]
    if "accounting standard" in str(result.differences).lower():
        alternatives.append("Request both GAAP and non-GAAP versions if available")
    if "currency" in str(result.differences).lower():
        alternatives.append("Request values in a common currency")
    
    return QueryRefusal(
        reason=reason,
        explanation="\n".join(explanation_parts),
        what_was_found=what_was_found,
        what_is_missing=what_is_missing,
        suggested_alternatives=alternatives
    )


def validate_operands_comparable(operand_definitions: list[FieldDefinition]) -> ComparabilityResult | None:
    """
    Validate that all operands in a calculation are mutually comparable.
    
    Compares all pairs of operands and returns a ComparabilityResult
    if any pair is not comparable. Returns None if all are compatible.
    
    Args:
        operand_definitions: List of field definitions for all operands
        
    Returns:
        None if all operands are compatible, or a ComparabilityResult
        describing the first incompatibility found
    """
    if len(operand_definitions) < 2:
        return None
    
    # Check all pairs
    for i, field_a in enumerate(operand_definitions):
        for field_b in operand_definitions[i + 1:]:
            result = check_field_comparability(field_a, field_b)
            if not result.comparable:
                return result
    
    return None
