"""Extract temporal metadata from document filenames and PDF metadata."""

import re
from typing import Any


# Patterns for fiscal quarters
FISCAL_QUARTER_PATTERNS = [
    r'(Q[1-4])',  # Q1, Q2, Q3, Q4
    r'(?:quarter|qtr)[_\s-]?([1-4])',  # quarter1, qtr_2
]

# Report type patterns
REPORT_TYPE_PATTERNS = {
    'annual': [r'annual', r'10-?k(?!\d)', r'yearly'],
    'quarterly': [r'quarterly', r'10-?q(?!\d)'],
    'earnings': [r'earnings', r'8-?k'],
}


def extract_temporal_metadata(filename: str, pdf_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Extract temporal metadata from a document filename and PDF metadata.
    
    Args:
        filename: The document filename (e.g., "JPMorgan_2024_Annual_Report.pdf")
        pdf_metadata: Optional PDF metadata dict
        
    Returns:
        Dict with temporal fields:
        - fiscal_year: int or None
        - fiscal_quarter: str or None (Q1, Q2, Q3, Q4)
        - report_type: str or None (annual, quarterly, earnings)
        - company_name: str or None (extracted company name)
    """
    result: dict[str, Any] = {}
    filename_lower = filename.lower()
    
    # Extract fiscal year
    fiscal_year = _extract_fiscal_year(filename)
    if fiscal_year:
        result['fiscal_year'] = fiscal_year
    
    # Extract fiscal quarter
    fiscal_quarter = _extract_fiscal_quarter(filename)
    if fiscal_quarter:
        result['fiscal_quarter'] = fiscal_quarter
    
    # Determine report type
    report_type = _extract_report_type(filename_lower)
    if report_type:
        result['report_type'] = report_type
    elif fiscal_quarter:
        result['report_type'] = 'quarterly'
    
    # Extract company name (first segment before year/numbers)
    company_name = _extract_company_name(filename)
    if company_name:
        result['company_name'] = company_name
    
    # Try to enhance from PDF metadata if available
    if pdf_metadata:
        # Check PDF creation/modification date for fiscal year hints
        if 'fiscal_year' not in result:
            for key in ['CreationDate', 'ModDate']:
                if key in pdf_metadata:
                    year = _extract_year_from_pdf_date(pdf_metadata[key])
                    if year:
                        result['fiscal_year'] = year
                        break
        
        # Use PDF title for company name if not found
        if 'company_name' not in result and 'title' in pdf_metadata:
            company = _extract_company_name(pdf_metadata['title'])
            if company:
                result['company_name'] = company
    
    return result


def _extract_fiscal_year(text: str) -> int | None:
    """Extract fiscal year from text."""
    # Try 4-digit year first
    match = re.search(r'(?:FY|fy)?[_\s-]?(20\d{2})', text)
    if match:
        return int(match.group(1))
    
    # Try 2-digit FY format (FY25 = 2025)
    match = re.search(r'(?:FY|fy)[_\s-]?(\d{2})(?!\d)', text)
    if match:
        year_short = int(match.group(1))
        # Assume 2000s (FY25 = 2025, FY99 would be 2099)
        return 2000 + year_short
    
    return None


def _extract_fiscal_quarter(text: str) -> str | None:
    """Extract fiscal quarter from text."""
    # Look for Q1, Q2, Q3, Q4
    match = re.search(r'(Q[1-4])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for "quarter 1", "qtr_2", etc.
    match = re.search(r'(?:quarter|qtr)[_\s-]?([1-4])', text, re.IGNORECASE)
    if match:
        return f"Q{match.group(1)}"
    
    return None


def _extract_report_type(text_lower: str) -> str | None:
    """Determine report type from text."""
    for report_type, patterns in REPORT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return report_type
    return None


def _extract_company_name(text: str) -> str | None:
    """
    Extract company name from filename.
    
    Assumes the company name is the first segment before numbers/dates.
    """
    # Remove extension
    name = re.sub(r'\.[^.]+$', '', text)
    
    # Split on common delimiters
    parts = re.split(r'[_\s-]+', name)
    
    # Take segments before first year/quarter indicator
    company_parts = []
    for part in parts:
        # Stop at year-like patterns
        if re.match(r'^(20\d{2}|FY\d{2}|Q[1-4]|\d{4})$', part, re.IGNORECASE):
            break
        # Skip common report type words
        if part.lower() in ['annual', 'quarterly', 'report', '10k', '10q', '8k']:
            continue
        company_parts.append(part)
    
    if company_parts:
        return ' '.join(company_parts)
    return None


def _extract_year_from_pdf_date(pdf_date: str) -> int | None:
    """Extract year from PDF date string (D:20241215...)."""
    match = re.search(r'D:(20\d{2})', str(pdf_date))
    if match:
        return int(match.group(1))
    return None
