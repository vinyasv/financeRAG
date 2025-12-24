"""Pydantic models for Finance RAG."""

from __future__ import annotations
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# Tool Models
# =============================================================================

class ToolName(str, Enum):
    """Available tools for the agent."""
    SQL_QUERY = "sql_query"
    VECTOR_SEARCH = "vector_search"
    CALCULATOR = "calculator"
    GET_DOCUMENT = "get_document"


class ToolCall(BaseModel):
    """A single tool call in the execution plan."""
    id: str = Field(..., description="Unique identifier (e.g., 'step_1')")
    tool: ToolName = Field(..., description="Which tool to call")
    input: str = Field(..., description="Tool input (query, expression, etc.)")
    depends_on: list[str] = Field(default_factory=list, description="IDs of steps this depends on")
    description: str = Field(..., description="What this step accomplishes")


class ExecutionPlan(BaseModel):
    """DAG of tool calls to answer the query."""
    query: str = Field(..., description="Original user query")
    reasoning: str = Field(..., description="Why this plan was chosen")
    steps: list[ToolCall] = Field(..., description="Ordered list of tool calls")
    
    def get_execution_layers(self) -> list[list[ToolCall]]:
        """Group steps into parallelizable layers based on dependencies."""
        completed: set[str] = set()
        layers: list[list[ToolCall]] = []
        remaining = list(self.steps)
        
        while remaining:
            # Find all steps whose dependencies are satisfied
            ready = [
                step for step in remaining
                if all(dep in completed for dep in step.depends_on)
            ]
            
            if not ready:
                # Circular dependency or missing step
                raise ValueError(f"Cannot resolve dependencies. Remaining: {[s.id for s in remaining]}")
            
            layers.append(ready)
            for step in ready:
                completed.add(step.id)
                remaining.remove(step)
        
        return layers


class ToolResult(BaseModel):
    """Result from a tool execution."""
    step_id: str = Field(..., description="ID of the step that produced this result")
    tool: ToolName = Field(..., description="Tool that was called")
    success: bool = Field(..., description="Whether the tool call succeeded")
    result: Any = Field(default=None, description="The result data")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(default=0, description="Time taken to execute")


# =============================================================================
# Document Models
# =============================================================================

class Document(BaseModel):
    """A source document."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    title: str | None = Field(default=None, description="Document title")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    page_count: int = Field(default=0, description="Number of pages")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    """A chunk of text from a document."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="The text content")
    page_number: int | None = Field(default=None, description="Page number in source")
    section_title: str | None = Field(default=None, description="Section this belongs to")
    chunk_index: int = Field(default=0, description="Index within document")
    start_line: int | None = Field(default=None, description="Starting line number in source document")
    end_line: int | None = Field(default=None, description="Ending line number in source document")
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Table/Structured Data Models
# =============================================================================

class ColumnType(str, Enum):
    """Data types for table columns."""
    TEXT = "text"
    NUMBER = "number"
    CURRENCY = "currency"
    DATE = "date"
    PERCENTAGE = "percentage"


class TableColumn(BaseModel):
    """A column in an extracted table."""
    name: str = Field(..., description="Column name")
    data_type: ColumnType = Field(default=ColumnType.TEXT)
    description: str | None = Field(default=None)


class TableSchema(BaseModel):
    """Schema for an extracted table."""
    table_name: str = Field(..., description="Unique table name")
    description: str = Field(..., description="What this table contains")
    columns: list[TableColumn] = Field(default_factory=list)
    source_document_id: str = Field(..., description="Document this came from")
    page_number: int | None = Field(default=None)


class ExtractedTable(BaseModel):
    """A table extracted from a document."""
    id: str = Field(..., description="Unique table identifier")
    document_id: str = Field(..., description="Source document ID")
    table_name: str = Field(..., description="Descriptive name for the table")
    page_number: int | None = Field(default=None)
    schema_description: str = Field(..., description="What this table represents")
    columns: list[str] = Field(default_factory=list, description="Column names")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Row data")
    raw_text: str | None = Field(default=None, description="Original text representation")


class TableDataRow(BaseModel):
    """A single row of data in a table."""
    table_id: str = Field(..., description="Parent table ID")
    row_index: int = Field(..., description="Row number")
    data: dict[str, Any] = Field(default_factory=dict, description="Column name -> value")


# =============================================================================
# Query/Response Models
# =============================================================================

class QueryType(str, Enum):
    """Types of queries the system can handle."""
    COMPUTATIONAL = "computational"  # Needs SQL + calculation
    FACTUAL = "factual"              # Needs text retrieval
    HYBRID = "hybrid"                # Needs both


class Citation(BaseModel):
    """A citation to source material."""
    document_id: str
    document_name: str
    page_number: int | None = None
    section: str | None = None
    start_line: int | None = Field(default=None, description="Starting line/row number")
    end_line: int | None = Field(default=None, description="Ending line/row number")
    text_snippet: str | None = None
    
    def format_reference(self) -> str:
        """
        Format as a human-readable citation string.
        
        Examples:
            - "report.pdf, p.42, L15-18"
            - "data.csv, Sheet: Revenue, Rows 1-25"
            - "report.pdf, p.5"
        """
        parts = [self.document_name]
        
        # Check if this is a spreadsheet (has section/sheet name but no page)
        is_spreadsheet = self.section and not self.page_number
        
        if is_spreadsheet:
            # Spreadsheet citation: Sheet name + row range
            parts.append(f"Sheet: {self.section}")
            if self.start_line:
                if self.end_line and self.end_line != self.start_line:
                    parts.append(f"Rows {self.start_line}-{self.end_line}")
                else:
                    parts.append(f"Row {self.start_line}")
        else:
            # PDF citation: Page + line range
            if self.page_number:
                parts.append(f"p.{self.page_number}")
            
            if self.start_line:
                if self.end_line and self.end_line != self.start_line:
                    parts.append(f"L{self.start_line}-{self.end_line}")
                else:
                    parts.append(f"L{self.start_line}")
        
        return ", ".join(parts)


class QueryResponse(BaseModel):
    """Final response to a user query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    citations: list[Citation] = Field(default_factory=list)
    execution_plan: ExecutionPlan | None = Field(default=None)
    tool_results: list[ToolResult] = Field(default_factory=list)
    total_time_ms: float = Field(default=0)


# =============================================================================
# SQL Query Models
# =============================================================================

class SQLQueryResult(BaseModel):
    """Result from a SQL query."""
    query: str = Field(..., description="The SQL query that was executed")
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)


class VectorSearchResult(BaseModel):
    """Result from vector search."""
    chunks: list[TextChunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


# =============================================================================
# Calculation Transparency Models
# =============================================================================

class OperandBinding(BaseModel):
    """A single operand in a calculation with its source provenance."""
    reference: str = Field(..., description="Original reference, e.g., '{step_1.revenue}'")
    resolved_value: float = Field(..., description="The actual numeric value used")
    source_step: str = Field(..., description="Step ID that produced this value")
    source_description: str = Field(..., description="Human-readable source, e.g., 'SQL: revenue from jpmorgan_financials'")
    

class CalculationTranscript(BaseModel):
    """
    Full transparency record for a calculation.
    
    Provides complete audit trail showing:
    - What expression was requested
    - What values were bound and where they came from
    - The resolved numeric expression
    - The final result
    """
    original_expression: str = Field(..., description="Expression with references, e.g., '{step_1.revenue} - {step_2.revenue}'")
    bindings: list[OperandBinding] = Field(default_factory=list, description="All operand bindings used")
    resolved_expression: str = Field(..., description="Expression with values substituted, e.g., '145600000000 - 128695000000'")
    result: float = Field(..., description="The computed result")
    formula_description: str | None = Field(default=None, description="Optional description of what this calculates")
    
    def format_for_display(self) -> str:
        """Format as a human-readable calculation block."""
        lines = [f"**Calculation:** {self.formula_description or 'Expression'}"]
        for binding in self.bindings:
            # Format large numbers with appropriate units
            value_str = self._format_number(binding.resolved_value)
            lines.append(f"  • {binding.reference}: {value_str} ({binding.source_description})")
        lines.append(f"  • Expression: `{self.resolved_expression}`")
        lines.append(f"  • **Result:** {self._format_number(self.result)}")
        return "\n".join(lines)
    
    @staticmethod
    def _format_number(value: float) -> str:
        """Format number with appropriate units (B/M/K)."""
        abs_val = abs(value)
        if abs_val >= 1_000_000_000:
            return f"${value / 1_000_000_000:,.2f}B"
        elif abs_val >= 1_000_000:
            return f"${value / 1_000_000:,.2f}M"
        elif abs_val >= 1_000:
            return f"${value / 1_000:,.2f}K"
        else:
            return f"{value:,.2f}"


# =============================================================================
# Refusal Models (Audit Transparency)
# =============================================================================

class RefusalReason(str, Enum):
    """Reasons for refusing to answer a query."""
    DEFINITION_MISMATCH = "definition_mismatch"      # Comparing incompatible metric definitions
    INSUFFICIENT_DATA = "insufficient_data"          # Required data not available
    PERIOD_DISCONTINUITY = "period_discontinuity"    # Time periods don't align or have gaps
    INCOMPARABLE_METRICS = "incomparable_metrics"    # Metrics from different standards (GAAP vs non-GAAP)
    MISSING_CONTEXT = "missing_context"              # Can't determine what user is asking for


class QueryRefusal(BaseModel):
    """
    Structured refusal with full explanation.
    
    Used when the system cannot reliably answer a query due to
    data quality, comparability, or completeness concerns.
    Treating refusal as a success mode improves trust.
    """
    reason: RefusalReason = Field(..., description="Category of refusal")
    explanation: str = Field(..., description="Clear explanation of why the query cannot be answered")
    what_was_found: str = Field(..., description="What data/context WAS successfully retrieved")
    what_is_missing: list[str] = Field(default_factory=list, description="Specific missing elements")
    suggested_alternatives: list[str] = Field(default_factory=list, description="Alternative queries that could work")
    
    def format_for_display(self) -> str:
        """Format as a professional analyst-style refusal."""
        lines = [
            "**Unable to Complete Analysis**",
            "",
            f"**Reason:** {self.reason.value.replace('_', ' ').title()}",
            "",
            self.explanation,
            "",
            f"**What was found:** {self.what_was_found}",
        ]
        
        if self.what_is_missing:
            lines.append("")
            lines.append("**Missing information:**")
            for item in self.what_is_missing:
                lines.append(f"  • {item}")
        
        if self.suggested_alternatives:
            lines.append("")
            lines.append("**Suggested alternatives:**")
            for alt in self.suggested_alternatives:
                lines.append(f"  • {alt}")
        
        return "\n".join(lines)


# =============================================================================
# Field Definition Models (Comparability Tracking)
# =============================================================================

class AccountingStandard(str, Enum):
    """Accounting standards for financial data."""
    GAAP = "gaap"
    NON_GAAP = "non_gaap"
    IFRS = "ifrs"
    UNKNOWN = "unknown"


class FieldDefinition(BaseModel):
    """
    Semantic definition of a data field for comparability checking.
    
    Tracks the meaning and context of a field to determine whether
    it can be meaningfully compared with another field.
    """
    field_name: str = Field(..., description="Name of the field/column")
    definition_hash: str = Field(..., description="SHA256 hash of canonical definition for quick matching")
    
    # Temporal context
    fiscal_period: str | None = Field(default=None, description="Period covered, e.g., 'FY2024', 'Q3-2024'")
    fiscal_year: int | None = Field(default=None, description="Fiscal year")
    
    # Accounting context
    accounting_standard: AccountingStandard = Field(default=AccountingStandard.UNKNOWN)
    
    # Scope context
    segment_scope: str | None = Field(default=None, description="Business segment, e.g., 'Consolidated', 'North America'")
    currency: str | None = Field(default=None, description="Currency, e.g., 'USD', 'EUR'")
    
    # Definition details
    definition_text: str | None = Field(default=None, description="Human-readable definition")
    includes_items: list[str] = Field(default_factory=list, description="What is included")
    excludes_items: list[str] = Field(default_factory=list, description="What is excluded, e.g., 'one-time items'")
    
    # Source tracking
    source_document_id: str | None = Field(default=None)
    source_table: str | None = Field(default=None)
    
    @classmethod
    def compute_hash(cls, field_name: str, accounting_standard: str, segment_scope: str | None, 
                     includes: list[str], excludes: list[str]) -> str:
        """Compute a deterministic hash for field definition matching."""
        import hashlib
        canonical = f"{field_name.lower()}|{accounting_standard}|{segment_scope or ''}|{','.join(sorted(includes))}|{','.join(sorted(excludes))}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ComparabilityResult(BaseModel):
    """
    Result of comparing two field definitions.
    
    Used to determine if two metrics can be meaningfully compared
    or computed together.
    """
    comparable: bool = Field(..., description="Whether the fields can be meaningfully compared")
    confidence: float = Field(default=1.0, description="Confidence in the assessment (0.0 to 1.0)")
    
    # Analysis details
    differences: list[str] = Field(default_factory=list, description="What differs between the fields")
    warnings: list[str] = Field(default_factory=list, description="Potential issues to be aware of")
    
    # Recommendation
    recommendation: str = Field(
        default="Safe to compare",
        description="Human-readable recommendation"
    )
    
    @classmethod
    def check_comparability(cls, field_a: FieldDefinition, field_b: FieldDefinition) -> "ComparabilityResult":
        """
        Compare two field definitions and determine if they are comparable.
        
        Checks:
        - Accounting standard compatibility
        - Currency matching
        - Segment scope alignment
        - Definition hash matching
        """
        differences: list[str] = []
        warnings: list[str] = []
        comparable = True
        confidence = 1.0
        
        # Check accounting standard
        if field_a.accounting_standard != field_b.accounting_standard:
            if AccountingStandard.UNKNOWN not in (field_a.accounting_standard, field_b.accounting_standard):
                differences.append(
                    f"Accounting standards differ: {field_a.accounting_standard.value} vs {field_b.accounting_standard.value}"
                )
                comparable = False
        
        # Check currency
        if field_a.currency and field_b.currency and field_a.currency != field_b.currency:
            differences.append(f"Currencies differ: {field_a.currency} vs {field_b.currency}")
            comparable = False
        
        # Check segment scope
        if field_a.segment_scope and field_b.segment_scope:
            if field_a.segment_scope.lower() != field_b.segment_scope.lower():
                differences.append(
                    f"Segment scopes differ: {field_a.segment_scope} vs {field_b.segment_scope}"
                )
                comparable = False
        
        # Check definition hash for exact semantic match
        if field_a.definition_hash != field_b.definition_hash:
            # Not an automatic fail, but note it
            warnings.append("Field definitions may not be identical—verify semantic equivalence")
            confidence = 0.7
        
        # Check excludes/includes for compatibility
        if set(field_a.excludes_items) != set(field_b.excludes_items):
            diff_a = set(field_a.excludes_items) - set(field_b.excludes_items)
            diff_b = set(field_b.excludes_items) - set(field_a.excludes_items)
            if diff_a or diff_b:
                warnings.append(f"Different exclusions applied to metrics")
                confidence = min(confidence, 0.6)
        
        # Build recommendation
        if comparable and not warnings:
            recommendation = "Safe to compare"
        elif comparable and warnings:
            recommendation = "Comparable with caveats—review warnings"
        else:
            recommendation = f"Not directly comparable: {'; '.join(differences)}"
        
        return cls(
            comparable=comparable,
            confidence=confidence,
            differences=differences,
            warnings=warnings,
            recommendation=recommendation
        )
