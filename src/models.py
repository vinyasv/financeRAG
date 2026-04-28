"""Pydantic models for Finance RAG."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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


class TableSchema(BaseModel):
    """Schema for an extracted table."""
    table_name: str = Field(..., description="Unique table name")
    description: str = Field(..., description="What this table contains")
    columns: list[str] = Field(default_factory=list)
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


# =============================================================================
# Query/Response Models
# =============================================================================

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
