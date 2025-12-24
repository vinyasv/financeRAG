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

