"""Response synthesizer that combines tool results into coherent answers."""

import logging
from typing import Any

from ..common.prompts import STANDARD_PROMPT_GUARD, prepare_prompt_user_content
from ..models import (
    CalculationTranscript,
    Citation,
    ExecutionPlan,
    QueryResponse,
    ToolName,
    ToolResult,
)
from ..storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """You are a SENIOR FINANCIAL ANALYST at a top-tier investment bank (Goldman Sachs, Morgan Stanley, JPMorgan level).

Your role: Synthesize research findings into a clear, professional response for senior partners and institutional clients.

═══════════════════════════════════════════════════════════════════════════════
SECURITY NOTICE
═══════════════════════════════════════════════════════════════════════════════
{security_notice}

═══════════════════════════════════════════════════════════════════════════════
USER QUERY (UNTRUSTED INPUT)
═══════════════════════════════════════════════════════════════════════════════
{query}

═══════════════════════════════════════════════════════════════════════════════
RESEARCH FINDINGS (VERIFIED DATA)
═══════════════════════════════════════════════════════════════════════════════
{results_text}

═══════════════════════════════════════════════════════════════════════════════
ANALYST RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

CORE PRINCIPLES:
1. USE ONLY the data from RESEARCH FINDINGS above - never fabricate numbers
2. Be PRECISE with figures - quote exact numbers from the research
3. CITE sources with page and line numbers when available (e.g., "(10-K, p.42, L15-18)" or "(Budget.xlsx, Sheet: Q4)")
4. ACKNOWLEDGE limitations - if data is incomplete, say so professionally

RESPONSE STRUCTURE (when appropriate):
• Lead with the KEY FINDING or direct answer
• Support with SPECIFIC DATA POINTS from the research
• Add CONTEXT and IMPLICATIONS where relevant
• For comparisons, use TABLES or structured formats
• Include CAVEATS for any limitations in the data

═══════════════════════════════════════════════════════════════════════════════
CRITICAL: ARITHMETIC PROHIBITION
═══════════════════════════════════════════════════════════════════════════════
• ALL arithmetic operations MUST come from calculator tool results in RESEARCH FINDINGS
• You may EXPLAIN what a calculation represents but NEVER perform arithmetic yourself
• If a needed calculation wasn't performed, state: "This would require computing [X]"
• Quote numbers EXACTLY as they appear in the research findings—do not round, adjust, or compute
• Calculator results include a full audit trail showing operand sources—reference this when explaining

FINANCIAL FORMULA REFERENCE (for explanation only, not computation):
• Operating Margin = Operating Income / Revenue
• Net Margin = Net Income / Revenue
• YoY Growth = (Current - Prior) / Prior × 100
• ROE = Net Income / Shareholders' Equity
• P/E Ratio = Stock Price / Earnings Per Share
• Free Cash Flow = Operating Cash Flow - Capital Expenditures

PROFESSIONAL STANDARDS:
• Use proper financial terminology
• Format large numbers with appropriate units ($X billion, $X million)
• Round percentages to one decimal place
• Present comparative data in parallel structure
• Distinguish between GAAP and non-GAAP metrics when relevant

WHEN DATA IS INSUFFICIENT:
• State clearly what WAS found
• Explain what IS missing
• DO NOT speculate or make up data
• Suggest what additional data would be needed

═══════════════════════════════════════════════════════════════════════════════
YOUR ANALYSIS
═══════════════════════════════════════════════════════════════════════════════"""


class ResponseSynthesizer:
    """
    Synthesizes coherent responses from execution results.
    
    Combines:
    - SQL query results (numbers, data)
    - Vector search results (context, explanations)
    - Calculator results (computed values)
    - Document content (full context)
    """
    
    def __init__(self, llm_client: Any = None, document_store: DocumentStore | None = None):
        """
        Initialize synthesizer.
        
        Args:
            llm_client: LLM client for generating responses. If None, uses templates.
            document_store: Document store for looking up document filenames.
        """
        self.llm_client = llm_client
        self.document_store = document_store
        self._doc_name_cache: dict[str, str] = {}  # Cache doc_id -> filename
    
    async def synthesize(
        self,
        plan: ExecutionPlan,
        results: dict[str, ToolResult]
    ) -> QueryResponse:
        """
        Synthesize a response from execution results.
        
        Args:
            plan: The executed plan
            results: Results from each step
            
        Returns:
            QueryResponse with synthesized answer
        """
        if not self.llm_client:
            raise RuntimeError("Response synthesis requires an LLM client")
        answer = await self._synthesize_with_llm(plan, results)
        
        # Extract citations
        citations = self._extract_citations(results)
        
        # Calculate total time
        total_time = sum(r.execution_time_ms for r in results.values())
        
        return QueryResponse(
            query=plan.query,
            answer=answer,
            citations=citations,
            execution_plan=plan,
            tool_results=list(results.values()),
            total_time_ms=total_time
        )
    
    async def _synthesize_with_llm(
        self,
        plan: ExecutionPlan,
        results: dict[str, ToolResult]
    ) -> str:
        """Generate response using LLM with suspicious-input advisory wrapping."""
        # Format results
        results_parts = []
        for step in plan.steps:
            result = results.get(step.id)
            if result:
                if result.success:
                    results_parts.append(f"[{step.id}] {step.tool.value}:\n{self._format_result(result.result)}")
                else:
                    results_parts.append(f"[{step.id}] {step.tool.value}: ERROR - {result.error}")
        results_text = "\n\n".join(results_parts)
        
        wrapped_query = prepare_prompt_user_content(plan.query, "user_query")

        prompt = SYNTHESIS_PROMPT.format(
            query=wrapped_query,
            results_text=results_text,
            security_notice=STANDARD_PROMPT_GUARD,
        )
        
        return await self.llm_client.generate(prompt)
    
    def _format_result(self, result: Any) -> str:
        """Format a result for display."""
        if result is None:
            return "No result"
        
        # Handle CalculationTranscript with full audit transparency
        if isinstance(result, CalculationTranscript):
            return result.format_for_display()
        
        if isinstance(result, (int, float)):
            # Format large numbers with commas
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"{result:,}"
        
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            
            # Format dict nicely
            parts = []
            for k, v in result.items():
                if k in ("columns", "row_count") or k.startswith("__"):
                    continue
                if isinstance(v, (int, float)):
                    parts.append(f"{k}: {v:,}")
                else:
                    parts.append(f"{k}: {v}")
            return ", ".join(parts) if parts else str(result)
        
        if isinstance(result, list):
            if len(result) == 0:
                return "No results"
            # For vector search results, format each chunk with its content
            if len(result) > 0 and isinstance(result[0], dict) and "content" in result[0]:
                chunks = []
                for i, item in enumerate(result[:5], 1):  # Limit to 5 chunks
                    content = item.get("content", "")
                    page = item.get("page_number", "?")
                    section = item.get("section_title", "")
                    header = f"[Chunk {i}, Page {page}"
                    if section:
                        header += f", {section}"
                    header += "]"
                    chunks.append(f"{header}\n{content}")
                return "\n\n".join(chunks)
            if len(result) == 1:
                return self._format_result(result[0])
            return f"{len(result)} results"
        
        return str(result)
    
    def _resolve_document_name(self, doc_id: str) -> str:
        """
        Resolve a document ID to its human-readable filename.
        
        Uses caching to avoid repeated lookups.
        
        Args:
            doc_id: The document ID (hash)
            
        Returns:
            The document filename, or the doc_id if not found
        """
        if doc_id in self._doc_name_cache:
            return self._doc_name_cache[doc_id]
        
        # Try to look up from document store
        if self.document_store:
            doc = self.document_store.get_document(doc_id)
            if doc:
                self._doc_name_cache[doc_id] = doc.filename
                return doc.filename
        
        # Fallback to doc_id if not found
        return doc_id
    
    def _extract_citations(self, results: dict[str, ToolResult]) -> list[Citation]:
        """Extract citations from results."""
        citations = []
        seen_citations = set()  # Avoid duplicate citations

        def add_citation(citation: Citation) -> None:
            cite_key = (
                citation.document_id,
                citation.document_name,
                citation.page_number,
                citation.section,
                citation.start_line,
                citation.end_line,
            )
            if cite_key in seen_citations:
                return
            seen_citations.add(cite_key)
            citations.append(citation)

        def add_sql_citations(result_value: Any) -> None:
            if not isinstance(result_value, dict):
                return
            provenance = result_value.get("__sql_provenance")
            if not isinstance(provenance, dict):
                return
            for table in provenance.get("tables", []):
                if not isinstance(table, dict):
                    continue
                doc_id = table.get("document_id") or "unknown"
                table_name = table.get("table_name") or "unknown_table"
                columns = table.get("columns") or []
                add_citation(Citation(
                    document_id=doc_id,
                    document_name=self._resolve_document_name(doc_id),
                    section=table_name,
                    text_snippet=f"SQL table {table_name}; columns: {', '.join(columns[:10])}",
                ))
        
        for result in results.values():
            if not result.success:
                continue
            
            if result.tool == ToolName.VECTOR_SEARCH:
                # Extract from search results
                if isinstance(result.result, list):
                    for item in result.result:
                        if isinstance(item, dict):
                            doc_id = item.get("document_id", "unknown")
                            
                            # Resolve document name from ID
                            doc_name = self._resolve_document_name(doc_id)
                            
                            snippet = item.get("content", "")[:200]
                            if len(item.get("content", "")) > 200:
                                snippet += "..."
                            add_citation(Citation(
                                document_id=doc_id,
                                document_name=doc_name,
                                page_number=item.get("page_number"),
                                section=item.get("section_title"),
                                start_line=item.get("start_line"),
                                end_line=item.get("end_line"),
                                text_snippet=snippet
                            ))

            elif result.tool == ToolName.SQL_QUERY:
                add_sql_citations(result.result)

            elif result.tool == ToolName.CALCULATOR:
                if isinstance(result.result, CalculationTranscript):
                    for binding in result.result.bindings:
                        source = results.get(binding.source_step)
                        if source and source.tool == ToolName.SQL_QUERY:
                            add_sql_citations(source.result)

            elif result.tool == ToolName.GET_DOCUMENT:
                if isinstance(result.result, dict):
                    doc_id = result.result.get("document_id", "unknown")
                    doc_name = result.result.get("filename") or self._resolve_document_name(doc_id)

                    add_citation(Citation(
                        document_id=doc_id,
                        document_name=doc_name,
                        section=result.result.get("section")
                    ))
        
        return citations
