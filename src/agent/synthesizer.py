"""Response synthesizer that combines tool results into coherent answers."""

from typing import Any

from ..models import ExecutionPlan, ToolResult, QueryResponse, Citation, ToolName
from ..storage.document_store import DocumentStore


SYNTHESIS_PROMPT = """You are a SENIOR FINANCIAL ANALYST at a top-tier investment bank (Goldman Sachs, Morgan Stanley, JPMorgan level).

Your role: Synthesize research findings into a clear, professional response for senior partners and institutional clients.

═══════════════════════════════════════════════════════════════════════════════
USER QUERY
═══════════════════════════════════════════════════════════════════════════════
{query}

═══════════════════════════════════════════════════════════════════════════════
RESEARCH FINDINGS
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

FINANCIAL CALCULATIONS (if performing any):
• Operating Margin = Operating Income / Revenue
• Net Margin = Net Income / Revenue
• YoY Growth = (Current - Prior) / Prior × 100
• ROE = Net Income / Shareholders' Equity
• P/E Ratio = Stock Price / Earnings Per Share
• Free Cash Flow = Operating Cash Flow - Capital Expenditures
• ALWAYS show your calculation steps

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
        if self.llm_client:
            answer = await self._synthesize_with_llm(plan, results)
        else:
            answer = self._synthesize_with_template(plan, results)
        
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
        """Generate response using LLM."""
        # Format plan description
        plan_lines = []
        for step in plan.steps:
            status = "✓" if results.get(step.id, ToolResult(step_id="", tool=ToolName.CALCULATOR, success=False)).success else "✗"
            plan_lines.append(f"{status} {step.id}: {step.description}")
        plan_description = "\n".join(plan_lines)
        
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
        
        prompt = SYNTHESIS_PROMPT.format(
            query=plan.query,
            plan_description=plan_description,
            results_text=results_text
        )
        
        return await self.llm_client.generate(prompt)
    
    def _synthesize_with_template(
        self,
        plan: ExecutionPlan,
        results: dict[str, ToolResult]
    ) -> str:
        """Generate response using templates (no LLM)."""
        parts = []
        
        # Add data results
        for step in plan.steps:
            result = results.get(step.id)
            if not result or not result.success:
                continue
            
            if step.tool == ToolName.SQL_QUERY:
                parts.append(f"**Data:** {self._format_result(result.result)}")
            
            elif step.tool == ToolName.CALCULATOR:
                parts.append(f"**Calculation result:** {result.result}")
            
            elif step.tool == ToolName.VECTOR_SEARCH:
                if isinstance(result.result, list) and result.result:
                    context = result.result[0].get("content", "")[:500]
                    parts.append(f"**Context:** {context}...")
            
            elif step.tool == ToolName.GET_DOCUMENT:
                if isinstance(result.result, dict):
                    content = result.result.get("content", "")[:500]
                    parts.append(f"**Document content:** {content}...")
        
        # Check for errors
        errors = [r for r in results.values() if not r.success]
        if errors:
            error_msgs = [f"{e.step_id}: {e.error}" for e in errors]
            parts.append(f"\n**Note:** Some steps failed: {'; '.join(error_msgs)}")
        
        if not parts:
            return "I couldn't find relevant information to answer your query."
        
        return "\n\n".join(parts)
    
    def _format_result(self, result: Any) -> str:
        """Format a result for display."""
        if result is None:
            return "No result"
        
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
                if k in ("columns", "row_count"):
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
        
        for result in results.values():
            if not result.success:
                continue
            
            if result.tool == ToolName.VECTOR_SEARCH:
                # Extract from search results
                if isinstance(result.result, list):
                    for item in result.result:
                        if isinstance(item, dict):
                            doc_id = item.get("document_id", "unknown")
                            
                            # Create a key to deduplicate
                            cite_key = (
                                doc_id,
                                item.get("page_number"),
                                item.get("start_line")
                            )
                            if cite_key in seen_citations:
                                continue
                            seen_citations.add(cite_key)
                            
                            # Resolve document name from ID
                            doc_name = self._resolve_document_name(doc_id)
                            
                            citations.append(Citation(
                                document_id=doc_id,
                                document_name=doc_name,
                                page_number=item.get("page_number"),
                                section=item.get("section_title"),
                                start_line=item.get("start_line"),
                                end_line=item.get("end_line"),
                                text_snippet=item.get("content", "")[:200]
                            ))
            
            elif result.tool == ToolName.GET_DOCUMENT:
                if isinstance(result.result, dict):
                    doc_id = result.result.get("document_id", "unknown")
                    doc_name = result.result.get("filename") or self._resolve_document_name(doc_id)
                    
                    citations.append(Citation(
                        document_id=doc_id,
                        document_name=doc_name,
                        section=result.result.get("section")
                    ))
        
        return citations
