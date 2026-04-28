"""Query planner that creates execution DAGs."""

import json
import logging
from typing import Any

from ..common.json_utils import parse_json_response
from ..common.prompts import STANDARD_PROMPT_GUARD, prepare_prompt_user_content
from ..models import ExecutionPlan, ToolCall, ToolName
from ..storage.schema_cluster import CompanyRegistry
from ..validation import detect_injection_attempt

logger = logging.getLogger(__name__)


PLANNER_PROMPT = """You are a SENIOR FINANCIAL ANALYST at a top-tier investment bank, planning research queries.

Your expertise includes:
- Deep knowledge of financial statements (10-K, 10-Q, annual reports)
- Understanding of GAAP/IFRS accounting standards
- Regulatory compliance (SEC, banking regulations, export controls)
- Cross-company comparative analysis
- Valuation methodologies (DCF, multiples, sum-of-parts)

═══════════════════════════════════════════════════════════════════════════════
SECURITY NOTICE
═══════════════════════════════════════════════════════════════════════════════
{security_notice}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE RESEARCH TOOLS
═══════════════════════════════════════════════════════════════════════════════

1. vector_search (PREFERRED - use this first for most queries)
   → Semantic search across all financial documents
   → Best for: finding specific facts, numbers, quotes, risk disclosures
   → Searches: full text of 10-Ks, earnings releases, annual reports
   → Example: "NVIDIA Q1 FY2026 revenue" or "JPMorgan credit loss provisions"
   
2. sql_query
   → Query structured tables extracted from financial statements
   → Use when: you need data from well-defined tables with known columns
   → Best for: time-series data, comparative metrics across periods
   → Example: "SELECT revenue, net_income FROM income_statement WHERE year=2024"

3. calculator
   → Perform financial calculations with precision
   → Use for: margins, ratios, growth rates, scenario analysis
   → Can reference previous results: {{step_id.field}}
   → Example: "{{step_1.revenue}} * 0.7" or "({{step_2.value}} - {{step_3.value}}) / {{step_3.value}} * 100"

4. get_document
   → Retrieve complete document sections
   → Use when: you need full context (e.g., MD&A, risk factors)
   → Example: "doc_123"

═══════════════════════════════════════════════════════════════════════════════
FINANCIAL ANALYST BEST PRACTICES
═══════════════════════════════════════════════════════════════════════════════

• For REVENUE/EARNINGS queries → vector_search the relevant company's filings
• For COMPARATIVE analysis → parallel vector_search for each company, then synthesize
• For CALCULATIONS → always use calculator tool (never rely on SQL aggregations)
• For RATIO analysis → fetch components first, then calculate
• For RISK FACTORS → vector_search with specific terms (regulatory, cybersecurity, etc.)
• For CROSS-COMPANY comparisons → search each company separately for consistency

EXECUTION OPTIMIZATION:
1. Make steps PARALLEL where possible (independent searches can run together)
2. Only add dependencies when a step TRULY needs another's output
3. For multi-company queries, create parallel searches for each company
4. Keep each step focused on ONE specific piece of information

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE DATA
═══════════════════════════════════════════════════════════════════════════════
{available_data}

═══════════════════════════════════════════════════════════════════════════════
USER QUERY (UNTRUSTED INPUT)
═══════════════════════════════════════════════════════════════════════════════
{query}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════
Return your research plan as JSON:
{{
  "query": "the original query",
  "reasoning": "Senior analyst reasoning: what information is needed and why",
  "steps": [
    {{
      "id": "step_1",
      "tool": "vector_search",
      "input": "precise search query",
      "depends_on": [],
      "description": "what this retrieves"
    }}
  ]
}}

Respond with ONLY the JSON, no other text."""


class Planner:
    """
    Creates execution plans from user queries.
    
    Uses an LLM to decompose queries into tool calls with dependencies.
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        sqlite_store: Any = None,
        company_registry: CompanyRegistry | None = None,
    ):
        """
        Initialize the planner.
        
        Args:
            llm_client: LLM client for generating plans. If None, uses basic heuristics.
            sqlite_store: Optional SQLite store for schema information.
        """
        self.llm_client = llm_client
        self.sqlite_store = sqlite_store
        self.company_registry = company_registry or CompanyRegistry()
    
    async def create_plan(
        self,
        query: str,
        available_tables: list[str] | None = None,
        available_documents: list[str] | None = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for a query.
        
        Args:
            query: The user's natural language query
            available_tables: List of available table names
            available_documents: List of available document IDs
        Returns:
            ExecutionPlan with steps to execute
        """
        if not self.llm_client:
            raise RuntimeError("Query planning requires an LLM client")

        # Fast path: Skip LLM for simple queries
        if self._is_simple_query(query):
            return self._create_fast_plan(query)

        return await self._plan_with_llm(query, available_tables, available_documents)
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Detect if a query is simple enough to skip LLM planning.
        
        Simple queries are single-topic lookups that don't need multi-step reasoning.
        Complex queries involve calculations, comparisons, or multi-company analysis.
        """
        query_lower = query.lower()
        
        # COMPLEX indicators - need LLM planning
        complex_indicators = [
            # Calculations
            'calculate', 'compute', 'ratio', 'percentage', 'growth rate',
            'margin', 'change', 'difference', 'compare',
            # Multi-step reasoning
            'and also', 'as well as', 'in addition',
            # Comparisons
            'vs', 'versus', 'compared to', 'relative to',
            'better than', 'worse than', 'higher than', 'lower than',
            # Multi-company
            'both', 'all three', 'each company',
            # Time-series analysis
            'trend', 'over time', 'year over year', 'yoy', 'qoq',
        ]
        
        for indicator in complex_indicators:
            if indicator in query_lower:
                return False  # Complex query - use LLM
        
        # SIMPLE indicators - can skip LLM
        simple_patterns = [
            # Direct lookups
            'what is', 'what was', 'what are', 'what were',
            'how much', 'how many',
            'tell me about', 'show me',
            # Single entity questions
            'who is', 'when did', 'where is',
        ]
        
        for pattern in simple_patterns:
            if query_lower.startswith(pattern):
                # Check it's not actually complex (multiple companies mentioned)
                known_companies = set(self.company_registry._registry.keys())
                company_count = sum(1 for c in known_companies if c.lower() in query_lower)
                if company_count <= 1:
                    return True  # Simple single-entity query
        
        # Default to LLM for ambiguous queries
        return False
    
    def _create_fast_plan(self, query: str) -> ExecutionPlan:
        """
        Create a fast default plan without LLM call.
        
        Uses parallel vector search + SQL query for comprehensive coverage.
        """
        query_lower = query.lower()
        steps = []
        
        # Detect if this is a stock/data query (use SQL)
        is_stock_query = any(term in query_lower for term in [
            'stock', 'price', 'volume', 'aapl', 'msft', 'nvda', 'googl', 'amzn',
            'average', 'total', 'sum', 'count'
        ])
        
        # Always do vector search (fast and comprehensive)
        steps.append(ToolCall(
            id="step_1",
            tool=ToolName.VECTOR_SEARCH,
            input=query,
            depends_on=[],
            description="Search documents for relevant information"
        ))
        
        # Add SQL query for data-oriented queries
        if is_stock_query:
            steps.append(ToolCall(
                id="step_2", 
                tool=ToolName.SQL_QUERY,
                input=query,
                depends_on=[],
                description="Query structured data tables"
            ))
        
        return ExecutionPlan(
            query=query,
            reasoning="[FAST PATH] Simple query - skipped LLM planning for speed",
            steps=steps
        )
    
    async def _plan_with_llm(
        self,
        query: str,
        available_tables: list[str] | None,
        available_documents: list[str] | None
    ) -> ExecutionPlan:
        """Create plan using LLM with suspicious-input advisory wrapping."""
        # Suspicious-input telemetry is advisory only; prompt wrapping carries
        # the actual boundary between instructions and untrusted user text.
        is_suspicious, patterns = detect_injection_attempt(query)
        if is_suspicious:
            logger.warning(f"Suspicious input advisory matched patterns: {patterns}")
        
        wrapped_query = prepare_prompt_user_content(query, "user_query")
        
        # Format available data
        available_data_parts = []
        
        if available_tables:
            available_data_parts.append(f"Tables: {', '.join(available_tables)}")
        else:
            available_data_parts.append("Tables: (query to discover)")
        
        if available_documents:
            available_data_parts.append(f"Documents: {', '.join(available_documents)}")
        else:
            available_data_parts.append("Documents: (search to discover)")
        
        available_data = "\n".join(available_data_parts)
        
        prompt = PLANNER_PROMPT.format(
            query=wrapped_query,
            available_data=available_data,
            security_notice=STANDARD_PROMPT_GUARD,
        )
        
        # Call LLM
        response = await self.llm_client.generate(prompt)
        
        # Parse response
        try:
            plan_data = parse_json_response(response)
            
            # Validate required structure
            if "steps" not in plan_data or not isinstance(plan_data["steps"], list):
                raise ValueError("Missing or invalid 'steps' array in LLM response")
            
            # Valid tool names for validation
            valid_tools = {t.value for t in ToolName}
            
            # Validate and convert steps
            steps = []
            for i, step in enumerate(plan_data["steps"]):
                # Validate required fields
                if "id" not in step:
                    raise ValueError(f"Step {i} missing required 'id' field")
                if "tool" not in step:
                    raise ValueError(f"Step {i} missing required 'tool' field")
                if "input" not in step:
                    raise ValueError(f"Step {i} missing required 'input' field")
                
                # Validate tool name before creating ToolName enum
                tool_name = step["tool"]
                if tool_name not in valid_tools:
                    logger.warning(f"Invalid tool name '{tool_name}' in step {step['id']}, skipping step")
                    continue
                
                steps.append(ToolCall(
                    id=step["id"],
                    tool=ToolName(tool_name),
                    input=step["input"],
                    depends_on=step.get("depends_on", []),
                    description=step.get("description", "")
                ))
            
            if not steps:
                raise ValueError("No valid steps parsed from LLM response")
            
            plan = ExecutionPlan(
                query=plan_data.get("query", query),
                reasoning=plan_data.get("reasoning", ""),
                steps=steps
            )
            errors = self.validate_plan(plan)
            if errors:
                raise ValueError("; ".join(errors))
            return plan
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM plan as JSON: {e}")
            return self._create_fast_plan(query)
        except ValueError as e:
            logger.warning(f"LLM plan validation failed: {e}")
            return self._create_fast_plan(query)
        except Exception as e:
            # Fall back to heuristics on parse error
            logger.warning(f"Failed to parse LLM plan: {e}")
            return self._create_fast_plan(query)
    
    def validate_plan(self, plan: ExecutionPlan) -> list[str]:
        """
        Validate an execution plan.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        step_ids = set()
        for step in plan.steps:
            if step.id in step_ids:
                errors.append(f"Duplicate step ID: {step.id}")
            step_ids.add(step.id)
        
        for step in plan.steps:
            # Check dependencies exist
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.id} depends on non-existent step {dep}")
            
            # Check for self-dependency
            if step.id in step.depends_on:
                errors.append(f"Step {step.id} depends on itself")
        
        # Check for cycles (simple check)
        try:
            plan.get_execution_layers()
        except ValueError as e:
            errors.append(str(e))
        
        return errors
