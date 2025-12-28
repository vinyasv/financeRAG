#!/usr/bin/env python3
"""
Multi-Document Analyst Evaluation Script
=========================================
Runs complex, real-world analyst queries across multiple financial documents
(Berkshire 2024, JPMorgan 2024, NVIDIA) and documents:
- Query text
- Response
- Response time
- Accuracy assessment

Usage:
    python scripts/analyst_evaluation.py
    python scripts/analyst_evaluation.py --output results.md
"""

import sys
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.rag_agent import RAGAgent
from src.llm_client import get_llm_client

@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query_id: int
    category: str
    query: str
    expected_sources: list[str]
    answer: str
    response_time_ms: float
    citations_count: int
    sources_used: list[str]
    accuracy_notes: str = ""


# Multi-document analyst queries organized by complexity and type
ANALYST_QUERIES = [
    # ============================================================================
    # CROSS-COMPANY COMPARISONS (require multiple documents)
    # ============================================================================
    {
        "category": "Cross-Company Revenue Comparison",
        "query": "Compare the total revenue growth rates between NVIDIA, JPMorgan, and Berkshire Hathaway for their most recent fiscal year. Which company had the highest growth?",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["NVIDIA had highest growth due to AI boom", "JPMorgan ~10-15% growth", "Berkshire ~20% growth"]
    },
    {
        "category": "Cross-Company Profitability",
        "query": "What are the operating margins for NVIDIA, JPMorgan, and Berkshire Hathaway? Rank them from highest to lowest.",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["NVIDIA has high margins due to GPU pricing power", "Banks typically have lower margins"]
    },
    {
        "category": "Cross-Company Cash Position",
        "query": "Which company has the strongest cash position - NVIDIA, JPMorgan, or Berkshire Hathaway? Provide specific cash and cash equivalents figures.",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["Berkshire known for large cash reserves", "~$150B+ cash at Berkshire"]
    },
    {
        "category": "Cross-Company Shareholder Returns",
        "query": "Compare the share buyback programs of NVIDIA, JPMorgan, and Berkshire Hathaway. Which company returned the most capital to shareholders?",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["All three have significant buyback programs"]
    },
    
    # ============================================================================
    # MULTI-FACTOR ANALYSIS (complex reasoning across data types)
    # ============================================================================
    {
        "category": "Risk Correlation Analysis",
        "query": "Both NVIDIA and JPMorgan face regulatory risks. Compare the specific regulatory challenges each company disclosed in their annual reports. How might these risks be correlated?",
        "expected_sources": ["NVIDIA", "JPMorgan"],
        "ground_truth_hints": ["NVIDIA: export controls, China restrictions", "JPMorgan: banking regulations, capital requirements"]
    },
    {
        "category": "Segment Revenue Breakdown",
        "query": "For both NVIDIA and JPMorgan, what percentage of total revenue comes from their largest business segment? Compare their revenue concentration.",
        "expected_sources": ["NVIDIA", "JPMorgan"],
        "ground_truth_hints": ["NVIDIA: Data Center ~88%", "JPMorgan: Consumer & Community Banking or Corporate & Investment Bank"]
    },
    {
        "category": "Geographic Diversification",
        "query": "Compare the international revenue exposure of NVIDIA and JPMorgan. Which company is more dependent on non-US markets?",
        "expected_sources": ["NVIDIA", "JPMorgan"],
        "ground_truth_hints": ["NVIDIA has significant Asia exposure", "China market concerns"]
    },
    
    # ============================================================================
    # SECTOR-SPECIFIC DEEP DIVES
    # ============================================================================
    {
        "category": "Technology AI Investment",
        "query": "What does NVIDIA's annual report reveal about their AI infrastructure investments? How much are they spending on R&D and capital expenditures?",
        "expected_sources": ["NVIDIA"],
        "ground_truth_hints": ["Significant R&D spending", "AI factory investments"]
    },
    {
        "category": "Banking Sector Analysis",
        "query": "What is JPMorgan's net interest income and how has it changed? What credit quality metrics do they report?",
        "expected_sources": ["JPMorgan"],
        "ground_truth_hints": ["NII growth", "Credit loss provisions"]
    },
    {
        "category": "Conglomerate Structure",
        "query": "List Berkshire Hathaway's major operating subsidiaries and their contribution to overall insurance float and operating earnings.",
        "expected_sources": ["Berkshire"],
        "ground_truth_hints": ["GEICO", "BNSF Railway", "Berkshire Hathaway Energy"]
    },
    
    # ============================================================================
    # QUANTITATIVE ANALYSIS (require calculations)
    # ============================================================================
    {
        "category": "Valuation Ratios",
        "query": "Calculate and compare the price-to-earnings ratios for NVIDIA and JPMorgan based on their reported earnings per share. Use current approximate stock prices if needed.",
        "expected_sources": ["NVIDIA", "JPMorgan"],
        "ground_truth_hints": ["NVIDIA trades at premium P/E", "Banks typically lower P/E"]
    },
    {
        "category": "Free Cash Flow Yield",
        "query": "What is the free cash flow for both NVIDIA and Berkshire Hathaway? Calculate free cash flow as operating cash flow minus capex.",
        "expected_sources": ["NVIDIA", "Berkshire"],
        "ground_truth_hints": ["Strong FCF from both companies"]
    },
    {
        "category": "Return on Equity",
        "query": "Compare the return on equity (ROE) between JPMorgan and Berkshire's insurance operations. Which generates better returns on invested capital?",
        "expected_sources": ["JPMorgan", "Berkshire"],
        "ground_truth_hints": ["JPMorgan typically 15-17% ROE", "Insurance varies"]
    },
    
    # ============================================================================
    # FORWARD-LOOKING ANALYSIS
    # ============================================================================
    {
        "category": "Growth Outlook Comparison",
        "query": "What forward guidance or outlook statements did NVIDIA, JPMorgan, and Berkshire provide? Which seems most optimistic about future growth?",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["NVIDIA strong AI demand outlook", "Buffett typically cautious"]
    },
    {
        "category": "Capital Allocation Priorities",
        "query": "Compare the stated capital allocation priorities of all three companies. Which is focused on dividends, buybacks, acquisitions, or organic growth?",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["Berkshire: acquisitions and buybacks", "JPMorgan: dividends and buybacks"]
    },
    
    # ============================================================================
    # COMPLEX MULTI-HOP QUERIES
    # ============================================================================
    {
        "category": "Supply Chain & Partner Exposure",
        "query": "NVIDIA mentions various manufacturing and cloud partners. Does any of these partners appear significant to JPMorgan or Berkshire's operations or investments?",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["Apple is Berkshire investment", "Cloud providers mentioned by all"]
    },
    {
        "category": "Executive Compensation Comparison",
        "query": "Compare the CEO compensation packages disclosed by NVIDIA, JPMorgan, and Berkshire Hathaway. Which CEO receives the highest total compensation?",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["Jamie Dimon high compensation", "Buffett famously takes $100K salary"]
    },
    {
        "category": "Workforce & Employment",
        "query": "How many employees do NVIDIA, JPMorgan, and Berkshire Hathaway have? Calculate revenue per employee for each.",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["NVIDIA ~30K employees with high revenue per employee", "JPMorgan ~300K", "Berkshire ~400K"]
    },
    
    # ============================================================================
    # EDGE CASES & STRESS TESTS
    # ============================================================================
    {
        "category": "Specific Table Lookup",
        "query": "What were the exact quarterly revenue figures for NVIDIA for each quarter in the most recent fiscal year? Present as a table.",
        "expected_sources": ["NVIDIA"],
        "ground_truth_hints": ["Should show Q1-Q4 breakdown", "Needs SQL query capability"]
    },
    {
        "category": "Risk Factor Search",
        "query": "Find and summarize all mentions of 'cybersecurity' or 'data security' risks across all three companies' annual reports.",
        "expected_sources": ["NVIDIA", "JPMorgan", "Berkshire"],
        "ground_truth_hints": ["All companies should mention cyber risks", "JPMorgan especially given banking"]
    },
]


async def run_evaluation():
    """Run the full multi-document evaluation."""
    
    print("=" * 80)
    print("ULTIMRAG MULTI-DOCUMENT ANALYST EVALUATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize agent
    llm_client = get_llm_client(provider="auto", model=None)
    if llm_client:
        print(f"✓ LLM client initialized")
    else:
        print("⚠ No LLM configured - results may be limited")
    
    agent = RAGAgent(llm_client=llm_client)
    
    # Get stats
    stats = agent.get_stats()
    print(f"✓ Knowledge base: {stats['document_count']} documents, {stats['table_count']} tables, {stats['chunk_count']} chunks")
    print()
    
    if stats['documents']:
        print("Documents in corpus:")
        for doc in stats['documents']:
            print(f"  • {doc['filename']} ({doc['pages']} pages)")
    print()
    
    # Run queries
    results: list[QueryResult] = []
    total_time = 0
    
    print("-" * 80)
    print(f"Running {len(ANALYST_QUERIES)} analyst queries...")
    print("-" * 80)
    print()
    
    for idx, q in enumerate(ANALYST_QUERIES, 1):
        print(f"[{idx}/{len(ANALYST_QUERIES)}] {q['category']}")
        print(f"Query: {q['query'][:80]}...")
        
        start_time = time.time()
        try:
            response = await agent.query(q['query'], verbose=False)
            elapsed_ms = (time.time() - start_time) * 1000
            total_time += elapsed_ms
            
            # Extract sources from citations
            sources = []
            if response.citations:
                for cite in response.citations:
                    doc_name = cite.document_name if cite.document_name else "Unknown"
                    if doc_name not in sources:
                        sources.append(doc_name)
            
            result = QueryResult(
                query_id=idx,
                category=q['category'],
                query=q['query'],
                expected_sources=q['expected_sources'],
                answer=response.answer,
                response_time_ms=elapsed_ms,
                citations_count=len(response.citations) if response.citations else 0,
                sources_used=sources
            )
            results.append(result)
            
            # Check if expected sources were used
            expected_hit = any(
                any(exp.lower() in src.lower() for src in sources) 
                for exp in q['expected_sources']
            )
            
            status = "✓" if expected_hit else "⚠"
            print(f"{status} Response: {elapsed_ms:.0f}ms | {len(sources)} sources | {len(response.answer)} chars")
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            result = QueryResult(
                query_id=idx,
                category=q['category'],
                query=q['query'],
                expected_sources=q['expected_sources'],
                answer=f"ERROR: {str(e)}",
                response_time_ms=elapsed_ms,
                citations_count=0,
                sources_used=[]
            )
            results.append(result)
            print(f"✗ Error: {str(e)[:50]}...")
        
        print()
    
    # Generate report
    avg_time = total_time / len(results) if results else 0
    
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total queries: {len(results)}")
    print(f"Total time: {total_time/1000:.1f}s")
    print(f"Average response time: {avg_time:.0f}ms")
    print()
    
    # Analyze source coverage
    multi_source_queries = [r for r in results if len(r.sources_used) > 1]
    print(f"Multi-source responses: {len(multi_source_queries)}/{len(results)}")
    
    # Time distribution
    fast_queries = [r for r in results if r.response_time_ms < 5000]
    medium_queries = [r for r in results if 5000 <= r.response_time_ms < 10000]
    slow_queries = [r for r in results if r.response_time_ms >= 10000]
    print(f"Fast (<5s): {len(fast_queries)} | Medium (5-10s): {len(medium_queries)} | Slow (>10s): {len(slow_queries)}")
    print()
    
    # Generate detailed markdown report
    report_path = Path(__file__).parent.parent / "analyst_evaluation_results.md"
    generate_markdown_report(results, stats, report_path)
    print(f"✓ Detailed report saved to: {report_path}")
    
    # Also save JSON for programmatic access
    json_path = Path(__file__).parent.parent / "analyst_evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_queries": len(results),
                "avg_response_time_ms": avg_time,
                "total_time_s": total_time / 1000,
                "multi_source_responses": len(multi_source_queries),
                "fast_queries": len(fast_queries),
                "medium_queries": len(medium_queries),
                "slow_queries": len(slow_queries),
                "timestamp": datetime.now().isoformat()
            },
            "results": [asdict(r) for r in results]
        }, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")
    
    return results


def generate_markdown_report(results: list[QueryResult], stats: dict, output_path: Path):
    """Generate a detailed markdown report."""
    
    lines = [
        "# Finance RAG Multi-Document Analyst Evaluation",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Queries | {len(results)} |",
        f"| Documents in Corpus | {stats['document_count']} |",
        f"| Total Chunks | {stats['chunk_count']} |",
        f"| Average Response Time | {sum(r.response_time_ms for r in results)/len(results) if results else 0:.0f}ms |",
        f"| Multi-Source Responses | {len([r for r in results if len(r.sources_used) > 1])} |",
        "",
        "## Query Results",
        "",
    ]
    
    for r in results:
        lines.extend([
            f"### {r.query_id}. {r.category}",
            "",
            f"**Query:** {r.query}",
            "",
            f"**Response Time:** {r.response_time_ms:.0f}ms",
            "",
            f"**Expected Sources:** {', '.join(r.expected_sources)}",
            "",
            f"**Sources Used:** {', '.join(r.sources_used) if r.sources_used else 'None detected'}",
            "",
            f"**Citations:** {r.citations_count}",
            "",
            "**Answer:**",
            "",
            r.answer,
            "",
            "---",
            "",
        ])
    
    # Performance breakdown
    lines.extend([
        "## Performance Analysis",
        "",
        "### Response Time Distribution",
        "",
        "| Query | Category | Time (ms) | Sources |",
        "|-------|----------|-----------|---------|",
    ])
    
    for r in sorted(results, key=lambda x: x.response_time_ms, reverse=True):
        lines.append(f"| {r.query_id} | {r.category[:30]} | {r.response_time_ms:.0f} | {len(r.sources_used)} |")
    
    lines.extend([
        "",
        "## Notes",
        "",
        "- Queries requiring multiple documents (cross-company analysis) typically take longer",
        "- Multi-source responses indicate successful retrieval from multiple documents",
        "- Response times include planning, retrieval, reranking, and synthesis",
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    asyncio.run(run_evaluation())
