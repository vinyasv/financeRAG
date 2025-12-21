#!/usr/bin/env python3
"""Detailed timing analysis of RAG query execution."""

import sys
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.rag_agent import RAGAgent
from src.llm_client import get_llm_client
from src.agent.executor import ExecutionMonitor


async def detailed_timing_analysis():
    """Run a query with detailed timing breakdown."""
    
    # Test query
    query = "Compare the total revenue growth rates between NVIDIA, JPMorgan, and Berkshire Hathaway for their most recent fiscal year."
    
    print("=" * 70)
    print("DETAILED TIMING ANALYSIS")
    print("=" * 70)
    print(f"Query: {query[:60]}...")
    print()
    
    # Initialize
    llm_client = get_llm_client(provider="auto", model=None)
    agent = RAGAgent(llm_client=llm_client)
    
    # Phase 1: Planning
    print("PHASE 1: PLANNING")
    print("-" * 40)
    planning_start = time.perf_counter()
    
    tables = agent.sqlite_store.list_tables()
    table_names = [t.table_name for t in tables]
    documents = agent.document_store.list_documents()
    doc_ids = [d.id for d in documents]
    
    plan = await agent.planner.create_plan(
        query=query,
        available_tables=table_names,
        available_documents=doc_ids
    )
    
    planning_time = (time.perf_counter() - planning_start) * 1000
    print(f"  Planning time: {planning_time:.0f}ms")
    print(f"  Steps created: {len(plan.steps)}")
    for step in plan.steps:
        print(f"    • {step.id}: {step.tool.value} - {step.description[:50]}...")
    print()
    
    # Phase 2: Execution (with monitoring)
    print("PHASE 2: TOOL EXECUTION")
    print("-" * 40)
    
    monitor = ExecutionMonitor()
    execution_start = time.perf_counter()
    results, timing = await monitor.execute_with_monitoring(agent.executor, plan)
    execution_time = (time.perf_counter() - execution_start) * 1000
    
    print(f"  Total execution time: {execution_time:.0f}ms")
    print(f"  Layers: {timing['layer_count']}")
    for i, layer_time in enumerate(timing['layer_times_ms']):
        print(f"    Layer {i+1}: {layer_time:.0f}ms")
    print()
    print("  Per-step timing:")
    for step_id, step_time in sorted(timing['step_times_ms'].items(), key=lambda x: -x[1]):
        print(f"    {step_id}: {step_time:.0f}ms")
    print()
    
    # Phase 3: Synthesis
    print("PHASE 3: RESPONSE SYNTHESIS")
    print("-" * 40)
    synthesis_start = time.perf_counter()
    response = await agent.synthesizer.synthesize(plan, results)
    synthesis_time = (time.perf_counter() - synthesis_start) * 1000
    
    print(f"  Synthesis time: {synthesis_time:.0f}ms")
    print(f"  Answer length: {len(response.answer)} chars")
    print()
    
    # Summary
    total_time = planning_time + execution_time + synthesis_time
    
    print("=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Phase':<25} {'Time (ms)':>12} {'Percentage':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'1. Planning (LLM)':<25} {planning_time:>12.0f} {planning_time/total_time*100:>11.1f}%")
    print(f"  {'2. Tool Execution':<25} {execution_time:>12.0f} {execution_time/total_time*100:>11.1f}%")
    print(f"  {'3. Synthesis (LLM)':<25} {synthesis_time:>12.0f} {synthesis_time/total_time*100:>11.1f}%")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'TOTAL':<25} {total_time:>12.0f} {'100.0%':>12}")
    print()
    
    # LLM time breakdown
    llm_time = planning_time + synthesis_time
    non_llm_time = execution_time
    
    print("=" * 70)
    print("LLM VS NON-LLM TIME")
    print("=" * 70)
    print(f"  LLM calls (Planning + Synthesis):  {llm_time:.0f}ms ({llm_time/total_time*100:.1f}%)")
    print(f"  Tool execution (Retrieval + Calc):  {non_llm_time:.0f}ms ({non_llm_time/total_time*100:.1f}%)")
    print()
    print("BOTTLENECK ANALYSIS:")
    if llm_time > non_llm_time:
        print(f"  ⚠️  LLM calls are the BOTTLENECK ({llm_time/non_llm_time:.1f}x slower than retrieval)")
        print("  Recommendations:")
        print("    • Use a faster LLM (e.g., gpt-4o-mini instead of gpt-4o)")
        print("    • Enable streaming responses")  
        print("    • Cache planning results for similar queries")
        print("    • Consider using local models for planning")
    else:
        print(f"  ⚠️  Tool execution is the BOTTLENECK")
        print("  Recommendations:")
        print("    • Optimize vector search (reduce k, better indexing)")
        print("    • Add caching for repeated searches")
    
    return {
        "planning_ms": planning_time,
        "execution_ms": execution_time,
        "synthesis_ms": synthesis_time,
        "total_ms": total_time,
        "llm_percentage": llm_time / total_time * 100
    }


if __name__ == "__main__":
    asyncio.run(detailed_timing_analysis())
