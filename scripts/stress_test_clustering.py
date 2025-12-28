#!/usr/bin/env python3
"""Stress test for schema clustering at scale."""

import time
import sys
from pathlib import Path

# Add src to path (works from any directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.schema_cluster import SchemaClusterManager


def stress_test(num_companies: int = 50, tables_per_company: int = 20) -> None:
    """
    Simulate large-scale clustering.
    
    Args:
        num_companies: Number of distinct companies
        tables_per_company: Tables per company
    """
    print(f"=== STRESS TEST: {num_companies} companies × {tables_per_company} tables ===\n")
    
    manager = SchemaClusterManager()  # No persistence, memory only
    
    # Simulate ingestion
    print("1. Simulating table ingestion...")
    start = time.time()
    
    # Use unique prefixes that look like company names (will use first word extraction)
    companies = [f"acme{i:03d}" for i in range(num_companies)]  # acme000, acme001, etc.
    domains = ["income", "balance", "cashflow", "metrics", "segment"]
    
    for company in companies:
        for j in range(tables_per_company):
            domain = domains[j % len(domains)]
            manager.assign_table(
                table_name=f"{company}_{domain}_table_{j}",
                columns=[f"col_{k}" for k in range(20)] + ["revenue", "income", "assets"],
                schema_description=f"Financial data for {company}"
            )
    
    total_tables = num_companies * tables_per_company
    ingestion_time = (time.time() - start) * 1000
    print(f"   Ingested {total_tables} tables in {ingestion_time:.1f}ms")
    print(f"   Created {len(manager.clusters)} clusters\n")
    
    # Test query performance
    print("2. Testing query performance...")
    
    # Single company query
    start = time.time()
    for _ in range(100):
        manager.get_relevant_clusters("company_25 revenue data")
    single_time = (time.time() - start) * 1000
    print(f"   100 single-company queries: {single_time:.1f}ms ({single_time/100:.2f}ms each)")
    
    # Cross-company query (no match, triggers fallback)
    start = time.time()
    for _ in range(100):
        manager.get_relevant_clusters("tech sector comparison growth")
    cross_time = (time.time() - start) * 1000
    print(f"   100 cross-company queries: {cross_time:.1f}ms ({cross_time/100:.2f}ms each)")
    
    # Memory analysis
    print("\n3. Memory analysis...")
    total_keywords = sum(len(c.keywords) for c in manager.clusters.values())
    total_table_refs = sum(len(c.table_names) for c in manager.clusters.values())
    print(f"   Total keywords in memory: {total_keywords}")
    print(f"   Total table references: {total_table_refs}")
    print(f"   Avg keywords per cluster: {total_keywords / len(manager.clusters):.1f}")
    
    # Find largest cluster
    largest = max(manager.clusters.values(), key=lambda c: len(c.keywords))
    print(f"   Largest cluster: {largest.cluster_id} ({len(largest.keywords)} keywords)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    stats = manager.get_stats()
    print(f"Total clusters: {stats['total_clusters']}")
    print(f"Tables indexed: {stats['total_tables_indexed']}")
    print(f"Query time (single company): {single_time/100:.2f}ms")
    print(f"Query time (cross-company): {cross_time/100:.2f}ms")
    
    # Warn about potential issues
    if total_keywords > 10000:
        print(f"\n⚠️  WARNING: {total_keywords} keywords may cause memory issues")
    if single_time / 100 > 10:
        print(f"\n⚠️  WARNING: Query time > 10ms, consider adding company index")


if __name__ == "__main__":
    # Default test
    stress_test(num_companies=50, tables_per_company=20)
    
    print("\n" + "="*60)
    print("EXTREME SCALE TEST")
    print("="*60 + "\n")
    
    # Extreme test
    stress_test(num_companies=100, tables_per_company=50)
