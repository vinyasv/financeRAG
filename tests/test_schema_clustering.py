"""Tests for schema clustering functionality."""

import pytest
from src.storage.schema_cluster import (
    SchemaClusterManager,
    SchemaCluster,
    extract_company_from_table_name,
    extract_company_from_query,
    FINANCIAL_DOMAINS,
    DOMAIN_MAP
)


class TestCompanyExtraction:
    """Tests for company name extraction."""
    
    def test_extract_nvidia_from_table(self):
        assert extract_company_from_table_name("nvidia_quarterly_revenue") == "nvidia"
        assert extract_company_from_table_name("nvda_performance") == "nvidia"
        
    def test_extract_jpmorgan_from_table(self):
        assert extract_company_from_table_name("jpmorgan_balance_sheet") == "jpmorgan"
        assert extract_company_from_table_name("jpm_income_2024") == "jpmorgan"
        
    def test_extract_general_from_unknown(self):
        assert extract_company_from_table_name("stock_details_5_years") == "general"
        
    def test_extract_nvidia_from_query(self):
        assert extract_company_from_query("What was NVIDIA's revenue in 2024?") == "nvidia"
        
    def test_extract_none_from_cross_company_query(self):
        assert extract_company_from_query("Compare tech sector performance") is None
    
    # New edge case tests for false positive prevention
    def test_word_boundary_matching_no_false_positives(self):
        """Test that substring matches don't cause false positives (Issue 14 fix)."""
        # "apples" should NOT match "apple"
        assert extract_company_from_query("Compare apples to oranges approach") is None
        # "googlesheet" should NOT match "google"
        assert extract_company_from_query("Export to googlesheet format") is None
        # But "google" alone should still match
        assert extract_company_from_query("What is Google's revenue?") == "google"
    
    def test_generic_prefix_not_treated_as_company(self):
        """Test that financial terms aren't treated as company names (Issue 13 fix)."""
        # These should return "general", not the first word
        assert extract_company_from_table_name("income_statement_2024") == "general"
        assert extract_company_from_table_name("balance_sheet_consolidated") == "general"
        assert extract_company_from_table_name("revenue_by_segment") == "general"
        assert extract_company_from_table_name("quarterly_earnings_report") == "general"
        assert extract_company_from_table_name("annual_performance_summary") == "general"


class TestDomainMatching:
    """Tests for domain matching logic."""
    
    def test_financial_domain_keywords(self):
        domain = DOMAIN_MAP["financial_statements"]
        assert "revenue" in domain.keywords
        assert "income" in domain.keywords
        assert "balance" in domain.keywords
        
    def test_stock_domain_keywords(self):
        domain = DOMAIN_MAP["stock_market"]
        assert "price" in domain.keywords
        assert "volume" in domain.keywords


class TestSchemaClusterManager:
    """Tests for the cluster manager."""
    
    @pytest.mark.asyncio
    async def test_assign_table_to_cluster(self):
        manager = SchemaClusterManager()
        
        cluster_id = await manager.assign_table(
            table_name="nvidia_income_statement",
            columns=["year", "revenue", "net_income"],
            schema_description="NVIDIA annual income statement"
        )
        
        assert cluster_id == "nvidia_financial_statements"
        assert "nvidia_income_statement" in manager.clusters[cluster_id].table_names
    
    @pytest.mark.asyncio
    async def test_assign_multiple_tables_same_cluster(self):
        manager = SchemaClusterManager()
        
        await manager.assign_table(
            table_name="nvidia_income_statement",
            columns=["year", "revenue"],
            schema_description=""
        )
        await manager.assign_table(
            table_name="nvidia_balance_sheet",
            columns=["assets", "liabilities"],
            schema_description=""
        )
        
        cluster = manager.clusters["nvidia_financial_statements"]
        assert "nvidia_income_statement" in cluster.table_names
        assert "nvidia_balance_sheet" in cluster.table_names
    
    @pytest.mark.asyncio
    async def test_entity_isolation(self):
        """NVIDIA data should never be in same cluster as JPMorgan."""
        manager = SchemaClusterManager()
        
        await manager.assign_table("nvidia_revenue", ["revenue"], "")
        await manager.assign_table("jpmorgan_revenue", ["revenue"], "")
        
        nvidia_cluster = manager.clusters.get("nvidia_financial_statements")
        jpmorgan_cluster = manager.clusters.get("jpmorgan_financial_statements")
        
        assert nvidia_cluster is not None
        assert jpmorgan_cluster is not None
        
        # Entity isolation: tables should be in separate clusters
        assert "nvidia_revenue" in nvidia_cluster.table_names
        assert "nvidia_revenue" not in jpmorgan_cluster.table_names
        assert "jpmorgan_revenue" in jpmorgan_cluster.table_names
        assert "jpmorgan_revenue" not in nvidia_cluster.table_names
    
    @pytest.mark.asyncio
    async def test_get_relevant_clusters_company_filter(self):
        """Query mentioning NVIDIA should only return NVIDIA clusters."""
        manager = SchemaClusterManager()
        
        await manager.assign_table("nvidia_revenue", ["revenue"], "")
        await manager.assign_table("jpmorgan_revenue", ["revenue"], "")
        
        clusters = manager.get_relevant_clusters("What was NVIDIA's revenue?")
        
        # Should only get NVIDIA cluster
        assert all(c.company == "nvidia" for c in clusters)
        
    def test_get_relevant_clusters_fallback_empty(self):
        """Query with no match returns empty (triggering fallback)."""
        manager = SchemaClusterManager()
        
        clusters = manager.get_relevant_clusters("Some unrelated query")
        
        assert clusters == []
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        manager = SchemaClusterManager()
        
        await manager.assign_table("nvidia_revenue", ["revenue"], "")
        await manager.assign_table("jpmorgan_balance", ["assets"], "")
        
        stats = manager.get_stats()
        
        assert stats["total_clusters"] == 2
        assert stats["total_tables_indexed"] == 2
        assert "nvidia" in stats["clusters_by_company"]
        assert "jpmorgan" in stats["clusters_by_company"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
