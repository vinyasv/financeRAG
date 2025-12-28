"""Schema clustering for scalable table context management.

Groups tables by Company + Domain hierarchy to ensure:
1. Entity isolation: NVIDIA numbers never confused with other companies
2. Semantic grouping: Related tables (income, balance sheet) grouped within company
3. Scalability: Only relevant schemas included in LLM prompts
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Domain Definitions
# =============================================================================

@dataclass
class DomainDefinition:
    """Definition of a semantic domain for table clustering."""
    domain_id: str
    name: str
    keywords: set[str]
    description: str
    priority: int = 0  # Higher = preferred when multiple domains match


# Pre-defined financial domains
FINANCIAL_DOMAINS = [
    DomainDefinition(
        domain_id="financial_statements",
        name="Financial Statements",
        keywords={
            "revenue", "income", "expense", "profit", "loss", "earnings",
            "balance", "assets", "liabilities", "equity", "cash", "flow",
            "operating", "net", "gross", "margin", "ebitda", "ebit"
        },
        description="Income statements, balance sheets, cash flow statements",
        priority=10
    ),
    DomainDefinition(
        domain_id="stock_market",
        name="Stock Market Data",
        keywords={
            "stock", "price", "close", "open", "high", "low", "volume",
            "ticker", "symbol", "share", "trading", "market", "adj"
        },
        description="Stock prices, trading volumes, market data",
        priority=8
    ),
    DomainDefinition(
        domain_id="performance_metrics",
        name="Performance Metrics",
        keywords={
            "ratio", "growth", "yoy", "qoq", "percentage", "change",
            "roi", "roe", "roa", "eps", "pe", "margin", "rate"
        },
        description="Financial ratios, growth metrics, performance indicators",
        priority=7
    ),
    DomainDefinition(
        domain_id="segment_data",
        name="Segment/Geographic Data",
        keywords={
            "segment", "region", "geographic", "division", "unit",
            "americas", "emea", "apac", "asia", "europe", "domestic", "international"
        },
        description="Business segment and geographic breakdown data",
        priority=6
    ),
    DomainDefinition(
        domain_id="quarterly_data",
        name="Quarterly Data",
        keywords={
            "quarter", "q1", "q2", "q3", "q4", "quarterly", "fiscal",
            "fy", "period", "ytd"
        },
        description="Quarterly financial data and period-specific metrics",
        priority=5
    ),
    DomainDefinition(
        domain_id="general",
        name="General Data",
        keywords=set(),  # Catch-all
        description="General tables not matching specific domains",
        priority=0
    ),
]

DOMAIN_MAP = {d.domain_id: d for d in FINANCIAL_DOMAINS}


# =============================================================================
# Schema Cluster Model
# =============================================================================

@dataclass
class SchemaCluster:
    """
    A cluster of semantically related tables.
    
    Hierarchy: Company -> Domain -> Tables
    Example: "nvidia" -> "financial_statements" -> ["nvidia_income_stmt", "nvidia_balance_sheet"]
    """
    cluster_id: str          # e.g., "nvidia_financial_statements"
    company: str             # e.g., "nvidia"
    domain_id: str           # e.g., "financial_statements"
    table_names: list[str] = field(default_factory=list)
    keywords: set[str] = field(default_factory=set)
    centroid: list[float] | None = None  # For embedding-based matching
    
    @property
    def display_name(self) -> str:
        domain = DOMAIN_MAP.get(self.domain_id)
        domain_name = domain.name if domain else self.domain_id
        return f"{self.company.upper()} - {domain_name}"
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "company": self.company,
            "domain_id": self.domain_id,
            "table_names": self.table_names,
            "keywords": list(self.keywords),
            "centroid": self.centroid
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SchemaCluster":
        return cls(
            cluster_id=data["cluster_id"],
            company=data["company"],
            domain_id=data["domain_id"],
            table_names=data.get("table_names", []),
            keywords=set(data.get("keywords", [])),
            centroid=data.get("centroid")
        )


# =============================================================================
# Company Registry - Dynamic LLM-based Extraction
# =============================================================================

# Simple prompt for company extraction from filename
COMPANY_EXTRACTION_PROMPT = """Extract the company name from this document filename.

Filename: {filename}

Rules:
1. Return ONLY the company name in lowercase, nothing else
2. If it's a multi-word company name, separate with underscore (e.g., "berkshire_hathaway")
3. If no clear company name, return "general"
4. Common examples:
   - "NVIDIA_Q1_FY2026_Earnings.pdf" -> "nvidia"
   - "JPMorgan_2024_Annual_Report.pdf" -> "jpmorgan"
   - "stock_details_5_years.csv" -> "general"
   - "Berkshire_Hathaway_2024_Letter.pdf" -> "berkshire_hathaway"

Company name:"""


# Prompt for LLM-based table classification during ingestion
TABLE_CLASSIFICATION_PROMPT = """Classify this database table.

Table name: {table_name}
Columns: {columns}
Source document: {source_document}

Respond with JSON only, no other text:
{{"company": "company_name", "domain": "domain_id"}}

COMPANY:
- Use lowercase with underscores for spaces (e.g., "berkshire_hathaway")
- Extract from source document name if unclear from table name
- Use "general" only if truly unknown

DOMAIN (pick one):
- "financial_statements" - Income statements, balance sheets, cash flow
- "stock_market" - Stock prices, trading volume, ticker data
- "performance_metrics" - Ratios, growth rates, ROE/ROA
- "segment_data" - Geographic or business segment breakdowns
- "quarterly_data" - Q1-Q4 specific data
- "general" - Only if none of the above fit

JSON:"""


class CompanyRegistry:
    """
    Dynamic registry of companies learned from ingested documents.
    
    Instead of hardcoded company lists, this:
    1. Uses LLM to extract company names from document filenames during ingestion
    2. Stores learned companies in SQLite for persistence
    3. Provides fast lookup at query time using the learned registry
    """
    
    # Common variations for fallback matching (when no registry or LLM available)
    # Includes both ticker symbols and full company names
    COMMON_VARIATIONS = {
        # NVIDIA
        "nvidia": "nvidia",
        "nvda": "nvidia",
        # JPMorgan
        "jpmorgan": "jpmorgan",
        "jpm": "jpmorgan",
        "chase": "jpmorgan",
        # Berkshire
        "berkshire": "berkshire",
        "brk": "berkshire",
        # Apple
        "apple": "apple",
        "aapl": "apple",
        # Microsoft
        "microsoft": "microsoft",
        "msft": "microsoft",
        # Google
        "google": "google",
        "googl": "google",
        "alphabet": "google",
        # Amazon
        "amazon": "amazon",
        "amzn": "amazon",
        # Tesla
        "tesla": "tesla",
        "tsla": "tesla",
        # Meta
        "meta": "meta",
        "facebook": "meta",
    }
    
    def __init__(self, sqlite_store: Any = None, llm_client: Any = None):
        self.sqlite_store = sqlite_store
        self.llm_client = llm_client
        
        # In-memory cache: pattern -> canonical_name
        # Pre-load common variations so ticker lookups work immediately
        self._registry: dict[str, str] = dict(self.COMMON_VARIATIONS)
        
        # Ensure table exists and load registry (may override defaults)
        if sqlite_store:
            self._init_table()
            self._load_registry()
    
    def _init_table(self) -> None:
        """Create company registry table if not exists."""
        with self.sqlite_store._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS company_registry (
                    pattern TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    source TEXT DEFAULT 'llm',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _load_registry(self) -> None:
        """Load registry from SQLite."""
        with self.sqlite_store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pattern, canonical_name FROM company_registry")
            for row in cursor.fetchall():
                self._registry[row["pattern"]] = row["canonical_name"]
        
        logger.debug(f"Loaded {len(self._registry)} company patterns from registry")
    
    async def extract_company_from_filename(self, filename: str) -> str:
        """
        Extract company name from filename using LLM.
        
        This is called during document ingestion to learn new companies.
        """
        # Check if we already know this pattern
        filename_lower = filename.lower()
        for pattern, canonical in sorted(self._registry.items(), key=lambda x: len(x[0]), reverse=True):
            if pattern in filename_lower:
                return canonical
        
        # Use LLM to extract company name
        if self.llm_client:
            try:
                prompt = COMPANY_EXTRACTION_PROMPT.format(filename=filename)
                response = await self.llm_client.generate(prompt)
                company = response.strip().lower().replace(" ", "_")
                
                # Validate response
                if company and len(company) >= 2 and company != "general":
                    # Register the new company
                    self._register_company(company, filename_lower)
                    return company
                elif company == "general":
                    return "general"
            except Exception as e:
                logger.warning(f"LLM company extraction failed: {e}")
        
        # Fallback: use heuristics
        return self._extract_with_heuristics(filename)
    
    def _register_company(self, canonical_name: str, source_filename: str) -> None:
        """Register a new company in the registry."""
        # Extract potential patterns from filename
        patterns = self._derive_patterns(canonical_name, source_filename)
        
        for pattern in patterns:
            if pattern not in self._registry:
                self._registry[pattern] = canonical_name
                
                # Persist to SQLite
                if self.sqlite_store:
                    try:
                        with self.sqlite_store._get_connection() as conn:
                            conn.execute(
                                "INSERT OR IGNORE INTO company_registry (pattern, canonical_name) VALUES (?, ?)",
                                (pattern, canonical_name)
                            )
                            conn.commit()
                    except Exception as e:
                        logger.warning(f"Failed to persist company pattern: {e}")
        
        logger.info(f"Registered new company: {canonical_name} with patterns: {patterns}")
    
    def _derive_patterns(self, canonical_name: str, source_filename: str) -> list[str]:
        """Derive searchable patterns from canonical name."""
        patterns = [canonical_name]
        
        # Add variations without underscores
        if "_" in canonical_name:
            patterns.append(canonical_name.replace("_", ""))
            # Also add first word as pattern
            first_word = canonical_name.split("_")[0]
            if len(first_word) >= 4:
                patterns.append(first_word)
        
        # Check for common ticker variations
        for ticker, company in self.COMMON_VARIATIONS.items():
            if company == canonical_name:
                patterns.append(ticker)
        
        return list(set(patterns))
    
    def _extract_with_heuristics(self, filename: str) -> str:
        """Fallback heuristic extraction when LLM unavailable."""
        filename_lower = filename.lower()
        
        # Check common variations
        for pattern, canonical in self.COMMON_VARIATIONS.items():
            if pattern in filename_lower:
                return canonical
        
        # Try first word before underscore/hyphen
        parts = re.split(r'[_\-\s]', filename_lower.replace('.pdf', '').replace('.xlsx', '').replace('.csv', ''))
        if parts and len(parts[0]) >= 3:
            first_part = parts[0]
            # Filter out generic terms
            generic_terms = {"annual", "quarterly", "report", "financial", "stock", "data", "sheet"}
            if first_part not in generic_terms:
                return first_part
        
        return "general"
    
    def lookup_company(self, query: str) -> str | None:
        """
        Look up company in query using learned registry.
        
        Uses word boundary matching for accuracy.
        Returns None if no company found.
        """
        query_lower = query.lower()
        
        # Sort patterns by length (longest first) to match specific patterns first
        sorted_patterns = sorted(self._registry.keys(), key=len, reverse=True)
        
        for pattern in sorted_patterns:
            # Word boundary matching to avoid false positives
            if re.search(rf'\b{re.escape(pattern)}\b', query_lower):
                return self._registry[pattern]
        
        return None
    
    def lookup_from_table_name(self, table_name: str) -> str:
        """
        Look up company from table name.
        
        Returns 'general' if no company found.
        """
        table_lower = table_name.lower()
        
        # Check registry
        sorted_patterns = sorted(self._registry.keys(), key=len, reverse=True)
        for pattern in sorted_patterns:
            if pattern in table_lower:
                return self._registry[pattern]
        
        # Fallback to first word if not generic
        parts = re.split(r'[_\s-]', table_lower)
        if parts and len(parts[0]) >= 3:
            generic_prefixes = {
                "table", "data", "sheet", "document", "report", "summary",
                "annual", "quarterly", "monthly", "weekly", "daily", "fiscal", "year", "ytd",
                "stock", "income", "balance", "cash", "revenue", "expense", "profit", "loss",
                "assets", "liability", "liabilities", "equity", "earnings", "margin",
                "notes", "segment", "geographic", "consolidated", "audited",
                "ratio", "growth", "performance", "metrics"
            }
            if parts[0] not in generic_prefixes:
                return parts[0]
        
        return "general"
    
    def get_all_companies(self) -> list[str]:
        """Get list of all known canonical company names."""
        return list(set(self._registry.values()))
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        canonical_companies = set(self._registry.values())
        return {
            "total_patterns": len(self._registry),
            "unique_companies": len(canonical_companies),
            "companies": sorted(canonical_companies)
        }


# Global registry instance (initialized by SchemaClusterManager)
_company_registry: CompanyRegistry | None = None


def get_company_registry() -> CompanyRegistry | None:
    """Get the global company registry instance."""
    return _company_registry


def extract_company_from_table_name(table_name: str) -> str:
    """
    Extract company identifier from table name.
    
    Uses the dynamic CompanyRegistry if available, otherwise falls back to heuristics.
    """
    if _company_registry:
        return _company_registry.lookup_from_table_name(table_name)
    
    # Fallback to heuristics if no registry
    table_lower = table_name.lower()
    
    # Check common variations first (tickers)
    for pattern, canonical in CompanyRegistry.COMMON_VARIATIONS.items():
        if pattern in table_lower:
            return canonical
    
    # Try first word if not generic
    parts = re.split(r'[_\s-]', table_lower)
    if parts and len(parts[0]) >= 3:
        generic_prefixes = {
            "table", "data", "sheet", "document", "report", "summary",
            "annual", "quarterly", "monthly", "weekly", "daily", "fiscal", "year", "ytd",
            "stock", "income", "balance", "cash", "revenue", "expense", "profit", "loss",
            "assets", "liability", "liabilities", "equity", "earnings", "margin",
            "notes", "segment", "geographic", "consolidated", "audited",
            "ratio", "growth", "performance", "metrics"
        }
        if parts[0] not in generic_prefixes:
            return parts[0]
    return "general"


def extract_company_from_query(query: str) -> str | None:
    """
    Extract company identifier from user query.
    
    Uses the dynamic CompanyRegistry if available, falls back to COMMON_VARIATIONS.
    Returns None if no specific company mentioned (cross-company query).
    
    For queries mentioning multiple companies, use extract_all_companies_from_query instead.
    """
    if _company_registry:
        return _company_registry.lookup_company(query)
    
    # Fallback: check common variations with word boundary matching
    query_lower = query.lower()
    for pattern, canonical in sorted(CompanyRegistry.COMMON_VARIATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf'\b{re.escape(pattern)}\b', query_lower):
            return canonical
    
    return None


def extract_all_companies_from_query(query: str) -> list[str]:
    """
    Extract ALL company identifiers mentioned in a user query.
    
    This is essential for cross-company comparison queries like:
    - "Compare NVIDIA revenue to Apple revenue" → ["nvidia", "apple"]
    - "Which is bigger, JPMorgan or Berkshire?" → ["jpmorgan", "berkshire"]
    
    Returns empty list if no companies mentioned.
    """
    companies_found = set()
    query_lower = query.lower()
    
    # Use registry if available
    if _company_registry:
        registry = _company_registry._registry
    else:
        registry = CompanyRegistry.COMMON_VARIATIONS
    
    # Check all patterns with word boundary matching
    for pattern, canonical in registry.items():
        if re.search(rf'\b{re.escape(pattern)}\b', query_lower):
            companies_found.add(canonical)
    
    return list(companies_found)



# =============================================================================
# Schema Cluster Manager
# =============================================================================

# Configuration constants for scale optimization
MAX_CLUSTER_KEYWORDS = 300  # Limit keyword set size per cluster
MAX_TABLES_PER_CLUSTER = 100  # Warn if cluster exceeds this
MAX_FALLBACK_TABLES = 50  # Limit tables in fallback mode to prevent context overflow
CACHE_TTL_SECONDS = 300  # 5-minute TTL for table metadata cache
MAX_DESCRIPTION_KEYWORDS = 100  # Limit keywords extracted from descriptions

class SchemaClusterManager:
    """
    Manages semantic clustering of table schemas.
    
    Provides:
    1. Table assignment to Company+Domain clusters during ingestion
    2. Fast cluster lookup during query (keyword-based with company index)
    3. Fallback to all schemas if no good match
    4. Caching to avoid repeated DB calls
    5. Dynamic company learning via LLM during ingestion
    """
    
    def __init__(self, sqlite_store: Any = None, embedding_client: Any = None, llm_client: Any = None):
        """
        Initialize cluster manager.
        
        Args:
            sqlite_store: SQLite store for persistence
            embedding_client: Optional embedding client for semantic matching
            llm_client: Optional LLM client for company name extraction
        """
        global _company_registry
        
        self.sqlite_store = sqlite_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        
        # Initialize company registry for dynamic company learning
        _company_registry = CompanyRegistry(sqlite_store, llm_client)
        self.company_registry = _company_registry
        
        # In-memory cluster cache
        self.clusters: dict[str, SchemaCluster] = {}
        self.table_to_cluster: dict[str, str] = {}  # table_name -> cluster_id
        
        # FIX #2: Company index for O(1) lookup instead of O(n) scan
        self.clusters_by_company: dict[str, list[str]] = {}  # company -> [cluster_ids]
        
        # FIX #3: Table metadata cache to avoid DB calls on every query
        self._table_metadata_cache: dict[str, dict] = {}  # table_name -> metadata
        self._cache_valid = False
        self._cache_timestamp: float = 0.0  # Time when cache was last refreshed
        
        # Query analytics for optimization insights
        self.query_hits: dict[str, int] = {}  # cluster_id -> hit count
        self.total_queries = 0
        self.fallback_count = 0
        
        # Load from persistence if available
        if sqlite_store:
            self._load_clusters()
    
    def _load_clusters(self) -> None:
        """Load clusters from SQLite persistence."""
        try:
            clusters_data = self.sqlite_store.get_all_clusters()
            for data in clusters_data:
                cluster = SchemaCluster.from_dict(data)
                self.clusters[cluster.cluster_id] = cluster
                
                # Build company index
                if cluster.company not in self.clusters_by_company:
                    self.clusters_by_company[cluster.company] = []
                self.clusters_by_company[cluster.company].append(cluster.cluster_id)
                
                for table_name in cluster.table_names:
                    self.table_to_cluster[table_name] = cluster.cluster_id
            
            logger.info(f"Loaded {len(self.clusters)} schema clusters")
        except Exception as e:
            logger.warning(f"Could not load clusters: {e}")
    
    async def learn_company_from_document(self, filename: str) -> str:
        """
        Learn company name from a document filename using LLM.
        
        This should be called during document ingestion to populate
        the company registry dynamically.
        
        Args:
            filename: Document filename (e.g., "NVIDIA_Q1_FY2026_Earnings.pdf")
            
        Returns:
            Canonical company name (e.g., "nvidia")
        """
        if self.company_registry:
            return await self.company_registry.extract_company_from_filename(filename)
        return "general"
    
    # Valid domains for LLM classification validation
    VALID_DOMAINS = {
        "financial_statements", "stock_market", "performance_metrics",
        "segment_data", "quarterly_data", "general"
    }
    
    async def classify_table_with_llm(
        self,
        table_name: str,
        columns: list[str],
        source_document: str = "",
        description: str = ""
    ) -> tuple[str, str]:
        """
        Use LLM to classify a table into company + domain.
        
        This provides much more accurate classification than heuristics,
        especially when the source document context is provided.
        
        Args:
            table_name: Name of the table
            columns: List of column names
            source_document: Source filename (e.g., "NVIDIA_Annual_Report.pdf")
            description: Optional table description
            
        Returns:
            (company, domain_id) tuple
        """
        if not self.llm_client:
            # Fallback to heuristics
            return self._classify_with_heuristics(table_name, columns, description)
        
        try:
            # Build prompt
            columns_str = ", ".join(columns[:20])  # Limit to first 20 columns
            if len(columns) > 20:
                columns_str += f" (and {len(columns) - 20} more)"
            
            prompt = TABLE_CLASSIFICATION_PROMPT.format(
                table_name=table_name,
                columns=columns_str,
                source_document=source_document or "unknown"
            )
            
            # Call LLM
            response = await self.llm_client.generate(prompt)
            response = response.strip()
            
            # Parse JSON response
            # Try to extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON found in response: {response[:100]}")
            
            # Validate and normalize company
            company = result.get("company", "general").lower().strip()
            company = re.sub(r'[^a-z0-9_]', '_', company)  # Sanitize
            if not company or company == "unknown":
                company = "general"
            
            # Validate domain
            domain = result.get("domain", "general").lower().strip()
            if domain not in self.VALID_DOMAINS:
                logger.warning(f"LLM returned invalid domain '{domain}', using heuristics")
                domain = self._match_domain(
                    self._extract_keywords(table_name, columns, description)
                )
            
            # Register new company if not "general"
            if company != "general" and self.company_registry:
                self.company_registry._register_company(
                    canonical_name=company,
                    source_filename=source_document or "llm_classification"
                )
            
            logger.debug(f"LLM classified '{table_name}' as company='{company}', domain='{domain}'")
            return (company, domain)
            
        except Exception as e:
            logger.warning(f"LLM classification failed for '{table_name}': {e}, using heuristics")
            return self._classify_with_heuristics(table_name, columns, description)
    
    def _classify_with_heuristics(
        self,
        table_name: str,
        columns: list[str],
        description: str = ""
    ) -> tuple[str, str]:
        """Classify table using heuristic rules (fallback)."""
        company = extract_company_from_table_name(table_name)
        table_keywords = self._extract_keywords(table_name, columns, description)
        domain_id = self._match_domain(table_keywords)
        return (company, domain_id)
    
    def get_company_stats(self) -> dict:
        """Get statistics about learned companies."""
        if self.company_registry:
            return self.company_registry.get_stats()
        return {"total_patterns": 0, "unique_companies": 0, "companies": []}
    
    async def assign_table(
        self,
        table_name: str,
        columns: list[str],
        schema_description: str = "",
        source_document: str = ""
    ) -> str:
        """
        Assign a table to a cluster based on its metadata.
        
        Uses LLM classification when available for accurate clustering,
        falls back to heuristics otherwise.
        
        Args:
            table_name: Name of the table
            columns: List of column names
            schema_description: Optional description of the table
            source_document: Source filename (improves LLM classification)
            
        Returns:
            cluster_id the table was assigned to
        """
        # Classify table (LLM if available, otherwise heuristics)
        if self.llm_client:
            company, domain_id = await self.classify_table_with_llm(
                table_name, columns, source_document, schema_description
            )
        else:
            company, domain_id = self._classify_with_heuristics(
                table_name, columns, schema_description
            )
        
        # Build keyword set for cluster matching
        table_keywords = self._extract_keywords(table_name, columns, schema_description)
        
        # Create cluster ID
        cluster_id = f"{company}_{domain_id}"
        
        # Get or create cluster
        is_new_cluster = cluster_id not in self.clusters
        if is_new_cluster:
            self.clusters[cluster_id] = SchemaCluster(
                cluster_id=cluster_id,
                company=company,
                domain_id=domain_id,
                table_names=[],
                keywords=set()
            )
            # Update company index
            if company not in self.clusters_by_company:
                self.clusters_by_company[company] = []
            self.clusters_by_company[company].append(cluster_id)
        
        cluster = self.clusters[cluster_id]
        
        # Add table to cluster if not already there
        if table_name not in cluster.table_names:
            cluster.table_names.append(table_name)
            
            # Only add keywords if under limit
            if len(cluster.keywords) < MAX_CLUSTER_KEYWORDS:
                remaining_capacity = MAX_CLUSTER_KEYWORDS - len(cluster.keywords)
                keywords_to_add = list(table_keywords)[:remaining_capacity]
                cluster.keywords.update(keywords_to_add)
            
            # Warn on large clusters
            if len(cluster.table_names) > MAX_TABLES_PER_CLUSTER:
                logger.warning(
                    f"Cluster '{cluster_id}' has {len(cluster.table_names)} tables, "
                    f"exceeds recommended limit of {MAX_TABLES_PER_CLUSTER}"
                )
        
        # Update index
        self.table_to_cluster[table_name] = cluster_id
        
        # Invalidate cache since data changed
        self._cache_valid = False
        
        # Persist if store available
        if self.sqlite_store:
            self.sqlite_store.save_cluster(cluster.to_dict())
        
        logger.debug(f"Assigned table '{table_name}' to cluster '{cluster_id}'")
        return cluster_id
    
    def _extract_keywords(
        self,
        table_name: str,
        columns: list[str],
        description: str
    ) -> set[str]:
        """Extract keywords from table metadata for matching."""
        keywords = set()
        
        # From table name
        parts = re.split(r'[_\s-]', table_name.lower())
        keywords.update(p for p in parts if len(p) >= 3)
        
        # From columns
        for col in columns:
            col_parts = re.split(r'[_\s-]', col.lower())
            keywords.update(p for p in col_parts if len(p) >= 3)
        
        # From description (limited to prevent bloat)
        if description:
            desc_words = re.findall(r'\b\w{3,}\b', description.lower())
            keywords.update(desc_words[:MAX_DESCRIPTION_KEYWORDS])
        
        return keywords
    
    def _match_domain(self, keywords: set[str]) -> str:
        """Match keywords to best domain."""
        best_domain = "general"
        best_score = 0
        best_priority = -1
        
        for domain in FINANCIAL_DOMAINS:
            if not domain.keywords:  # Skip catch-all
                continue
                
            overlap = len(keywords & domain.keywords)
            
            # Score = overlap count, with priority as tiebreaker
            if overlap > best_score or (overlap == best_score and domain.priority > best_priority):
                best_score = overlap
                best_priority = domain.priority
                best_domain = domain.domain_id
        
        return best_domain
    
    def get_relevant_clusters(
        self,
        query: str,
        top_k: int = 3
    ) -> list[SchemaCluster]:
        """
        Get clusters relevant to a query.
        
        Uses optimized matching:
        1. O(1) company lookup via company index (if company mentioned)
        2. Keyword overlap with cluster keywords
        3. Analytics tracking for optimization insights
        
        Args:
            query: User's natural language query
            top_k: Maximum clusters to return
            
        Returns:
            List of relevant clusters, or empty if none found
        """
        self.total_queries += 1
        
        if not self.clusters:
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        # Step 1: Check for ALL companies mentioned (supports cross-company queries)
        target_companies = extract_all_companies_from_query(query)
        
        # Step 2: Use company index for efficient lookup
        if target_companies:
            # Collect clusters for ALL mentioned companies + general
            candidate_cluster_ids = []
            for company in target_companies:
                candidate_cluster_ids.extend(self.clusters_by_company.get(company, []))
            # Always include general clusters as fallback
            candidate_cluster_ids.extend(self.clusters_by_company.get("general", []))
            # Deduplicate
            candidate_cluster_ids = list(set(candidate_cluster_ids))
        else:
            # No specific company: check all clusters
            candidate_cluster_ids = list(self.clusters.keys())
        
        # Step 3: Score only candidate clusters
        scores: dict[str, float] = {}
        
        for cluster_id in candidate_cluster_ids:
            cluster = self.clusters.get(cluster_id)
            if not cluster:
                continue
                
            score = 0.0
            
            # Company match bonus (strong signal)
            if target_companies and cluster.company in target_companies:
                score += 10.0  # Strong boost for company match
            elif cluster.company == "general":
                score += 1.0  # General tables might still be relevant
            
            # Keyword overlap
            keyword_overlap = len(query_words & cluster.keywords)
            score += keyword_overlap * 2.0
            
            # Domain keyword match
            domain = DOMAIN_MAP.get(cluster.domain_id)
            if domain:
                domain_overlap = len(query_words & domain.keywords)
                score += domain_overlap * 1.5
            
            if score > 0:
                scores[cluster_id] = score
        
        if not scores:
            # No good match - return empty (will trigger fallback)
            self.fallback_count += 1
            logger.debug(f"No cluster match for query: {query[:50]}...")
            return []
        
        # Sort by score and return top K
        sorted_clusters = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = [self.clusters[c] for c in sorted_clusters[:top_k]]
        
        # Track hits for analytics
        for c in result:
            self.query_hits[c.cluster_id] = self.query_hits.get(c.cluster_id, 0) + 1
        
        logger.debug(f"Query matched {len(result)} clusters: {[c.cluster_id for c in result]}")
        return result
    
    def get_schemas_for_query(
        self,
        query: str,
        sqlite_store: Any = None
    ) -> str:
        """
        Get formatted schema string for relevant tables only.
        
        This is the main entry point for SQL generation.
        
        Args:
            query: User's natural language query
            sqlite_store: SQLite store to fetch full schemas
            
        Returns:
            Formatted schema string for LLM prompt
        """
        store = sqlite_store or self.sqlite_store
        if not store:
            return "-- No database connection available"
        
        # Get relevant clusters
        relevant_clusters = self.get_relevant_clusters(query)
        
        if not relevant_clusters:
            # Fallback: return limited schemas to prevent context overflow
            logger.info("No cluster match - falling back to limited schemas")
            return self._build_fallback_schema(store)
        
        # Collect table names from relevant clusters
        relevant_tables = set()
        for cluster in relevant_clusters:
            relevant_tables.update(cluster.table_names)
        
        # Build focused schema string
        return self._build_schema_string(relevant_tables, relevant_clusters, store)
    
    def _build_schema_string(
        self,
        table_names: set[str],
        clusters: list[SchemaCluster],
        store: Any
    ) -> str:
        """Build formatted schema string for selected tables."""
        parts = []
        
        # Header
        parts.append("""
-- ═══════════════════════════════════════════════════════════════════════════
-- RELEVANT DATABASE SCHEMA (Filtered by Query Context)
-- ═══════════════════════════════════════════════════════════════════════════
--
-- Tables below are selected based on your query.
-- Query them directly: SELECT column FROM table_name WHERE condition
--
-- ═══════════════════════════════════════════════════════════════════════════
""")
        
        # Show which clusters matched
        cluster_names = [c.display_name for c in clusters]
        parts.append(f"-- Matched contexts: {', '.join(cluster_names)}")
        parts.append(f"-- Tables available: {len(table_names)}")
        parts.append("")
        
        # FIX #3: Use cached table metadata to avoid DB call
        table_map = self._get_table_metadata_cache(store)
        
        for table_name in sorted(table_names):
            table_info = table_map.get(table_name)
            if not table_info:
                continue
                
            columns = table_info.get('columns', [])
            row_count = table_info.get('row_count', 0)
            
            parts.append(f"-- TABLE: {table_name} ({row_count} rows)")
            if columns:
                parts.append(f"--   Columns: {', '.join(columns[:15])}")
                if len(columns) > 15:
                    parts.append(f"--   ... and {len(columns) - 15} more columns")
            parts.append("")
        
        return "\n".join(parts)
    
    def _get_table_metadata_cache(self, store: Any) -> dict[str, dict]:
        """
        Get or build table metadata cache with TTL.
        
        Cache expires after CACHE_TTL_SECONDS to prevent stale data.
        """
        cache_age = time.time() - self._cache_timestamp
        cache_expired = cache_age > CACHE_TTL_SECONDS
        
        if not self._cache_valid or not self._table_metadata_cache or cache_expired:
            all_tables = store.list_spreadsheet_tables()
            self._table_metadata_cache = {t['table_name']: t for t in all_tables}
            self._cache_valid = True
            self._cache_timestamp = time.time()
            logger.debug(f"Refreshed table metadata cache: {len(self._table_metadata_cache)} tables")
        return self._table_metadata_cache
    
    def _build_fallback_schema(self, store: Any) -> str:
        """
        Build a limited schema string when no clusters match.
        
        Limits output to MAX_FALLBACK_TABLES tables to prevent context overflow.
        Prioritizes tables with more rows (likely more important data).
        """
        all_tables = store.list_spreadsheet_tables()
        
        if not all_tables:
            return "-- No tables available"
        
        total_tables = len(all_tables)
        
        # If within limit, use normal method
        if total_tables <= MAX_FALLBACK_TABLES:
            return store.get_schema_for_llm()
        
        # Sort by row_count (descending) to prioritize tables with more data
        sorted_tables = sorted(
            all_tables,
            key=lambda t: t.get('row_count', 0),
            reverse=True
        )[:MAX_FALLBACK_TABLES]
        
        logger.warning(
            f"Fallback schema truncated: showing {MAX_FALLBACK_TABLES} of {total_tables} tables. "
            f"Consider improving cluster matching."
        )
        
        parts = [f"""
-- ═══════════════════════════════════════════════════════════════════════════
-- DATABASE SCHEMA (FALLBACK MODE - LIMITED)
-- ═══════════════════════════════════════════════════════════════════════════
--
-- WARNING: No cluster matched your query. Showing top {MAX_FALLBACK_TABLES} tables
-- out of {total_tables} total. Results may be incomplete.
--
-- ═══════════════════════════════════════════════════════════════════════════
"""]
        
        for table in sorted_tables:
            table_name = table['table_name']
            columns = table.get('columns', [])
            row_count = table.get('row_count', 0)
            
            parts.append(f"-- TABLE: {table_name} ({row_count} rows)")
            if columns:
                parts.append(f"--   Columns: {', '.join(columns[:10])}")
                if len(columns) > 10:
                    parts.append(f"--   ... and {len(columns) - 10} more columns")
            parts.append("")
        
        return "\n".join(parts)
    
    def get_stats(self) -> dict:
        """Get clustering statistics including analytics."""
        # Calculate fallback rate
        fallback_rate = (
            (self.fallback_count / self.total_queries * 100) 
            if self.total_queries > 0 else 0.0
        )
        
        # Get top 5 most-hit clusters
        top_clusters = sorted(
            self.query_hits.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "total_clusters": len(self.clusters),
            "total_tables_indexed": len(self.table_to_cluster),
            "clusters_by_company": self._count_by_company(),
            "clusters_by_domain": self._count_by_domain(),
            # Analytics
            "total_queries": self.total_queries,
            "fallback_count": self.fallback_count,
            "fallback_rate_pct": round(fallback_rate, 1),
            "top_clusters": top_clusters,
            "cache_valid": self._cache_valid
        }
    
    def _count_by_company(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cluster in self.clusters.values():
            counts[cluster.company] = counts.get(cluster.company, 0) + len(cluster.table_names)
        return counts
    
    def _count_by_domain(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cluster in self.clusters.values():
            counts[cluster.domain_id] = counts.get(cluster.domain_id, 0) + len(cluster.table_names)
        return counts
