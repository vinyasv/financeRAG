# Product Requirements Document (PRD)

## FinanceRAG — Enterprise Financial Document Analysis with RAG

**Version:** 1.0
**Last Updated:** 2026-02-11
**Status:** Final

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [User Personas](#4-user-personas)
5. [System Architecture](#5-system-architecture)
6. [Functional Requirements](#6-functional-requirements)
   - 6.1 [Document Ingestion Pipeline](#61-document-ingestion-pipeline)
   - 6.2 [Storage Layer](#62-storage-layer)
   - 6.3 [Query Engine](#63-query-engine)
   - 6.4 [Tool System](#64-tool-system)
   - 6.5 [LLM Integration](#65-llm-integration)
   - 6.6 [Security Layer](#66-security-layer)
   - 6.7 [CLI Interface](#67-cli-interface)
   - 6.8 [Audit & Transparency](#68-audit--transparency)
7. [Data Models](#7-data-models)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Configuration & Environment](#9-configuration--environment)
10. [Testing Requirements](#10-testing-requirements)
11. [File & Module Inventory](#11-file--module-inventory)
12. [Build Phases](#12-build-phases)
13. [Risks & Mitigations](#13-risks--mitigations)
14. [Glossary](#14-glossary)

---

## 1. Executive Summary

FinanceRAG is an enterprise-grade Retrieval-Augmented Generation (RAG) system purpose-built for financial document analysis. It ingests PDFs, spreadsheets, and CSVs; extracts structured tables and unstructured text; stores them across dual backends (SQL + vector); and answers natural language questions with cited, auditable responses.

**Core value proposition:** "Ask questions about financial documents. Get cited, auditable answers."

**Key differentiators:**
- DAG-based parallel query execution
- Three-tier table extraction (VLM → Docling → Rule-based)
- Calculation audit trails with operand provenance
- Field comparability checking (GAAP vs non-GAAP, currency, standards)
- Graceful refusal when data is insufficient or incomparable
- Defense-in-depth security (SQL injection, prompt injection, path traversal)

---

## 2. Problem Statement

Financial analysts routinely spend hours manually searching through 10-K filings, earnings reports, and spreadsheets to answer questions like "How did NVIDIA's gross margin compare to AMD's in FY2024?" This requires:

1. Finding the right documents
2. Locating the relevant tables and passages
3. Extracting the correct numbers
4. Performing calculations
5. Verifying definitions match (e.g., GAAP vs non-GAAP)
6. Citing sources for audit

Existing RAG systems fail at financial analysis because they:
- Cannot reliably extract tables from PDFs
- Let LLMs hallucinate arithmetic
- Don't track where numbers come from
- Don't check whether metrics are actually comparable
- Lack structured data querying alongside semantic search

FinanceRAG solves all of these.

---

## 3. Goals & Success Metrics

### Primary Goals
| Goal | Metric | Target |
|------|--------|--------|
| Accurate answers | Factual correctness on evaluation suite | >90% |
| Cited responses | Every numeric claim has a source citation | 100% |
| Table extraction | Tables correctly parsed from financial PDFs | >85% |
| Arithmetic integrity | All calculations via audited tool, never LLM | 100% |
| Query latency | End-to-end for simple queries | <5 seconds |
| Query latency | End-to-end for complex multi-tool queries | <15 seconds |
| Ingestion throughput | 20-page PDF fully ingested | <60 seconds |

### Secondary Goals
| Goal | Metric | Target |
|------|--------|--------|
| Security | Pass all injection test suites | 100% |
| Graceful degradation | System works without LLM (heuristic mode) | Yes |
| Multi-format support | PDF, XLSX, XLS, CSV | All four |
| Refusal quality | Structured refusal for unanswerable queries | Yes |

---

## 4. User Personas

### Persona 1: Financial Analyst
- **Use case:** Compare revenue, margins, and growth across companies and periods
- **Needs:** Accurate numbers, source citations, correct metric definitions
- **Interaction:** CLI queries, batch exports (PDF/CSV/JSON)

### Persona 2: Compliance Officer
- **Use case:** Verify disclosures, cross-reference filings
- **Needs:** Audit trail, calculation transparency, document provenance
- **Interaction:** Interactive REPL mode with follow-up questions

### Persona 3: Investment Researcher
- **Use case:** Screen companies, extract KPIs from filings
- **Needs:** Batch ingestion, multi-document queries, data export
- **Interaction:** Folder ingestion, programmatic JSON output

### Persona 4: Developer/Integrator
- **Use case:** Extend the tool system, add new document types, integrate into pipelines
- **Needs:** Clean abstractions, pluggable architecture, comprehensive tests
- **Interaction:** Python API, tool base class

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                           │
│              (scripts/query.py, scripts/ingest.py)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                       RAG Agent                                 │
│                    (src/rag_agent.py)                            │
│              Main orchestrator for all operations               │
└───────┬──────────────────────────────────┬──────────────────────┘
        │                                  │
        ▼                                  ▼
┌───────────────────┐          ┌──────────────────────────────────┐
│  Ingestion        │          │  Query Engine                    │
│  Pipeline         │          │  ┌────────┐ ┌────────┐ ┌──────┐ │
│                   │          │  │Planner │→│Executor│→│Synth.│ │
│  ┌─────────────┐  │          │  └────────┘ └────────┘ └──────┘ │
│  │ PDF Parser  │  │          │       │          │               │
│  │ VLM Extract │  │          │       ▼          ▼               │
│  │ Docling     │  │          │  ┌─────────────────────────────┐ │
│  │ Rule-based  │  │          │  │      Tool System            │ │
│  │ Spreadsheet │  │          │  │  SQL │ Vector │ Calc │ Doc  │ │
│  │ Chunker     │  │          │  └─────────────────────────────┘ │
│  │ Schema Det. │  │          └──────────────┬───────────────────┘
│  │ Temporal Ex.│  │                         │
│  └─────────────┘  │                         │
└───────┬───────────┘                         │
        │                                     │
        ▼                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Storage Layer                             │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  ┌────────┐ │
│  │  SQLite     │  │  ChromaDB   │  │ Document Store│  │Schema  │ │
│  │ (Structured)│  │  (Vectors)  │  │ (Full content)│  │Cluster │ │
│  └────────────┘  └────────────┘  └───────────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Cross-Cutting Concerns                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Security │  │  Config  │  │  Models  │  │  LLM Client    │  │
│  │          │  │          │  │ (Pydantic)│  │ (Multi-provider│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow — Ingestion

```
Input File (PDF / XLSX / CSV)
  │
  ├─[PDF Path]─────────────────────────────────────────────────┐
  │   1. pdfplumber extracts raw text + per-page content       │
  │   2. Table extraction (3-tier strategy):                   │
  │      a. VLM Extractor (OpenRouter Gemini/GPT-4o vision)    │
  │      b. Docling (local IBM TableFormer)                    │
  │      c. Rule-based regex fallback                          │
  │   3. Schema Detector enhances table schemas via LLM        │
  │   4. Temporal Extractor pulls fiscal year/quarter           │
  │   5. Chunker splits text into semantic chunks              │
  │   6. Store:                                                │
  │      - Tables → SQLite (structured rows)                   │
  │      - Chunks → ChromaDB (vector embeddings)               │
  │      - Full text → DocumentStore (content + metadata)      │
  │      - Schema Cluster assigns company + domain             │
  │                                                            │
  ├─[Spreadsheet Path]────────────────────────────────────────┐│
  │   1. pandas reads all sheets                              ││
  │   2. Each sheet becomes a document                        ││
  │   3. Store:                                               ││
  │      - Native SQL table (direct column queries)           ││
  │      - Text chunk (for vector search)                     ││
  │      - DocumentStore entry                                ││
  │      - Schema Cluster assignment                          ││
  └───────────────────────────────────────────────────────────┘│
  └────────────────────────────────────────────────────────────┘
```

### 5.3 Data Flow — Query

```
Natural Language Query
  │
  ▼
┌──────────────────┐
│ Input Validation │  ← Security: length check, injection scan
└────────┬─────────┘
         ▼
┌──────────────────┐
│     Planner      │  ← Decomposes query into DAG of tool calls
│                  │     Uses LLM for complex queries, heuristics for simple
│  Outputs:        │
│  ExecutionPlan   │     [{tool, input, depends_on}, ...]
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Executor      │  ← Groups steps into parallel layers
│                  │     Resolves {step_id.field} references
│  Layer 1: ────── │     Independent steps run concurrently
│  Layer 2: ────── │     Dependent steps wait for predecessors
│  Layer N: ────── │
└────────┬─────────┘
         ▼
┌──────────────────┐
│   Synthesizer    │  ← Combines all tool results
│                  │     Extracts citations (doc, page, line)
│                  │     Formats final response
│  Output:         │
│  QueryResponse   │     {answer, citations, metadata, timing}
└──────────────────┘
```

---

## 6. Functional Requirements

### 6.1 Document Ingestion Pipeline

#### FR-ING-01: PDF Parsing
- **Module:** `src/ingestion/pdf_parser.py`
- **Description:** Extract text and metadata from PDF files using pdfplumber.
- **Inputs:** PDF file path
- **Outputs:** `List[TextChunk]`, document metadata (page count, title)
- **Behavior:**
  - Extract full text per page
  - Preserve page numbers and approximate line ranges
  - Handle multi-column layouts gracefully
  - Extract document title from first page or filename
- **Dependencies:** `pdfplumber>=0.10.0`

#### FR-ING-02: VLM Table Extraction (Primary)
- **Module:** `src/ingestion/vlm_extractor.py`
- **Description:** Use vision-language models to extract tables from PDF pages rendered as images.
- **Inputs:** PDF file path, LLM client
- **Outputs:** `List[ExtractedTable]`
- **Behavior:**
  - Render each PDF page as an image using PyMuPDF (fitz)
  - Send images to a vision-capable LLM via OpenRouter
  - Prompt the model to return structured JSON with headers and rows
  - Parse and validate the response
  - Process all pages in a single batch call where possible
- **Dependencies:** `PyMuPDF>=1.23.0`, LLM client with vision support
- **Fallback:** If VLM fails or no API key, fall through to Docling

#### FR-ING-03: Docling Table Extraction (Secondary)
- **Module:** `src/ingestion/vision_table_extractor.py`
- **Description:** Use IBM's Docling library with TableFormer for local, free table extraction.
- **Inputs:** PDF file path
- **Outputs:** `List[ExtractedTable]`
- **Behavior:**
  - Use Docling's `DocumentConverter` with `TableFormerMode.ACCURATE`
  - Extract tables with headers and data rows
  - Associate each table with its page number
  - Runs entirely locally — no API calls
- **Dependencies:** `docling>=2.66.0`
- **Fallback:** If Docling fails, fall through to rule-based

#### FR-ING-04: Rule-Based Table Extraction (Fallback)
- **Module:** `src/ingestion/table_extractor.py`
- **Description:** Regex-based heuristic table detection as a last resort.
- **Inputs:** Raw text per page
- **Outputs:** `List[ExtractedTable]`
- **Behavior:**
  - Detect tabular patterns using regex (aligned whitespace, delimiters)
  - Parse headers and data rows
  - Basic but always available — no external dependencies
- **Dependencies:** None (stdlib only)

#### FR-ING-05: Schema Detection & Enhancement
- **Module:** `src/ingestion/schema_detector.py`
- **Description:** Use LLM to enhance extracted table schemas with semantic column names, data types, and descriptions.
- **Inputs:** `ExtractedTable` with raw headers/rows, LLM client
- **Outputs:** Enhanced `ExtractedTable` with improved column metadata
- **Behavior:**
  - Send table sample (headers + first few rows) to LLM
  - Request: clean column names, inferred data types, unit detection
  - Concurrency-controlled with semaphore (max 10 concurrent LLM calls)
  - Graceful fallback: use raw headers if LLM unavailable
- **Dependencies:** LLM client

#### FR-ING-06: Temporal Metadata Extraction
- **Module:** `src/ingestion/temporal_extractor.py`
- **Description:** Extract fiscal year, quarter, and period information from filenames and document content.
- **Inputs:** Filename string, optional document text
- **Outputs:** `TemporalMetadata` (fiscal_year, quarter, period_type)
- **Behavior:**
  - Match patterns: "10-K 2024", "Q3 2024", "FY2024", "2024-Q1"
  - Filename patterns take priority over content patterns
  - Store as document metadata for temporal reasoning
- **Dependencies:** None (regex-based)

#### FR-ING-07: Semantic Text Chunking
- **Module:** `src/ingestion/chunker.py`
- **Description:** Split extracted text into overlapping semantic chunks suitable for embedding.
- **Inputs:** Full document text, page boundaries
- **Outputs:** `List[TextChunk]`
- **Configuration:**
  - `chunk_size`: 500 tokens (~385 words)
  - `chunk_overlap`: 50 tokens
  - `min_chunk_size`: 50 tokens
- **Behavior:**
  - Split on paragraph boundaries (double newlines)
  - Maintain overlap between adjacent chunks for context continuity
  - Preserve metadata: page number, section title, line range (start_line, end_line)
  - Skip chunks below minimum size threshold
- **Dependencies:** None (token estimation via word count heuristic)

#### FR-ING-08: Spreadsheet Parsing
- **Module:** `src/ingestion/spreadsheet_parser.py`
- **Description:** Parse Excel and CSV files into structured tables.
- **Inputs:** File path (.xlsx, .xls, .csv)
- **Outputs:** `List[ExtractedTable]` (one per sheet)
- **Behavior:**
  - Excel: read all sheets via pandas + openpyxl
  - CSV: read as single sheet
  - Each sheet produces:
    - A native SQL table (for direct column queries)
    - A text chunk (for vector search)
  - Track metadata: row count, column count, header names, sheet name
- **Dependencies:** `pandas`, `openpyxl`

#### FR-ING-09: Ingestion Deduplication
- **Description:** Prevent duplicate ingestion of the same document.
- **Behavior:**
  - Check DocumentStore for existing document by filename
  - Skip ingestion if document already exists
  - Log a warning when duplicate detected

#### FR-ING-10: File Validation
- **Description:** Validate files before ingestion.
- **Rules:**
  - Maximum file size: 500 MB
  - Supported extensions: `.pdf`, `.xlsx`, `.xls`, `.csv`
  - File must exist and be readable
  - Reject empty files

---

### 6.2 Storage Layer

#### FR-STO-01: SQLite Store (Structured Data)
- **Module:** `src/storage/sqlite_store.py`
- **Description:** Store and query structured tabular data extracted from documents.
- **Schema:**
  ```sql
  -- Core metadata tables
  CREATE TABLE documents (
      id TEXT PRIMARY KEY,
      filename TEXT,
      file_type TEXT,
      page_count INTEGER,
      ingested_at TIMESTAMP,
      metadata TEXT  -- JSON
  );

  CREATE TABLE extracted_tables (
      id TEXT PRIMARY KEY,
      document_id TEXT REFERENCES documents(id),
      table_name TEXT,
      page_number INTEGER,
      headers TEXT,  -- JSON array
      row_count INTEGER,
      schema_info TEXT  -- JSON
  );

  -- EAV-style table data
  CREATE TABLE table_data (
      table_id TEXT REFERENCES extracted_tables(id),
      row_index INTEGER,
      column_name TEXT,
      value TEXT
  );

  -- Dynamic tables: one per extracted/spreadsheet table
  -- e.g., CREATE TABLE "nvidia_revenue_2024" (...)
  ```
- **Operations:**
  - `store_table(table_id, headers, rows)` — Create dynamic table
  - `execute_query(sql)` — Run validated SELECT queries
  - `get_table_schemas()` — Return all table schemas for LLM context
  - `get_document_tables(doc_id)` — Tables for a specific document
- **Security:** See FR-SEC-02 (SQL injection prevention)
- **Dependencies:** `sqlite3` (stdlib)

#### FR-STO-02: ChromaDB Store (Vector Embeddings)
- **Module:** `src/storage/chroma_store.py`
- **Description:** Store and search document text chunks via vector similarity.
- **Configuration:**
  - Collection name: `"documents"` (configurable)
  - Distance metric: cosine similarity
  - HNSW parameters tuned for large collections
- **Operations:**
  - `add_chunks(chunks, embeddings)` — Batch insert with metadata
  - `search(query_embedding, top_k, filters)` — Similarity search
  - `delete_document(doc_id)` — Remove all chunks for a document
- **Embedding Functions:**
  - **Local:** sentence-transformers (`BAAI/bge-small-en-v1.5`, 133MB model)
  - **Remote:** OpenRouter (`qwen/qwen3-embedding-8b`)
  - **Auto-detection:** Use remote if API key available, else local
- **Dependencies:** `chromadb>=0.4.0`, `sentence-transformers>=2.2.0`

#### FR-STO-03: Document Store (Full Content)
- **Module:** `src/storage/document_store.py`
- **Description:** Store original file references, full extracted text, and metadata.
- **Directory Structure:**
  ```
  data/documents/
  ├── .metadata/{doc_id}.json    # Document metadata (JSON)
  ├── .content/{doc_id}.txt      # Full extracted text
  └── {original_filename}        # Original file (optional copy)
  ```
- **Operations:**
  - `store_document(doc_id, filename, content, metadata)`
  - `get_document(doc_id)` → content + metadata
  - `get_document_by_name(filename)` → lookup by original name
  - `list_documents()` → all documents with metadata summaries
  - `document_exists(filename)` → deduplication check
- **Security:** Path traversal prevention (no `..`, no null bytes)

#### FR-STO-04: Schema Cluster Manager
- **Module:** `src/storage/schema_cluster.py`
- **Description:** Organize table schemas hierarchically by company and domain for scalable LLM context.
- **Hierarchy:** Company → Domain → Table
- **Domains:**
  - Financial Statements
  - Stock Market
  - Performance Metrics
  - Segments
  - Quarterly
  - General
- **Operations:**
  - `assign_table(table_name, company, domain)` — Assign table to cluster
  - `get_relevant_schemas(query)` — Return only schemas relevant to a query
  - `detect_company(filename)` — Extract company name from filename via LLM
  - `classify_domain(table_name, headers)` — Determine domain from table content
- **Benefits:**
  - Entity isolation (NVIDIA tables separate from AMD tables)
  - Reduced LLM prompt size (only relevant schemas included)
  - Semantic grouping (related tables discovered together)

---

### 6.3 Query Engine

#### FR-QRY-01: Query Planner
- **Module:** `src/agent/planner.py`
- **Description:** Decompose natural language queries into a DAG (Directed Acyclic Graph) of tool calls.
- **Inputs:** Query string, available tools, table schemas
- **Outputs:** `ExecutionPlan` containing ordered `ToolCall` steps
- **Behavior:**
  - **Complexity Detection:** Classify query as simple or complex
    - Complex triggers: calculations, comparisons, multi-company, time-series, aggregations
    - Simple: direct lookups, single-entity questions
  - **LLM Planning (complex queries):**
    - Send query + tool descriptions + available schemas to LLM
    - Parse structured response into `ExecutionPlan`
    - Validate: no circular dependencies, all tool names valid, all references valid
  - **Heuristic Planning (simple queries or no LLM):**
    - Default to single `vector_search` step
    - Add `sql_query` step if structured data keywords detected
  - **Company Registry:** Dynamically detect company names mentioned in query
    - Used to scope schema context to relevant companies
- **Planning Prompt Requirements:**
  - 100+ line financial analyst system prompt
  - Emphasis on parallel execution of independent steps
  - Tool descriptions with usage examples
  - Arithmetic prohibition (must use calculator tool)
  - Security disclaimer for untrusted input handling

#### FR-QRY-02: DAG Executor
- **Module:** `src/agent/executor.py`
- **Description:** Execute the planned DAG with maximum parallelism.
- **Inputs:** `ExecutionPlan`, tool registry, previous step results
- **Outputs:** `Dict[step_id, ToolResult]`
- **Behavior:**
  - **Layer Detection:** Group steps into parallelizable layers
    - Layer 1: all steps with no dependencies
    - Layer 2: steps whose dependencies are all in Layer 1
    - Layer N: steps whose dependencies are all in Layers 1..N-1
  - **Parallel Execution:** Execute all steps within a layer concurrently via `asyncio.gather`
  - **Reference Resolution:**
    - Pattern: `{step_id}` or `{step_id.field.subfield}`
    - Navigate dicts and object attributes
    - Substitute resolved values into tool inputs
    - Fallback to `"0"` if reference cannot be resolved
  - **Error Handling:**
    - Individual step failures don't abort the entire plan
    - Failed steps recorded with error messages
    - Dependent steps receive error context
  - **Timing:** Record execution time for each step

#### FR-QRY-03: Response Synthesizer
- **Module:** `src/agent/synthesizer.py`
- **Description:** Combine tool results into a coherent, cited response.
- **Inputs:** All `ToolResult` objects, original query
- **Outputs:** `QueryResponse` with answer, citations, metadata
- **Behavior:**
  - **LLM Synthesis:**
    - Format all tool results into context
    - Send to LLM with financial analyst synthesis prompt
    - Prompt enforces: no LLM arithmetic, cite sources, use exact numbers
  - **Template Synthesis (no LLM):**
    - Concatenate tool results with headers
    - Basic formatting without natural language generation
  - **Citation Extraction:**
    - Pull `document_id`, `page_number`, `section_title`, `snippet` from tool results
    - Resolve document IDs to human-readable filenames (cached)
    - Deduplicate citations
  - **Refusal Handling:**
    - If any tool returns a `QueryRefusal`, surface it as the primary response
    - Include refusal reason and suggestions
- **Synthesis Prompt Requirements:**
  - Financial analyst persona
  - CRITICAL: arithmetic prohibition ("never compute, always cite calculator results")
  - Citation format specification
  - Professional standards (GAAP/non-GAAP labeling, units, periods)
  - 80+ lines of guidelines

---

### 6.4 Tool System

#### FR-TOOL-00: Tool Base Class
- **Module:** `src/tools/base.py`
- **Description:** Abstract base class for all pluggable tools.
- **Interface:**
  ```python
  class Tool(ABC):
      name: ToolName
      description: str

      @abstractmethod
      async def execute(self, input: dict, context: dict) -> ToolResult:
          """Execute the tool with given input and return results."""
          pass

      def get_schema(self) -> dict:
          """Return JSON schema describing tool inputs."""
          pass
  ```
- **Contract:**
  - Every tool returns a `ToolResult` (success or failure)
  - Tools are stateless — all state in storage layer
  - Tools receive execution context (previous results, config)

#### FR-TOOL-01: SQL Query Tool
- **Module:** `src/tools/sql_query.py`
- **Enum Name:** `ToolName.SQL_QUERY`
- **Description:** Convert natural language to SQL and execute against structured data.
- **Input:** `{"query": "What was NVIDIA's revenue in 2024?"}`
- **Process:**
  1. Retrieve relevant table schemas (via SchemaClusterManager or all schemas)
  2. Send NL query + schemas to LLM → receive SQL
  3. Validate SQL (security checks — see FR-SEC-02)
  4. Execute against SQLite
  5. Format results as `SQLQueryResult`
- **Output:** `ToolResult` containing rows, column names, row count
- **Heuristic Mode (no LLM):**
  - Attempt simple pattern matching for basic queries
  - Fall back to returning schema information
- **Context:** Uses SchemaClusterManager to limit schemas to relevant companies/domains

#### FR-TOOL-02: Vector Search Tool
- **Module:** `src/tools/vector_search.py`
- **Enum Name:** `ToolName.VECTOR_SEARCH`
- **Description:** Semantic similarity search over document text chunks.
- **Input:** `{"query": "NVIDIA gross margin discussion", "top_k": 5}`
- **Process:**
  1. Embed the query text
  2. Retrieve `3 * top_k` candidates from ChromaDB (over-fetch for reranking)
  3. Optionally rerank with cross-encoder
  4. Filter by relevance score threshold (0.2–0.3)
  5. Return top-k chunks with metadata
- **Output:** `ToolResult` containing list of chunks, each with:
  - `content`: chunk text
  - `document_id`: source document
  - `page_number`: page in original document
  - `section_title`: detected section heading
  - `start_line`, `end_line`: line range
  - `score`: relevance score
- **Reranking:**
  - Model: FlashRank (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - Lazy-loaded on first use
  - Configurable on/off
  - ~25% precision improvement

#### FR-TOOL-02a: Reranker
- **Module:** `src/tools/reranker.py`
- **Description:** Cross-encoder reranking for improved retrieval precision.
- **Input:** Query string, list of candidate chunks
- **Output:** Reranked list with updated scores
- **Behavior:**
  - Lazy initialization (model loaded on first call)
  - Score each (query, chunk) pair with cross-encoder
  - Sort by cross-encoder score descending
- **Dependencies:** `flashrank>=0.2.0`

#### FR-TOOL-03: Calculator Tool
- **Module:** `src/tools/calculator.py`
- **Enum Name:** `ToolName.CALCULATOR`
- **Description:** Safe arithmetic evaluation with full audit transparency.
- **Input:** `{"expression": "{step_1.revenue} - {step_2.revenue}", "description": "Revenue difference"}`
- **Process:**
  1. Resolve `{step_id.field}` references from execution context
  2. Record each binding: reference → resolved value → source document
  3. Parse resolved expression into AST
  4. Evaluate using safe AST walker (no `eval()`)
  5. Build `CalculationTranscript`
- **Output:** `ToolResult` containing `CalculationTranscript`:
  ```json
  {
    "original_expression": "{step_1.revenue} - {step_2.revenue}",
    "bindings": [
      {"reference": "{step_1.revenue}", "resolved_value": 145600000000, "source": "nvidia_10k.pdf p.42"},
      {"reference": "{step_2.revenue}", "resolved_value": 128695000000, "source": "amd_10k.pdf p.38"}
    ],
    "resolved_expression": "145600000000 - 128695000000",
    "result": 16905000000,
    "formula_description": "Revenue difference"
  }
  ```
- **Supported Operators:** `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Security:**
  - AST-based evaluation — never uses `eval()` or `exec()`
  - AST depth limit: 50 levels (DoS protection)
  - Only numeric literals and operators allowed
- **Error Cases:**
  - Division by zero → error result
  - Unresolvable reference → fallback to `0` with warning
  - `ComparabilityError` → converts to `QueryRefusal`

#### FR-TOOL-04: Get Document Tool
- **Module:** `src/tools/get_document.py`
- **Enum Name:** `ToolName.GET_DOCUMENT`
- **Description:** Retrieve full document content by ID.
- **Input:** `{"document_id": "doc_abc123"}`
- **Output:** `ToolResult` containing full text content + metadata
- **Use case:** When synthesis needs more context than chunks provide

#### FR-TOOL-05: Comparability Checker
- **Module:** `src/tools/comparability.py`
- **Description:** Check whether two financial fields are semantically comparable.
- **Inputs:** Two `FieldDefinition` objects
- **Output:** `ComparabilityResult` with confidence score and warnings
- **Checks:**
  - Accounting standard match (GAAP vs non-GAAP vs IFRS)
  - Currency match
  - Segment scope match (company-wide vs segment-level)
  - Definition semantic similarity
- **Confidence:** 0.0–1.0 score
- **Behavior:**
  - `confidence >= 0.8` → comparable
  - `0.5 <= confidence < 0.8` → comparable with warnings
  - `confidence < 0.5` → not comparable → triggers `QueryRefusal`

---

### 6.5 LLM Integration

#### FR-LLM-01: Multi-Provider Client
- **Module:** `src/llm_client.py`
- **Description:** Unified LLM client abstracting over multiple providers.
- **Providers (in priority order):**
  1. **OpenRouter** — Single API key for all models (recommended)
  2. **OpenAI** — Direct API access
  3. **Anthropic** — Direct API access
- **Auto-Detection:** Try providers in order; use first one with valid API key
- **Interface:**
  ```python
  class LLMClient:
      async def generate(self, prompt: str, system: str = None, model: str = None) -> str
      async def generate_with_image(self, prompt: str, image_b64: str, model: str = None) -> str
      def get_available_models(self) -> List[str]
  ```

#### FR-LLM-02: Model Registry
- **Description:** Named aliases for commonly used models.
- **Tiers:**

  | Tier | Alias | Model ID |
  |------|-------|----------|
  | Fast | `gemini-flash` | `google/gemini-2.0-flash-001` |
  | Fast | `gpt-4o-mini` | `openai/gpt-4o-mini` |
  | Fast | `claude-haiku` | `anthropic/claude-3-haiku` |
  | Fast | `llama-8b` | `meta-llama/llama-3.1-8b-instruct` |
  | Balanced | `gpt-4o` | `openai/gpt-4o` |
  | Balanced | `claude-sonnet` | `anthropic/claude-3.5-sonnet` |
  | Balanced | `gemini-pro` | `google/gemini-pro-1.5` |
  | Balanced | `llama-70b` | `meta-llama/llama-3.1-70b-instruct` |
  | Best | `claude-opus` | `anthropic/claude-3-opus` |
  | Best | `gpt-4-turbo` | `openai/gpt-4-turbo` |
  | Best | `llama-405b` | `meta-llama/llama-3.1-405b-instruct` |
  | Free | `free` | `meta-llama/llama-3.1-8b-instruct:free` |

- **Default Model:** `google/gemini-3-flash-preview`

#### FR-LLM-03: Embedding Provider
- **Module:** `src/embeddings.py`
- **Description:** Manage embedding generation for vector search.
- **Providers:**
  - **Local:** `sentence-transformers` with `BAAI/bge-small-en-v1.5` (133MB)
  - **Remote:** OpenRouter with `qwen/qwen3-embedding-8b`
- **Auto-Detection:** Use remote if OpenRouter API key is available; otherwise use local
- **Interface:**
  ```python
  class EmbeddingProvider:
      def embed(self, texts: List[str]) -> List[List[float]]
      def embed_query(self, query: str) -> List[float]
  ```

---

### 6.6 Security Layer

#### FR-SEC-01: Prompt Injection Defense
- **Module:** `src/security.py`
- **Description:** Detect and neutralize prompt injection attempts in user input.
- **Detection Patterns (13 regex patterns):**
  - `"ignore previous instructions"`
  - `"disregard above"`
  - `"you are now a"`
  - `"your new role is"`
  - `"act as if"`
  - `"reveal your system prompt"`
  - `"forget everything"`
  - `"override your instructions"`
  - And 5 additional patterns
- **Mitigation:**
  1. **Sanitization:** Remove control characters, null bytes
  2. **Escaping:** Escape `{` and `}` format string markers
  3. **Wrapping:** Enclose user input in clear delimiters (`<user_query>...</user_query>`)
  4. **Logging:** Log suspicious patterns with severity level
- **Behavior:**
  - Detection does NOT block the query (to avoid false positive censorship)
  - Detected patterns are logged and the input is sanitized before use

#### FR-SEC-02: SQL Injection Prevention
- **Module:** `src/storage/sqlite_store.py`
- **Description:** Prevent destructive SQL from reaching the database.
- **Forbidden Keywords (word-boundary matched):**
  - Modification: `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`
  - Destructive: `TRUNCATE`, `EXEC`, `EXECUTE`, `GRANT`, `REVOKE`
  - Schema: `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`
  - Merge: `REPLACE`, `MERGE`
- **Validation Function:** `validate_sql_query(sql: str) -> bool`
  - Must start with `SELECT` (case-insensitive)
  - Must not contain forbidden keywords (word-boundary regex)
  - Must not contain multiple statements (no `;` followed by more SQL)
  - Must not contain SQL comments (`--`, `/* */`)
- **Automatic Safeguards:**
  - Append `LIMIT 10000` to all queries without explicit LIMIT
  - Read-only database connection mode where supported

#### FR-SEC-03: Path Traversal Protection
- **Module:** `src/storage/document_store.py`
- **Description:** Prevent directory traversal attacks in document storage.
- **Rules:**
  - Reject paths containing `..`
  - Reject paths containing null bytes (`\x00`)
  - Resolve all paths to absolute paths within the data directory
  - Verify resolved path is under the allowed base directory

#### FR-SEC-04: Input Validation
- **Description:** Validate all external inputs.
- **Rules:**
  - Query length: maximum 5,000 characters
  - File size: maximum 500 MB
  - File type: must be in allowed extensions list
  - API keys: accessed via properties (not stored in object fields)

---

### 6.7 CLI Interface

#### FR-CLI-01: Query Script
- **Module:** `scripts/query.py`
- **Description:** Interactive and single-shot query interface.
- **Usage:**
  ```bash
  # Interactive REPL
  python scripts/query.py

  # Single query
  python scripts/query.py "What was NVIDIA's revenue in FY2024?"

  # With model selection
  python scripts/query.py -m claude-sonnet "Complex query here"

  # Export to file
  python scripts/query.py -o report.pdf "Query"
  python scripts/query.py -o results.csv "Query"
  python scripts/query.py -o data.json "Query"

  # List available models
  python scripts/query.py --list-models
  ```
- **Features:**
  - Interactive REPL with prompt and history
  - Single-shot mode with exit code
  - Model selection via `-m`/`--model` flag
  - Output export: PDF (via fpdf2), CSV, JSON via `-o`/`--output` flag
  - Execution timing display
  - Citation display with document names, pages, sections
  - Progress indicators via Rich
  - Graceful error handling

#### FR-CLI-02: Ingest Script
- **Module:** `scripts/ingest.py`
- **Description:** Document ingestion interface.
- **Usage:**
  ```bash
  # Single file
  python scripts/ingest.py report.pdf

  # Multiple files
  python scripts/ingest.py doc1.pdf data.xlsx sheet.csv

  # Folder ingestion
  python scripts/ingest.py -f ./reports

  # Pattern matching within folder
  python scripts/ingest.py -f ./data --pattern "*.pdf"
  ```
- **Features:**
  - Single or multiple file arguments
  - Recursive folder ingestion via `-f`/`--folder`
  - Glob pattern filtering via `--pattern`
  - File validation (size, type, existence)
  - Progress tracking with Rich panels
  - Per-file success/failure reporting
  - Summary statistics on completion

#### FR-CLI-03: Terminal UI
- **Module:** `src/ui/console.py`
- **Description:** Rich-based terminal UI components.
- **Components:**
  - Themed console (custom color palette)
  - Section panels with borders
  - Progress bars with spinners
  - Result tables
  - Status messages (info, success, warning, error)
  - Citation formatting
- **Dependencies:** `rich>=13.0.0`

---

### 6.8 Audit & Transparency

#### FR-AUD-01: Calculation Transcripts
- **Description:** Every arithmetic operation produces a full audit trail.
- **Content:**
  - Original expression with symbolic references
  - Each operand binding: reference → resolved value → source document/page
  - Final resolved numeric expression
  - Computed result
  - Human-readable formula description
- **Enforcement:** LLM is explicitly instructed to never perform arithmetic — all computation goes through the Calculator tool.

#### FR-AUD-02: Citation Tracking
- **Description:** Every answer includes source citations.
- **Citation Fields:**
  - `document_name`: Human-readable filename
  - `document_id`: Internal unique ID
  - `page_number`: Page in original document
  - `section_title`: Detected section heading
  - `snippet`: Relevant text excerpt
  - `start_line`, `end_line`: Line range in document
- **Behavior:**
  - Citations extracted from tool results during synthesis
  - Document IDs resolved to filenames (with caching)
  - Deduplicated before display

#### FR-AUD-03: Graceful Refusal
- **Description:** When data is insufficient or incomparable, return structured refusal instead of hallucinating.
- **Model:** `QueryRefusal`
  ```python
  class QueryRefusal:
      reason: RefusalReason  # Enum
      explanation: str       # Human-readable explanation
      suggestions: List[str] # What the user could do instead
      partial_data: dict     # Any data that WAS found
  ```
- **Refusal Reasons (Enum):**
  - `DEFINITION_MISMATCH` — Metrics not semantically comparable
  - `INSUFFICIENT_DATA` — Required data not found in knowledge base
  - `PERIOD_DISCONTINUITY` — Time periods don't align
  - `INCOMPARABLE_METRICS` — Different accounting standards or scopes
  - `MISSING_CONTEXT` — Need more information to answer
- **Behavior:**
  - Refusal is treated as a success mode (not an error)
  - Calculator tool raises `ComparabilityError` → converted to `QueryRefusal`
  - Synthesizer surfaces refusals prominently in response

#### FR-AUD-04: Field Comparability
- **Description:** Before comparing metrics across companies or periods, verify they are semantically comparable.
- **Checks:**
  - Accounting standard: GAAP vs non-GAAP vs IFRS
  - Currency: USD vs EUR vs other
  - Segment scope: Company-wide vs business segment
  - Metric definition: Semantic similarity of field descriptions
- **Output:** `ComparabilityResult` with confidence score (0.0–1.0) and warnings

---

## 7. Data Models

### 7.1 Core Models (`src/models.py`)

All models use Pydantic v2 for runtime validation and serialization.

#### Execution Models
```python
class ExecutionPlan(BaseModel):
    steps: List[ToolCall]
    metadata: dict = {}

class ToolCall(BaseModel):
    step_id: str                    # Unique step identifier
    tool: ToolName                  # Enum: SQL_QUERY, VECTOR_SEARCH, CALCULATOR, GET_DOCUMENT
    input: dict                     # Tool-specific input parameters
    depends_on: List[str] = []      # Step IDs this depends on
    description: str = ""           # Human-readable step description

class ToolResult(BaseModel):
    step_id: str
    tool: ToolName
    success: bool
    result: Any = None              # Tool-specific output
    error: str = None
    execution_time_ms: float = 0
    citations: List[Citation] = []

class ToolName(str, Enum):
    SQL_QUERY = "sql_query"
    VECTOR_SEARCH = "vector_search"
    CALCULATOR = "calculator"
    GET_DOCUMENT = "get_document"
```

#### Response Models
```python
class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    execution_plan: ExecutionPlan
    tool_results: List[ToolResult]
    total_time_ms: float
    model_used: str = ""
    refusal: QueryRefusal = None

class Citation(BaseModel):
    document_id: str
    document_name: str = ""
    page_number: int = None
    section_title: str = ""
    snippet: str = ""
    start_line: int = None
    end_line: int = None
```

#### Document Models
```python
class Document(BaseModel):
    id: str
    filename: str
    file_type: str
    page_count: int = 0
    ingested_at: datetime
    metadata: dict = {}

class TextChunk(BaseModel):
    content: str
    document_id: str
    page_number: int = None
    section_title: str = ""
    start_line: int = None
    end_line: int = None
    metadata: dict = {}

class ExtractedTable(BaseModel):
    id: str
    document_id: str
    table_name: str
    page_number: int = None
    headers: List[str]
    rows: List[List[str]]
    row_count: int = 0
    schema_info: dict = {}
```

#### Calculation Models
```python
class CalculationTranscript(BaseModel):
    original_expression: str
    bindings: List[OperandBinding]
    resolved_expression: str
    result: float
    formula_description: str = ""

class OperandBinding(BaseModel):
    reference: str           # e.g., "{step_1.revenue}"
    resolved_value: float
    source: str              # e.g., "nvidia_10k.pdf p.42"

class QueryRefusal(BaseModel):
    reason: RefusalReason
    explanation: str
    suggestions: List[str] = []
    partial_data: dict = {}

class RefusalReason(str, Enum):
    DEFINITION_MISMATCH = "definition_mismatch"
    INSUFFICIENT_DATA = "insufficient_data"
    PERIOD_DISCONTINUITY = "period_discontinuity"
    INCOMPARABLE_METRICS = "incomparable_metrics"
    MISSING_CONTEXT = "missing_context"
```

#### Comparability Models
```python
class FieldDefinition(BaseModel):
    name: str
    description: str = ""
    accounting_standard: AccountingStandard = AccountingStandard.UNKNOWN
    currency: str = "USD"
    segment: str = "company-wide"

class ComparabilityResult(BaseModel):
    comparable: bool
    confidence: float           # 0.0 - 1.0
    warnings: List[str] = []
    details: dict = {}

class AccountingStandard(str, Enum):
    GAAP = "gaap"
    NON_GAAP = "non_gaap"
    IFRS = "ifrs"
    UNKNOWN = "unknown"
```

#### SQL Models
```python
class SQLQueryResult(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    query_executed: str
```

---

## 8. Non-Functional Requirements

### NFR-01: Performance
| Metric | Target |
|--------|--------|
| Simple query latency (single vector search) | <5 seconds |
| Complex query latency (multi-tool, parallel) | <15 seconds |
| SQL query execution (after SQL generation) | <100 ms |
| Vector search (with optional reranking) | 50–200 ms |
| PDF ingestion (20 pages) | <60 seconds |
| Spreadsheet ingestion | <15 seconds per sheet |

### NFR-02: Scalability
- Handle knowledge bases with 100+ documents
- Schema clustering reduces LLM context linearly with document count
- ChromaDB HNSW parameters tuned for large collections
- Batch embedding generation for ingestion efficiency
- Concurrent table processing with semaphore (10 concurrent)

### NFR-03: Reliability
- Graceful degradation: system works without LLM (heuristic mode)
- Three-tier table extraction: always a fallback available
- Individual tool failures don't abort entire query
- Automatic retry not implemented (fail fast, inform user)

### NFR-04: Security
- Defense-in-depth across all input boundaries
- No `eval()` or `exec()` anywhere in codebase
- SQL queries restricted to SELECT only
- API keys never stored in object fields (property-based access)
- Input validation on all user-facing interfaces

### NFR-05: Maintainability
- Pydantic models for all data structures (type safety + validation)
- Abstract base class for tools (pluggable architecture)
- Comprehensive test coverage across security and core functionality
- Modular architecture (ingestion, storage, tools, agent, UI are independent)

### NFR-06: Resource Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| Disk | 1 GB + documents | 5 GB+ |
| Python | 3.11+ | 3.12 |
| GPU | Not required | Optional (for local embeddings) |

---

## 9. Configuration & Environment

### 9.1 Environment Variables

```bash
# ── Required (at least one LLM provider) ──
OPENROUTER_API_KEY=sk-or-...        # Recommended: single key for all models

# ── Optional LLM providers ──
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# ── Model selection ──
LLM_MODEL=google/gemini-3-flash-preview    # Default LLM model
EMBEDDING_MODEL=qwen/qwen3-embedding-8b    # Embedding model
EMBEDDING_PROVIDER=auto                     # auto | local | openrouter

# ── Feature flags ──
USE_VISION_TABLES=true                      # Enable VLM table extraction
```

### 9.2 Config Module (`src/config.py`)

```python
@dataclass
class Config:
    # Paths (auto-created on init)
    base_dir: Path
    data_dir: Path              # {base_dir}/data
    sqlite_path: Path           # {data_dir}/db/structured.db
    chroma_path: Path           # {data_dir}/db/chroma
    documents_dir: Path         # {data_dir}/data/documents
    logs_dir: Path              # {data_dir}/logs

    # Model settings
    llm_model: str              # From LLM_MODEL env var
    embedding_model: str        # From EMBEDDING_MODEL env var
    embedding_provider: str     # From EMBEDDING_PROVIDER env var

    # Feature flags
    use_vision_tables: bool     # From USE_VISION_TABLES env var

    # API keys (property-based, not stored)
    @property
    def openrouter_api_key(self) -> str | None
    @property
    def openai_api_key(self) -> str | None
    @property
    def anthropic_api_key(self) -> str | None
```

### 9.3 File: `env.example`
Template for users to copy to `.env` with all supported variables documented.

---

## 10. Testing Requirements

### 10.1 Test Suites

| Suite | File | Purpose | Key Tests |
|-------|------|---------|-----------|
| Calculator | `tests/test_calculator.py` | Audit transparency | Transcript generation, operand binding, division by zero |
| Executor | `tests/test_executor.py` | DAG execution | Parallel layers, dependency resolution, circular dependency detection |
| SQL Security | `tests/test_sql_security.py` | Injection prevention | All forbidden keywords, SELECT-only enforcement, multi-statement blocking |
| Prompt Injection | `tests/test_prompt_injection.py` | Input defense | All 13 patterns detected, sanitization, content wrapping |
| Schema Clustering | `tests/test_schema_clustering.py` | Organization | Company-domain assignment, relevant schema retrieval |
| Docling Extractor | `tests/test_docling_extractor.py` | Table extraction | Docling integration, fallback behavior |
| Storage Fixes | `tests/test_storage_fixes.py` | Storage integrity | SQLite operations, ChromaDB operations, document store |
| Tools Fixes | `tests/test_tools_fixes.py` | Tool functionality | Each tool's execute method, error handling |
| Scripts Fixes | `tests/test_scripts_fixes.py` | CLI validation | Argument parsing, file validation, output handling |

### 10.2 Test Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 10.3 Running Tests
```bash
pytest                          # All tests
pytest tests/test_calculator.py # Single suite
pytest -v                       # Verbose output
pytest -x                       # Stop on first failure
```

---

## 11. File & Module Inventory

### Complete File Tree (to build from scratch)

```
FinanceRAG/
│
├── pyproject.toml                          # Project metadata, pytest config, ruff config
├── requirements.txt                        # Pinned dependencies
├── env.example                             # Environment variable template
├── README.md                               # User-facing documentation
├── .gitignore                              # Git ignore rules
│
├── src/
│   ├── __init__.py                         # Package init
│   ├── config.py                           # Configuration management
│   ├── models.py                           # 60+ Pydantic data models
│   ├── llm_client.py                       # Multi-provider LLM abstraction
│   ├── embeddings.py                       # Embedding provider management
│   ├── security.py                         # Input validation, injection defense
│   ├── rag_agent.py                        # Main orchestrator
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py                   # PDF text extraction (pdfplumber)
│   │   ├── vlm_extractor.py                # Vision LLM table extraction
│   │   ├── vision_table_extractor.py       # Docling local table extraction
│   │   ├── table_extractor.py              # Rule-based fallback extraction
│   │   ├── spreadsheet_parser.py           # Excel/CSV parsing
│   │   ├── chunker.py                      # Semantic text chunking
│   │   ├── schema_detector.py              # LLM schema enhancement
│   │   ├── temporal_extractor.py           # Fiscal period extraction
│   │   └── utils.py                        # Ingestion utilities
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py                 # Structured data store
│   │   ├── chroma_store.py                 # Vector embedding store
│   │   ├── document_store.py               # Full content store
│   │   └── schema_cluster.py               # Company-domain clustering
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── planner.py                      # Query → execution DAG
│   │   ├── executor.py                     # Parallel DAG execution
│   │   └── synthesizer.py                  # Results → cited response
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                         # Abstract Tool base class
│   │   ├── sql_query.py                    # NL-to-SQL tool
│   │   ├── vector_search.py                # Semantic search tool
│   │   ├── calculator.py                   # Safe arithmetic tool
│   │   ├── get_document.py                 # Document retrieval tool
│   │   ├── reranker.py                     # Cross-encoder reranking
│   │   └── comparability.py                # Field comparability checking
│   │
│   └── ui/
│       ├── __init__.py
│       └── console.py                      # Rich terminal UI
│
├── scripts/
│   ├── query.py                            # Query CLI
│   ├── ingest.py                           # Ingestion CLI
│   └── analyst_evaluation.py               # Evaluation suite
│
├── tests/
│   ├── test_calculator.py
│   ├── test_executor.py
│   ├── test_sql_security.py
│   ├── test_prompt_injection.py
│   ├── test_schema_clustering.py
│   ├── test_docling_extractor.py
│   ├── test_storage_fixes.py
│   ├── test_tools_fixes.py
│   └── test_scripts_fixes.py
│
└── data/                                   # Created at runtime
    ├── db/
    │   ├── structured.db                   # SQLite database
    │   └── chroma/                         # ChromaDB persistence
    ├── documents/
    │   ├── .metadata/                      # Document metadata JSONs
    │   └── .content/                       # Extracted full text
    └── logs/
```

### Dependencies (`requirements.txt`)

```
pydantic>=2.0.0
pydantic-ai>=0.0.12
chromadb>=0.4.0
sentence-transformers>=2.2.0
pdfplumber>=0.10.0
PyMuPDF>=1.23.0
docling>=2.66.0
openai>=1.0.0
anthropic>=0.18.0
aiohttp>=3.9.0
flashrank>=0.2.0
rich>=13.0.0
python-dotenv>=1.0.0
pandas
openpyxl
fpdf2
```

---

## 12. Build Phases

### Phase 1: Foundation (Week 1–2)
**Goal:** Core infrastructure that everything else builds on.

| Task | Files | Description |
|------|-------|-------------|
| 1.1 | `pyproject.toml`, `requirements.txt`, `env.example` | Project setup, dependencies, config files |
| 1.2 | `src/__init__.py`, `src/config.py` | Configuration management with secure API key handling |
| 1.3 | `src/models.py` | All Pydantic data models (60+ models) |
| 1.4 | `src/security.py` | Input validation, prompt injection detection, sanitization |
| 1.5 | `src/llm_client.py` | Multi-provider LLM client (OpenRouter, OpenAI, Anthropic) |
| 1.6 | `src/embeddings.py` | Embedding provider (local sentence-transformers + remote) |
| 1.7 | `src/ui/console.py` | Rich-based terminal UI components |

**Tests:** `test_prompt_injection.py`

### Phase 2: Storage Layer (Week 2–3)
**Goal:** All three storage backends operational.

| Task | Files | Description |
|------|-------|-------------|
| 2.1 | `src/storage/sqlite_store.py` | SQLite store with SQL validation and security |
| 2.2 | `src/storage/chroma_store.py` | ChromaDB store with embedding integration |
| 2.3 | `src/storage/document_store.py` | File-based document content and metadata store |
| 2.4 | `src/storage/schema_cluster.py` | Company-domain schema clustering |

**Tests:** `test_sql_security.py`, `test_storage_fixes.py`, `test_schema_clustering.py`

### Phase 3: Ingestion Pipeline (Week 3–4)
**Goal:** Ingest PDFs, spreadsheets, and CSVs into all storage backends.

| Task | Files | Description |
|------|-------|-------------|
| 3.1 | `src/ingestion/pdf_parser.py` | PDF text extraction with pdfplumber |
| 3.2 | `src/ingestion/table_extractor.py` | Rule-based table extraction (fallback) |
| 3.3 | `src/ingestion/vision_table_extractor.py` | Docling local table extraction |
| 3.4 | `src/ingestion/vlm_extractor.py` | VLM-based table extraction via OpenRouter |
| 3.5 | `src/ingestion/chunker.py` | Semantic text chunking |
| 3.6 | `src/ingestion/schema_detector.py` | LLM-powered schema enhancement |
| 3.7 | `src/ingestion/temporal_extractor.py` | Fiscal period metadata extraction |
| 3.8 | `src/ingestion/spreadsheet_parser.py` | Excel/CSV parsing |
| 3.9 | `scripts/ingest.py` | Ingestion CLI script |

**Tests:** `test_docling_extractor.py`, `test_scripts_fixes.py`

### Phase 4: Tool System (Week 4–5)
**Goal:** All four tools operational with full audit transparency.

| Task | Files | Description |
|------|-------|-------------|
| 4.1 | `src/tools/base.py` | Abstract Tool base class |
| 4.2 | `src/tools/vector_search.py` | Semantic search with reranking |
| 4.3 | `src/tools/reranker.py` | FlashRank cross-encoder reranking |
| 4.4 | `src/tools/sql_query.py` | NL-to-SQL with schema context |
| 4.5 | `src/tools/calculator.py` | Safe AST-based arithmetic with transcripts |
| 4.6 | `src/tools/get_document.py` | Full document retrieval |
| 4.7 | `src/tools/comparability.py` | Field comparability checking |

**Tests:** `test_calculator.py`, `test_tools_fixes.py`

### Phase 5: Query Engine (Week 5–6)
**Goal:** End-to-end query pipeline from NL question to cited answer.

| Task | Files | Description |
|------|-------|-------------|
| 5.1 | `src/agent/planner.py` | Query decomposition into DAG |
| 5.2 | `src/agent/executor.py` | Parallel DAG execution |
| 5.3 | `src/agent/synthesizer.py` | Response synthesis with citations |

**Tests:** `test_executor.py`

### Phase 6: Orchestration & CLI (Week 6–7)
**Goal:** Full system integration, CLI interface, and evaluation.

| Task | Files | Description |
|------|-------|-------------|
| 6.1 | `src/rag_agent.py` | Main RAG agent orchestrator |
| 6.2 | `scripts/query.py` | Query CLI (REPL + single-shot + export) |
| 6.3 | `scripts/analyst_evaluation.py` | Evaluation suite |
| 6.4 | `README.md` | User documentation |

**Tests:** All test suites pass, end-to-end integration testing

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VLM table extraction fails on complex layouts | Medium | Medium | Three-tier fallback (VLM → Docling → Rule-based) |
| LLM generates incorrect SQL | Medium | High | SQL validation layer, SELECT-only enforcement, LIMIT safeguards |
| Prompt injection via document content | Low | High | Defense-in-depth: sanitization, wrapping, pattern detection |
| LLM hallucinated arithmetic | Medium | Critical | Arithmetic prohibition in prompts + Calculator tool enforcement |
| ChromaDB performance degrades at scale | Low | Medium | HNSW tuning, schema clustering for context reduction |
| API rate limits from LLM providers | Medium | Medium | Semaphore-limited concurrency, model tier selection |
| Large PDFs exceed memory | Low | Medium | 500MB file size limit, streaming where possible |
| Incomparable metrics compared without warning | Medium | High | Comparability checker, field definitions, refusal system |
| No LLM API key configured | Medium | Low | Heuristic fallback mode for all components |

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — enhance LLM answers with retrieved context |
| **DAG** | Directed Acyclic Graph — dependency graph for execution ordering |
| **VLM** | Vision-Language Model — LLM that can process images |
| **ChromaDB** | Open-source vector database for embedding storage and similarity search |
| **pdfplumber** | Python library for extracting text and tables from PDFs |
| **Docling** | IBM's document understanding library with TableFormer table extraction |
| **FlashRank** | Lightweight cross-encoder reranking library |
| **HNSW** | Hierarchical Navigable Small World — approximate nearest neighbor algorithm |
| **EAV** | Entity-Attribute-Value — flexible schema pattern for storing tabular data |
| **Cross-encoder** | Neural model that scores (query, document) pairs for reranking |
| **GAAP** | Generally Accepted Accounting Principles |
| **IFRS** | International Financial Reporting Standards |
| **10-K** | Annual SEC filing with comprehensive financial data |
| **Operand Binding** | Mapping from a symbolic reference to its resolved numeric value and source |
| **Schema Clustering** | Organizing tables by company and domain for targeted LLM context |
| **Query Refusal** | Structured response when the system cannot reliably answer a question |
| **Comparability** | Whether two financial metrics can be meaningfully compared |

---

*End of PRD*
