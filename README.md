<h1 align="center">Finance RAG</h1>

<p align="center">
  <strong>Ask questions about financial documents. Get cited, auditable answers.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="technicaloverview.md">Technical Docs</a>
</p>

---

Finance RAG ingests PDFs, spreadsheets, and CSVs, then answers complex financial questions using parallel tool execution, structured SQL queries, and semantic search — with full audit trails so you can verify every number.

```
"Compare NVIDIA and AMD gross margins for FY2024"

  ✓ Planning query...         220ms
  ✓ SQL: nvidia_income_stmt    12ms  ┐
  ✓ SQL: amd_income_stmt        9ms  ├─ parallel
  ✓ Vector search + rerank     85ms  ┘
  ✓ Synthesizing response...  480ms

  NVIDIA reported 75.0% gross margin vs AMD at 47.6% for FY2024...
  Sources: nvidia_10k_2024.pdf (p.42), amd_10k_2024.pdf (p.38)
```

## Quick Start

```bash
pip install -r requirements.txt

cp env.example .env
# Add your OPENROUTER_API_KEY (get one at openrouter.ai/keys)

# Ingest a document
python scripts/ingest.py annual_report.pdf

# Ask questions
python scripts/query.py "What was total revenue for Q4 2024?"
```

## How It Works

```
Query → Planner → Execution DAG → Tools (parallel) → Synthesizer → Response
                       │
         ┌─────────────┼─────────────┐
         │             │             │
     SQL Query   Vector Search   Calculator
     (tables)    (unstructured)  (arithmetic)
         │             │             │
         └─────────────┴─────────────┘
                       │
              Cited, auditable answer
```

1. **Ingest** — PDFs are parsed, tables extracted (via VLM or Docling), text chunked and embedded into ChromaDB, structured data stored in SQLite
2. **Plan** — The query planner decomposes questions into a DAG of tool calls with dependencies
3. **Execute** — Independent tools run in parallel: SQL for structured data, vector search for unstructured text, calculator for arithmetic
4. **Synthesize** — Results are combined into a cited response with full audit trails

## Features

**Document Processing**
- Three-tier table extraction: Gemini VLM → Docling (local) → rule-based fallback
- Native CSV and multi-sheet Excel support
- Automatic fiscal year/quarter metadata extraction
- Schema clustering — tables grouped by company and domain

**Query Engine**
- DAG-based parallel execution for fast, multi-step queries
- Three planning modes: LLM-planned, heuristic fallback, fast-path for simple queries
- FlashRank cross-encoder reranking (~10-50ms, +25% precision)
- Natural language to SQL with schema clustering for focused context
- Multi-provider LLM support: OpenRouter, OpenAI, Anthropic
- Execution monitoring with per-step timing

**Audit & Trust**
- Calculation transcripts with operand provenance — every number traceable to its source
- LLM prohibited from arithmetic; all math done via AST-based deterministic calculator
- Structured refusal with reasons (insufficient data, definition mismatch, period discontinuity, incomparable metrics, missing context)
- Field comparability checking (GAAP vs non-GAAP, currency, segment scope)

**Security**
- SQL injection prevention: 18 forbidden keywords, SELECT-only enforcement
- Prompt injection defense: 16 detection patterns with input sanitization
- Path traversal protection, API key isolation, AST depth limiting

**Output**
- PDF, CSV, and JSON export
- Rich terminal UI with progress indicators

## Usage

### Ingestion

```bash
python scripts/ingest.py report.pdf                # Single file
python scripts/ingest.py report.pdf data.xlsx       # Multiple files
python scripts/ingest.py -f ./earnings/             # Entire folder
python scripts/ingest.py -f ./reports --pattern "*.pdf"
```

Supports: PDF, XLSX, XLS, CSV

### Querying

```bash
python scripts/query.py                                 # Interactive mode
python scripts/query.py "What was 2024 revenue growth?" # Single query
python scripts/query.py -m claude-sonnet "..."          # Choose model
python scripts/query.py -o report.pdf "..."             # Export to PDF
python scripts/query.py --list-models                   # Available models
```

<details>
<summary><strong>Example queries</strong></summary>

```
"What was total revenue for Q4 2024?"
"Compare gross margins between NVIDIA and AMD"
"Calculate YoY growth rate for each quarter"
"What risks are mentioned regarding supply chain?"
"What would profit be if operating costs increased 15%?"
```

</details>

## Configuration

```bash
cp env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | **Required.** Get one at [openrouter.ai/keys](https://openrouter.ai/keys) |
| `OPENAI_API_KEY` | — | Optional. Alternative LLM provider |
| `ANTHROPIC_API_KEY` | — | Optional. Alternative LLM provider |
| `LLM_MODEL` | `google/gemini-3-flash-preview` | LLM model for planning and synthesis |
| `VISION_MODEL` | `google/gemini-2.5-flash-lite` | Vision model for table extraction |
| `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model (remote) |
| `EMBEDDING_PROVIDER` | `auto` | `auto` (prefers OpenRouter if key available), `local` (BAAI/bge-small-en-v1.5), or `openrouter` |
| `USE_VISION_TABLES` | `true` | Enable VLM-based table extraction |

## Architecture

```
src/
├── agent/              # Planner (3 modes), DAG executor, ExecutionMonitor, synthesizer
├── tools/              # SQL, vector search, calculator, reranker, comparability checker
├── ingestion/          # PDF/CSV parsers, VLM + Docling + rule-based table extraction
│                       # Schema detector, temporal extractor, semantic chunker
├── storage/            # SQLite, ChromaDB, document store, schema clustering
│                       # CompanyRegistry, SchemaClusterManager
├── rag_agent.py        # Main orchestrator
├── models.py           # 60+ Pydantic v2 data models
├── llm_client.py       # Multi-provider LLM abstraction (12 model aliases)
├── embeddings.py       # Embedding providers (local + remote)
├── security.py         # Input validation, injection prevention
├── config.py           # Configuration management
└── ui/console.py       # Rich terminal UI with custom theme
scripts/
├── ingest.py           # Document ingestion CLI
├── query.py            # Query CLI (REPL, single-shot, export)
├── analyst_evaluation.py  # Multi-document evaluation suite
└── benchmark_vlm.py    # VLM extraction benchmarking
tests/                  # 9 test suites (security, tools, storage, execution, clustering)
```

See [technicaloverview.md](technicaloverview.md) for the full architecture deep-dive.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector store | [ChromaDB](https://www.trychroma.com/) (HNSW, cosine similarity) |
| Reranking | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (ms-marco-MiniLM-L-6-v2) |
| Table extraction | Gemini 2.5 Flash Lite VLM → [Docling](https://github.com/DS4SD/docling) TableFormer (local) → rule-based |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (BAAI/bge-small-en-v1.5) or OpenRouter (qwen3-embedding-8b) |
| Structured storage | SQLite with SQL injection prevention |
| LLM routing | [OpenRouter](https://openrouter.ai/) (12 model aliases across 4 tiers) |
| PDF parsing | PyMuPDF + pdfplumber |
| Data models | [Pydantic](https://docs.pydantic.dev/) v2 (60+ models) |
| Terminal UI | [Rich](https://github.com/Textualize/rich) |
| PDF export | fpdf2 |

## Development

```bash
# Run all tests (9 suites)
pytest tests/

# Run specific suite
pytest tests/test_sql_security.py
pytest tests/test_calculator.py

# Lint
ruff check src/
```

Requires Python 3.11+ (3.12 recommended).

## License

MIT
