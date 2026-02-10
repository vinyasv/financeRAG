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
- FlashRank cross-encoder reranking (~10-50ms) for precise retrieval
- Natural language to SQL for structured financial data
- Multi-provider LLM support: OpenRouter, OpenAI, Anthropic

**Audit & Trust**
- Calculation transcripts with operand provenance — every number traceable to its source
- LLM prohibited from arithmetic; all math done via deterministic calculator
- Structured refusal when data is insufficient
- Field comparability checking (GAAP vs non-GAAP, etc.)

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
| `LLM_MODEL` | `google/gemini-3-flash-preview` | LLM model for planning and synthesis |
| `EMBEDDING_PROVIDER` | `local` | `local` (free, BAAI/bge-small-en-v1.5) or `openrouter` |

## Architecture

```
src/
├── agent/              # Planner, executor, synthesizer
├── tools/              # SQL, vector search, calculator, reranker
├── ingestion/          # PDF/CSV parsers, VLM + Docling table extraction
├── storage/            # SQLite, ChromaDB, schema clustering
├── rag_agent.py        # Main orchestrator
├── llm_client.py       # Multi-provider LLM abstraction
├── security.py         # Input validation, injection prevention
└── ui/console.py       # Rich terminal UI
scripts/
├── ingest.py           # Document ingestion CLI
├── query.py            # Query CLI
└── analyst_evaluation.py
tests/                  # Unit tests (security, tools, storage, execution)
```

See [technicaloverview.md](technicaloverview.md) for the full architecture deep-dive.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Reranking | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) |
| Table extraction | [Docling](https://github.com/DS4SD/docling) (local) + Gemini VLM |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (BAAI/bge-small-en-v1.5) |
| Structured storage | SQLite |
| LLM routing | [OpenRouter](https://openrouter.ai/) |
| PDF parsing | PyMuPDF + pdfplumber |
| Terminal UI | [Rich](https://github.com/Textualize/rich) |

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check src/
```

Requires Python 3.11+.

## License

MIT
