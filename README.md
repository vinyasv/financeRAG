<p align="center">
  <h1 align="center">Finance RAG</h1>
  <p align="center">
    High-performance RAG for financial documents with parallel execution and audit transparency
    <br />
    <a href="#quick-start">Quick Start</a>
    ·
    <a href="#features">Features</a>
    ·
    <a href="technicaloverview.md">Technical Docs</a>
  </p>
</p>

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp env.example .env
# Add your OPENROUTER_API_KEY to .env

# Ingest documents
python scripts/ingest.py path/to/annual_report.pdf

# Query
python scripts/query.py "What was 2024 revenue growth?"
```

## Features

| Feature | Description |
|---------|-------------|
| **Parallel Execution** | DAG-based tool execution for maximum speed |
| **Smart Table Extraction** | Vision-based (VLM) or rule-based PDF table parsing |
| **FlashRank Reranking** | Ultra-fast reranking (~10-50ms) for precise retrieval |
| **Audit Transparency** | Calculation transcripts with operand provenance |
| **Multi-Provider LLM** | OpenRouter, OpenAI, Anthropic, local models |
| **Export** | PDF, CSV, JSON output formats |

<details>
<summary><strong>Audit Transparency Features</strong></summary>

- **Operand binding**: Every calculation shows source values with provenance
- **Computation enforcement**: LLM prohibited from arithmetic—all math via calculator
- **Refusal mode**: System refuses with explanation when data is insufficient
- **Definition hashing**: Field comparability checking (GAAP vs non-GAAP, etc.)

</details>

## Architecture

```
Query → Planner → Execution DAG → Parallel Tools → Response
                        │
          ┌─────────────┼─────────────┐
          │             │             │
      SQL Query   Vector Search   Calculator
          │             │             │
          └─────────────┴─────────────┘
                        │
                   Synthesizer
```

## Usage

### Ingestion

```bash
python scripts/ingest.py document.pdf          # Single file
python scripts/ingest.py -f ./reports/         # Entire folder
```

### Queries

```bash
python scripts/query.py                                    # Interactive
python scripts/query.py "Compare margins 2023 vs 2024"     # Single query
python scripts/query.py -m claude-sonnet "..."             # Specific model
python scripts/query.py -o report.pdf "..."                # Export to PDF
```

<details>
<summary><strong>Example Queries</strong></summary>

```
"What was the total revenue for Q4 2024?"
"Compare gross margins between 2023 and 2024"
"Calculate profit if operating costs increased by 15%"
"What risks are mentioned regarding supply chain?"
"Calculate the YoY growth rate for each quarter"
```

</details>

## Configuration

Create `.env` from `env.example`:

```bash
OPENROUTER_API_KEY=your-key-here   # Required (get at openrouter.ai/keys)

# Optional
LLM_MODEL=gpt-4o-mini              # Default model
USE_VISION_TABLES=true             # Enable vision table extraction
EMBEDDING_PROVIDER=local           # Use local embeddings (free)
```

## Tools

| Tool | Purpose |
|------|---------|
| `sql_query` | Query structured data from extracted tables |
| `vector_search` | Semantic search with FlashRank reranking |
| `calculator` | Deterministic math with audit transcripts |
| `get_document` | Retrieve full document content |
| `comparability` | Check field definition compatibility |

## Performance

| Operation | Latency |
|-----------|---------|
| Query Planning | 200-500ms |
| Vector Search + Rerank | 50-150ms |
| SQL Query | 5-50ms |
| **End-to-end** | **500-1500ms** |

## Project Structure

```
src/
├── agent/          # Planner, executor, synthesizer
├── tools/          # SQL, vector search, calculator
├── ingestion/      # PDF parser, table extractor, chunker
└── storage/        # SQLite, ChromaDB, document store
scripts/
├── ingest.py       # Document ingestion CLI
└── query.py        # Query CLI
```

See [technicaloverview.md](technicaloverview.md) for detailed architecture.

## Contributing

```bash
pytest tests/       # Run tests before submitting PRs
```

## License

MIT

## Acknowledgements

[ChromaDB](https://www.trychroma.com/) · [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) · [thepipe](https://github.com/emcf/thepipe) · [OpenRouter](https://openrouter.ai/)
