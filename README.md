# Finance RAG

A high-performance RAG (Retrieval Augmented Generation) system designed for financial documents. Extracts structured data from PDFs, enables semantic search, and provides accurate answers to complex queries using parallel tool execution.

## Key Features

### Parallel Execution
- Independent tool calls run simultaneously for maximum speed
- DAG-based execution planning with dependency resolution
- Real-time execution monitoring with per-step timing

### Smart Table Extraction
- **Vision-based extraction**: Uses VLMs via `thepipe` for 5-6x faster, more accurate table parsing
- **Fallback rule-based extraction**: Works without API keys using `pdfplumber`
- **Schema enhancement**: LLM-powered column and table name normalization

### FlashRank Reranking
- Ultra-fast reranking (~10-50ms latency) using distilled T5 models
- Hybrid scoring combining vector similarity + reranker scores
- Dramatically improved retrieval precision over pure embedding search

### Flexible Embeddings
- **OpenRouter**: Qwen3-8B, OpenAI, Cohere, Google embeddings via single API key
- **Local**: Free sentence-transformers (BGE-large, all-MiniLM-L6-v2)
- Auto-detection based on available API keys

### Deterministic Calculations
- Calculator tool for safe arithmetic evaluation
- Avoids LLM hallucination on numerical computations
- Supports complex expressions with percentages and unit conversions

### Batch Processing & Export
- **Batch folder ingestion**: Recursively ingest entire directories with progress bar
- **Export to PDF/CSV/JSON**: Generate professional reports from query results
- **Accurate citations**: Shows filename, page number, and line references

### Multi-Provider LLM Support
- **OpenRouter** (recommended): Single API key for all models
- **Direct APIs**: OpenAI, Anthropic
- **Model shortcuts**: `gpt-4o`, `claude-sonnet`, `gemini-flash`, `llama-70b`, etc.

## Architecture

```
Query -> Planner (1 LLM call) -> Execution DAG -> Parallel Tool Calls -> Synthesize Response
                                    |
                    +---------------+---------------+
                    |               |               |
              SQL Query      Vector Search    Calculator
                    |               |               |
                    +---------------+---------------+
                                    |
                            Response Synthesis
```

## Tools

| Tool | Purpose |
|------|---------|
| `sql_query` | Query structured data from extracted tables |
| `vector_search` | Semantic search with FlashRank reranking |
| `calculator` | Safe deterministic math expression evaluation |
| `get_document` | Retrieve full document or section content |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FlashRank for reranking (optional but recommended)
pip install flashrank

# Set up environment variables
cp env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file in the project root:

```bash
# =============================================================================
# API Key (required for full functionality)
# =============================================================================
OPENROUTER_API_KEY=your-key-here  # Get at https://openrouter.ai/keys

# Or use direct provider APIs:
# OPENAI_API_KEY=your-key
# ANTHROPIC_API_KEY=your-key

# =============================================================================
# LLM Model (optional)
# =============================================================================
# Options: gpt-4o-mini, claude-sonnet, gemini-flash, llama-70b
# LLM_MODEL=gpt-4o-mini

# =============================================================================
# Vision Model for Table Extraction (optional)
# =============================================================================
# VISION_MODEL=google/gemini-2.0-flash-001  # Fast, accurate, cheap
# USE_VISION_TABLES=true                     # Enable vision extraction

# =============================================================================
# Embedding Model (optional)
# =============================================================================
# EMBEDDING_MODEL=qwen/qwen3-embedding-8b    # Default (OpenRouter)
# EMBEDDING_PROVIDER=local                   # Force local embeddings
```

## Usage

### Ingest Documents

```bash
# Ingest a single PDF
python scripts/ingest.py path/to/document.pdf

# Ingest multiple documents
python scripts/ingest.py doc1.pdf doc2.pdf doc3.pdf

# Batch ingest entire folder (with progress bar)
python scripts/ingest.py -f /path/to/documents/

# Only PDFs from a folder
python scripts/ingest.py -f ./reports --pattern "*.pdf"
```

### Run Queries

```bash
# Interactive mode
python scripts/query.py

# Single query
python scripts/query.py "What was Q4 revenue growth?"

# Specify model
python scripts/query.py -m claude-sonnet "Compare operating margins"

# Verbose mode (shows execution details)
python scripts/query.py -v "Calculate profit if costs increased 20%"

# Export to PDF report
python scripts/query.py -o report.pdf "What was NVIDIA's revenue?"

# Export to CSV (for Excel)
python scripts/query.py -o results.csv "Compare margins across companies"

# Export to JSON
python scripts/query.py -o data.json "List all risk factors"

# List available models
python scripts/query.py --list-models
```

### Interactive Commands

```
Query: quit     # Exit interactive mode
Query: stats    # Show knowledge base statistics
Query: models   # List available LLM models
```

## Project Structure

```
financeRAG/
├── src/
│   ├── agent/
│   │   ├── planner.py        # Query planning with LLM
│   │   ├── executor.py       # DAG-based parallel execution
│   │   └── synthesizer.py    # Response synthesis
│   ├── tools/
│   │   ├── calculator.py     # Safe math evaluation
│   │   ├── sql_query.py      # Structured data queries
│   │   ├── vector_search.py  # Semantic search + reranking
│   │   ├── reranker.py       # FlashRank reranking
│   │   └── get_document.py   # Document retrieval
│   ├── ingestion/
│   │   ├── pdf_parser.py     # PDF text extraction
│   │   ├── table_extractor.py        # Rule-based table extraction
│   │   ├── vision_table_extractor.py # VLM-based table extraction
│   │   ├── chunker.py        # Semantic text chunking
│   │   └── schema_detector.py # LLM schema enhancement
│   ├── storage/
│   │   ├── sqlite_store.py   # Structured data storage
│   │   ├── chroma_store.py   # Vector storage (ChromaDB)
│   │   └── document_store.py # Full document storage
│   ├── embeddings.py         # Multi-provider embedding support
│   ├── llm_client.py         # LLM client abstraction
│   ├── rag_agent.py          # Main agent orchestration
│   ├── models.py             # Pydantic data models
│   └── config.py             # Configuration management
├── data/
│   ├── documents/            # Full document storage
│   └── db/                   # SQLite + ChromaDB databases
├── scripts/
│   ├── ingest.py             # Document ingestion CLI
│   └── query.py              # Query CLI
├── tests/                    # Test suite
├── requirements.txt
└── env.example               # Configuration template
```

## Performance

| Operation | Typical Latency |
|-----------|----------------|
| Query Planning | ~200-500ms |
| Vector Search + Rerank | ~50-150ms |
| SQL Query | ~5-50ms |
| Calculator | ~1ms |
| Response Synthesis | ~300-800ms |
| **Total Query** | **~500-1500ms** |

## Example Queries

```
# Financial analysis
"What was the total revenue for Q4 2024?"
"Compare gross margins between 2023 and 2024"
"Calculate what profit would be if operating costs increased by 15%"

# Semantic search
"What risks are mentioned regarding supply chain?"
"Summarize the key strategic initiatives"

# Complex queries
"What percentage of total revenue came from the top 3 segments?"
"Calculate the YoY growth rate for each quarter"
```

## Advanced Configuration

### ChromaDB HNSW Parameters

The vector store is optimized for large knowledge bases:

```python
{
    "hnsw:space": "cosine",      # Cosine similarity
    "hnsw:M": 32,                # Connections per node (higher = better recall)
    "hnsw:construction_ef": 200, # Index quality
    "hnsw:search_ef": 100,       # Search depth for large KBs
}
```

### Reranking Configuration

```python
from src.tools.reranker import Reranker

# Fast model (default)
reranker = Reranker(model_name="rank-T5-flan")

# More accurate model
reranker = Reranker(model_name="rank_zephyr_7b_v1_full")
```

## Contributing

Contributions are welcome! Please ensure tests pass before submitting PRs.

```bash
# Run tests
pytest tests/
```

## License

MIT

## Acknowledgements

- [ChromaDB](https://www.trychroma.com/) - Vector database
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - Ultra-fast reranking
- [thepipe](https://github.com/emcf/thepipe) - Vision-based PDF extraction
- [OpenRouter](https://openrouter.ai/) - Multi-model LLM gateway
