#!/usr/bin/env python3
"""CLI script for querying the document corpus."""

import sys
import asyncio
import argparse
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env first
from src.config import config
from src.rag_agent import RAGAgent
from src.llm_client import get_llm_client, OpenRouterClient
from src.security import MAX_QUERY_LENGTH, detect_injection_attempt

# Configure logging (filter sensitive data)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)


def validate_query_input(query: str) -> tuple[bool, str]:
    """
    Validate user query input.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long ({len(query)} chars). Maximum: {MAX_QUERY_LENGTH}"
    
    # Log potential injection attempts (but don't block - might be false positive)
    is_suspicious, patterns = detect_injection_attempt(query)
    if is_suspicious:
        logger.warning(f"Potential injection attempt detected. Query length: {len(query)}")
        # Note: We don't log the actual query content for privacy
    
    return True, ""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query the UltimateRAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s "What was Q3 revenue?"             # Single query
  %(prog)s -m claude-sonnet "Complex query"   # Use specific model
  %(prog)s -o results.csv "Query here"        # Export to CSV
  %(prog)s -o report.pdf "Query here"         # Export to PDF
  %(prog)s -o results.json "Query here"       # Export to JSON
  %(prog)s --list-models                      # Show available models

Environment Variables:
  OPENROUTER_API_KEY    OpenRouter API key (recommended)
  OPENAI_API_KEY        Direct OpenAI API key
  ANTHROPIC_API_KEY     Direct Anthropic API key
"""
    )
    
    parser.add_argument(
        "query",
        nargs="*",
        help="Query to run (omit for interactive mode)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="LLM model to use (e.g., gpt-4o-mini, claude-sonnet, llama-70b)"
    )
    
    parser.add_argument(
        "-p", "--provider",
        choices=["auto", "openrouter", "openai", "anthropic", "none"],
        default="auto",
        help="LLM provider (default: auto-detect from env vars)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show execution details"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Export results to file (supports .csv, .json, and .pdf)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available OpenRouter model shortcuts"
    )
    
    return parser.parse_args()


def format_citations(citations) -> str:
    """Format citations as a semicolon-separated string."""
    if not citations:
        return ""
    
    parts = []
    for cite in citations[:5]:  # Limit to 5 citations
        ref = cite.format_reference()
        parts.append(ref)
    
    return "; ".join(parts)


def export_to_csv(output_path: Path, results: list[dict]):
    """Export query results to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "answer", "citations", "response_time_ms", "timestamp"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📁 Results exported to: {output_path}")


def export_to_json(output_path: Path, results: list[dict]):
    """Export query results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Results exported to: {output_path}")


def export_to_pdf(output_path: Path, results: list[dict]):
    """Export query results to PDF file."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("❌ PDF export requires fpdf2: pip install fpdf2")
        return
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "UltimateRAG Query Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)
    
    for i, result in enumerate(results, 1):
        # Query header
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"Query {i}", ln=True, fill=True)
        
        # Query text
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, result.get("query", ""))
        pdf.ln(3)
        
        # Answer
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Answer:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        # Handle long answers with multi_cell
        answer = result.get("answer", "")
        # Encode to latin-1 compatible (replace non-latin chars)
        answer_safe = answer.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 5, answer_safe)
        pdf.ln(3)
        
        # Citations
        if result.get("citations"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Sources:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            citations_safe = result["citations"].encode('latin-1', errors='replace').decode('latin-1')
            pdf.multi_cell(0, 5, citations_safe)
        
        # Response time
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, f"Response time: {result.get('response_time_ms', 0):.1f}ms", ln=True)
        pdf.set_text_color(0, 0, 0)
        
        pdf.ln(8)
    
    pdf.output(str(output_path))
    print(f"\n📄 PDF report exported to: {output_path}")


async def interactive_mode(agent: RAGAgent, model_name: str | None):
    """Run in interactive mode."""
    print("UltimateRAG Interactive Query Mode")
    if model_name:
        print(f"Model: {model_name}")
    print("Commands: 'quit', 'stats', 'models'")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        # Validate query
        is_valid, error_msg = validate_query_input(query)
        if not is_valid:
            print(f"⚠️ {error_msg}")
            continue
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if query.lower() == 'stats':
            stats = agent.get_stats()
            print(f"\nDocuments: {stats['document_count']}")
            print(f"Tables: {stats['table_count']}")
            print(f"Chunks: {stats['chunk_count']}")
            for doc in stats['documents']:
                print(f"  - {doc['filename']} ({doc['pages']} pages)")
            continue
        
        if query.lower() == 'models':
            print("\nAvailable OpenRouter models:")
            for short, full in OpenRouterClient.list_models().items():
                print(f"  {short:15} -> {full}")
            continue
        
        try:
            response = await agent.query(query, verbose=True)
            print("\n" + "=" * 50)
            print("ANSWER:")
            print(response.answer)
            
            if response.citations:
                print("\nCitations:")
                for cite in response.citations[:3]:
                    print(f"  - {cite.format_reference()}")
        except Exception as e:
            print(f"Error: {e}")


async def single_query(agent: RAGAgent, query: str, verbose: bool = False, output_path: Path | None = None):
    """Run a single query with input validation."""
    # Validate query
    is_valid, error_msg = validate_query_input(query)
    if not is_valid:
        print(f"⚠️ {error_msg}")
        return
    
    logger.info(f"Processing query of length {len(query)} chars")
    
    response = await agent.query(query, verbose=verbose)
    
    print("\nAnswer:")
    print(response.answer)
    
    if response.citations:
        print("\nCitations:")
        for cite in response.citations[:5]:
            print(f"  - {cite.format_reference()}")
    
    print(f"\nExecution time: {response.total_time_ms:.1f}ms")
    
    # Export if output path specified
    if output_path:
        result = {
            "query": query,
            "answer": response.answer,
            "citations": format_citations(response.citations),
            "response_time_ms": round(response.total_time_ms, 1),
            "timestamp": datetime.now().isoformat()
        }
        
        ext = output_path.suffix.lower()
        if ext == ".json":
            export_to_json(output_path, [result])
        elif ext == ".pdf":
            export_to_pdf(output_path, [result])
        else:
            # Default to CSV
            export_to_csv(output_path, [result])


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("Available OpenRouter model shortcuts:")
        print()
        print("Fast & Cheap:")
        for name in ["gemini-flash", "gpt-4o-mini", "claude-haiku", "llama-8b"]:
            print(f"  {name:15} -> {OpenRouterClient.list_models()[name]}")
        print()
        print("Balanced:")
        for name in ["gpt-4o", "claude-sonnet", "gemini-pro", "llama-70b"]:
            print(f"  {name:15} -> {OpenRouterClient.list_models()[name]}")
        print()
        print("Best Quality:")
        for name in ["claude-opus", "gpt-4-turbo", "llama-405b"]:
            print(f"  {name:15} -> {OpenRouterClient.list_models()[name]}")
        print()
        print("Free (rate limited):")
        print(f"  {'free':15} -> {OpenRouterClient.list_models()['free']}")
        return
    
    # Get LLM client
    llm_client = get_llm_client(provider=args.provider, model=args.model)
    
    if llm_client:
        model_info = args.model or "auto"
        print(f"Using LLM: {args.provider} ({model_info})")
    else:
        print("No LLM configured - using heuristic mode")
        print("Set OPENROUTER_API_KEY for full functionality")
    
    # Initialize agent
    agent = RAGAgent(llm_client=llm_client)
    
    # Check if we have any documents
    stats = agent.get_stats()
    if stats['document_count'] == 0:
        print("\nNo documents ingested yet!")
        print("Run: python scripts/ingest.py <pdf_path>")
        return
    
    print(f"Knowledge base: {stats['document_count']} docs, {stats['table_count']} tables, {stats['chunk_count']} chunks")
    print()
    
    # Run query or interactive mode
    if args.query:
        query = " ".join(args.query)
        await single_query(agent, query, args.verbose, args.output)
    else:
        await interactive_mode(agent, args.model)


if __name__ == "__main__":
    asyncio.run(main())
