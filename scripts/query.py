#!/usr/bin/env python3
"""CLI script for querying the document corpus."""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import _bootstrap

from src.cli.common import (
    configure_cli_logging,
    export_results_to_csv,
    export_results_to_json,
    export_results_to_pdf,
    format_citations,
    silence_third_party_loggers,
)
from src.llm_client import OpenRouterClient, get_llm_client
from src.rag_agent import RAGAgent
from src.ui import (
    console,
    create_status,
    print_answer,
    print_citations,
    print_error,
    print_header,
    print_models_table,
    print_stats_table,
    print_warning,
)
from src.validation import validate_query

configure_cli_logging("query.log")
logger = logging.getLogger(__name__)
silence_third_party_loggers()
PROJECT_ROOT = _bootstrap.PROJECT_ROOT


def validate_query_input(query: str) -> tuple[bool, str]:
    """
    Validate user query input.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    is_valid, error_msg, suspicious_patterns = validate_query(query)
    if suspicious_patterns:
        logger.warning(f"Potential injection attempt detected. Query length: {len(query)}")
    return is_valid, error_msg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query the Finance RAG knowledge base",
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
async def interactive_mode(agent: RAGAgent, model_name: str | None, provider: str | None):
    """Run in interactive mode."""
    stats = agent.get_stats()
    print_header(
        doc_count=stats['document_count'],
        table_count=stats['table_count'],
        chunk_count=stats['chunk_count'],
        model=model_name,
        provider=provider
    )
    
    console.print("[muted]Commands: quit, stats, models[/muted]")
    console.print()
    
    while True:
        try:
            query = console.input("[accent]>[/accent] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[muted]Goodbye.[/muted]")
            break
        
        if not query:
            continue
        
        # Validate query
        is_valid, error_msg = validate_query_input(query)
        if not is_valid:
            print_warning(error_msg)
            continue
        
        if query.lower() in ('quit', 'exit', 'q'):
            console.print("[muted]Goodbye.[/muted]")
            break
        
        if query.lower() == 'stats':
            stats = agent.get_stats()
            print_stats_table(stats)
            continue
        
        if query.lower() == 'models':
            models = OpenRouterClient.list_models()
            print_models_table(models)
            continue
        
        try:
            with create_status("Planning query..."):
                response = await agent.query(query, verbose=True)
            
            console.print()
            print_answer(response.answer, response.total_time_ms)
            
            if response.citations:
                print_citations(response.citations)
                
        except Exception as e:
            print_error(str(e))


async def single_query(agent: RAGAgent, query: str, verbose: bool = False, output_path: Path | None = None):
    """Run a single query with input validation."""
    # Validate query
    is_valid, error_msg = validate_query_input(query)
    if not is_valid:
        print_warning(error_msg)
        return
    
    logger.info(f"Processing query of length {len(query)} chars")
    
    with create_status("Processing query..."):
        response = await agent.query(query, verbose=verbose)
    
    console.print()
    print_answer(response.answer, response.total_time_ms)
    
    if response.citations:
        print_citations(response.citations)
    
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
            export_results_to_json(output_path, [result])
        elif ext == ".pdf":
            export_results_to_pdf(output_path, [result])
        else:
            # Default to CSV
            export_results_to_csv(output_path, [result])


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle --list-models
    if args.list_models:
        console.print("\n[accent]Available OpenRouter Models[/accent]")
        
        models = OpenRouterClient.list_models()
        categories = [
            ("Fast & Cheap", ["gemini-flash", "gpt-4o-mini", "claude-haiku", "llama-8b"]),
            ("Balanced", ["gpt-4o", "claude-sonnet", "gemini-pro", "llama-70b"]),
            ("Best Quality", ["claude-opus", "gpt-4-turbo", "llama-405b"]),
            ("Free (rate limited)", ["free"]),
        ]
        
        print_models_table(models, categories)
        return
    
    # Get LLM client
    llm_client = get_llm_client(provider=args.provider, model=args.model)
    
    provider_name = args.provider if args.provider != "auto" else None
    if llm_client:
        if hasattr(llm_client, 'provider'):
            provider_name = llm_client.provider
    
    if not llm_client:
        print_warning("No LLM configured - using heuristic mode")
        console.print("[muted]Set OPENROUTER_API_KEY for full functionality[/muted]")
    
    # Initialize agent
    agent = RAGAgent(llm_client=llm_client)
    
    # Check if we have any documents
    stats = agent.get_stats()
    if stats['document_count'] == 0:
        print_warning("No documents ingested yet")
        console.print("[muted]Run: python scripts/ingest.py <pdf_path>[/muted]")
        return
    
    # Run query or interactive mode
    if args.query:
        query = " ".join(args.query)
        await single_query(agent, query, args.verbose, args.output)
    else:
        await interactive_mode(agent, args.model, provider_name)


if __name__ == "__main__":
    asyncio.run(main())
