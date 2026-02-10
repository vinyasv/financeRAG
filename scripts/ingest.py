#!/usr/bin/env python3
"""CLI script for ingesting documents (PDFs and spreadsheets)."""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env first
from src.config import config
from src.rag_agent import RAGAgent
from src.llm_client import get_llm_client
from src.security import validate_file_size, validate_path_safety
from src.ui import (
    console,
    print_ingestion_summary,
    print_error,
    print_warning,
    print_success,
    print_info,
    create_progress,
)

# Configure logging - write to data directory, not CWD
log_dir = config.data_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'ingest.log'),
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy loggers  
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.csv'}

# Maximum file size (500MB)
MAX_FILE_SIZE_MB = 500


def validate_file_for_ingestion(file_path: Path) -> tuple[bool, str]:
    """
    Validate a file is safe and appropriate for ingestion.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {file_path.suffix}"
    
    # Check path safety (no traversal)
    is_safe, error = validate_path_safety(str(file_path.name), SUPPORTED_EXTENSIONS)
    if not is_safe:
        return False, f"Path security issue: {error}"
    
    # Check file size
    try:
        size_bytes = file_path.stat().st_size
        is_valid_size, error = validate_file_size(size_bytes, MAX_FILE_SIZE_MB)
        if not is_valid_size:
            return False, error
    except OSError as e:
        return False, f"Cannot read file: {e}"
    
    return True, ""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Finance RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    # Ingest single file
  %(prog)s doc1.pdf doc2.pdf               # Ingest multiple files
  %(prog)s -f /path/to/folder              # Ingest all files in folder
  %(prog)s -f ./reports --pattern "*.pdf"  # Only PDFs from folder
  %(prog)s *.pdf *.xlsx                    # Shell glob expansion
"""
    )
    
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to ingest"
    )
    
    parser.add_argument(
        "-f", "--folder",
        type=Path,
        help="Recursively ingest all supported files from folder"
    )
    
    parser.add_argument(
        "--pattern",
        default="*",
        help="File pattern for folder mode (default: '*' for all supported types)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    
    return parser.parse_args()


def discover_files(folder: Path, pattern: str = "*") -> list[Path]:
    """
    Recursively discover all supported files in a folder.
    
    Args:
        folder: Root folder to search
        pattern: Glob pattern to filter files (e.g., "*.pdf")
        
    Returns:
        List of file paths
    """
    if not folder.exists():
        print_error(f"Folder not found: {folder}")
        return []
    
    if not folder.is_dir():
        print_error(f"Not a directory: {folder}")
        return []
    
    files = []
    
    # If pattern is *, find all supported extensions
    if pattern == "*":
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(folder.rglob(f"*{ext}"))
    else:
        # Use the specific pattern
        for path in folder.rglob(pattern):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
    
    # Sort for consistent ordering
    files.sort()
    return files


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Collect files to process
    files_to_process: list[Path] = []
    
    # From folder mode
    if args.folder:
        discovered = discover_files(args.folder, args.pattern)
        files_to_process.extend(discovered)
        print_info(f"Discovered {len(discovered)} files in {args.folder}")
    
    # From explicit file arguments
    for file_path in args.files:
        path = Path(file_path)
        if path.exists():
            files_to_process.append(path)
        else:
            print_warning(f"File not found: {path}")
    
    if not files_to_process:
        print_error("No files to ingest")
        console.print()
        console.print("[muted]Usage:[/muted]")
        console.print("  python ingest.py <file_path> [file_path2] ...")
        console.print("  python ingest.py -f <folder_path>")
        console.print()
        console.print("[muted]Supported file types:[/muted]")
        console.print("  - PDF documents (.pdf)")
        console.print("  - Excel workbooks (.xlsx, .xls)")
        console.print("  - CSV files (.csv)")
        sys.exit(1)
    
    # Initialize LLM client
    llm_client = get_llm_client()
    
    # Docling table extraction status
    print_info("Table extraction: Docling (local, free)")
    console.print()
    
    # Initialize agent with LLM
    agent = RAGAgent(llm_client=llm_client)
    
    # Track ingestion results
    ingested_files = []
    skipped_files = []
    failed_files = []
    
    # Create progress bar
    show_progress = not args.no_progress and len(files_to_process) > 1
    
    if show_progress:
        with create_progress() as progress:
            task = progress.add_task("Ingesting files...", total=len(files_to_process))
            
            for path in files_to_process:
                ext = path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    skipped_files.append((path.name, f"unsupported type {ext}"))
                    progress.advance(task)
                    continue
                
                # Validate file before ingestion
                is_valid, error = validate_file_for_ingestion(path)
                if not is_valid:
                    skipped_files.append((path.name, error))
                    logger.warning(f"Skipping invalid file: {error}")
                    progress.advance(task)
                    continue
                
                # Update progress description
                progress.update(task, description=f"Processing {path.name[:40]}...")
                
                logger.info(f"Ingesting file: {path.name} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
                
                try:
                    result = await agent.ingest_document(path)
                    
                    # Handle both single doc (PDF) and list of docs (spreadsheet)
                    if isinstance(result, list):
                        ingested_files.append((path.name, f"{len(result)} sheets"))
                    else:
                        ingested_files.append((path.name, "1 document"))
                    
                except Exception as e:
                    failed_files.append((path.name, str(e)))
                
                progress.advance(task)
    else:
        # Single file mode - no progress bar
        for path in files_to_process:
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped_files.append((path.name, f"unsupported type {ext}"))
                continue
            
            # Validate file before ingestion
            is_valid, error = validate_file_for_ingestion(path)
            if not is_valid:
                skipped_files.append((path.name, error))
                logger.warning(f"Skipping invalid file: {error}")
                continue
            
            logger.info(f"Ingesting file: {path.name} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
            console.print(f"[muted]Processing:[/muted] {path.name}")
            
            try:
                result = await agent.ingest_document(path)
                
                # Handle both single doc (PDF) and list of docs (spreadsheet)
                if isinstance(result, list):
                    ingested_files.append((path.name, f"{len(result)} sheets"))
                else:
                    ingested_files.append((path.name, "1 document"))
                
                print_success(f"Ingested: {path.name}")
                
            except Exception as e:
                failed_files.append((path.name, str(e)))
                print_error(f"{path.name}: {e}")
    
    # Print summary
    stats = agent.get_stats()
    print_ingestion_summary(ingested_files, skipped_files, failed_files, stats)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Ingestion cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error during ingestion:[/bold red] {e}")
        sys.exit(1)
