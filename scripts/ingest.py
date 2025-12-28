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

# Configure logging - write to data directory, not CWD
log_dir = config.data_path / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy loggers  
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback for tqdm if not installed."""
        return iterable

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
        print(f"❌ Folder not found: {folder}")
        return []
    
    if not folder.is_dir():
        print(f"❌ Not a directory: {folder}")
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
        print(f"📁 Discovered {len(discovered)} files in {args.folder}")
    
    # From explicit file arguments
    for file_path in args.files:
        path = Path(file_path)
        if path.exists():
            files_to_process.append(path)
        else:
            print(f"⚠️  File not found: {path}")
    
    if not files_to_process:
        print("No files to ingest!")
        print("\nUsage:")
        print("  python ingest.py <file_path> [file_path2] ...")
        print("  python ingest.py -f <folder_path>")
        print("\nSupported file types:")
        print("  - PDF documents (.pdf)")
        print("  - Excel workbooks (.xlsx, .xls)")
        print("  - CSV files (.csv)")
        sys.exit(1)
    
    # Initialize LLM client for vision table extraction
    llm_client = get_llm_client()
    if llm_client and config.use_vision_tables:
        print(f"Vision table extraction: enabled (model: {config.vision_model})")
    else:
        print("Vision table extraction: disabled (no LLM client or USE_VISION_TABLES=false)")
    print()
    
    # Initialize agent with LLM
    agent = RAGAgent(llm_client=llm_client)
    
    # Track ingestion results
    ingested_files = []
    skipped_files = []
    failed_files = []
    
    # Create progress bar
    show_progress = HAS_TQDM and not args.no_progress and len(files_to_process) > 1
    
    # Process each file
    iterator = tqdm(
        files_to_process,
        desc="Ingesting",
        unit="file",
        disable=not show_progress
    )
    
    for path in iterator:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped_files.append((path.name, f"unsupported type {ext}"))
            continue
        
        # Validate file before ingestion (SEC-006)
        is_valid, error = validate_file_for_ingestion(path)
        if not is_valid:
            skipped_files.append((path.name, error))
            logger.warning(f"Skipping invalid file: {error}")
            continue
        
        # Update progress bar description
        if show_progress:
            iterator.set_postfix_str(path.name[:30])
        
        logger.info(f"Ingesting file: {path.name} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
        
        try:
            result = await agent.ingest_document(path)
            
            # Handle both single doc (PDF) and list of docs (spreadsheet)
            if isinstance(result, list):
                ingested_files.append((path.name, f"{len(result)} sheets"))
            else:
                ingested_files.append((path.name, "1 document"))
            
            if not show_progress:
                print(f"✅ {path.name}")
            
        except Exception as e:
            failed_files.append((path.name, str(e)))
            if not show_progress:
                print(f"❌ {path.name}: {e}")
    
    # Print summary
    stats = agent.get_stats()
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Results: {len(ingested_files)} ingested, {len(skipped_files)} skipped, {len(failed_files)} failed")
    
    if ingested_files and len(ingested_files) <= 10:
        print("\n✅ Successfully ingested:")
        for name, detail in ingested_files:
            print(f"   • {name} ({detail})")
    elif ingested_files:
        print(f"\n✅ Successfully ingested {len(ingested_files)} files")
    
    if skipped_files:
        print("\n⚠️  Skipped:")
        for name, reason in skipped_files[:5]:
            print(f"   • {name} ({reason})")
        if len(skipped_files) > 5:
            print(f"   ... and {len(skipped_files) - 5} more")
    
    if failed_files:
        print("\n❌ Failed:")
        for name, error in failed_files[:5]:
            print(f"   • {name}: {error}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
    
    print("\n" + "-" * 60)
    print("KNOWLEDGE BASE STATISTICS")
    print("-" * 60)
    print(f"  Total Documents: {stats['document_count']}")
    print(f"    - PDFs: {stats['pdf_count']}")
    print(f"    - Spreadsheet Sheets: {stats['spreadsheet_sheet_count']}")
    print(f"  SQL Tables: {stats['table_count']}")
    print(f"  Text Chunks: {stats['chunk_count']}")


if __name__ == "__main__":
    asyncio.run(main())
