"""Main RAG agent that integrates all components."""

from pathlib import Path
from typing import Any
import asyncio

from .models import ExecutionPlan, QueryResponse, Document, TextChunk, ExtractedTable
from .config import config
from .agent.planner import Planner
from .agent.executor import DAGExecutor, ExecutionMonitor
from .agent.synthesizer import ResponseSynthesizer
from .tools.calculator import CalculatorTool
from .tools.sql_query import SQLQueryTool
from .tools.vector_search import VectorSearchTool
from .tools.get_document import GetDocumentTool
from .storage.sqlite_store import SQLiteStore
from .storage.chroma_store import ChromaStore
from .storage.document_store import DocumentStore
from .ingestion.pdf_parser import PDFParser
from .ingestion.table_extractor import TableExtractor
from .ingestion.vision_table_extractor import VisionTableExtractor
from .ingestion.chunker import SemanticChunker
from .ingestion.schema_detector import SchemaDetector
from .ingestion.spreadsheet_parser import SpreadsheetParser
from .ingestion.temporal_extractor import extract_temporal_metadata


class RAGAgent:
    """
    The main RAG agent that handles document ingestion and queries.
    
    Provides a simple interface for:
    - Ingesting PDF documents and spreadsheets (Excel/CSV)
    - Querying with natural language
    - Getting structured responses with citations
    """
    
    # Supported file types for ingestion
    SUPPORTED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.csv'}
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize the RAG agent.
        
        Args:
            llm_client: Optional LLM client for planning and synthesis.
                       If None, uses heuristic-based fallbacks.
        """
        self.llm_client = llm_client
        
        # Initialize storage
        self.sqlite_store = SQLiteStore()
        self.chroma_store = ChromaStore()
        self.document_store = DocumentStore()
        
        # Initialize tools
        self.calculator = CalculatorTool()
        self.sql_query = SQLQueryTool(
            sqlite_store=self.sqlite_store,
            llm_client=llm_client
        )
        self.vector_search = VectorSearchTool(chroma_store=self.chroma_store)
        self.get_document = GetDocumentTool(document_store=self.document_store)
        
        # Initialize agent components
        self.planner = Planner(llm_client=llm_client, sqlite_store=self.sqlite_store)
        self.executor = DAGExecutor(
            calculator=self.calculator,
            sql_query=self.sql_query,
            vector_search=self.vector_search,
            get_document=self.get_document
        )
        self.synthesizer = ResponseSynthesizer(llm_client=llm_client, document_store=self.document_store)
        
        # Ingestion components
        self.pdf_parser = PDFParser()
        self.spreadsheet_parser = SpreadsheetParser()
        self.table_extractor = TableExtractor()
        self.vision_table_extractor = VisionTableExtractor(llm_client=llm_client)
        self.chunker = SemanticChunker()
        self.schema_detector = SchemaDetector(llm_client=llm_client)
        
        # Vision table extraction enabled?
        self.use_vision_tables = config.use_vision_tables and llm_client is not None
    
    async def ingest_document(self, file_path: Path) -> Document | list[Document]:
        """
        Ingest a document (PDF or spreadsheet).
        
        For spreadsheets, each sheet becomes a separate document.
        
        Args:
            file_path: Path to the file (.pdf, .xlsx, .xls, or .csv)
            
        Returns:
            Document for PDFs, or list of Documents for spreadsheets (one per sheet)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        if ext == '.pdf':
            return await self._ingest_pdf(file_path)
        else:
            return await self._ingest_spreadsheet(file_path)
    
    async def _ingest_pdf(self, pdf_path: Path) -> Document:
        """
        Ingest a PDF document.
        
        Extracts:
        - Text content (chunked for vector search)
        - Tables (stored as structured data)
        - Metadata
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            The created Document record
        """
        # Parse PDF
        parsed = self.pdf_parser.parse(pdf_path)
        
        # Generate document ID
        doc_id = PDFParser.generate_document_id(pdf_path.name)
        
        # Extract temporal metadata from filename
        temporal_meta = extract_temporal_metadata(pdf_path.name, parsed.metadata)
        
        # Create document record with combined metadata
        doc = Document(
            id=doc_id,
            filename=pdf_path.name,
            title=parsed.metadata.get("title", pdf_path.stem),
            page_count=parsed.page_count,
            metadata={**parsed.metadata, **temporal_meta}
        )
        
        # Save document
        self.sqlite_store.save_document(doc)
        self.document_store.save_document(
            doc,
            source_path=pdf_path,
            full_text=parsed.full_text
        )
        
        # Extract and save tables
        tables = []
        
        if self.use_vision_tables:
            # Use vision-based extraction for better accuracy
            print(f"  Using vision LLM for table extraction...")
            try:
                tables = await self.vision_table_extractor.extract_tables_from_pdf(
                    pdf_path, doc_id
                )
            except Exception as e:
                print(f"  Vision extraction failed: {e}")
                print(f"  Falling back to rule-based extraction...")
                tables = []
        
        if not tables:
            # Fall back to rule-based extraction
            tables = self.table_extractor.extract_tables(parsed, doc_id)
        
        for table in tables:
            # Always enhance schema with LLM for better column/table names
            if self.llm_client:
                print(f"    Enhancing schema for table: {table.table_name[:40]}...")
                table = await self.schema_detector.detect_schema(table)
                print(f"    → Renamed to: {table.table_name}")
            self.sqlite_store.save_table(table)
        
        # Chunk text and store in vector DB
        chunks = self.chunker.chunk_document(parsed, doc_id)
        self.chroma_store.add_chunks(chunks)
        
        extraction_method = "vision" if self.use_vision_tables and tables else "rule-based"
        print(f"Ingested: {pdf_path.name}")
        print(f"  - {parsed.page_count} pages")
        print(f"  - {len(tables)} tables extracted ({extraction_method})")
        print(f"  - {len(chunks)} text chunks created")
        
        return doc
    
    async def _ingest_spreadsheet(self, file_path: Path) -> list[Document]:
        """
        Ingest a spreadsheet file (Excel or CSV).
        
        Each sheet becomes a separate document with its own ID.
        Data is stored both as SQL tables (for structured queries) and
        text chunks (for semantic search).
        
        Args:
            file_path: Path to .xlsx, .xls, or .csv file
            
        Returns:
            List of Document records (one per sheet)
        """
        parsed = self.spreadsheet_parser.parse(file_path)
        documents = []
        
        print(f"Parsing spreadsheet: {file_path.name}")
        
        for sheet in parsed.sheets:
            # Generate unique ID for this sheet
            doc_id = SpreadsheetParser.generate_document_id(
                file_path.name, 
                sheet.sheet_name
            )
            
            # Create document record
            doc = Document(
                id=doc_id,
                filename=file_path.name,
                title=f"{file_path.stem} - {sheet.sheet_name}",
                page_count=1,  # Sheets don't have pages
                metadata={
                    "source_type": "spreadsheet",
                    "sheet_name": sheet.sheet_name,
                    "row_count": sheet.row_count,
                    "col_count": sheet.col_count,
                    "columns": sheet.headers
                }
            )
            
            # Save document record
            self.sqlite_store.save_document(doc)
            self.document_store.save_document(
                doc,
                source_path=file_path,
                full_text=sheet.raw_text
            )
            
            # Save as native SQL table for fast queries
            table_name = SpreadsheetParser.sanitize_table_name(sheet.sheet_name)
            
            # Use native table storage for large spreadsheets (much faster)
            if sheet.dataframe is not None:
                self.sqlite_store.save_spreadsheet_native(
                    table_name=table_name,
                    df=sheet.dataframe,
                    doc_id=doc_id
                )
            else:
                # Fallback to EAV format for compatibility
                table = ExtractedTable(
                    id=f"{doc_id}_data",
                    document_id=doc_id,
                    table_name=table_name,
                    page_number=None,
                    schema_description=f"Data from '{sheet.sheet_name}' ({sheet.row_count} rows, {sheet.col_count} columns)",
                    columns=sheet.headers,
                    rows=sheet.rows
                )
                self.sqlite_store.save_table(table)
            
            # Create text chunk for vector search with spreadsheet-specific metadata
            # Note: For spreadsheets, we use section_title for sheet name and
            # start_line/end_line to indicate the row range in the sample
            sample_rows = min(25, sheet.row_count)  # We show first 25 rows in raw_text
            chunk = TextChunk(
                id=ChromaStore.generate_chunk_id(doc_id, 0),
                document_id=doc_id,
                content=sheet.raw_text,
                page_number=None,  # No page for spreadsheets
                section_title=sheet.sheet_name,  # Sheet name for section
                chunk_index=0,
                start_line=1,  # Row 1
                end_line=sample_rows,  # Up to 25 rows shown in sample
                metadata={
                    "source_type": "spreadsheet",
                    "total_rows": sheet.row_count,
                    "columns": sheet.headers[:5]  # First 5 columns for context
                }
            )
            self.chroma_store.add_chunks([chunk])
            
            documents.append(doc)
            print(f"  Sheet '{sheet.sheet_name}': {sheet.row_count} rows, {sheet.col_count} columns → table '{table_name}'")
        
        print(f"Ingested: {file_path.name}")
        print(f"  - {len(parsed.sheets)} sheets → {len(documents)} documents")
        print(f"  - {len(documents)} SQL tables created (native format)")
        print(f"  - {len(documents)} text chunks created")
        
        return documents
    
    async def query(self, query: str, verbose: bool = False) -> QueryResponse:
        """
        Query the document corpus.
        
        Args:
            query: Natural language query
            verbose: Whether to print execution details
            
        Returns:
            QueryResponse with answer and citations
        """
        # Get available data for planning
        tables = self.sqlite_store.list_tables()
        table_names = [t.table_name for t in tables]
        
        documents = self.document_store.list_documents()
        doc_ids = [d.id for d in documents]
        
        if verbose:
            print(f"Query: {query}")
            print(f"Available tables: {table_names}")
            print(f"Available documents: {len(doc_ids)}")
        
        # Create execution plan
        plan = await self.planner.create_plan(
            query=query,
            available_tables=table_names,
            available_documents=doc_ids
        )
        
        if verbose:
            print(f"\nExecution Plan ({len(plan.steps)} steps):")
            for step in plan.steps:
                deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
                print(f"  {step.id}: {step.tool.value} - {step.description}{deps}")
        
        # Execute plan
        if verbose:
            monitor = ExecutionMonitor()
            results, timing = await monitor.execute_with_monitoring(self.executor, plan)
            print(f"\nExecution completed in {timing['total_time_ms']:.1f}ms")
            for step_id, time_ms in timing['step_times_ms'].items():
                print(f"  {step_id}: {time_ms:.1f}ms")
        else:
            results = await self.executor.execute(plan)
        
        # Synthesize response
        response = await self.synthesizer.synthesize(plan, results)
        
        if verbose:
            print(f"\nTotal time: {response.total_time_ms:.1f}ms")
        
        return response
    
    def list_documents(self) -> list[Document]:
        """List all ingested documents."""
        return self.document_store.list_documents()
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge base."""
        docs = self.document_store.list_documents()
        tables = self.sqlite_store.list_tables()
        chunk_count = self.chroma_store.count()
        
        # Separate PDFs from spreadsheets
        pdf_docs = [d for d in docs if d.metadata.get("source_type") != "spreadsheet"]
        spreadsheet_docs = [d for d in docs if d.metadata.get("source_type") == "spreadsheet"]
        
        return {
            "document_count": len(docs),
            "pdf_count": len(pdf_docs),
            "spreadsheet_sheet_count": len(spreadsheet_docs),
            "table_count": len(tables),
            "chunk_count": chunk_count,
            "documents": [
                {
                    "id": d.id, 
                    "filename": d.filename, 
                    "pages": d.page_count,
                    "type": d.metadata.get("source_type", "pdf")
                }
                for d in docs
            ]
        }


# Convenience function for quick usage
async def query_documents(query: str, llm_client: Any = None) -> str:
    """
    Quick function to query documents.
    
    Args:
        query: Natural language query
        llm_client: Optional LLM client
        
    Returns:
        Answer string
    """
    agent = RAGAgent(llm_client=llm_client)
    response = await agent.query(query)
    return response.answer


# Run query from sync context
def query_sync(query: str, llm_client: Any = None) -> str:
    """Synchronous wrapper for query_documents."""
    return asyncio.run(query_documents(query, llm_client))
