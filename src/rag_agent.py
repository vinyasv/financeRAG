"""Main RAG agent that integrates all components."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from .agent.executor import DAGExecutor, ExecutionMonitor
from .agent.planner import Planner
from .agent.synthesizer import ResponseSynthesizer
from .common.ids import chunk_id, document_id_from_filename, sheet_document_id
from .common.naming import sanitize_table_name
from .config import config
from .ingestion.chunker import SemanticChunker
from .ingestion.exceptions import ExtractionFailed
from .ingestion.pdf_parser import PDFParser
from .ingestion.schema_detector import SchemaDetector
from .ingestion.spreadsheet_parser import SpreadsheetParser
from .ingestion.temporal_extractor import extract_temporal_metadata
from .ingestion.vision_table_extractor import VisionTableExtractor
from .ingestion.vlm_extractor import VLMTableExtractor
from .models import Document, ExtractedTable, QueryResponse, TextChunk
from .storage.chroma_store import ChromaStore
from .storage.document_store import DocumentStore
from .storage.schema_cluster import SchemaClusterManager
from .storage.sqlite_store import SQLiteStore
from .tools.calculator import CalculatorTool
from .tools.get_document import GetDocumentTool
from .tools.sql_query import SQLQueryTool
from .tools.vector_search import VectorSearchTool

logger = logging.getLogger(__name__)


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
    
    # Number of sample rows to include in vector search chunk for spreadsheets
    SPREADSHEET_SAMPLE_ROWS = 25
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize the RAG agent.
        
        Args:
            llm_client: Optional LLM client for ingestion helpers.
                       Querying requires an LLM client.
        """
        self.llm_client = llm_client
        
        # Initialize storage
        self.sqlite_store = SQLiteStore()
        self.chroma_store = ChromaStore()
        self.document_store = DocumentStore()
        
        # Schema clustering for scalable context (must be before SQL tool)
        # Pass llm_client for dynamic company learning
        self.schema_cluster_manager = SchemaClusterManager(
            sqlite_store=self.sqlite_store,
            llm_client=llm_client
        )
        
        # Initialize tools
        self.calculator = CalculatorTool()
        self.sql_query = SQLQueryTool(
            sqlite_store=self.sqlite_store,
            llm_client=llm_client,
            schema_cluster_manager=self.schema_cluster_manager
        )
        self.vector_search = VectorSearchTool(chroma_store=self.chroma_store)
        self.get_document = GetDocumentTool(document_store=self.document_store)
        
        # Initialize agent components
        self.planner = Planner(
            llm_client=llm_client,
            sqlite_store=self.sqlite_store,
            company_registry=self.schema_cluster_manager.company_registry,
        )
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
        self.docling_extractor = VisionTableExtractor()  # Secondary - Local Docling
        self.vlm_extractor = VLMTableExtractor() # Primary - Cloud VLM
        self.chunker = SemanticChunker()
        self.schema_detector = SchemaDetector(llm_client=llm_client)

    def _get_or_create_document(
        self,
        doc_id: str,
        *,
        filename: str,
        title: str,
        page_count: int,
        metadata: dict[str, Any],
        source_path: Path,
        full_text: str,
        kind: str,
        display_name: str,
        force: bool = False,
        requires_table: bool = False,
    ) -> tuple[Document, bool]:
        """Load an existing document or create and persist it."""
        existing_doc = self.sqlite_store.get_document(doc_id)
        if existing_doc and not force and self._document_ingestion_complete(doc_id, requires_table=requires_table):
            logger.info(f"{kind} already ingested: {display_name} (id={doc_id}), skipping")
            return existing_doc, False
        if existing_doc:
            logger.info(f"{kind} requires re-ingestion: {display_name} (id={doc_id})")
            self.chroma_store.delete_document_chunks(doc_id)

        document = Document(
            id=doc_id,
            filename=filename,
            title=title,
            page_count=page_count,
            metadata=metadata,
        )
        self._persist_document_payload(document, source_path=source_path, full_text=full_text)
        return document, True

    def _document_ingestion_complete(self, doc_id: str, *, requires_table: bool) -> bool:
        """Return whether all persisted surfaces exist for a document."""
        if not self.document_store.get_full_text(doc_id):
            return False
        if self.chroma_store.count(document_id=doc_id) < 1:
            return False
        if requires_table and not self.sqlite_store.list_tables(document_id=doc_id):
            return False
        return True

    def _persist_document_payload(
        self,
        document: Document,
        *,
        source_path: Path,
        full_text: str,
    ) -> None:
        """Persist document metadata and extracted text."""
        self.sqlite_store.save_document(document)
        self.document_store.save_document(
            document,
            source_path=source_path,
            full_text=full_text,
        )

    async def _learn_company_safe(self, source_document: str) -> None:
        """Learn company metadata without breaking ingestion on failures."""
        try:
            await self.schema_cluster_manager.learn_company_from_document(source_document)
        except Exception as exc:
            logger.warning(f"Company learning failed for {source_document}: {exc}")

    async def _assign_cluster_safe(
        self,
        *,
        table_name: str,
        columns: list[str],
        schema_description: str,
        source_document: str,
        queryable_table_name: str | None = None,
    ) -> None:
        """Assign a table to a schema cluster without failing ingestion."""
        try:
            await self.schema_cluster_manager.assign_table(
                table_name=table_name,
                columns=columns,
                schema_description=schema_description,
                source_document=source_document,
                queryable_table_name=queryable_table_name,
            )
        except Exception as exc:
            logger.warning(f"Table clustering failed for {table_name}, skipping assignment: {exc}")

    def _index_chunk_batch(self, chunks: list[TextChunk]) -> None:
        """Persist a batch of chunks into the vector store."""
        if chunks:
            self.chroma_store.add_chunks(chunks)

    async def _extract_pdf_tables(self, parsed, pdf_path: Path, doc_id: str) -> list[ExtractedTable]:
        """Extract PDF tables using VLM first, then Docling local fallback."""
        tables: list[ExtractedTable] = []

        if config.openrouter_api_key:
            try:
                logger.info("Using VLM Table Extractor (Cloud)")
                tables = await self.vlm_extractor.extract_tables_from_pdf(pdf_path, doc_id)
            except ExtractionFailed as exc:
                logger.warning(f"VLM extraction failed: {exc}, falling back to Docling")

        if not tables:
            try:
                logger.info("Using Docling for table extraction (Local Fast Mode)")
                tables = await self.docling_extractor.extract_tables_from_pdf(pdf_path, doc_id)
            except ExtractionFailed as exc:
                logger.warning(f"Docling extraction failed: {exc}; continuing without PDF tables")
                tables = []

        if not tables:
            logger.info("No PDF tables extracted; continuing with text chunks only")

        return tables

    async def _process_pdf_tables(self, tables: list[ExtractedTable], source_document: str) -> None:
        """Enhance, persist, and cluster extracted PDF tables.

        Uses ``asyncio.gather(..., return_exceptions=True)`` so that a single
        bad table (e.g. invalid identifier, schema enhancement failure, or
        clustering crash) does not abort the rest of the batch. Each failure
        is logged with full traceback context and processing continues.
        """
        logger.info(f"Processing {len(tables)} tables concurrently...")
        semaphore = asyncio.Semaphore(10)

        async def process_table(table: ExtractedTable) -> None:
            async with semaphore:
                if self.llm_client:
                    try:
                        logger.debug(f"Enhancing schema for table: {(table.table_name or '')[:40]}")
                        table = await self.schema_detector.detect_schema(table)
                        logger.debug(f"Renamed to: {table.table_name}")
                    except Exception as exc:
                        logger.warning(f"Schema enhancement failed for {table.id}, using original: {exc}")

                native_table_name = self.sqlite_store.save_table(table)
                if native_table_name:
                    await self._assign_cluster_safe(
                        table_name=table.table_name,
                        columns=table.columns,
                        schema_description=table.schema_description,
                        source_document=source_document,
                        queryable_table_name=native_table_name,
                    )

        tasks = [process_table(table) for table in tables]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = 0
        for table, result in zip(tables, results):
            if isinstance(result, BaseException):
                failures += 1
                logger.exception(
                    "Table %s (from %s) failed during ingestion",
                    table.id,
                    source_document,
                    exc_info=result,
                )
        if failures:
            logger.warning(
                "Processed %d tables from %s with %d failures",
                len(tables),
                source_document,
                failures,
            )

    async def _save_spreadsheet_table(
        self,
        *,
        doc_id: str,
        source_document: str,
        sheet,
        table_name: str,
    ) -> None:
        """Persist spreadsheet data and its cluster assignment."""
        if sheet.dataframe is not None:
            self.sqlite_store.save_spreadsheet_native(
                table_name=table_name,
                df=sheet.dataframe,
                doc_id=doc_id,
            )
            schema_description = f"Spreadsheet data from {sheet.sheet_name}"
        else:
            table = ExtractedTable(
                id=f"{doc_id}_data",
                document_id=doc_id,
                table_name=table_name,
                page_number=None,
                schema_description=f"Data from '{sheet.sheet_name}' ({sheet.row_count} rows, {sheet.col_count} columns)",
                columns=sheet.headers,
                rows=sheet.rows,
            )
            self.sqlite_store.save_table(table)
            schema_description = table.schema_description

        await self._assign_cluster_safe(
            table_name=table_name,
            columns=sheet.headers,
            schema_description=schema_description,
            source_document=source_document,
        )

    def _build_spreadsheet_chunks(self, doc_id: str, sheet, rows_per_chunk: int = 100) -> list[TextChunk]:
        """Create row-window vector-search chunks for a spreadsheet sheet."""
        chunks: list[TextChunk] = []
        headers = sheet.headers
        for start in range(0, sheet.row_count, rows_per_chunk):
            end = min(start + rows_per_chunk, sheet.row_count)
            rows = sheet.rows[start:end]
            lines = [
                f"Spreadsheet Sheet: {sheet.sheet_name}",
                f"Rows: {start + 1}-{end} of {sheet.row_count}",
                f"Columns: {', '.join(headers)}",
                "",
            ]
            for row_number, row in enumerate(rows, start=start + 1):
                row_values = [
                    f"{header}: {row.get(header)}"
                    for header in headers
                    if row.get(header) is not None
                ]
                if row_values:
                    lines.append(f"Row {row_number}: {' | '.join(row_values)}")

            chunk_index = len(chunks)
            chunks.append(TextChunk(
                id=chunk_id(doc_id, chunk_index),
                document_id=doc_id,
                content="\n".join(lines),
                page_number=None,
                section_title=sheet.sheet_name,
                chunk_index=chunk_index,
                start_line=start + 1,
                end_line=end,
                metadata={
                    "source_type": "spreadsheet",
                    "total_rows": sheet.row_count,
                    "columns": headers[:5],
                },
            ))
        return chunks
    
    async def ingest_document(self, file_path: Path, force: bool = False) -> Document | list[Document]:
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
            return await self._ingest_pdf(file_path, force=force)
        else:
            return await self._ingest_spreadsheet(file_path, force=force)
    
    async def _ingest_pdf(self, pdf_path: Path, force: bool = False) -> Document:
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
        parsed = self.pdf_parser.parse(pdf_path)
        doc_id = document_id_from_filename(pdf_path.name)
        temporal_meta = extract_temporal_metadata(pdf_path.name, parsed.metadata)

        document, created = self._get_or_create_document(
            doc_id,
            filename=pdf_path.name,
            title=parsed.metadata.get("title", pdf_path.stem),
            page_count=parsed.page_count,
            metadata={**parsed.metadata, **temporal_meta},
            source_path=pdf_path,
            full_text=parsed.full_text,
            kind="Document",
            display_name=pdf_path.name,
            force=force,
            requires_table=False,
        )
        if not created:
            return document

        await self._learn_company_safe(pdf_path.name)
        tables = await self._extract_pdf_tables(parsed, pdf_path, doc_id)
        await self._process_pdf_tables(tables, pdf_path.name)
        chunks = self.chunker.chunk_document(parsed, doc_id)
        self._index_chunk_batch(chunks)

        logger.info(
            f"Ingested PDF: {pdf_path.name} - {parsed.page_count} pages, {len(tables)} tables, {len(chunks)} chunks"
        )

        return document
    
    async def _ingest_spreadsheet(self, file_path: Path, force: bool = False) -> list[Document]:
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
        
        logger.info(f"Parsing spreadsheet: {file_path.name}")
        await self._learn_company_safe(file_path.name)
        
        for sheet in parsed.sheets:
            doc_id = sheet_document_id(file_path.name, sheet.sheet_name)
            document, created = self._get_or_create_document(
                doc_id,
                filename=file_path.name,
                title=f"{file_path.stem} - {sheet.sheet_name}",
                page_count=1,
                metadata={
                    "source_type": "spreadsheet",
                    "sheet_name": sheet.sheet_name,
                    "row_count": sheet.row_count,
                    "col_count": sheet.col_count,
                    "columns": sheet.headers,
                },
                source_path=file_path,
                full_text=sheet.raw_text,
                kind="Sheet",
                display_name=sheet.sheet_name,
                force=force,
                requires_table=True,
            )
            if not created:
                documents.append(document)
                continue

            table_name = sanitize_table_name(sheet.sheet_name, numeric_prefix="sheet_")
            await self._save_spreadsheet_table(
                doc_id=doc_id,
                source_document=file_path.name,
                sheet=sheet,
                table_name=table_name,
            )
            self._index_chunk_batch(self._build_spreadsheet_chunks(doc_id, sheet))

            documents.append(document)
            logger.debug(f"Sheet '{sheet.sheet_name}': {sheet.row_count} rows, {sheet.col_count} cols -> table '{table_name}'")
        
        logger.info(f"Ingested spreadsheet: {file_path.name} - {len(parsed.sheets)} sheets, {len(documents)} tables")
        
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
        if not self.llm_client:
            raise RuntimeError("Querying requires an LLM client. Configure OPENROUTER_API_KEY.")

        tables = self.sqlite_store.list_tables()
        table_names = [t.table_name for t in tables]
        
        documents = self.document_store.list_documents()
        doc_ids = [d.id for d in documents]
        
        if verbose:
            logger.info(f"Query: {query}")
            logger.info(f"Available tables: {table_names}, documents: {len(doc_ids)}")
        
        # Create execution plan
        plan = await self.planner.create_plan(
            query=query,
            available_tables=table_names,
            available_documents=doc_ids
        )
        
        if verbose:
            logger.info(f"Execution Plan ({len(plan.steps)} steps):")
            for step in plan.steps:
                deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
                logger.info(f"  {step.id}: {step.tool.value} - {step.description}{deps}")
        
        # Execute plan
        if verbose:
            monitor = ExecutionMonitor()
            results, timing = await monitor.execute_with_monitoring(self.executor, plan)
            logger.info(f"Execution completed in {timing['total_time_ms']:.1f}ms")
            for step_id, time_ms in timing['step_times_ms'].items():
                logger.debug(f"  {step_id}: {time_ms:.1f}ms")
        else:
            results = await self.executor.execute(plan)
        
        # Synthesize response
        response = await self.synthesizer.synthesize(plan, results)
        
        if verbose:
            logger.info(f"Total time: {response.total_time_ms:.1f}ms")
        
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
            "schema_clusters": self.schema_cluster_manager.get_stats(),
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
