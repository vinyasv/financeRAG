"""
VLM-based table extraction using OpenRouter (GPT-4o, Gemini, etc).

This module provides a fast, cloud-based alternative to local table extraction.
It converts PDF pages to images and sends them to a Vision Language Model
to extract tabular data in structured JSON format.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import fitz  # PyMuPDF

from ..common.json_utils import parse_json_response
from ..common.naming import sanitize_table_name
from ..config import config
from ..models import ExtractedTable
from .exceptions import ExtractionFailed

logger = logging.getLogger(__name__)

# Fraction of failed pages at or above which the whole VLM extraction is
# treated as failed, so the caller can fall back to local Docling extraction.
# Set below 1.0 deliberately: returning one table salvaged from a 20-page
# filing looks like success to the caller and permanently suppresses the
# fallback. A third is tolerant of a few transient 429s while still catching
# a systemic outage.
MAX_PAGE_FAILURE_RATIO = 1 / 3

class VLMTableExtractor:
    """
    Extract tables from PDFs using external VLM API (via OpenRouter).
    
    Features:
    - High concurrency for speed (sub-5s processing potential)
    - Uses cheap/fast models (GPT-4o-mini, Gemini Flash)
    - Robust to complex layouts by using vision capabilities
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "google/gemini-2.5-flash-lite",
        concurrency: int = 20
    ):
        """
        Initialize VLM extractor.
        
        Args:
            api_key: OpenRouter API key. Defaults to config value.
            model: Model ID to use (e.g., "openai/gpt-4o-mini")
            concurrency: Max number of concurrent API requests
        """
        self.api_key = api_key or config.openrouter_api_key
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("VLMTableExtractor initialized without API key. Extraction will fail.")

    async def extract_tables_from_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        max_tables: int = 100
    ) -> List[ExtractedTable]:
        """
        Extract tables from a PDF using VLM.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Document ID for metadata
            max_tables: Limit on tables (soft limit here)

        Returns:
            List of ExtractedTable objects

        Raises:
            ExtractionFailed: only if every page failed (so the caller can fall
                back to local Docling extraction). A PDF whose pages all succeed
                but contain no tables returns ``[]`` without raising.
        """
        if not self.api_key:
            logger.error("No OpenRouter API key found. Cannot perform VLM extraction.")
            raise ExtractionFailed("No OpenRouter API key configured")

        try:
            with fitz.open(pdf_path) as doc:
                tasks = []
                page_count = len(doc)

                logger.info(f"Starting VLM extraction for {pdf_path.name} ({page_count} pages) using {self.model}")

                # Create extraction task for each page
                for page_num in range(page_count):
                    tasks.append(self._process_page(doc, page_num, document_id))

                # Run all pages concurrently (rate limited by semaphore).
                # return_exceptions=True mirrors RAGAgent._process_pdf_tables:
                # one transient per-page failure (429, timeout, bad JSON) must
                # not discard the pages that succeeded.
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results, skipping (and logging) failed pages
            all_tables = []
            failures = 0
            for page_num, page_tables in enumerate(results):
                if isinstance(page_tables, BaseException):
                    # Cancellation is control flow, not a page error. Counting
                    # it as a recoverable failure would swallow the signal and
                    # let a cancelled extraction report partial success.
                    if isinstance(page_tables, asyncio.CancelledError):
                        raise page_tables
                    failures += 1
                    # Logged once, here, at WARNING: _process_page already
                    # records the detail, and the caller treats a sub-threshold
                    # failure as recoverable, so ERROR+traceback per page would
                    # emit 500 tracebacks for one outage on a 500-page filing.
                    logger.warning(
                        "VLM extraction failed for page %d of %s: %s",
                        page_num + 1,
                        pdf_path.name,
                        page_tables,
                    )
                    continue
                all_tables.extend(page_tables)

            # Partial-failure policy. Returning whatever survived is wrong past
            # a point: RAGAgent._extract_pdf_tables only falls back to Docling
            # on ExtractionFailed or an empty list, and _get_or_create_document
            # then marks the document ingested -- so a 19-of-20-page failure
            # would silently persist a near-empty extraction that a later
            # ingest run skips entirely without --force. Above the threshold we
            # raise so the local extractor gets its turn.
            if failures:
                failure_ratio = failures / page_count
                if failure_ratio > MAX_PAGE_FAILURE_RATIO:
                    raise ExtractionFailed(
                        f"{failures}/{page_count} pages failed VLM extraction for "
                        f"{pdf_path.name} (over the {MAX_PAGE_FAILURE_RATIO:.0%} threshold)"
                    )
                logger.warning(
                    "VLM extraction for %s completed with %d/%d failed pages; "
                    "returning partial results",
                    pdf_path.name,
                    failures,
                    page_count,
                )

            logger.info(f"VLM extracted {len(all_tables)} tables from {pdf_path.name}")
            return all_tables[:max_tables]

        except Exception as e:
            logger.error(f"VLM extraction failed for {pdf_path.name}: {e}")
            if isinstance(e, ExtractionFailed):
                raise
            raise ExtractionFailed(f"VLM extraction failed for {pdf_path.name}: {e}") from e

    async def _process_page(self, doc, page_num: int, document_id: str) -> List[ExtractedTable]:
        """Process a single page: Convert to image -> Call VLM -> Parse JSON."""
        async with self.semaphore:
            try:
                # 1. Convert page to image
                # Using 150 DPI is a good balance for VLM resolution vs token cost/speed
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                b64_img = base64.b64encode(img_data).decode("utf-8")
                
                # 2. Call VLM API
                raw_response = await self._call_vlm_api(b64_img)
                
                # 3. Parse Response
                tables = self._parse_response(raw_response, page_num + 1, document_id)
                return tables
                
            except Exception as e:
                logger.warning(f"Failed to process page {page_num + 1}: {e}")
                raise ExtractionFailed(f"Failed to process page {page_num + 1}: {e}") from e

    async def _call_vlm_api(self, b64_img: str) -> Optional[Dict]:
        """Send image to OpenRouter API."""
        prompt = """
        Analyze this document page image. Identify any data tables.
        If there are tables, extract them into a JSON format.
        
        Return a JSON object with this exact structure:
        {
            "tables": [
                {
                    "name": "Table Title or Brief Description",
                    "columns": ["Col1", "Col2", ...],
                    "rows": [
                        {"Col1": "Val1", "Col2": "Val2", ...},
                        ...
                    ]
                }
            ]
        }
        
        Rules:
        1. If no tables are present, return {"tables": []}.
        2. Ensure all keys in 'rows' match the 'columns' list exactly.
        3. Output ONLY valid JSON. Do not include markdown formatting like ```json.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ultimaterag.local", 
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0 # Deterministic
        }
        
        timeout = aiohttp.ClientTimeout(total=45) # 45s timeout per page
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.base_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    # Log first 200 chars to avoid huge HTML dumps
                    logger.error(f"VLM API Error {resp.status}: {text[:200]}...")
                    raise ExtractionFailed(f"VLM API returned status {resp.status}")
                    
                result = await resp.json()
                content = result['choices'][0]['message']['content']
                
                return parse_json_response(content)

    def _parse_response(self, data: Optional[Dict], page_number: int, document_id: str) -> List[ExtractedTable]:
        """Convert VLM JSON response to ExtractedTable objects."""
        if not data or 'tables' not in data:
            return []
            
        extracted = []
        for i, t in enumerate(data['tables']):
            try:
                # Generate a unique ID
                columns = t.get('columns', [])
                rows = t.get('rows', [])
                name = t.get('name', f"Table {i+1}")
                
                if not columns or not rows:
                    continue
                    
                table_id = f"{document_id}_p{page_number}_t{i}_{len(rows)}"
                
                # Generate description
                desc = f"Table '{name}' on page {page_number}. Columns: {', '.join(columns)}"
                
                # Simple raw text representation
                raw_text = f"| {' | '.join(columns)} |\n"
                raw_text += f"| {' | '.join(['---']*len(columns))} |\n"
                for row in rows:
                    row_vals = [str(row.get(c, '')) for c in columns]
                    raw_text += f"| {' | '.join(row_vals)} |\n"
                
                # Sanitize at the extractor boundary so downstream
                # validate_identifier (called from save_table) cannot
                # be tripped by free-form VLM names like "Q1 FY26 Revenue".
                safe_name = sanitize_table_name(name)
                table = ExtractedTable(
                    id=table_id,
                    document_id=document_id,
                    table_name=safe_name,
                    page_number=page_number,
                    schema_description=desc,
                    columns=columns,
                    rows=rows,
                    raw_text=raw_text
                )
                extracted.append(table)
            except Exception as e:
                logger.warning(f"Error parsing table {i} on page {page_number}: {e}")
                continue
                
        return extracted
