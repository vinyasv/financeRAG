"""Semantic text chunking for RAG."""

import re
from dataclasses import dataclass

from ..models import TextChunk
from .pdf_parser import ParsedPDF
from .utils import generate_chunk_id


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    max_chunk_size: int = 500  # tokens (approximated as words * 1.3)
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    
    @property
    def max_words(self) -> int:
        """Approximate max words (tokens / 1.3)."""
        return int(self.max_chunk_size / 1.3)
    
    @property
    def overlap_words(self) -> int:
        """Approximate overlap words."""
        return int(self.chunk_overlap / 1.3)


class SemanticChunker:
    """
    Chunk documents semantically based on structure.
    
    Strategy:
    1. Split on paragraph/section boundaries
    2. Merge small chunks
    3. Split large chunks at sentence boundaries
    4. Add overlap between chunks
    """
    
    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
    
    def chunk_document(
        self,
        parsed_pdf: ParsedPDF,
        document_id: str
    ) -> list[TextChunk]:
        """
        Chunk a parsed PDF into text chunks.
        
        Args:
            parsed_pdf: The parsed PDF
            document_id: ID of the document
            
        Returns:
            List of text chunks
        """
        chunks = []
        chunk_index = 0
        
        for page in parsed_pdf.pages:
            if not page.text.strip():
                continue
            
            # Track line positions for this page
            page_start_line = page.start_line_offset + 1  # 1-indexed
            current_line_in_page = 1
            
            # Split page into paragraphs
            paragraphs = self._split_into_paragraphs(page.text)
            
            # Process paragraphs into chunks
            current_chunk_text = ""
            current_section = None
            chunk_start_line = page_start_line
            
            for para in paragraphs:
                # Calculate line number for this paragraph
                para_line_count = para.count('\n') + 1
                para_start_line = page_start_line + current_line_in_page - 1
                
                # Detect section headers
                if self._is_section_header(para):
                    # Save current chunk if any
                    if current_chunk_text.strip():
                        chunk_end_line = para_start_line - 1
                        chunks.append(self._create_chunk(
                            document_id=document_id,
                            content=current_chunk_text.strip(),
                            page_number=page.page_number,
                            section_title=current_section,
                            chunk_index=chunk_index,
                            start_line=chunk_start_line,
                            end_line=chunk_end_line
                        ))
                        chunk_index += 1
                        current_chunk_text = ""
                    
                    current_section = para.strip()
                    current_line_in_page += para_line_count + 1  # +1 for paragraph gap
                    chunk_start_line = page_start_line + current_line_in_page - 1
                    continue
                
                # Check if adding this paragraph exceeds max size
                word_count = len((current_chunk_text + " " + para).split())
                
                if word_count > self.config.max_words:
                    # Save current chunk
                    if current_chunk_text.strip():
                        chunk_end_line = para_start_line - 1
                        chunks.append(self._create_chunk(
                            document_id=document_id,
                            content=current_chunk_text.strip(),
                            page_number=page.page_number,
                            section_title=current_section,
                            chunk_index=chunk_index,
                            start_line=chunk_start_line,
                            end_line=chunk_end_line
                        ))
                        chunk_index += 1
                    
                    # Handle large paragraph
                    if len(para.split()) > self.config.max_words:
                        # Split paragraph into sentences
                        para_chunks = self._split_large_text(para)
                        lines_per_chunk = max(1, para_line_count // len(para_chunks))
                        chunk_line = para_start_line
                        
                        for para_chunk in para_chunks:
                            chunks.append(self._create_chunk(
                                document_id=document_id,
                                content=para_chunk.strip(),
                                page_number=page.page_number,
                                section_title=current_section,
                                chunk_index=chunk_index,
                                start_line=chunk_line,
                                end_line=chunk_line + lines_per_chunk - 1
                            ))
                            chunk_index += 1
                            chunk_line += lines_per_chunk
                        current_chunk_text = ""
                        chunk_start_line = para_start_line + para_line_count
                    else:
                        # Start new chunk with overlap
                        overlap = self._get_overlap(current_chunk_text)
                        current_chunk_text = overlap + " " + para if overlap else para
                        chunk_start_line = para_start_line
                else:
                    # Add to current chunk
                    current_chunk_text = current_chunk_text + "\n\n" + para if current_chunk_text else para
                
                current_line_in_page += para_line_count + 1  # +1 for paragraph gap
            
            # Save remaining chunk for this page
            if current_chunk_text.strip():
                chunk_end_line = page_start_line + page.line_count - 1
                chunks.append(self._create_chunk(
                    document_id=document_id,
                    content=current_chunk_text.strip(),
                    page_number=page.page_number,
                    section_title=current_section,
                    chunk_index=chunk_index,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line
                ))
                chunk_index += 1
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or multiple whitespace
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up and filter empty
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header."""
        text = text.strip()
        
        # Short text that's all caps
        if len(text) < 100 and text.isupper():
            return True
        
        # Starts with number followed by text (like "1. Introduction")
        if re.match(r'^\d+\.?\s+\w', text) and len(text) < 100:
            return True
        
        # Markdown-style headers
        if text.startswith('#'):
            return True
        
        return False
    
    def _split_large_text(self, text: str) -> list[str]:
        """Split large text into smaller chunks at sentence boundaries."""
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk.split()) > self.config.max_words:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        words = text.split()
        if len(words) <= self.config.overlap_words:
            return text
        
        return " ".join(words[-self.config.overlap_words:])
    
    def _create_chunk(
        self,
        document_id: str,
        content: str,
        page_number: int,
        section_title: str | None,
        chunk_index: int,
        start_line: int | None = None,
        end_line: int | None = None
    ) -> TextChunk:
        """Create a TextChunk instance with contextual prefix for better embedding."""
        chunk_id = generate_chunk_id(document_id, chunk_index)
        
        # Add contextual prefix for better embedding quality
        # This helps the embedding model understand the context
        prefix_parts = []
        if section_title:
            prefix_parts.append(f"Section: {section_title}")
        prefix_parts.append(f"Page {page_number}")
        if start_line and end_line:
            prefix_parts.append(f"Lines {start_line}-{end_line}")
        
        # Create contextual content
        if prefix_parts:
            contextual_content = f"{' | '.join(prefix_parts)}\n\n{content}"
        else:
            contextual_content = content
        
        return TextChunk(
            id=chunk_id,
            document_id=document_id,
            content=contextual_content,
            page_number=page_number,
            section_title=section_title,
            chunk_index=chunk_index,
            start_line=start_line,
            end_line=end_line
        )

