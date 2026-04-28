"""Semantic text chunking for RAG."""

import re
from dataclasses import dataclass

from ..common.ids import chunk_id
from ..models import TextChunk
from .pdf_parser import ParsedPDF


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
            
            page_start_line = page.start_line_offset + 1  # 1-indexed
            
            paragraphs = self._split_into_paragraphs_with_lines(page.text)
            
            # Process paragraphs into chunks
            current_chunk_text = ""
            current_section = None
            chunk_start_line = page_start_line
            chunk_end_line = page_start_line
            
            for para, para_start_offset, para_end_offset in paragraphs:
                para_start_line = page_start_line + para_start_offset
                para_end_line = page_start_line + para_end_offset
                
                # Detect section headers
                if self._is_section_header(para):
                    # Save current chunk if any
                    if current_chunk_text.strip():
                        chunk_end_line = max(chunk_start_line, para_start_line - 1)
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
                    chunk_start_line = para_end_line + 1
                    continue
                
                # Check if adding this paragraph exceeds max size
                word_count = len((current_chunk_text + " " + para).split())
                
                if word_count > self.config.max_words:
                    # Save current chunk
                    if current_chunk_text.strip():
                        chunks.append(self._create_chunk(
                            document_id=document_id,
                            content=current_chunk_text.strip(),
                            page_number=page.page_number,
                            section_title=current_section,
                            chunk_index=chunk_index,
                            start_line=chunk_start_line,
                            end_line=max(chunk_start_line, chunk_end_line)
                        ))
                        chunk_index += 1
                    
                    # Handle large paragraph
                    if len(para.split()) > self.config.max_words:
                        # Split paragraph into sentences
                        para_chunks = self._split_large_text_with_line_ranges(
                            para,
                            para_start_line,
                            para_end_line,
                        )

                        for para_chunk, split_start_line, split_end_line in para_chunks:
                            chunks.append(self._create_chunk(
                                document_id=document_id,
                                content=para_chunk.strip(),
                                page_number=page.page_number,
                                section_title=current_section,
                                chunk_index=chunk_index,
                                start_line=split_start_line,
                                end_line=split_end_line,
                            ))
                            chunk_index += 1
                        current_chunk_text = ""
                        chunk_start_line = para_end_line + 1
                        chunk_end_line = para_end_line
                    else:
                        # Start new chunk with overlap
                        overlap = self._get_overlap(current_chunk_text)
                        current_chunk_text = overlap + " " + para if overlap else para
                        chunk_start_line = max(page_start_line, chunk_end_line) if overlap else para_start_line
                        chunk_end_line = para_end_line
                else:
                    # Add to current chunk
                    current_chunk_text = current_chunk_text + "\n\n" + para if current_chunk_text else para
                    chunk_end_line = para_end_line
            # Save remaining chunk for this page
            if current_chunk_text.strip():
                chunks.append(self._create_chunk(
                    document_id=document_id,
                    content=current_chunk_text.strip(),
                    page_number=page.page_number,
                    section_title=current_section,
                    chunk_index=chunk_index,
                    start_line=chunk_start_line,
                    end_line=max(chunk_start_line, chunk_end_line)
                ))
                chunk_index += 1
        
        return chunks
    
    def _split_into_paragraphs_with_lines(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into paragraphs with 0-indexed start/end line offsets."""
        paragraphs: list[tuple[str, int, int]] = []
        current: list[str] = []
        start_line = 0

        for line_no, line in enumerate(text.splitlines()):
            if line.strip():
                if not current:
                    start_line = line_no
                current.append(line)
            elif current:
                paragraphs.append(("\n".join(current).strip(), start_line, line_no - 1))
                current = []

        if current:
            paragraphs.append(("\n".join(current).strip(), start_line, len(text.splitlines()) - 1))

        return paragraphs
    
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

    def _split_large_text_with_line_ranges(
        self,
        text: str,
        start_line: int,
        end_line: int,
    ) -> list[tuple[str, int, int]]:
        """Split large text and distribute source line ranges across chunks."""
        chunks = self._split_large_text(text)
        if not chunks:
            return []

        total_lines = max(1, end_line - start_line + 1)
        ranged_chunks = []
        for index, chunk in enumerate(chunks):
            split_start = start_line + (index * total_lines) // len(chunks)
            split_end = start_line + ((index + 1) * total_lines) // len(chunks) - 1
            ranged_chunks.append((chunk, split_start, max(split_start, split_end)))
        return ranged_chunks
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if self.config.overlap_words <= 0:
            return ""

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
        generated_chunk_id = chunk_id(document_id, chunk_index)
        
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
            id=generated_chunk_id,
            document_id=document_id,
            content=contextual_content,
            page_number=page_number,
            section_title=section_title,
            chunk_index=chunk_index,
            start_line=start_line,
            end_line=end_line
        )
