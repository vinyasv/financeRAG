"""Vector search tool for semantic document retrieval."""

from typing import Any

from .base import Tool
from ..models import ToolName, TextChunk
from ..storage.chroma_store import ChromaStore


class VectorSearchTool(Tool):
    """
    Semantic search over document text chunks.
    
    Use for finding relevant context, explanations, and qualitative information.
    Supports optional cross-encoder reranking for improved precision.
    """
    
    name = ToolName.VECTOR_SEARCH
    description = "Search document text semantically. Use for context, explanations, qualitative info."
    
    def __init__(
        self, 
        chroma_store: ChromaStore | None = None, 
        n_results: int = 5,
        use_reranking: bool = True
    ):
        self.chroma_store = chroma_store or ChromaStore()
        self.n_results = n_results
        self.use_reranking = use_reranking
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy-load reranker only when needed."""
        if self._reranker is None and self.use_reranking:
            try:
                from .reranker import Reranker
                self._reranker = Reranker()
            except ImportError:
                # sentence-transformers not available
                self._reranker = None
                self.use_reranking = False
        return self._reranker
    
    async def execute(self, input_str: str, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Search for relevant text chunks.
        
        Args:
            input_str: The search query
            context: Optional context (not used for this tool)
            
        Returns:
            List of relevant chunks with content and metadata
        """
        # Retrieve more candidates for reranking (3x the final count)
        retrieval_count = self.n_results * 3 if self.use_reranking else self.n_results
        
        # Perform search with lower threshold (reranking will refine)
        min_score = 0.2 if self.use_reranking else 0.3
        chunks, scores = self.chroma_store.search(
            query=input_str,
            n_results=retrieval_count,
            min_score=min_score
        )
        
        if not chunks:
            return []
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker and len(chunks) > self.n_results:
            reranked = self.reranker.rerank_with_scores(
                query=input_str,
                chunks=chunks,
                scores=scores,
                top_k=self.n_results,
                vector_weight=0.3  # 30% vector, 70% cross-encoder
            )
            chunks = [c for c, s in reranked]
            scores = [s for c, s in reranked]
        else:
            # Limit to n_results
            chunks = chunks[:self.n_results]
            scores = scores[:self.n_results]
        
        # Format results
        results = []
        for chunk, score in zip(chunks, scores):
            results.append({
                "content": chunk.content,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "relevance_score": round(score, 3)
            })
        
        return results
    
    def search_sync(self, query: str, n_results: int | None = None) -> list[TextChunk]:
        """
        Synchronous search for use outside async context.
        
        Args:
            query: The search query
            n_results: Optional override for number of results
            
        Returns:
            List of TextChunk objects
        """
        chunks, _ = self.chroma_store.search(
            query=query,
            n_results=n_results or self.n_results
        )
        return chunks


class MultiQuerySearch:
    """
    Search with multiple query variations for better recall.
    
    Generates query variations and merges results.
    """
    
    def __init__(self, chroma_store: ChromaStore | None = None):
        self.chroma_store = chroma_store or ChromaStore()
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        generate_variations: bool = True
    ) -> list[dict[str, Any]]:
        """
        Search with query expansion.
        
        Args:
            query: The base query
            n_results: Number of results per query
            generate_variations: Whether to generate query variations
            
        Returns:
            Merged and deduplicated results
        """
        queries = [query]
        
        if generate_variations:
            queries.extend(self._generate_variations(query))
        
        # Collect all results
        all_chunks: dict[str, tuple[TextChunk, float]] = {}
        
        for q in queries:
            chunks, scores = self.chroma_store.search(q, n_results=n_results)
            
            for chunk, score in zip(chunks, scores):
                # Keep highest score for each chunk
                if chunk.id not in all_chunks or all_chunks[chunk.id][1] < score:
                    all_chunks[chunk.id] = (chunk, score)
        
        # Sort by score and return top results
        sorted_results = sorted(
            all_chunks.values(),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]
        
        return [
            {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "relevance_score": round(score, 3)
            }
            for chunk, score in sorted_results
        ]
    
    def _generate_variations(self, query: str) -> list[str]:
        """Generate query variations for better recall."""
        variations = []
        
        # Add question form if not already
        if not query.endswith("?"):
            variations.append(query + "?")
        
        # Add "what is" prefix
        if not query.lower().startswith(("what", "how", "why", "when", "where")):
            variations.append(f"What is {query.lower()}?")
        
        return variations

