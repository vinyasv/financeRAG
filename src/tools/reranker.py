"""FlashRank reranking for improved retrieval accuracy."""

import tempfile
from pathlib import Path
from typing import Any

from ..models import TextChunk


class Reranker:
    """
    Rerank retrieved chunks using FlashRank for higher precision.
    
    FlashRank is an ultra-fast, lightweight reranker using optimized
    distilled T5 models. Adds ~10-50ms latency for reranking.
    """
    
    def __init__(self, model_name: str = "rank-T5-flan"):
        """
        Initialize the reranker.
        
        Args:
            model_name: FlashRank model name. Options:
                - "rank-T5-flan" (default, fastest)
                - "rank_zephyr_7b_v1_full" (more accurate, slower)
        """
        self.model_name = model_name
        self._ranker = None
    
    def _ensure_model(self):
        """Lazy load the FlashRank model."""
        if self._ranker is None:
            from flashrank import Ranker
            cache_dir = Path(tempfile.gettempdir()) / "flashrank"
            self._ranker = Ranker(model_name=self.model_name, cache_dir=str(cache_dir))
    
    def rerank(
        self,
        query: str,
        chunks: list[TextChunk],
        top_k: int = 5
    ) -> list[tuple[TextChunk, float]]:
        """
        Rerank chunks by query-document relevance.
        
        Args:
            query: The search query
            chunks: List of TextChunk objects to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if not chunks:
            return []
        
        self._ensure_model()
        
        from flashrank import RerankRequest
        
        # Prepare passages for FlashRank
        passages = [
            {"id": i, "text": chunk.content}
            for i, chunk in enumerate(chunks)
        ]
        
        # Rerank
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)
        
        # Map back to chunks with scores
        ranked = []
        for result in results[:top_k]:
            idx = result["id"]
            score = result["score"]
            ranked.append((chunks[idx], score))
        
        return ranked
    
    def rerank_with_scores(
        self,
        query: str,
        chunks: list[TextChunk],
        scores: list[float],
        top_k: int = 5,
        vector_weight: float = 0.3
    ) -> list[tuple[TextChunk, float]]:
        """
        Rerank with combined vector + reranker scores.
        
        Args:
            query: The search query
            chunks: List of TextChunk objects
            scores: Original vector similarity scores
            top_k: Number of results to return
            vector_weight: Weight for original vector score (0-1)
            
        Returns:
            List of (chunk, combined_score) tuples
        """
        if not chunks:
            return []
        
        # Get reranker scores for all chunks
        reranked = self.rerank(query, chunks, top_k=len(chunks))
        
        # Create mapping of chunk id to reranker score
        rerank_scores = {id(c): s for c, s in reranked}
        
        # Normalize reranker scores to 0-1
        if reranked:
            score_values = [s for _, s in reranked]
            score_min, score_max = min(score_values), max(score_values)
            if score_max > score_min:
                rerank_scores_norm = {
                    cid: (s - score_min) / (score_max - score_min)
                    for cid, s in rerank_scores.items()
                }
            else:
                rerank_scores_norm = {cid: 0.5 for cid in rerank_scores}
        else:
            rerank_scores_norm = {}
        
        # Combine scores
        combined = []
        for chunk, vec_score in zip(chunks, scores):
            rerank_score = rerank_scores_norm.get(id(chunk), 0.0)
            final_score = vector_weight * vec_score + (1 - vector_weight) * rerank_score
            combined.append((chunk, final_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
