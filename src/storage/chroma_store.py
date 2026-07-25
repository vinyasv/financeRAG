"""ChromaDB storage for text embeddings and vector search."""

import os
import threading
from pathlib import Path

from ..config import config
from ..models import TextChunk

# Chroma enforces a maximum batch size per add(); stay comfortably under it.
MAX_ADD_BATCH = 4000


class ChromaStore:
    """
    ChromaDB-based vector storage for text chunks.
    
    Provides:
    - Storage of text chunks with embeddings
    - Semantic similarity search
    - Metadata filtering
    
    Supports both OpenRouter embeddings (API) and local sentence-transformers.
    """
    
    def __init__(
        self,
        persist_path: Path | None = None,
        collection_name: str = "documents",
        embedding_provider: str | None = None,
    ):
        self.persist_path = persist_path or config.chroma_path
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or config.embedding_provider
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._init_lock = threading.Lock()
    
    def _ensure_initialized(self):
        """
        Lazy initialization of ChromaDB client and collection.
        
        Thread-safe using double-checked locking pattern.
        """
        # Guard on _collection, not _client: _client used to be published
        # BEFORE the collection was built, so a second thread could pass the
        # fast-path check and use a still-None _collection.
        if self._collection is None:
            with self._init_lock:
                # Double-check after acquiring lock
                if self._collection is None:
                    import chromadb
                    from chromadb.config import Settings

                    # Create persist directory if needed
                    self.persist_path.mkdir(parents=True, exist_ok=True)

                    # Initialize client with persistence. Assigned to a local
                    # first so a partially-constructed store is never visible.
                    client = chromadb.PersistentClient(
                        path=str(self.persist_path),
                        settings=Settings(anonymized_telemetry=False)
                    )

                    # Get or create collection with embedding function
                    self._embedding_function = self._get_embedding_function()
                    collection = client.get_or_create_collection(
                        name=self.collection_name,
                        embedding_function=self._embedding_function,
                        metadata={
                            "hnsw:space": "cosine",      # Use cosine similarity
                            "hnsw:M": 32,                 # Increased from 16 for better recall
                            "hnsw:construction_ef": 200,  # Higher quality index construction
                            "hnsw:search_ef": 100,        # Much higher search depth for large KB
                        }
                    )

                    # Publish both together, collection last: it is the flag
                    # the fast path above reads.
                    self._client = client
                    self._collection = collection

    def _get_embedding_function(self):
        """Get the embedding function based on config."""
        from ..embeddings import ChromaEmbeddingFunction, get_embedding_provider
        
        # Check if we should use OpenRouter or local
        if self.embedding_provider == "auto":
            # Use OpenRouter if API key is available
            if os.getenv("OPENROUTER_API_KEY"):
                provider = get_embedding_provider("openrouter")
                return ChromaEmbeddingFunction(provider, name="openrouter")
        elif self.embedding_provider == "openrouter":
            provider = get_embedding_provider("openrouter")
            return ChromaEmbeddingFunction(provider, name="openrouter")
        
        # Fall back to local sentence-transformers
        from chromadb.utils import embedding_functions
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.local_embedding_model
        )
    
    # =========================================================================
    # Storage Operations
    # =========================================================================
    
    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Add multiple text chunks."""
        self._ensure_initialized()
        
        if not chunks:
            return
        
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                **{k: str(v) for k, v in chunk.metadata.items()},
                "document_id": chunk.document_id,
                "page_number": chunk.page_number or 0,
                "section_title": chunk.section_title or "",
                "chunk_index": chunk.chunk_index,
                "start_line": chunk.start_line or 0,
                "end_line": chunk.end_line or 0,
            }
            for chunk in chunks
        ]
        
        # Chroma rejects a single add() above its max batch size (~5.4k), so a
        # large spreadsheet -- 100 rows per chunk over a 600k-row sheet -- would
        # fail the whole ingest. Write in slices instead.
        for start in range(0, len(ids), MAX_ADD_BATCH):
            end = start + MAX_ADD_BATCH
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end]
            )
    
    def delete_document_chunks(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        self._ensure_initialized()
        self._collection.delete(where={"document_id": document_id})
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        document_id: str | None = None,
        min_score: float = 0.0
    ) -> tuple[list[TextChunk], list[float]]:
        """
        Search for similar chunks.
        
        Args:
            query: The search query
            n_results: Maximum number of results
            document_id: Optional filter by document
            min_score: Minimum similarity score (0-1)
            
        Returns:
            Tuple of (chunks, scores)
        """
        self._ensure_initialized()
        
        # Build where filter
        where = None
        if document_id:
            where = {"document_id": document_id}
        
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        scores = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity scores
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance
                
                if score < min_score:
                    continue
                
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""
                
                chunk = TextChunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=content,
                    page_number=int(metadata.get("page_number", 0)) or None,
                    section_title=metadata.get("section_title") or None,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    start_line=int(metadata.get("start_line", 0)) or None,
                    end_line=int(metadata.get("end_line", 0)) or None
                )
                
                chunks.append(chunk)
                scores.append(score)
        
        return chunks, scores
    
    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        """Get a specific chunk by ID."""
        self._ensure_initialized()
        
        results = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return None
        
        metadata = results["metadatas"][0] if results["metadatas"] else {}
        content = results["documents"][0] if results["documents"] else ""
        
        return TextChunk(
            id=chunk_id,
            document_id=metadata.get("document_id", ""),
            content=content,
            page_number=int(metadata.get("page_number", 0)) or None,
            section_title=metadata.get("section_title") or None,
            chunk_index=int(metadata.get("chunk_index", 0)),
            start_line=int(metadata.get("start_line", 0)) or None,
            end_line=int(metadata.get("end_line", 0)) or None
        )
    
    def count(self, document_id: str | None = None) -> int:
        """Get total number of chunks stored."""
        self._ensure_initialized()
        if document_id:
            results = self._collection.get(where={"document_id": document_id})
            return len(results.get("ids", []))
        return self._collection.count()
    
