"""Embedding providers for Finance RAG."""

import os
from abc import ABC, abstractmethod
from typing import Any
import asyncio

from .config import config


# OpenRouter embedding models
OPENROUTER_EMBEDDING_MODELS = {
    # Qwen (recommended)
    "qwen3-8b": "qwen/qwen3-embedding-8b",
    
    # OpenAI
    "text-embedding-3-small": "openai/text-embedding-3-small",
    "text-embedding-3-large": "openai/text-embedding-3-large",
    
    # Cohere
    "embed-english-v3": "cohere/embed-english-v3.0",
    "embed-multilingual-v3": "cohere/embed-multilingual-v3.0",
    
    # Google
    "text-embedding-004": "google/text-embedding-004",
}

# Default model
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        pass


class OpenRouterEmbeddings(EmbeddingProvider):
    """
    OpenRouter embedding provider.
    
    Uses OpenRouter's /embeddings endpoint to generate embeddings.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize OpenRouter embeddings.
        
        Args:
            api_key: OpenRouter API key (or uses OPENROUTER_API_KEY env var)
            model: Model name - can be short name or full OpenRouter name
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter embeddings")
        
        # Resolve model name
        model = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        if model in OPENROUTER_EMBEDDING_MODELS:
            self.model = OPENROUTER_EMBEDDING_MODELS[model]
        else:
            self.model = model
        
        self._session = None
    
    def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings asynchronously."""
        import aiohttp
        
        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Batch texts (OpenRouter supports array input)
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"OpenRouter embedding error: {response.status} - {error_text}")
                
                data = await response.json()
        
        # Extract embeddings in order
        embeddings = [None] * len(texts)
        for item in data["data"]:
            embeddings[item["index"]] = item["embedding"]
        
        return embeddings
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - use nest_asyncio or run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._embed_async(texts))
                return future.result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            return asyncio.run(self._embed_async(texts))
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        embeddings = self.embed([text])
        return embeddings[0]
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB-compatible callable interface."""
        return self.embed(input)


class LocalEmbeddings(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Free, works offline, no API key needed.
    Uses BGE-large by default for better retrieval performance.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize local embeddings.
        
        Args:
            model_name: sentence-transformers model name
        """
        self.model_name = model_name
        self._model = None
    
    def _ensure_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        self._ensure_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        return self.embed([text])[0]
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB-compatible callable interface."""
        return self.embed(input)


class ChromaEmbeddingFunction:
    """
    Wrapper that adapts our embedding providers to ChromaDB's interface.
    
    ChromaDB expects an embedding function with:
    - __call__(input: Documents) -> Embeddings
    - embed_documents(input: Documents) -> Embeddings
    - embed_query(input: str) -> Embedding
    - name() -> str (for caching/validation)
    """
    
    def __init__(self, provider: EmbeddingProvider, name: str = "openrouter"):
        self.provider = provider
        self._name = name
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.provider.embed(input)
    
    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed documents (for adding to collection)."""
        return self.provider.embed(input)
    
    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        """Embed query text (for searching)."""
        if isinstance(input, str):
            return self.provider.embed_query(input)
        else:
            return self.provider.embed(input)
    
    def name(self) -> str:
        """Return embedding function name for ChromaDB."""
        return self._name


def get_embedding_provider(
    provider: str = "auto",
    model: str | None = None,
) -> EmbeddingProvider:
    """
    Get an embedding provider.
    
    Args:
        provider: "openrouter", "local", or "auto"
        model: Model name (provider-specific)
        
    Returns:
        EmbeddingProvider instance
    """
    if provider == "auto":
        # Use OpenRouter if API key is available
        if os.getenv("OPENROUTER_API_KEY"):
            return OpenRouterEmbeddings(model=model)
        else:
            # Fall back to local
            return LocalEmbeddings(model_name=model or "all-MiniLM-L6-v2")
    
    if provider == "openrouter":
        return OpenRouterEmbeddings(model=model)
    
    if provider == "local":
        return LocalEmbeddings(model_name=model or "all-MiniLM-L6-v2")
    
    raise ValueError(f"Unknown embedding provider: {provider}")

