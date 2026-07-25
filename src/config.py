"""Configuration for Finance RAG."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv():
    """Load .env file if it exists."""
    try:
        from dotenv import load_dotenv
        
        # Look for .env in project root
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
    except ImportError:
        pass
    return False


# Load .env on module import
_dotenv_loaded = _load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean setting from the environment, tolerating common spellings."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an int setting from the environment, ignoring malformed values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    """
    Application configuration with secure credential handling.
    
    API keys are accessed via properties and retrieved from environment
    at runtime to minimize exposure in memory and stack traces.
    """
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default=None)
    documents_dir: Path = field(default=None)
    db_dir: Path = field(default=None)
    sqlite_path: Path = field(default=None)
    chroma_path: Path = field(default=None)
    
    # LLM settings (non-sensitive)
    llm_provider: str = "auto"
    llm_model: str = "google/gemini-3-flash-preview"  # Fast Gemini model
    
    # Embedding settings (non-sensitive)
    embedding_model: str = "qwen/qwen3-embedding-8b"  # OpenRouter default
    embedding_provider: str = "local"  # "auto", "openrouter", or "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"  # High performance, small size (133MB)
    
    # Retrieval settings
    use_reranking: bool = True  # Cross-encoder reranking for +25% precision
    # FlashRank model name (see src/tools/reranker.py). This is NOT a
    # sentence-transformers cross-encoder id -- the previous value here was
    # never read, so the docs described a model the code never loaded.
    reranker_model: str = "rank-T5-flan"
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    def __post_init__(self):
        """Initialize derived paths and load non-sensitive settings."""
        # Set up paths
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.documents_dir is None:
            self.documents_dir = self.data_dir / "documents"
        if self.db_dir is None:
            self.db_dir = self.data_dir / "db"
        if self.sqlite_path is None:
            self.sqlite_path = self.db_dir / "structured.db"
        if self.chroma_path is None:
            self.chroma_path = self.db_dir / "chroma"
        
        # Create directories
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Load non-sensitive settings from environment
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider)
        self.local_embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL", self.local_embedding_model
        )
        self.reranker_model = os.getenv("RERANKER_MODEL", self.reranker_model)
        self.use_reranking = _env_bool("USE_RERANKING", self.use_reranking)
        self.chunk_size = _env_int("CHUNK_SIZE", self.chunk_size)
        self.chunk_overlap = _env_int("CHUNK_OVERLAP", self.chunk_overlap)


    # =========================================================================
    # Secure API Key Access (retrieved at runtime, not stored)
    # =========================================================================
    
    @property
    def openrouter_api_key(self) -> str | None:
        """Get OpenRouter API key from environment."""
        return os.getenv("OPENROUTER_API_KEY")
    
    @property
    def has_llm_key(self) -> bool:
        """Check if an OpenRouter API key is configured."""
        return bool(self.openrouter_api_key)
    
    def __repr__(self) -> str:
        """
        Safe repr that doesn't expose API keys.
        
        This is important for logging and debugging where repr might
        be called automatically.
        """
        return (
            f"Config("
            f"llm_model='{self.llm_model}', "
            f"has_openrouter={bool(self.openrouter_api_key)})"
        )


# Global config instance
config = Config()
