"""Configuration for Finance RAG."""

import os
from pathlib import Path
from dataclasses import dataclass, field


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
    embedding_provider: str = "auto"  # "auto", "openrouter", or "local"
    local_embedding_model: str = "BAAI/bge-large-en-v1.5"  # Better local model
    
    # Retrieval settings
    use_reranking: bool = True  # Cross-encoder reranking for +25% precision
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Vision settings (for table extraction)
    vision_model: str = "google/gemini-3-flash-preview"  # Vision-capable model for table extraction
    use_vision_tables: bool = True  # Enable vision-based table extraction
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Internal: cached values from environment (not exposed)
    _cached_llm_model: str = field(default=None, repr=False)
    
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
        self._cached_llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.llm_model = self._cached_llm_model
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider)
        self.vision_model = os.getenv("VISION_MODEL", self.vision_model)
        self.use_vision_tables = os.getenv("USE_VISION_TABLES", "true").lower() == "true"
    
    # =========================================================================
    # Secure API Key Access (retrieved at runtime, not stored)
    # =========================================================================
    
    @property
    def openrouter_api_key(self) -> str | None:
        """Get OpenRouter API key from environment."""
        return os.getenv("OPENROUTER_API_KEY")
    
    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def anthropic_api_key(self) -> str | None:
        """Get Anthropic API key from environment."""
        return os.getenv("ANTHROPIC_API_KEY")
    
    @property
    def has_llm_key(self) -> bool:
        """Check if any LLM API key is configured."""
        return bool(
            self.openrouter_api_key or 
            self.openai_api_key or 
            self.anthropic_api_key
        )
    
    def __repr__(self) -> str:
        """
        Safe repr that doesn't expose API keys.
        
        This is important for logging and debugging where repr might
        be called automatically.
        """
        return (
            f"Config("
            f"llm_model='{self.llm_model}', "
            f"has_openrouter={bool(self.openrouter_api_key)}, "
            f"has_openai={bool(self.openai_api_key)}, "
            f"has_anthropic={bool(self.anthropic_api_key)})"
        )


# Global config instance
config = Config()

