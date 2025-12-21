"""Configuration for UltimateRAG."""

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
    """Application configuration."""
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default=None)
    documents_dir: Path = field(default=None)
    db_dir: Path = field(default=None)
    sqlite_path: Path = field(default=None)
    chroma_path: Path = field(default=None)
    
    # LLM settings
    llm_provider: str = "auto"
    llm_model: str = "google/gemini-3-flash-preview"  # Fast Gemini model
    openrouter_api_key: str = field(default=None)
    openai_api_key: str = field(default=None)
    anthropic_api_key: str = field(default=None)
    
    # Embedding settings
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
    
    def __post_init__(self):
        """Initialize derived paths and load environment variables."""
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
        
        # Load from environment (includes .env via dotenv)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.openrouter_api_key)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider)
        self.vision_model = os.getenv("VISION_MODEL", self.vision_model)
        self.use_vision_tables = os.getenv("USE_VISION_TABLES", "true").lower() == "true"
    
    @property
    def has_llm_key(self) -> bool:
        """Check if any LLM API key is configured."""
        return bool(
            self.openrouter_api_key or 
            self.openai_api_key or 
            self.anthropic_api_key
        )


# Global config instance
config = Config()

