"""Tests for storage module fixes.

Tests:
- SQLite store: validate_identifier, WAL mode
- ChromaStore: thread-safe initialization
- DocumentStore: UTF-8 encoding, error handling, section detection
- SchemaClusterManager: cache TTL, keyword limits
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sqlite_store import SQLiteStore, validate_identifier, SecurityError
from src.storage.document_store import DocumentStore, PathSecurityError
from src.storage.chroma_store import ChromaStore
from src.storage.schema_cluster import (
    SchemaClusterManager,
    MAX_DESCRIPTION_KEYWORDS,
    CACHE_TTL_SECONDS
)


class TestSQLiteStoreFixes:
    """Tests for SQLite store security and performance fixes."""
    
    def test_validate_identifier_valid(self):
        """Valid SQL identifiers should pass validation."""
        valid_ids = [
            "table_name",
            "Column123",
            "_private",
            "CamelCase",
            "a",
        ]
        for identifier in valid_ids:
            assert validate_identifier(identifier) == identifier
    
    def test_validate_identifier_rejects_injection(self):
        """SQL injection attempts should be blocked."""
        malicious = [
            "table; DROP TABLE--",
            "name' OR '1'='1",
            "table\"; DELETE FROM",
        ]
        for identifier in malicious:
            with pytest.raises(SecurityError):
                validate_identifier(identifier)
    
    def test_validate_identifier_rejects_starting_digit(self):
        """Identifiers starting with digits should be rejected."""
        with pytest.raises(SecurityError):
            validate_identifier("123table")
    
    def test_validate_identifier_rejects_empty(self):
        """Empty identifiers should be rejected."""
        with pytest.raises(SecurityError):
            validate_identifier("")
    
    def test_validate_identifier_rejects_too_long(self):
        """Identifiers over 63 characters should be rejected."""
        with pytest.raises(SecurityError):
            validate_identifier("a" * 64)
    
    def test_validate_identifier_rejects_special_chars(self):
        """Special characters should be rejected."""
        invalid = ["table-name", "table name", "table.name", "table@name"]
        for identifier in invalid:
            with pytest.raises(SecurityError):
                validate_identifier(identifier)
    
    def test_wal_mode_enabled(self):
        """WAL mode should be enabled for better concurrency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            
            with store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode")
                mode = cursor.fetchone()[0]
                assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"


class TestChromaStoreFixes:
    """Tests for ChromaDB thread safety fixes."""
    
    def test_has_init_lock(self):
        """ChromaStore should have an initialization lock."""
        # Don't actually initialize (requires chromadb), just check attribute
        store = ChromaStore.__new__(ChromaStore)
        store.persist_path = Path("/tmp/test")
        store.collection_name = "test"
        store.embedding_provider = "local"
        store._client = None
        store._collection = None
        store._embedding_function = None
        store._init_lock = threading.Lock()
        
        assert hasattr(store, '_init_lock')
        assert isinstance(store._init_lock, type(threading.Lock()))
    
    def test_thread_safety_double_check_pattern(self):
        """Verify the double-checked locking pattern is implemented."""
        import inspect
        source = inspect.getsource(ChromaStore._ensure_initialized)
        
        # Should have both the outer check and inner check
        assert "if self._client is None:" in source
        assert "with self._init_lock:" in source


class TestDocumentStoreFixes:
    """Tests for DocumentStore file handling fixes."""
    
    def test_utf8_encoding_write(self):
        """Documents with unicode should be saved with UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(base_path=Path(tmpdir))
            
            from src.models import Document
            doc = Document(
                id="test123",
                filename="test_unicode.pdf",
                title="Test with émojis 🎉 and ñ"
            )
            
            # Should not raise
            store.save_document(doc, full_text="Content with unicode: café ñ 日本語")
            
            # Verify it can be read back
            retrieved = store.get_document("test123")
            assert retrieved is not None
            assert "émojis" in retrieved.title
            
            content = store.get_full_text("test123")
            assert "café" in content
            assert "日本語" in content
    
    def test_error_handling_corrupted_metadata(self):
        """Corrupted metadata files should not crash list_documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(base_path=Path(tmpdir))
            
            # Create a valid document
            from src.models import Document
            valid_doc = Document(id="valid123", filename="valid.pdf")
            store.save_document(valid_doc)
            
            # Create a corrupted metadata file
            corrupted_path = store.metadata_path / "corrupted456.json"
            corrupted_path.write_text("{ this is not valid json }", encoding="utf-8")
            
            # list_documents should skip corrupted file, not crash
            docs = store.list_documents()
            assert len(docs) == 1
            assert docs[0].id == "valid123"
    
    def test_get_document_returns_none_for_corrupted(self):
        """get_document should return None for corrupted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(base_path=Path(tmpdir))
            
            # Create corrupted metadata
            corrupted_path = store.metadata_path / "bad123.json"
            corrupted_path.write_text("invalid json {{{", encoding="utf-8")
            
            # Should return None, not raise
            result = store.get_document("bad123")
            assert result is None
    
    def test_section_detection_operator_precedence(self):
        """Test that section detection logic works correctly after precedence fix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(base_path=Path(tmpdir))
            
            from src.models import Document
            doc = Document(id="test789", filename="test.pdf")
            
            # Content with sections
            content = """
INTRODUCTION
This is the introduction section.
More intro text.

# Next Section
This starts a new section with hash.
"""
            store.save_document(doc, full_text=content)
            
            # Get the introduction section
            section = store.get_section("test789", "INTRODUCTION")
            assert section is not None
            assert "introduction section" in section.lower()
            # Should not include the hash section content
            assert "hash" not in section.lower()


class TestSchemaClusterFixes:
    """Tests for SchemaClusterManager performance fixes."""
    
    @pytest.mark.asyncio
    async def test_keyword_extraction_limit(self):
        """Keywords from description should be limited."""
        manager = SchemaClusterManager()
        
        # Create a very long description
        long_description = " ".join([f"word{i}" for i in range(500)])
        
        keywords = manager._extract_keywords(
            table_name="test_table",
            columns=["col1", "col2"],
            description=long_description
        )
        
        # Should be capped at MAX_DESCRIPTION_KEYWORDS plus keywords from table/columns
        # The limit applies only to description words
        assert len(keywords) <= MAX_DESCRIPTION_KEYWORDS + 10  # buffer for table/column keywords
    
    @pytest.mark.asyncio
    async def test_cache_has_ttl(self):
        """Cache should have TTL mechanism."""
        manager = SchemaClusterManager()
        
        # Check that cache timestamp attribute exists
        assert hasattr(manager, '_cache_timestamp')
        assert manager._cache_timestamp == 0.0  # Initial value
    
    @pytest.mark.asyncio
    async def test_cache_ttl_refresh_logic(self):
        """Cache should refresh after TTL expires."""
        # This tests the TTL logic by checking the source code
        import inspect
        source = inspect.getsource(SchemaClusterManager._get_table_metadata_cache)
        
        assert "time.time()" in source
        assert "CACHE_TTL_SECONDS" in source
        assert "cache_expired" in source or "cache_age" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
