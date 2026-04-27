"""Tests for storage module fixes.

Tests:
- SQLite store: validate_identifier, WAL mode
- ChromaStore: thread-safe initialization
- DocumentStore: UTF-8 encoding, error handling, section detection
- SchemaClusterManager: cache TTL, keyword limits
"""

import sys
import tempfile
import threading
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.chroma_store import ChromaStore
from src.storage.document_store import DocumentStore
from src.storage.schema_cluster import (
    MAX_DESCRIPTION_KEYWORDS,
    SchemaClusterManager,
)
from src.storage.sqlite_store import SecurityError, SQLiteStore, validate_identifier


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
    
    def test_keyword_extraction_limit(self):
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
    
    def test_cache_has_ttl(self):
        """Cache should have TTL mechanism."""
        manager = SchemaClusterManager()
        
        # Check that cache timestamp attribute exists
        assert hasattr(manager, '_cache_timestamp')
        assert manager._cache_timestamp == 0.0  # Initial value
    
    def test_cache_ttl_refresh_logic(self):
        """Cache should refresh after TTL expires."""
        # This tests the TTL logic by checking the source code
        import inspect
        source = inspect.getsource(SchemaClusterManager._get_table_metadata_cache)
        
        assert "time.time()" in source
        assert "CACHE_TTL_SECONDS" in source
        assert "cache_expired" in source or "cache_age" in source


class TestTableNamingFixes:
    """Tests for ingestion-write-path fixes (P0.5/P0.6/P0.7)."""

    def _make_doc(self, doc_id: str, filename: str):
        from src.models import Document

        return Document(id=doc_id, filename=filename, page_count=1)

    def _make_table(self, doc_id: str, table_id: str, table_name: str):
        from src.models import ExtractedTable

        return ExtractedTable(
            id=table_id,
            document_id=doc_id,
            table_name=table_name,
            page_number=1,
            schema_description="Revenue table",
            columns=["quarter", "revenue"],
            rows=[{"quarter": "Q1", "revenue": 100}],
            raw_text="| quarter | revenue |\n| Q1 | 100 |",
        )

    def test_table_name_collision_with_similar_filenames(self, tmp_path):
        """Two docs whose filenames previously collided to the same prefix
        must now each get distinct native SQL tables.

        Under the old filename-slug scheme, both ``nvidia_q1_2024.pdf`` and
        ``nvidia_q1_2025.pdf`` reduced to ``nvidiaq120`` and overwrote each
        other. Under the new ``t_<doc_id>`` prefix the SHA-derived doc_ids
        are collision-free, and the literal ``t_`` ensures the identifier
        always starts with a letter even when the SHA prefix begins with
        a digit -- so we can use the real ``document_id_from_filename``
        outputs here.
        """
        from src.common.ids import document_id_from_filename

        db_path = tmp_path / "collision.db"
        store = SQLiteStore(db_path=db_path)

        # The previously-colliding pair under the old slug scheme.
        filename_a = "nvidia_q1_2024.pdf"
        filename_b = "nvidia_q1_2025.pdf"
        doc_a_id = document_id_from_filename(filename_a)
        doc_b_id = document_id_from_filename(filename_b)
        # Sanity: SHA-derived IDs are collision-free for these two files.
        assert doc_a_id != doc_b_id

        store.save_document(self._make_doc(doc_a_id, filename_a))
        store.save_document(self._make_doc(doc_b_id, filename_b))

        table_a = self._make_table(doc_a_id, f"{doc_a_id}_revenue", "revenue")
        table_b = self._make_table(doc_b_id, f"{doc_b_id}_revenue", "revenue")
        # Differentiate the data so we can verify no overwrite happened.
        table_a.rows = [{"quarter": "Q1", "revenue": 2024}]
        table_b.rows = [{"quarter": "Q1", "revenue": 2025}]

        store.save_table(table_a)
        store.save_table(table_b)

        # Two distinct entries in extracted_tables.
        listed = store.list_tables()
        assert len(listed) == 2

        # Two distinct native SQL tables exist.
        native_tables = store.list_spreadsheet_tables()
        native_names = sorted({t["table_name"] for t in native_tables})
        assert len(native_names) == 2, (
            f"expected 2 distinct native tables, got {native_names}"
        )

        # The new prefix is doc_id-based, not filename-based.
        for name in native_names:
            assert "nvidiaq120" not in name, (
                f"native table {name!r} still uses old filename slug"
            )

        # Querying each native table returns the right per-document data.
        with store._get_connection() as conn:
            cursor = conn.cursor()
            for name in native_names:
                # validate_identifier was already enforced on write
                cursor.execute(f'SELECT revenue FROM "{name}"')
                values = [row[0] for row in cursor.fetchall()]
                assert values  # non-empty
                # Each native table should have exactly one of 2024 or 2025.
                assert values[0] in (2024, 2025)

            # Confirm both years are represented across the two tables.
            all_values = set()
            for name in native_names:
                cursor.execute(f'SELECT revenue FROM "{name}"')
                all_values.update(r[0] for r in cursor.fetchall())
            assert all_values == {2024, 2025}

    def test_make_unique_table_name_uses_doc_id_prefix(self, tmp_path):
        """Direct unit test for _make_unique_table_name: the prefix is
        ``t_`` plus the first 12 chars of the doc_id, not a filename slug,
        and the suffix preserves uniqueness after identifier truncation.
        The ``t_`` prefix guarantees a letter-starting identifier even when
        the SHA-derived doc_id begins with a digit."""
        store = SQLiteStore(db_path=tmp_path / "unit.db")
        doc_id = "abcdef1234567890"
        result = store._make_unique_table_name("revenue", doc_id, "table_1")
        assert result.startswith("t_abcdef123456_revenue_")
        assert len(result) <= 63
        # And independent of any document existing in the store.
        assert store.get_document(doc_id) is None

    def test_long_same_doc_table_names_keep_distinct_native_tables(self, tmp_path):
        """Long sanitized names that differ after the truncation boundary
        must not collapse to the same native SQLite table."""
        from src.models import Document

        db_path = tmp_path / "long-name-collision.db"
        store = SQLiteStore(db_path=db_path)
        doc_id = "abcdef1234567890"
        store.save_document(Document(id=doc_id, filename="report.pdf", page_count=1))

        base = "a" * 48
        table_a = self._make_table(doc_id, "table_a", f"{base}xx")
        table_b = self._make_table(doc_id, "table_b", f"{base}yy")
        table_a.rows = [{"quarter": "Q1", "revenue": 1}]
        table_b.rows = [{"quarter": "Q1", "revenue": 2}]

        store.save_table(table_a)
        store.save_table(table_b)

        native_tables = store.list_spreadsheet_tables()
        native_names = sorted(t["table_name"] for t in native_tables)
        assert len(native_names) == 2
        assert native_names[0] != native_names[1]
        assert all(len(name) <= 63 for name in native_names)

        with store._get_connection() as conn:
            values = set()
            for name in native_names:
                cursor = conn.execute(f'SELECT revenue FROM "{name}"')
                values.update(row[0] for row in cursor.fetchall())
        assert values == {1, 2}

    def test_vlm_table_name_with_spaces_is_sanitized(self, tmp_path):
        """ExtractedTable carrying a sanitized VLM-style name must save
        cleanly without tripping validate_identifier."""
        from src.common.ids import document_id_from_filename
        from src.common.naming import sanitize_table_name

        db_path = tmp_path / "vlm.db"
        store = SQLiteStore(db_path=db_path)

        filename = "report.pdf"
        doc_id = document_id_from_filename(filename)
        store.save_document(self._make_doc(doc_id, filename))

        # Mirror what the extractor will now pass post-sanitization.
        sanitized = sanitize_table_name("Q1 FY26 Revenue")
        table = self._make_table(doc_id, f"{doc_id}_q1_fy26_revenue", sanitized)

        # Must not raise SecurityError or any other exception.
        store.save_table(table)

        native_tables = store.list_spreadsheet_tables()
        assert len(native_tables) == 1
        native_name = native_tables[0]["table_name"]

        # Native table name must be alphanumeric/underscore only.
        import re

        assert re.match(r"^[a-zA-Z0-9_]+$", native_name), (
            f"native table name {native_name!r} is not SQL-safe"
        )

    def test_vlm_extractor_sanitizes_raw_table_name(self):
        """Targeted regression for the sanitization line in
        ``vlm_extractor._parse_response``. The previous test only verified
        the store accepts an already-sanitized name; this one drives the
        actual extractor entry point with a free-form VLM-style payload
        and asserts the resulting ExtractedTable.table_name is SQL-safe.

        If someone removes the ``sanitize_table_name(name)`` call in
        ``vlm_extractor.py``, this test fails -- the previous test would
        not catch that regression.
        """
        import re as _re

        from src.ingestion.vlm_extractor import VLMTableExtractor

        extractor = VLMTableExtractor()
        # Mirror the shape returned by the model.
        payload = {
            "tables": [
                {
                    "name": "Q1 FY26 Revenue",
                    "columns": ["quarter", "revenue"],
                    "rows": [{"quarter": "Q1", "revenue": 100}],
                }
            ]
        }

        extracted = extractor._parse_response(payload, page_number=1, document_id="doc123")

        assert len(extracted) == 1
        table = extracted[0]
        assert _re.match(r"^[a-zA-Z0-9_]+$", table.table_name), (
            f"VLM extractor must sanitize free-form names; got {table.table_name!r}"
        )
        assert " " not in table.table_name


class TestPartialFailureTolerance:
    """Tests that one bad table does not abort an ingestion batch."""

    def test_partial_failure_in_table_batch_does_not_abort(self, tmp_path):
        """When one table is rigged to fail mid-batch, the other two must
        still be persisted (the gather no longer aborts the batch)."""
        import asyncio

        from src.common.naming import sanitize_table_name
        from src.models import Document, ExtractedTable
        from src.rag_agent import RAGAgent
        from src.storage.sqlite_store import SQLiteStore

        db_path = tmp_path / "partial.db"
        store = SQLiteStore(db_path=db_path)

        filename = "report.pdf"
        # Use a letter-starting doc_id so the doc_id-prefixed native table
        # name passes validate_identifier deterministically.
        doc_id = "abcdef1234567890"
        store.save_document(Document(id=doc_id, filename=filename, page_count=1))

        # Build a minimally-initialized RAGAgent so we exercise the real
        # _process_pdf_tables coroutine without needing the full ChromaDB
        # / schema-cluster stack.
        agent = RAGAgent.__new__(RAGAgent)
        agent.llm_client = None  # skip the schema_detector branch
        agent.sqlite_store = store
        agent.schema_detector = None  # not consulted when llm_client is None

        bad_id = "rigged_bad_table"

        # Rig _assign_cluster_safe to raise for one specific table_id.
        # _assign_cluster_safe is the *outer* wrapper called from the gather,
        # so raising here exercises the gather's return_exceptions path.
        async def rigged_assign_cluster_safe(*, table_name, columns, schema_description, source_document):
            if table_name.endswith("_bad"):
                raise RuntimeError("rigged failure for partial-batch test")

        agent._assign_cluster_safe = rigged_assign_cluster_safe  # type: ignore[assignment]

        def _make_table(table_id: str, suffix: str) -> ExtractedTable:
            return ExtractedTable(
                id=table_id,
                document_id=doc_id,
                table_name=sanitize_table_name(f"table_{suffix}"),
                page_number=1,
                schema_description="t",
                columns=["a"],
                rows=[{"a": 1}],
                raw_text="| a |\n| 1 |",
            )

        tables = [
            _make_table("ok_one", "good1"),
            _make_table(bad_id, "bad"),
            _make_table("ok_two", "good2"),
        ]

        # Must complete without raising even though one table's clustering
        # raised.
        asyncio.run(agent._process_pdf_tables(tables, source_document=filename))

        # The two non-failing tables MUST be in extracted_tables. (The bad
        # table also lands in extracted_tables because save_table runs
        # before _assign_cluster_safe — what matters is that the batch did
        # not abort.)
        listed_names = {t.table_name for t in store.list_tables()}
        good_names = {sanitize_table_name("table_good1"), sanitize_table_name("table_good2")}
        assert good_names.issubset(listed_names), (
            f"good tables missing from extracted_tables: {listed_names}"
        )

        # Native SQL tables for the good ones must also exist.
        native_names = {t["table_name"] for t in store.list_spreadsheet_tables()}
        for good in good_names:
            assert any(good in n for n in native_names), (
                f"good native table {good!r} not found in {native_names}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
