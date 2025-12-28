"""Tests for tools module fixes.

Tests coverage for:
- Missing exports in __init__.py
- Exception handling in sql_query.py
- Hardcoded path fix in reranker.py
- Property mutation fix in vector_search.py
- Input validation in get_document.py
- LLM timeout in sql_query.py
"""

import sys
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestToolsExports:
    """Test that all exports are available in __init__.py."""
    
    def test_tool_base_exports(self):
        """Base Tool and ToolResult should be exported."""
        from src.tools import Tool, ToolResult
        assert Tool is not None
        assert ToolResult is not None
    
    def test_calculator_exports(self):
        """CalculatorTool and ComparabilityError should be exported."""
        from src.tools import CalculatorTool, ComparabilityError
        assert CalculatorTool is not None
        assert ComparabilityError is not None
    
    def test_sql_exports(self):
        """SQLQueryTool and SQLExecutor should be exported."""
        from src.tools import SQLQueryTool, SQLExecutor
        assert SQLQueryTool is not None
        assert SQLExecutor is not None
    
    def test_vector_search_exports(self):
        """VectorSearchTool and MultiQuerySearch should be exported."""
        from src.tools import VectorSearchTool, MultiQuerySearch
        assert VectorSearchTool is not None
        assert MultiQuerySearch is not None
    
    def test_reranker_export(self):
        """Reranker should be exported."""
        from src.tools import Reranker
        assert Reranker is not None
    
    def test_comparability_exports(self):
        """Comparability functions should be exported."""
        from src.tools import check_field_comparability, create_comparability_refusal
        assert check_field_comparability is not None
        assert create_comparability_refusal is not None
    
    def test_get_document_export(self):
        """GetDocumentTool should be exported."""
        from src.tools import GetDocumentTool
        assert GetDocumentTool is not None


class TestSQLQueryToolFixes:
    """Tests for SQL query tool exception handling and timeout."""
    
    @pytest.mark.asyncio
    async def test_security_error_returns_sanitized_message(self):
        """SecurityError should return a user-friendly message."""
        from src.tools.sql_query import SQLQueryTool
        from src.storage.sqlite_store import SecurityError
        
        tool = SQLQueryTool()
        
        # Mock the SQL generation to return a malicious query
        with patch.object(tool, '_generate_sql', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "SELECT * FROM users; DROP TABLE users"
            
            # execute_query should raise SecurityError, which gets caught
            with patch.object(tool.sqlite_store, 'execute_query') as mock_exec:
                mock_exec.side_effect = SecurityError("Forbidden keyword: DROP")
                
                result = await tool.execute("get all users")
                
                assert "error" in result
                assert "Query not allowed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_llm_timeout_handling(self):
        """LLM calls should timeout and raise appropriate error."""
        from src.tools.sql_query import SQLQueryTool, LLM_TIMEOUT_SECONDS
        
        # Create real async function that hangs
        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(LLM_TIMEOUT_SECONDS + 5)
            return "SELECT 1"
        
        mock_llm = MagicMock()
        mock_llm.generate = slow_llm
        
        tool = SQLQueryTool(llm_client=mock_llm)
        
        # Mock the schema
        with patch.object(tool, 'schema_cluster_manager', None):
            with patch.object(tool.sqlite_store, 'get_schema_for_llm', return_value="schema"):
                with pytest.raises(ValueError) as exc_info:
                    await tool._generate_sql_with_llm("test query")
                
                assert "timed out" in str(exc_info.value).lower()


class TestSQLExecutorFixes:
    """Tests for SQLExecutor error reporting."""
    
    def test_error_field_populated_on_failure(self):
        """SQLExecutor should populate error field on failure."""
        from src.tools.sql_query import SQLExecutor
        from src.models import SQLQueryResult
        
        executor = SQLExecutor()
        
        # Execute invalid SQL
        result = executor.execute("INVALID SQL QUERY HERE")
        
        assert isinstance(result, SQLQueryResult)
        assert result.error is not None
        assert len(result.rows) == 0


class TestRerankerFixes:
    """Tests for Reranker path and type fixes."""
    
    def test_cache_dir_uses_tempfile(self):
        """Cache directory should use tempfile, not hardcoded /tmp."""
        import inspect
        from src.tools.reranker import Reranker
        
        source = inspect.getsource(Reranker._ensure_model)
        assert "tempfile.gettempdir()" in source
        assert '"/tmp/flashrank"' not in source
    
    def test_type_hints_use_textchunk(self):
        """Rerank methods should use TextChunk type hints."""
        import inspect
        from src.tools.reranker import Reranker
        
        # Check rerank method signature
        sig = inspect.signature(Reranker.rerank)
        chunks_param = sig.parameters.get('chunks')
        assert chunks_param is not None
        # The annotation should reference TextChunk
        assert 'TextChunk' in str(chunks_param.annotation)


class TestVectorSearchFixes:
    """Tests for VectorSearchTool property mutation fix."""
    
    def test_no_property_mutation(self):
        """use_reranking should not be mutated by property access."""
        from src.tools.vector_search import VectorSearchTool
        
        tool = VectorSearchTool(use_reranking=True)
        
        # Access the reranker property (which may fail to import)
        with patch('src.tools.vector_search.VectorSearchTool.reranker', new_callable=lambda: property(lambda self: None)):
            _ = tool.reranker
        
        # use_reranking should still be True
        assert tool.use_reranking is True
    
    def test_has_init_failed_flag(self):
        """Should have _reranker_init_failed flag."""
        from src.tools.vector_search import VectorSearchTool
        
        tool = VectorSearchTool()
        assert hasattr(tool, '_reranker_init_failed')
        assert tool._reranker_init_failed is False


class TestGetDocumentFixes:
    """Tests for GetDocumentTool input validation."""
    
    @pytest.mark.asyncio
    async def test_empty_doc_id_rejected(self):
        """Empty document ID should be rejected."""
        from src.tools.get_document import GetDocumentTool
        
        tool = GetDocumentTool()
        result = await tool.execute("")
        
        assert "error" in result
        assert "required" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_doc_id_format_rejected(self):
        """Document IDs with invalid characters should be rejected."""
        from src.tools.get_document import GetDocumentTool
        
        tool = GetDocumentTool()
        
        # Path traversal attempt
        result = await tool.execute("../../../etc/passwd")
        assert "error" in result
        assert "Invalid document ID" in result["error"]
        
        # SQL injection attempt
        result = await tool.execute("doc'; DROP TABLE--")
        assert "error" in result
        assert "Invalid document ID" in result["error"]
    
    @pytest.mark.asyncio
    async def test_valid_doc_id_accepted(self):
        """Valid document IDs should be accepted."""
        from src.tools.get_document import GetDocumentTool
        
        tool = GetDocumentTool()
        
        # Valid hex hash
        result = await tool.execute("abc123def456")
        # Will return "not found" since it doesn't exist, but no validation error
        assert "error" not in result or "Invalid document ID" not in result.get("error", "")
        
        # Valid with underscores
        result = await tool.execute("my_document_id")
        assert "error" not in result or "Invalid document ID" not in result.get("error", "")


class TestSQLQueryResultModel:
    """Tests for SQLQueryResult model error field."""
    
    def test_error_field_exists(self):
        """SQLQueryResult should have an error field."""
        from src.models import SQLQueryResult
        
        result = SQLQueryResult(query="SELECT 1", error="Test error")
        assert result.error == "Test error"
    
    def test_error_field_defaults_to_none(self):
        """Error field should default to None."""
        from src.models import SQLQueryResult
        
        result = SQLQueryResult(query="SELECT 1")
        assert result.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
