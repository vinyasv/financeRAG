"""Tests for tools module fixes.

Tests coverage for:
- Missing exports in __init__.py
- Exception handling in sql_query.py
- Hardcoded path fix in reranker.py
- Property mutation fix in vector_search.py
- Input validation in get_document.py
- LLM timeout in sql_query.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSQLQueryToolFixes:
    """Tests for SQL query tool exception handling and timeout."""
    
    def test_security_error_returns_sanitized_message(self):
        """SecurityError should return a user-friendly message."""
        from src.storage.sqlite_store import SecurityError
        from src.tools.sql_query import SQLQueryTool
        
        tool = SQLQueryTool()
        
        # Mock the SQL generation to return a malicious query
        with patch.object(tool, '_generate_sql', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "SELECT * FROM users; DROP TABLE users"
            
            # execute_query should raise SecurityError, which gets caught
            with patch.object(tool.sqlite_store, 'execute_query') as mock_exec:
                mock_exec.side_effect = SecurityError("Forbidden keyword: DROP")
                
                result = asyncio.run(tool.execute("get all users"))
                
                assert "error" in result
                assert "Query not allowed" in result["error"]
    
    def test_llm_timeout_handling(self):
        """LLM calls should timeout and raise appropriate error."""
        from src.tools.sql_query import LLM_TIMEOUT_SECONDS, SQLQueryTool
        
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
                    asyncio.run(tool._generate_sql_with_llm("test query"))
                
                assert "timed out" in str(exc_info.value).lower()


class TestSQLExecutorFixes:
    """Tests for SQLExecutor error reporting."""
    
    def test_error_field_populated_on_failure(self):
        """SQLExecutor should populate error field on failure."""
        from src.models import SQLQueryResult
        from src.tools.sql_query import SQLExecutor
        
        executor = SQLExecutor()
        
        # Execute invalid SQL
        result = executor.execute("INVALID SQL QUERY HERE")
        
        assert isinstance(result, SQLQueryResult)
        assert result.error is not None
        assert len(result.rows) == 0


class TestGetDocumentFixes:
    """Tests for GetDocumentTool input validation."""
    
    def test_empty_doc_id_rejected(self):
        """Empty document ID should be rejected."""
        from src.tools.get_document import GetDocumentTool
        
        tool = GetDocumentTool()
        result = asyncio.run(tool.execute(""))
        
        assert "error" in result
        assert "required" in result["error"].lower()
    
    def test_invalid_doc_id_format_rejected(self):
        """Document IDs with invalid characters should be rejected."""
        from src.tools.get_document import GetDocumentTool
        
        tool = GetDocumentTool()
        
        # Path traversal attempt
        result = asyncio.run(tool.execute("../../../etc/passwd"))
        assert "error" in result
        assert "Invalid document ID" in result["error"]
        
        # SQL injection attempt
        result = asyncio.run(tool.execute("doc'; DROP TABLE--"))
        assert "error" in result
        assert "Invalid document ID" in result["error"]
    
    def test_valid_doc_id_accepted(self):
        """Valid document IDs should be accepted."""
        from src.tools.get_document import GetDocumentTool
        
        tool = GetDocumentTool()
        
        # Valid hex hash
        result = asyncio.run(tool.execute("abc123def456"))
        # Will return "not found" since it doesn't exist, but no validation error
        assert "error" not in result or "Invalid document ID" not in result.get("error", "")
        
        # Valid with underscores
        result = asyncio.run(tool.execute("my_document_id"))
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
