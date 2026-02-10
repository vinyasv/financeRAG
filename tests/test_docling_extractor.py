"""Tests for VisionTableExtractor (Docling)."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.ingestion.vision_table_extractor import VisionTableExtractor
from src.models import ExtractedTable


class TestVisionTableExtractor:
    """Tests for VisionTableExtractor (Docling)."""
    
    @pytest.fixture
    def extractor(self):
        """Create a VisionTableExtractor instance."""
        return VisionTableExtractor()
    
    def test_init(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor._converter is None
    
    def test_ensure_docling_imports(self, extractor):
        """Test lazy loading of Docling."""
        # Should work without error
        extractor._ensure_docling()
        assert extractor._converter is not None
    
    def test_process_value_number(self, extractor):
        """Test processing numeric values."""
        assert extractor._process_value(42) == 42
        assert extractor._process_value(3.14) == 3.14
    
    def test_process_value_string(self, extractor):
        """Test processing string values."""
        assert extractor._process_value("hello") == "hello"
        assert extractor._process_value("  trimmed  ") == "trimmed"
    
    def test_process_value_currency(self, extractor):
        """Test processing currency strings."""
        assert extractor._process_value("$1,234") == 1234.0
        assert extractor._process_value("$1.5M") == 1500000.0
        assert extractor._process_value("-$500") == -500.0
    
    def test_generate_table_name(self, extractor):
        """Test table name generation."""
        columns = ["revenue", "profit", "margin"]
        name = extractor._generate_table_name(columns)
        assert name == "revenue_profit_margin"
    
    def test_generate_description_financial(self, extractor):
        """Test description generation for financial tables."""
        columns = ["revenue", "profit", "year"]
        rows = [{"revenue": 100, "profit": 20, "year": 2024}]
        desc = extractor._generate_description(columns, rows)
        assert "Financial data" in desc
    
    def test_dataframe_to_extracted_table_basic(self, extractor):
        """Test DataFrame conversion."""
        df = pd.DataFrame({
            "Revenue": [100, 200, 300],
            "Profit": [10, 20, 30]
        })
        
        table = extractor._dataframe_to_extracted_table(
            df=df,
            document_id="test_doc",
            table_index=0
        )
        
        assert table is not None
        assert len(table.rows) == 3
        assert "revenue" in table.columns
        assert "profit" in table.columns

    @pytest.mark.asyncio
    async def test_extract_tables_docling_mock(self, extractor, tmp_path):
        """Test extraction using mocked Docling converter."""
        # Mock file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")
        
        # Setup mocks
        mock_converter = Mock()
        mock_result = Mock()
        mock_doc = Mock()
        mock_table = Mock()
        
        # Mock DataFrame
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_table.export_to_dataframe.return_value = df
        
        # Mock provenance for page number
        mock_prov = Mock()
        mock_prov.page_no = 1
        mock_table.prov = [mock_prov]
        
        mock_doc.tables = [mock_table]
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        
        extractor._converter = mock_converter
        
        # Run extraction
        tables = await extractor.extract_tables_from_pdf(
            pdf_path=pdf_path,
            document_id="test_doc"
        )
        
        # Verify
        mock_converter.convert.assert_called_once()
        assert len(tables) == 1
        assert tables[0].document_id == "test_doc"
        assert len(tables[0].rows) == 2


class TestIntegration:
    """Integration tests with real PDFs."""
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Get path to a sample PDF for testing."""
        path = Path("data/documents/NVIDIAAn.pdf")
        if path.exists():
            return path
        pytest.skip("No sample PDF available for integration test")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_pdf_extraction(self, sample_pdf_path):
        """Test extraction on a real PDF file."""
        extractor = VisionTableExtractor()
        
        tables = await extractor.extract_tables_from_pdf(
            pdf_path=sample_pdf_path,
            document_id="integration_test",
            max_tables=3  # Limit for speed
        )
        
        assert isinstance(tables, list)
        assert len(tables) > 0
        assert isinstance(tables[0], ExtractedTable)
