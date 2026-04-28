"""Regression tests for post-Phase-1 remediation work."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.agent.planner import Planner
from src.agent.synthesizer import ResponseSynthesizer
from src.common.json_utils import parse_json_response
from src.ingestion.exceptions import ExtractionFailed
from src.models import CalculationTranscript, OperandBinding, ToolName, ToolResult
from src.rag_agent import RAGAgent


def test_parse_json_response_handles_bare_fence():
    assert parse_json_response('```\n{"ok": true}\n```') == {"ok": True}


def test_pdf_table_fallback_distinguishes_empty_from_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    agent = RAGAgent.__new__(RAGAgent)
    agent.vlm_extractor = SimpleNamespace(extract_tables_from_pdf=AsyncMock(return_value=[]))
    agent.docling_extractor = SimpleNamespace(extract_tables_from_pdf=AsyncMock(return_value=["docling"]))
    agent.table_extractor = SimpleNamespace(extract_tables=MagicMock(return_value=["rule"]))

    result = asyncio.run(agent._extract_pdf_tables(object(), MagicMock(name="report.pdf"), "doc"))

    assert result == ["docling"]
    agent.docling_extractor.extract_tables_from_pdf.assert_awaited_once()

    agent.vlm_extractor.extract_tables_from_pdf = AsyncMock(side_effect=ExtractionFailed("boom"))
    agent.docling_extractor.extract_tables_from_pdf = AsyncMock(return_value=[])

    result = asyncio.run(agent._extract_pdf_tables(object(), MagicMock(name="report.pdf"), "doc"))

    assert result == ["rule"]
    agent.table_extractor.extract_tables.assert_called_once()


def test_planner_rejects_no_llm_and_falls_back_on_invalid_plan():
    with pytest.raises(RuntimeError, match="requires an LLM"):
        asyncio.run(Planner().create_plan("compare revenue"))

    llm = MagicMock()
    llm.generate = AsyncMock(return_value='{"query":"q","reasoning":"bad","steps":[{"id":"a","tool":"vector_search","input":"x","depends_on":["missing"],"description":"x"}]}')
    planner = Planner(llm_client=llm)

    plan = asyncio.run(planner.create_plan("compare revenue"))

    assert plan.reasoning.startswith("[FAST PATH]")
    assert plan.steps


def test_planner_uses_supplied_company_registry_for_fast_path():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value='{"query":"q","reasoning":"llm","steps":[{"id":"step_1","tool":"vector_search","input":"x","depends_on":[],"description":"x"}]}')
    planner = Planner(llm_client=llm)
    planner.company_registry._registry["acme"] = "acme"
    planner.company_registry._registry["globex"] = "globex"

    asyncio.run(planner.create_plan("what is acme vs globex revenue?"))

    llm.generate.assert_awaited_once()


def test_sql_and_calculator_citations_use_sql_provenance():
    synthesizer = ResponseSynthesizer(llm_client=MagicMock(), document_store=MagicMock())
    synthesizer.document_store.get_document.return_value = SimpleNamespace(filename="report.xlsx")
    sql_result = {
        "value": 100,
        "column": "revenue",
        "__sql_provenance": {
            "sql": 'SELECT revenue FROM "revenue_table"',
            "tables": [
                {"table_name": "revenue_table", "document_id": "doc_1", "columns": ["revenue"]}
            ],
        },
    }
    calc = CalculationTranscript(
        original_expression="{step_1.value} * 2",
        bindings=[
            OperandBinding(
                reference="{step_1.value}",
                resolved_value=100,
                source_step="step_1",
                source_description="SQL query: value from step_1",
            )
        ],
        resolved_expression="100 * 2",
        result=200,
    )
    results = {
        "step_1": ToolResult(step_id="step_1", tool=ToolName.SQL_QUERY, success=True, result=sql_result),
        "step_2": ToolResult(step_id="step_2", tool=ToolName.CALCULATOR, success=True, result=calc),
    }

    citations = synthesizer._extract_citations(results)

    assert len(citations) == 1
    assert citations[0].document_name == "report.xlsx"
    assert citations[0].section == "revenue_table"


def test_excel_title_row_header_inference(tmp_path):
    from src.ingestion.spreadsheet_parser import SpreadsheetParser

    path = tmp_path / "title.xlsx"
    df = pd.DataFrame([
        ["Annual Revenue Report", None, None],
        ["Quarter", "Revenue", "Cost"],
        ["Q1", 100, 50],
    ])
    df.to_excel(path, index=False, header=False)

    parsed = SpreadsheetParser().parse(path)

    assert parsed.sheets[0].headers == ["Quarter", "Revenue", "Cost"]
    assert parsed.sheets[0].rows[0]["Revenue"] == 100
