"""End-to-end smoke test: ingest the sample PDF, then run a query through the agent.

Follows the existing test pattern in this repo (sync `def test_*` with
`asyncio.run(...)` inline; no `pytest-asyncio`). Uses `MockLLMClient` so no
network calls reach OpenRouter, and pins the embedding provider to "local"
to keep the vector store self-contained.
"""

import asyncio
from pathlib import Path

import pytest


# Distinctive substring inside src/agent/synthesizer.py SYNTHESIS_PROMPT that
# does NOT appear in the planner prompt. We rely on this to route the mock
# response to synthesis vs planning. ("SENIOR FINANCIAL ANALYST" is shared by
# both prompts, so we check it last.)
SYNTHESIZER_DISTINCT_SUBSTRING = "RESEARCH FINDINGS (VERIFIED DATA)"

PLANNER_RESPONSE_JSON = (
    '{"query":"smoke","reasoning":"smoke","steps":['
    '{"id":"step_1","tool":"vector_search","input":"Acme revenue",'
    '"depends_on":[],"description":"smoke"}]}'
)
SYNTHESIZER_RESPONSE_TEXT = "Acme Corp Q1 revenue was $100 million."

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_ingest_and_query_smoke(tmp_path, monkeypatch):
    """Smoke test: ingest sample.pdf, run a query, get a non-empty answer.

    Verifies that planner -> executor -> synthesizer wires up end-to-end
    against a mocked LLM, and that ingestion persists at least one document.
    """
    # Force local embeddings so initialization never reaches OpenRouter.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

    # Redirect every config-derived path under tmp_path so this test never
    # touches the real data/ directory.
    from src import config as config_module

    cfg = config_module.config
    test_data_dir = tmp_path / "data"
    test_documents_dir = test_data_dir / "documents"
    test_db_dir = test_data_dir / "db"
    test_documents_dir.mkdir(parents=True, exist_ok=True)
    test_db_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cfg, "data_dir", test_data_dir)
    monkeypatch.setattr(cfg, "documents_dir", test_documents_dir)
    monkeypatch.setattr(cfg, "db_dir", test_db_dir)
    monkeypatch.setattr(cfg, "sqlite_path", test_db_dir / "structured.db")
    monkeypatch.setattr(cfg, "chroma_path", test_db_dir / "chroma")
    monkeypatch.setattr(cfg, "embedding_provider", "local")

    # Build the mock LLM. NOTE: MockLLMClient iterates dict items in order and
    # returns the first key whose lowercased form is a substring of the prompt
    # (see src/llm_client.py::MockLLMClient.generate). Because both the
    # planner and synthesizer prompts contain "SENIOR FINANCIAL ANALYST",
    # we must list the synthesizer-distinctive key FIRST so it wins for
    # synthesis prompts; the planner prompt then falls through to the
    # broader "SENIOR FINANCIAL ANALYST" key.
    from src.llm_client import MockLLMClient

    mock_llm = MockLLMClient(
        responses={
            SYNTHESIZER_DISTINCT_SUBSTRING: SYNTHESIZER_RESPONSE_TEXT,
            "SENIOR FINANCIAL ANALYST": PLANNER_RESPONSE_JSON,
        }
    )

    # Import RAGAgent AFTER monkeypatching so storage classes pick up the
    # patched config paths during __init__.
    from src.rag_agent import RAGAgent

    agent = RAGAgent(llm_client=mock_llm)

    # Ingest the sample fixture.
    asyncio.run(agent.ingest_document(FIXTURE_PDF))

    # Run the query.
    response = asyncio.run(agent.query("What was Acme revenue?"))

    # Assertions.
    assert len(agent.list_documents()) >= 1
    assert isinstance(response.answer, str)
    assert response.answer  # non-empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
