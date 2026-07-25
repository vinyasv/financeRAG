"""Tests for the native SQL table registry (`spreadsheet_tables`).

`SQLiteStore.list_tables()` reads `extracted_tables`, whose `table_name` is a
LOGICAL PDF table name and is never populated for spreadsheets. The registry of
REAL queryable tables lives in `spreadsheet_tables`, exposed via
`SQLiteStore.list_native_tables()`. These tests pin the three RAGAgent call
sites that must consult the native registry:

  * `_document_ingestion_complete` -> spreadsheet re-ingest is a no-op
  * `get_stats()["table_count"]`   -> counts real tables
  * `query()` available_tables     -> passes queryable names to the planner

Follows the repo test style: sync `def test_*` with `asyncio.run(...)` inline
(no pytest-asyncio), config paths redirected under tmp_path as in
tests/test_smoke.py, and RAGAgent imported AFTER patching.
"""

import asyncio
from pathlib import Path

CSV_CONTENT = (
    "company,quarter,revenue_musd\n"
    "Acme,Q1,100\n"
    "Acme,Q2,125\n"
    "Globex,Q1,80\n"
    "Globex,Q2,95\n"
)


def _make_agent(tmp_path, monkeypatch, llm_client=None):
    """Redirect all config paths under tmp_path and build a RAGAgent."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

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

    # Import AFTER patching so storage picks up the patched paths in __init__.
    from src.rag_agent import RAGAgent

    return RAGAgent(llm_client=llm_client)


def _write_csv(tmp_path: Path, name: str = "quarterly.csv") -> Path:
    csv_path = tmp_path / name
    csv_path.write_text(CSV_CONTENT)
    return csv_path


def test_ingestion_complete_true_after_first_csv_ingest(tmp_path, monkeypatch):
    """Regression: spreadsheet ingest must register as complete.

    `_document_ingestion_complete` previously gated on `list_tables()`, which
    is never written by `save_spreadsheet_native()`, so it always returned
    False and every re-ingest re-parsed the file from scratch.
    """
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = _write_csv(tmp_path)

    documents = asyncio.run(agent.ingest_document(csv_path))
    assert isinstance(documents, list) and documents

    for document in documents:
        assert agent._document_ingestion_complete(document.id, requires_table=True) is True
        # And the native registry knows about a real table for this document.
        assert agent.sqlite_store.list_native_tables(document_id=document.id)


def test_reingest_csv_is_a_genuine_noop(tmp_path, monkeypatch):
    """Second ingest of the same CSV must not re-save tables or re-embed."""
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = _write_csv(tmp_path)

    asyncio.run(agent.ingest_document(csv_path))

    from src.storage.sqlite_store import SQLiteStore

    save_calls: list[str] = []
    original_save = SQLiteStore.save_spreadsheet_native

    def spy_save(self, table_name, df, doc_id):
        save_calls.append(table_name)
        return original_save(self, table_name, df, doc_id)

    monkeypatch.setattr(SQLiteStore, "save_spreadsheet_native", spy_save)

    index_calls: list[int] = []
    original_index = agent._index_chunk_batch

    def spy_index(chunks):
        index_calls.append(len(chunks))
        return original_index(chunks)

    monkeypatch.setattr(agent, "_index_chunk_batch", spy_index)

    documents = asyncio.run(agent.ingest_document(csv_path))

    assert documents, "re-ingest should still return the existing documents"
    assert save_calls == [], f"sheet was reprocessed on re-ingest: {save_calls}"
    assert index_calls == [], f"chunks were re-indexed on re-ingest: {index_calls}"


def test_force_reingest_still_reprocesses(tmp_path, monkeypatch):
    """`force=True` must bypass the completeness gate and reprocess."""
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = _write_csv(tmp_path)

    asyncio.run(agent.ingest_document(csv_path))

    from src.storage.sqlite_store import SQLiteStore

    save_calls: list[str] = []
    original_save = SQLiteStore.save_spreadsheet_native

    def spy_save(self, table_name, df, doc_id):
        save_calls.append(table_name)
        return original_save(self, table_name, df, doc_id)

    monkeypatch.setattr(SQLiteStore, "save_spreadsheet_native", spy_save)

    asyncio.run(agent.ingest_document(csv_path, force=True))

    assert save_calls, "force=True should reprocess the sheet"


def test_get_stats_table_count_matches_native_tables(tmp_path, monkeypatch):
    """`table_count` must reflect real native tables, not extracted_tables."""
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = _write_csv(tmp_path)

    asyncio.run(agent.ingest_document(csv_path))

    native_tables = agent.sqlite_store.list_native_tables()
    assert len(native_tables) > 0

    stats = agent.get_stats()
    assert stats["table_count"] == len(native_tables)


def test_native_table_names_are_queryable(tmp_path, monkeypatch):
    """Every registered native name must resolve to a real SQL table."""
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = _write_csv(tmp_path)

    asyncio.run(agent.ingest_document(csv_path))

    native_tables = agent.sqlite_store.list_native_tables()
    assert native_tables

    for table in native_tables:
        rows = agent.sqlite_store.execute_query(
            f'SELECT * FROM "{table["table_name"]}"'
        )
        assert len(rows) == 4


def test_list_native_tables_filters_by_document(tmp_path, monkeypatch):
    """document_id filtering must narrow to that document's tables only."""
    agent = _make_agent(tmp_path, monkeypatch)
    first = _write_csv(tmp_path, "first.csv")
    second = _write_csv(tmp_path, "second.csv")

    first_docs = asyncio.run(agent.ingest_document(first))
    asyncio.run(agent.ingest_document(second))

    all_tables = agent.sqlite_store.list_native_tables()
    assert len(all_tables) >= 2

    doc_id = first_docs[0].id
    filtered = agent.sqlite_store.list_native_tables(document_id=doc_id)
    assert filtered
    assert {t["document_id"] for t in filtered} == {doc_id}
    assert len(filtered) < len(all_tables)


def test_list_tables_still_returns_extracted_tables_rows(tmp_path, monkeypatch):
    """Guard against over-correction: list_tables() must not change shape.

    It still reads `extracted_tables` and returns TableSchema objects with the
    LOGICAL table name, which is what tests/test_storage_fixes.py asserts on.
    """
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = _write_csv(tmp_path)
    asyncio.run(agent.ingest_document(csv_path))

    from src.models import ExtractedTable, TableSchema

    # A CSV writes only spreadsheet_tables, so extracted_tables stays empty.
    assert agent.sqlite_store.list_tables() == []

    table = ExtractedTable(
        id="tbl_1",
        document_id="doc_1",
        table_name="income_statement",
        page_number=3,
        schema_description="Logical PDF table",
        columns=["year", "revenue"],
        rows=[{"year": 2024, "revenue": 10}],
    )
    native_name = agent.sqlite_store.save_table(table)

    listed = agent.sqlite_store.list_tables()
    assert len(listed) == 1
    assert isinstance(listed[0], TableSchema)
    assert listed[0].table_name == "income_statement"
    assert listed[0].source_document_id == "doc_1"
    assert listed[0].page_number == 3

    # Filtering by document still works on extracted_tables.
    assert len(agent.sqlite_store.list_tables(document_id="doc_1")) == 1
    assert agent.sqlite_store.list_tables(document_id="nope") == []

    # The same save registers the QUERYABLE name in the native registry.
    native_names = {t["table_name"] for t in agent.sqlite_store.list_native_tables()}
    assert native_name in native_names
    assert "income_statement" not in native_names

    # list_spreadsheet_tables() delegates to list_native_tables() unfiltered.
    assert agent.sqlite_store.list_spreadsheet_tables() == agent.sqlite_store.list_native_tables()


# =============================================================================
# Round-2 regression: document-scoped native table names.
#
# Every test above ingests CSV, where spreadsheet_parser sets
# sheet_name = file_path.stem -- so names are unique BY CONSTRUCTION and the
# collision below is structurally impossible to hit. XLSX is where it bites,
# because "Sheet1" is the pandas/Excel default.
# =============================================================================

def _write_workbook(path: Path, sheet_name: str, marker: str) -> None:
    import pandas as pd

    frame = pd.DataFrame({"company": [marker], "revenue_musd": [42]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)


def test_two_workbooks_sharing_a_sheet_name_do_not_clobber_each_other(
    tmp_path, monkeypatch
):
    """`Sheet1` in two workbooks used to destroy each other's rows.

    spreadsheet_tables.table_name is the PRIMARY KEY and
    save_spreadsheet_native issues DROP TABLE IF EXISTS, so the second ingest
    replaced the first's registry row AND dropped its data -- which then made
    _document_ingestion_complete report False again, so re-ingesting the first
    clobbered the second, forever.
    """
    agent = _make_agent(tmp_path, monkeypatch)

    book_a = tmp_path / "alpha_report.xlsx"
    book_b = tmp_path / "beta_report.xlsx"
    _write_workbook(book_a, "Sheet1", "AlphaCorp")
    _write_workbook(book_b, "Sheet1", "BetaCorp")

    docs_a = asyncio.run(agent.ingest_document(book_a))
    docs_b = asyncio.run(agent.ingest_document(book_b))

    native = agent.sqlite_store.list_native_tables()
    assert len(native) == 2, (
        f"Both workbooks must get their own table; got {[t['table_name'] for t in native]}"
    )

    # Both documents' data must still be readable.
    markers = set()
    for table in native:
        rows = agent.sqlite_store.execute_query(
            f'SELECT company FROM "{table["table_name"]}"'
        )
        markers.update(row["company"] for row in rows)
    assert markers == {"AlphaCorp", "BetaCorp"}, (
        f"One workbook's rows were destroyed by the other; found {markers}"
    )

    # And neither document is left looking incomplete (the re-ingest ping-pong).
    for doc in (*docs_a, *docs_b):
        assert agent._document_ingestion_complete(doc.id, requires_table=True), (
            f"{doc.id} reported incomplete after a successful ingest"
        )

    assert agent.get_stats()["table_count"] == 2


def test_native_table_names_are_document_scoped_and_valid_identifiers(
    tmp_path, monkeypatch
):
    from src.storage.sqlite_store import validate_identifier

    agent = _make_agent(tmp_path, monkeypatch)
    book = tmp_path / "gamma_report.xlsx"
    _write_workbook(book, "Sheet1", "GammaCorp")
    (doc,) = asyncio.run(agent.ingest_document(book))

    (table,) = agent.sqlite_store.list_native_tables(document_id=doc.id)
    name = table["table_name"]

    validate_identifier(name)  # raises SecurityError if malformed
    assert name != "sheet1", "Name must be document-scoped, not the bare sheet name"
    assert "sheet1" in name, "Scoped name should stay readable in LLM prompts"


def test_empty_document_id_filter_does_not_fail_open(tmp_path, monkeypatch):
    """`if document_id:` treated "" as 'no filter' and returned every row.

    Because the completeness gate is `not list_native_tables(document_id=...)`,
    a fail-open filter would report a document complete and skip ingestion.
    """
    agent = _make_agent(tmp_path, monkeypatch)
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")
    asyncio.run(agent.ingest_document(csv_path))

    assert agent.sqlite_store.list_native_tables() != []
    assert agent.sqlite_store.list_native_tables(document_id="") == []


# =============================================================================
# Regression: narrow-sheet header detection.
#
# `_detect_header_row` gated on `first_non_null <= 2`, an absolute width, so a
# genuine 2-column header was mistaken for a title row: the first DATA row was
# promoted to the header, and a short sheet then came back empty and was
# dropped by `_parse_excel`'s `if df.empty: continue` -- silent data loss for
# every narrow workbook. The heuristic is "strictly fewer cells than the row
# below", not "at most two cells".
# =============================================================================

def test_two_column_sheet_keeps_its_real_header():
    import tempfile

    import pandas as pd

    from src.ingestion.spreadsheet_parser import SpreadsheetParser

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "narrow.xlsx"
        pd.DataFrame({"company": ["GammaCorp"], "revenue_musd": [42]}).to_excel(
            path, sheet_name="Sheet1", index=False
        )

        parsed = SpreadsheetParser().parse(path)

    assert len(parsed.sheets) == 1, "Narrow sheet was silently dropped"
    assert parsed.sheets[0].headers == ["company", "revenue_musd"]
    assert parsed.sheets[0].row_count == 1


def test_title_row_above_header_is_still_detected():
    """Guard: the P1.8 behaviour this heuristic exists for must survive."""
    import tempfile

    import openpyxl

    from src.ingestion.spreadsheet_parser import SpreadsheetParser

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "title.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Annual Report 2024"])
        sheet.append(["company", "revenue", "year"])
        sheet.append(["Acme", 100, 2024])
        sheet.append(["Globex", 80, 2024])
        workbook.save(path)

        parsed = SpreadsheetParser().parse(path)

    assert parsed.sheets[0].headers == ["company", "revenue", "year"]
