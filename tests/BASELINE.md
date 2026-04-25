# Test Baseline

**Result (initial baseline):** 109 passed, 1 skipped (no failures, no errors)

**Result (after dropping import-existence-only tests in `test_tools_fixes.py`):** 98 passed, 1 skipped

**Result (after adding `tests/test_smoke.py`):** 99 passed, 1 skipped — smoke test passes today (no `xfail` needed).

## Per-suite results

- PASS — tests/test_calculator.py (17 passed)
- PASS — tests/test_common_refactor.py (6 passed)
- PASS — tests/test_docling_extractor.py (9 passed, 1 skipped — `test_real_pdf_extraction` is gated behind `pytest.mark.integration` and skipped by default)
- PASS — tests/test_executor.py (8 passed)
- PASS — tests/test_prompt_injection.py (5 passed)
- PASS — tests/test_rag_agent_refactor.py (4 passed)
- PASS — tests/test_schema_clustering.py (15 passed)
- PASS — tests/test_scripts_fixes.py (6 passed)
- PASS — tests/test_sql_security.py (4 passed)
- PASS — tests/test_storage_fixes.py (16 passed)
- PASS — tests/test_tools_fixes.py (initially 19 passed; now 8 passed after removing `TestToolsExports`, `TestRerankerFixes`, and `TestVectorSearchFixes` — those classes only asserted symbols could be imported, not that they worked)
- PASS — tests/test_smoke.py (1 passed) — added in step 5; ingests `tests/fixtures/sample.pdf` and runs an end-to-end query against `MockLLMClient` with `EMBEDDING_PROVIDER=local`

No failing tests in the baseline. Six warnings observed:
- 5x `DeprecationWarning` from SWIG-generated bindings (third-party, ignorable).
- 1x `PytestUnknownMarkWarning` for `pytest.mark.integration` in `test_docling_extractor.py` (custom mark not registered).

## How to reproduce

```bash
pip3 install -r requirements-dev.txt && pytest tests/ -v
```

(Project runtime dependencies from `requirements.txt` must also already be installed.)
