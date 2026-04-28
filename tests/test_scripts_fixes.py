"""Tests for script bootstrap and shared CLI helpers."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_bootstrap_module():
    """Load the script bootstrap helper as a module."""
    bootstrap_path = Path(__file__).parent.parent / "scripts" / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("finance_rag_bootstrap", bootstrap_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestScriptBootstrap:
    """Scripts should share one bootstrap path instead of inlining sys.path edits."""

    def test_bootstrap_project_root_adds_repo_root(self):
        """Bootstrap helper should return and register the repo root."""
        module = load_bootstrap_module()
        project_root = module.bootstrap_project_root()

        assert project_root == Path(__file__).parent.parent
        assert str(project_root) in sys.path

    def test_scripts_use_shared_bootstrap_and_cli_common(self):
        """Both scripts should rely on shared bootstrap and CLI helpers."""
        repo_root = Path(__file__).parent.parent
        ingest_source = (repo_root / "scripts" / "ingest.py").read_text(encoding="utf-8")
        query_source = (repo_root / "scripts" / "query.py").read_text(encoding="utf-8")

        assert "import _bootstrap" in ingest_source
        assert "import _bootstrap" in query_source
        assert "PROJECT_ROOT = _bootstrap.PROJECT_ROOT" in ingest_source
        assert "PROJECT_ROOT = _bootstrap.PROJECT_ROOT" in query_source
        assert 'configure_cli_logging("ingest.log")' in ingest_source
        assert 'configure_cli_logging("query.log")' in query_source

    def test_ingest_status_does_not_advertise_removed_rule_based_fallback(self):
        """The ingestion CLI should describe the supported VLM/Docling chain."""
        repo_root = Path(__file__).parent.parent
        ingest_source = (repo_root / "scripts" / "ingest.py").read_text(encoding="utf-8")

        assert "rule-based fallback" not in ingest_source
        assert "Table extraction: VLM cloud, then Docling local fallback" in ingest_source
        assert "Table extraction: Docling local" in ingest_source


class TestCliCommon:
    """Shared CLI helpers should handle logging and exports."""

    def test_configure_cli_logging_writes_to_configured_data_dir(self, tmp_path, monkeypatch):
        """Logging helper should create log files under config.data_dir/logs."""
        from src.cli import common as cli_common

        monkeypatch.setattr(cli_common.config, "data_dir", tmp_path)
        cli_common.configure_cli_logging("test.log")

        logger = logging.getLogger("finance_rag_test")
        logger.info("hello from cli helper")
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_path = tmp_path / "logs" / "test.log"
        assert log_path.exists()
        assert "hello from cli helper" in log_path.read_text(encoding="utf-8")

    def test_export_helpers_create_files(self, tmp_path):
        """Shared export helpers should create CSV and JSON output files."""
        from src.cli.common import export_results_to_csv, export_results_to_json

        results = [
            {
                "query": "test",
                "answer": "result",
                "citations": "report.pdf, p.1",
                "response_time_ms": 12.3,
                "timestamp": "2026-01-01T00:00:00",
            }
        ]

        csv_path = tmp_path / "out.csv"
        json_path = tmp_path / "out.json"

        assert export_results_to_csv(csv_path, results) is True
        assert export_results_to_json(json_path, results) is True
        assert "query,answer,citations,response_time_ms,timestamp" in csv_path.read_text(encoding="utf-8")
        assert '"query": "test"' in json_path.read_text(encoding="utf-8")


class TestAnalystEvaluation:
    """Existing script safeguards should remain intact."""

    def test_has_encoding_on_file_writes(self):
        """File writes should specify utf-8 encoding."""
        eval_path = Path(__file__).parent.parent / "scripts" / "analyst_evaluation.py"
        content = eval_path.read_text(encoding="utf-8")

        assert "open(json_path, 'w', encoding='utf-8')" in content
        assert "open(output_path, 'w', encoding='utf-8')" in content

    def test_division_by_zero_guard(self):
        """Average calculation should handle empty results."""
        eval_path = Path(__file__).parent.parent / "scripts" / "analyst_evaluation.py"
        content = eval_path.read_text(encoding="utf-8")

        assert "if results else 0" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
