"""Ingestion-specific exception types."""


class ExtractionFailed(RuntimeError):
    """Raised when an extractor failed, distinct from finding no tables."""
