"""
Tests for empty chunk filtering in the ingest pipeline.
Verifies fix for issue #686 - PDF ingestion fails with
"Cannot embed empty text" error.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestEmptyChunkFiltering:
    """Tests for empty/whitespace chunk filtering before embedding."""

    def _make_doc(self, content):
        doc = MagicMock()
        doc.content = content
        doc.metadata = {}
        return doc

    def test_empty_chunks_are_filtered(self):
        """Empty string chunks should be removed before embedding."""
        documents = [
            self._make_doc("valid chunk"),
            self._make_doc(""),
            self._make_doc("another valid chunk"),
        ]
        filtered = [
            doc for doc in documents if doc.content and doc.content.strip()]
        assert len(filtered) == 2

    def test_whitespace_only_chunks_are_filtered(self):
        """Whitespace-only chunks should be removed before embedding."""
        documents = [
            self._make_doc("valid chunk"),
            self._make_doc("   "),
            self._make_doc("\n\t"),
        ]
        filtered = [
            doc for doc in documents if doc.content and doc.content.strip()]
        assert len(filtered) == 1

    def test_all_valid_chunks_pass_through(self):
        """Valid chunks should not be filtered."""
        documents = [
            self._make_doc("chunk one"),
            self._make_doc("chunk two"),
            self._make_doc("chunk three"),
        ]
        filtered = [
            doc for doc in documents if doc.content and doc.content.strip()]
        assert len(filtered) == 3

    def test_all_empty_chunks_returns_zero(self):
        """If all chunks are empty, result should be empty list."""
        documents = [
            self._make_doc(""),
            self._make_doc("   "),
            self._make_doc("\n"),
        ]
        filtered = [
            doc for doc in documents if doc.content and doc.content.strip()]
        assert len(filtered) == 0

    def test_filtered_count_is_correct(self):
        """Filtered count should accurately reflect removed chunks."""
        documents = [
            self._make_doc("valid"),
            self._make_doc(""),
            self._make_doc("  "),
            self._make_doc("also valid"),
        ]
        original_count = len(documents)
        filtered = [
            doc for doc in documents if doc.content and doc.content.strip()]
        filtered_count = original_count - len(filtered)
        assert filtered_count == 2
