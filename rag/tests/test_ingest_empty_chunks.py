"""
Tests for empty chunk filtering in the ingest pipeline.
Verifies fix for issue #686 - PDF ingestion fails with
"Cannot embed empty text" error.
"""
from unittest.mock import MagicMock, patch

from core.base import Document
from core.ingest_handler import IngestHandler


def _make_doc(content):
    return Document(content=content, metadata={"parser": "test"})


def _make_handler():
    handler = IngestHandler.__new__(IngestHandler)
    handler.namespace = "test"
    handler.project = "test"
    handler.database = "test"
    handler.dataset_name = None
    handler.data_processing_strategy = "test"
    handler.blob_processor = MagicMock()
    handler.embedder = MagicMock()
    handler.vector_store = MagicMock()
    handler.config = MagicMock()
    return handler


class TestEmptyChunkFiltering:
    """Tests that ingest_file filters empty chunks before embedding."""

    def test_empty_chunks_are_filtered(self):
        """Empty string chunks should never reach the embedder."""
        handler = _make_handler()
        handler.blob_processor.process_blob.return_value = [
            _make_doc("valid chunk"),
            _make_doc(""),
            _make_doc("another valid chunk"),
        ]
        handler.embedder.get_embedding_dimension.return_value = 3
        handler.embedder.validate_config.return_value = True
        handler.embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        handler.vector_store.add_documents.return_value = ["id1"]

        with patch("core.ingest_handler.EventLogger"), \
             patch("core.ingest_handler.is_valid_embedding", return_value=(True, None)):
            handler.ingest_file(b"fake pdf", {"filename": "test.pdf"})

        embedded_args = [call.args[0] for call in handler.embedder.embed.call_args_list]
        assert len(embedded_args) == 2

    def test_whitespace_only_chunks_are_filtered(self):
        """Whitespace-only chunks should never reach the embedder."""
        handler = _make_handler()
        handler.blob_processor.process_blob.return_value = [
            _make_doc("valid chunk"),
            _make_doc("   "),
            _make_doc("\n\t"),
        ]
        handler.embedder.get_embedding_dimension.return_value = 3
        handler.embedder.validate_config.return_value = True
        handler.embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        handler.vector_store.add_documents.return_value = ["id1"]

        with patch("core.ingest_handler.EventLogger"), \
             patch("core.ingest_handler.is_valid_embedding", return_value=(True, None)):
            handler.ingest_file(b"fake pdf", {"filename": "test.pdf"})

        embedded_args = [call.args[0] for call in handler.embedder.embed.call_args_list]
        assert len(embedded_args) == 1

    def test_all_valid_chunks_pass_through(self):
        """Valid chunks should all reach the embedder."""
        handler = _make_handler()
        handler.blob_processor.process_blob.return_value = [
            _make_doc("chunk one"),
            _make_doc("chunk two"),
            _make_doc("chunk three"),
        ]
        handler.embedder.get_embedding_dimension.return_value = 3
        handler.embedder.validate_config.return_value = True
        handler.embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        handler.vector_store.add_documents.return_value = ["id1"]

        with patch("core.ingest_handler.EventLogger"), \
             patch("core.ingest_handler.is_valid_embedding", return_value=(True, None)):
            handler.ingest_file(b"fake pdf", {"filename": "test.pdf"})

        embedded_args = [call.args[0] for call in handler.embedder.embed.call_args_list]
        assert len(embedded_args) == 3

    def test_all_empty_chunks_returns_error(self):
        """If all chunks are empty, ingest_file should return error and never call embedder."""
        handler = _make_handler()
        handler.blob_processor.process_blob.return_value = [
            _make_doc(""),
            _make_doc("   "),
            _make_doc("\n"),
        ]

        with patch("core.ingest_handler.EventLogger"):
            result = handler.ingest_file(b"fake pdf", {"filename": "test.pdf"})

        assert result["status"] == "error"
        assert result["reason"] == "all_chunks_empty"
        handler.embedder.embed.assert_not_called()

    def test_mixed_valid_and_empty_chunks(self):
        """Embedder should only receive the valid chunks from a mixed input."""
        handler = _make_handler()
        handler.blob_processor.process_blob.return_value = [
            _make_doc("valid"),
            _make_doc(""),
            _make_doc("  "),
            _make_doc("also valid"),
        ]
        handler.embedder.get_embedding_dimension.return_value = 3
        handler.embedder.validate_config.return_value = True
        handler.embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        handler.vector_store.add_documents.return_value = ["id1"]

        with patch("core.ingest_handler.EventLogger"), \
             patch("core.ingest_handler.is_valid_embedding", return_value=(True, None)):
            handler.ingest_file(b"fake pdf", {"filename": "test.pdf"})

        embedded_args = [call.args[0] for call in handler.embedder.embed.call_args_list]
        assert len(embedded_args) == 2
