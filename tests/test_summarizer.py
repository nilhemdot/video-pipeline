"""Unit tests for summarizer.py — DistilBART summarization.

transformers + torch are mocked via conftest.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.search_and_index import summarizer


class TestGetSummarizer:
    def test_raises_if_model_missing(self, monkeypatch):
        monkeypatch.setattr(summarizer, "_tokenizer", None)
        monkeypatch.setattr(summarizer, "_model", None)
        monkeypatch.setattr(summarizer, "MODEL_SUMMARIZER_PATH", "/nonexistent/model")
        with pytest.raises(RuntimeError, match="Summarizer model not found"):
            summarizer.get_summarizer()

    @patch("backend.search_and_index.summarizer.AutoTokenizer")
    @patch("backend.search_and_index.summarizer.AutoModelForSeq2SeqLM")
    def test_lazy_loads_once(self, mock_model_cls, mock_tok_cls, monkeypatch, tmp_path):
        monkeypatch.setattr(summarizer, "_tokenizer", None)
        monkeypatch.setattr(summarizer, "_model", None)
        monkeypatch.setattr(summarizer, "MODEL_SUMMARIZER_PATH", str(tmp_path))

        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = mock_model

        t1, m1 = summarizer.get_summarizer()
        t2, m2 = summarizer.get_summarizer()
        assert t1 is t2
        assert m1 is m2
        assert mock_tok_cls.from_pretrained.call_count == 1
        assert mock_model_cls.from_pretrained.call_count == 1


class TestSummaryGenerator:
    @patch("backend.search_and_index.summarizer.get_summarizer")
    def test_generates_summary_from_segments(self, mock_get, monkeypatch):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get.return_value = (mock_tokenizer, mock_model)

        # Mock tokenizer behavior
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "Generated summary"

        # Mock model.generate
        mock_model.generate.return_value = MagicMock()
        # Need decode to work on the generated IDs
        mock_tokenizer.decode.return_value = "Summary text"

        segments = [{"text": "This is content to summarize."}]
        result = summarizer.summary_generator(segments)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("backend.search_and_index.summarizer.get_summarizer")
    def test_accepts_string_input(self, mock_get):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get.return_value = (mock_tokenizer, mock_model)

        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Summary"

        result = summarizer.summary_generator("raw text input")
        assert isinstance(result, str)

    @patch("backend.search_and_index.summarizer.get_summarizer")
    def test_chunks_long_input(self, mock_get):
        """Input exceeding max_tokens (1024) should be chunked."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get.return_value = (mock_tokenizer, mock_model)

        # Simulate 2000 tokens (needs 2 chunks of 1024)
        mock_tokenizer.encode.return_value = list(range(2000))
        mock_tokenizer.decode.return_value = "chunk summary"

        segments = [{"text": "long content"}]
        summarizer.summary_generator(segments)
        # model.generate should be called twice (2 chunks)
        assert mock_model.generate.call_count >= 2

    @patch("backend.search_and_index.summarizer.get_summarizer")
    def test_joins_multiple_chunk_summaries(self, mock_get):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get.return_value = (mock_tokenizer, mock_model)

        mock_tokenizer.encode.return_value = list(range(2000))
        # Each decode returns a different summary
        mock_tokenizer.decode.side_effect = ["First summary", "Second summary"]

        result = summarizer.summary_generator([{"text": "content"}])
        assert "First summary" in result
        assert "Second summary" in result

    @patch("backend.search_and_index.summarizer.get_summarizer")
    def test_empty_segments(self, mock_get):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get.return_value = (mock_tokenizer, mock_model)

        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = ""

        result = summarizer.summary_generator([])
        assert isinstance(result, str)
