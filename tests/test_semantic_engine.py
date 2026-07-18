"""Unit tests for semantic_engine.py — embeddings, sentence windowing, vector DB ops.

SentenceTransformer + LanceDB are mocked (no model download / GPU needed).
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.search_and_index import semantic_engine


class TestSentenceWindow:
    def test_single_sentence(self):
        data = [{"text": "hello", "start": 0, "end": 1}]
        result = semantic_engine.sentence_window(data, window_size=3)
        assert len(result) == 1
        assert result[0] == ["hello"]

    def test_window_size_3(self):
        data = [
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
            {"text": "d"},
            {"text": "e"},
        ]
        result = semantic_engine.sentence_window(data, window_size=3)
        assert len(result) == 5
        # Index 0: [a, b], index 2: [b, c, d], index 4: [d, e]
        assert result[0] == ["a", "b"]
        assert result[2] == ["b", "c", "d"]
        assert result[4] == ["d", "e"]

    def test_window_size_5(self):
        data = [{"text": str(i)} for i in range(5)]
        result = semantic_engine.sentence_window(data, window_size=5)
        assert len(result) == 5
        # Center item gets full window
        assert len(result[2]) == 5
        # Edge items get smaller windows
        assert len(result[0]) == 3  # indices 0, 1, 2
        assert len(result[4]) == 3  # indices 2, 3, 4

    def test_empty_input(self):
        assert semantic_engine.sentence_window([]) == []


class TestEmbed:
    @patch("backend.search_and_index.semantic_engine.get_model")
    def test_returns_list_of_lists(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_get_model.return_value = mock_model

        result = semantic_engine.embed(["test sentence"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("backend.search_and_index.semantic_engine.get_model")
    def test_multiple_sentences(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [[0.1], [0.2], [0.3]]
        mock_get_model.return_value = mock_model

        result = semantic_engine.embed(["a", "b", "c"])
        assert len(result) == 3


class TestGetModel:
    def test_raises_if_model_missing(self, monkeypatch):
        monkeypatch.setattr(semantic_engine, "_MODEL", None)
        monkeypatch.setattr(
            semantic_engine, "MODEL_SEMANTIC_PATH", "/nonexistent/model"
        )
        with pytest.raises(RuntimeError, match="Semantic model not found"):
            semantic_engine.get_model()

    @patch("backend.search_and_index.semantic_engine.SentenceTransformer")
    def test_lazy_loads_once(self, mock_st, monkeypatch):
        monkeypatch.setattr(semantic_engine, "_MODEL", None)
        import tempfile

        fake_model_dir = tempfile.mkdtemp()
        monkeypatch.setattr(semantic_engine, "MODEL_SEMANTIC_PATH", fake_model_dir)
        mock_st.return_value = MagicMock()
        m1 = semantic_engine.get_model()
        m2 = semantic_engine.get_model()
        assert m1 is m2
        assert mock_st.call_count == 1


class TestSaveToVectorDB:
    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_creates_new_table(self, mock_embed, mock_lancedb):
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db
        semantic_engine.save_to_vector_db(
            1, "v.mp4", "/v.mp4", [{"start": 0, "end": 5, "text": "hello"}]
        )
        mock_db.create_table.assert_called_once()

    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_adds_to_existing_table(self, mock_embed, mock_lancedb):
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["semantic_segments"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        semantic_engine.save_to_vector_db(
            1, "v.mp4", "/v.mp4", [{"start": 0, "end": 5, "text": "hello"}]
        )
        mock_table.add.assert_called_once()

    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_empty_transcript_returns_none(self, mock_embed, mock_lancedb):
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db
        result = semantic_engine.save_to_vector_db(1, "v.mp4", "/v.mp4", [])
        assert result is None


class TestSemanticSearch:
    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_returns_empty_if_no_table(self, mock_embed, mock_lancedb):
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db
        assert semantic_engine.semantic_search("query", 10) == []

    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_returns_formatted_results(self, mock_embed, mock_lancedb):
        mock_embed.return_value = [[0.1, 0.2]]
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["semantic_segments"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        result = semantic_engine.semantic_search("hello", 10)
        assert isinstance(result, list)


class TestSaveSummaryVector:
    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_creates_new_table(self, mock_embed, mock_lancedb):
        mock_embed.return_value = [[0.1, 0.2]]
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db
        semantic_engine.save_summary_vector(1, "v.mp4", "summary text")
        mock_db.create_table.assert_called_once()

    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_adds_to_existing_table(self, mock_embed, mock_lancedb):
        mock_embed.return_value = [[0.1, 0.2]]
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["summary_segments"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        semantic_engine.save_summary_vector(1, "v.mp4", "summary text")
        mock_table.add.assert_called_once()


class TestFileSearch:
    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_returns_empty_if_no_table(self, mock_embed, mock_lancedb):
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db
        assert semantic_engine.file_search("query", 5) == []

    @patch("backend.search_and_index.semantic_engine.lancedb")
    @patch("backend.search_and_index.semantic_engine.embed")
    def test_returns_results_without_vectors(self, mock_embed, mock_lancedb):
        mock_embed.return_value = [[0.1, 0.2]]
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["summary_segments"]
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.limit.return_value.to_pandas.return_value.to_dict.return_value = []
        mock_lancedb.connect.return_value = mock_db
        result = semantic_engine.file_search("test", 5)
        assert isinstance(result, list)
