"""Unit tests for model_downloader.py — model path config + ensure functions.

sentence_transformers + transformers + faster_whisper are mocked via conftest.
"""

import os
from unittest.mock import patch, MagicMock

from backend.search_and_index import model_downloader as md


class TestModelPaths:
    def test_model_dir_exists(self):
        assert hasattr(md, "MODEL_DIR")
        assert "models" in md.MODEL_DIR

    def test_visual_path(self):
        assert md.MODEL_VISUAL_PATH.endswith("clip-ViT-B-32")

    def test_semantic_path(self):
        assert md.MODEL_SEMANTIC_PATH.endswith("all-MiniLM-L6-v2")

    def test_summarizer_path(self):
        assert md.MODEL_SUMMARIZER_PATH.endswith("distilbart-cnn-6-6")

    def test_whisper_path(self):
        assert md.MODEL_WHISPER_PATH.endswith("whisper-distil-large-v3")


class TestEnsureSemanticModel:
    def test_skips_if_exists(self, monkeypatch):
        monkeypatch.setattr(md, "MODEL_SEMANTIC_PATH", "/fake/existing/path")
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        mock_st = MagicMock()
        monkeypatch.setattr(md, "SentenceTransformer", mock_st)

        md.ensure_semantic_model()
        mock_st.assert_not_called()  # already exists, skip download

    @patch("backend.search_and_index.model_downloader.SentenceTransformer")
    def test_downloads_if_missing(self, mock_st, monkeypatch, tmp_path):
        model_path = str(tmp_path / "semantic_model")
        monkeypatch.setattr(md, "MODEL_SEMANTIC_PATH", model_path)
        monkeypatch.setattr(os.path, "exists", lambda p: False)

        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        md.ensure_semantic_model()
        mock_st.assert_called_once_with(md.SEMANTIC_MODEL)
        mock_instance.save.assert_called_once_with(model_path)


class TestEnsureVisualModel:
    @patch("backend.search_and_index.model_downloader.SentenceTransformer")
    def test_downloads_if_missing(self, mock_st, monkeypatch, tmp_path):
        model_path = str(tmp_path / "visual_model")
        monkeypatch.setattr(md, "MODEL_VISUAL_PATH", model_path)
        monkeypatch.setattr(os.path, "exists", lambda p: False)

        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        md.ensure_visual_model()
        mock_st.assert_called_once_with(md.VISUAL_MODEL)
        mock_instance.save.assert_called_once_with(model_path)


class TestEnsureSummarizerModel:
    @patch("backend.search_and_index.model_downloader.AutoModelForSeq2SeqLM")
    @patch("backend.search_and_index.model_downloader.AutoTokenizer")
    def test_downloads_if_missing(self, mock_tok_cls, mock_model_cls, monkeypatch, tmp_path):
        model_path = str(tmp_path / "summarizer_model")
        monkeypatch.setattr(md, "MODEL_SUMMARIZER_PATH", model_path)
        monkeypatch.setattr(os.path, "exists", lambda p: False)

        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = mock_model

        md.ensure_summarizer_model()
        mock_tok_cls.from_pretrained.assert_called_once_with(md.SUMMARIZER_MODEL)
        mock_model_cls.from_pretrained.assert_called_once_with(md.SUMMARIZER_MODEL)
        mock_tok.save_pretrained.assert_called_once_with(model_path)
        mock_model.save_pretrained.assert_called_once_with(model_path)


class TestEnsureWhisperModel:
    @patch("backend.search_and_index.model_downloader.WhisperModel")
    def test_downloads_if_missing(self, mock_whisper_cls, monkeypatch, tmp_path):
        model_path = str(tmp_path / "whisper_model")
        monkeypatch.setattr(md, "MODEL_WHISPER_PATH", model_path)
        monkeypatch.setattr(os.path, "exists", lambda p: False)

        md.ensure_whisper_model()
        mock_whisper_cls.download_model.assert_called_once_with(
            md.WHISPER_MODEL,
            output_dir=model_path,
        )


class TestEnsureAllModels:
    @patch("backend.search_and_index.model_downloader.ensure_whisper_model")
    @patch("backend.search_and_index.model_downloader.ensure_summarizer_model")
    @patch("backend.search_and_index.model_downloader.ensure_visual_model")
    @patch("backend.search_and_index.model_downloader.ensure_semantic_model")
    def test_calls_all_ensure_functions(self, mock_sem, mock_vis, mock_sum, mock_whis):
        md.ensure_all_models()
        mock_sem.assert_called_once()
        mock_vis.assert_called_once()
        mock_sum.assert_called_once()
        mock_whis.assert_called_once()
