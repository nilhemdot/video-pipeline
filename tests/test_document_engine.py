"""Unit tests for document_engine.py — PDF + Markdown + TXT processing.

PyMuPDF (fitz) and frontmatter are mocked via conftest.
semantic_engine + sql_database functions are monkeypatched.
"""

from unittest.mock import MagicMock

from backend.search_and_index import document_engine


class TestProcessFile:
    def test_processes_txt_file(self, tmp_path, monkeypatch):
        f = tmp_path / "notes.txt"
        f.write_text("First paragraph.\n\nSecond paragraph.\n\nThird.")

        monkeypatch.setattr(document_engine, "summary_generator", lambda d: "summary")
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 42)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        media_id = document_engine.process_file(str(f))
        assert media_id == 42

    def test_processes_markdown_file(self, tmp_path, monkeypatch):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nContent here.\n\nMore content.")

        monkeypatch.setattr(document_engine, "summary_generator", lambda d: "summary")
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 10)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        media_id = document_engine.process_file(str(f))
        assert media_id == 10

    def test_splits_on_double_newline(self, tmp_path, monkeypatch):
        f = tmp_path / "multi.txt"
        f.write_text("Para one.\n\nPara two.\n\nPara three.")

        captured_segments = []
        def capture_summary(data):
            captured_segments.extend(data)
            return "summary"

        monkeypatch.setattr(document_engine, "summary_generator", capture_summary)
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 1)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        document_engine.process_file(str(f))
        assert len(captured_segments) == 3
        assert captured_segments[0]["text"] == "Para one."
        assert captured_segments[1]["text"] == "Para two."
        assert captured_segments[2]["text"] == "Para three."

    def test_skips_empty_paragraphs(self, tmp_path, monkeypatch):
        f = tmp_path / "empty_paras.txt"
        f.write_text("Real content.\n\n\n\n\n\nAlso real.")

        captured = []
        def capture(data):
            captured.extend(data)
            return "summary"

        monkeypatch.setattr(document_engine, "summary_generator", capture)
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 1)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        document_engine.process_file(str(f))
        assert len(captured) == 2  # empty paragraphs stripped

    def test_segments_have_page_field(self, tmp_path, monkeypatch):
        f = tmp_path / "page.txt"
        f.write_text("Content.")

        captured = []
        def capture(data):
            captured.extend(data)
            return "summary"

        monkeypatch.setattr(document_engine, "summary_generator", capture)
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 1)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        document_engine.process_file(str(f))
        assert captured[0]["page"] == 1  # flat files = page 1

    def test_returns_none_when_save_fails(self, tmp_path, monkeypatch):
        f = tmp_path / "fail.txt"
        f.write_text("Content.")

        monkeypatch.setattr(document_engine, "summary_generator", lambda d: "summary")
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: None)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        result = document_engine.process_file(str(f))
        assert result is None

    def test_calls_save_to_vector_db(self, tmp_path, monkeypatch):
        f = tmp_path / "call.txt"
        f.write_text("Content.")

        mock_save_vec = MagicMock()
        mock_save_summary = MagicMock()
        monkeypatch.setattr(document_engine, "summary_generator", lambda d: "summary")
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 5)
        monkeypatch.setattr(document_engine, "save_to_vector_db", mock_save_vec)
        monkeypatch.setattr(document_engine, "save_summary_vector", mock_save_summary)

        document_engine.process_file(str(f))
        mock_save_vec.assert_called_once()
        mock_save_summary.assert_called_once()


class TestProcessPdf:
    def test_processes_pdf(self, tmp_path, monkeypatch):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")

        # Mock fitz.open to return a fake document with 2 pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page one content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page two content"
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.load_page.side_effect = [mock_page1, mock_page2]
        mock_doc.close = MagicMock()

        import sys
        fitz_mock = sys.modules.get("fitz", MagicMock())
        fitz_mock.open.return_value = mock_doc

        captured = []
        def capture(data):
            captured.extend(data)
            return "summary"

        monkeypatch.setattr(document_engine, "summary_generator", capture)
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 7)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        media_id = document_engine.process_pdf(str(f))
        assert media_id == 7
        assert len(captured) == 2
        assert captured[0]["text"] == "Page one content"
        assert captured[0]["page"] == 1
        assert captured[1]["text"] == "Page two content"
        assert captured[1]["page"] == 2

    def test_skips_empty_pages(self, tmp_path, monkeypatch):
        f = tmp_path / "empty_pages.pdf"
        f.write_bytes(b"fake pdf")

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Has content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "   "  # whitespace only
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.load_page.side_effect = [mock_page1, mock_page2, mock_page3]
        mock_doc.close = MagicMock()

        import sys
        fitz_mock = sys.modules.get("fitz", MagicMock())
        fitz_mock.open.return_value = mock_doc

        captured = []
        def capture(data):
            captured.extend(data)
            return "summary"

        monkeypatch.setattr(document_engine, "summary_generator", capture)
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 1)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        document_engine.process_pdf(str(f))
        assert len(captured) == 1  # only non-empty pages
        assert captured[0]["text"] == "Has content"

    def test_returns_none_when_save_fails(self, tmp_path, monkeypatch):
        f = tmp_path / "fail.pdf"
        f.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content"
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()

        import sys
        fitz_mock = sys.modules.get("fitz", MagicMock())
        fitz_mock.open.return_value = mock_doc

        monkeypatch.setattr(document_engine, "summary_generator", lambda d: "summary")
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: None)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        result = document_engine.process_pdf(str(f))
        assert result is None

    def test_closes_document(self, tmp_path, monkeypatch):
        f = tmp_path / "close.pdf"
        f.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content"
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()

        import sys
        fitz_mock = sys.modules.get("fitz", MagicMock())
        fitz_mock.open.return_value = mock_doc

        monkeypatch.setattr(document_engine, "summary_generator", lambda d: "summary")
        monkeypatch.setattr(document_engine, "save_doc_to_db", lambda *a, **kw: 1)
        monkeypatch.setattr(document_engine, "save_to_vector_db", MagicMock())
        monkeypatch.setattr(document_engine, "save_summary_vector", MagicMock())

        document_engine.process_pdf(str(f))
        mock_doc.close.assert_called_once()
