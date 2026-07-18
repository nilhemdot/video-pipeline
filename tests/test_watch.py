"""Unit tests for watch.py — file watcher, debouncing, stability checks, initial scan.

watchdog is mocked via conftest. enqueue_job + initialize_db are monkeypatched.
"""

import os
from unittest.mock import MagicMock

from backend.search_and_index import watch


class TestFileHandlerHelpers:
    def test_supported_extensions(self):
        assert ".mp4" in watch.SUPPORTED_EXTENSIONS
        assert ".pdf" in watch.SUPPORTED_EXTENSIONS
        assert ".md" in watch.SUPPORTED_EXTENSIONS
        assert ".txt" in watch.SUPPORTED_EXTENSIONS

    def test_temp_suffixes(self):
        assert ".tmp" in watch.TEMP_SUFFIXES
        assert ".part" in watch.TEMP_SUFFIXES
        assert ".crdownload" in watch.TEMP_SUFFIXES


class TestIsTemporaryFile:
    def setup_method(self):
        self.handler = watch.FileHandler()

    def test_temp_suffix(self):
        assert self.handler._is_temporary_file("/path/file.tmp") is True
        assert self.handler._is_temporary_file("/path/file.part") is True
        assert self.handler._is_temporary_file("/path/file.partial") is True

    def test_temp_name(self):
        assert self.handler._is_temporary_file("/path/thumbs.db") is True
        assert self.handler._is_temporary_file("/path/.ds_store") is True

    def test_office_temp(self):
        assert self.handler._is_temporary_file("/path/.~file.docx") is True
        assert self.handler._is_temporary_file("/path/~$file.xlsx") is True

    def test_normal_file(self):
        assert self.handler._is_temporary_file("/path/video.mp4") is False
        assert self.handler._is_temporary_file("/path/doc.pdf") is False

    def test_swp_file(self):
        assert self.handler._is_temporary_file("/path/.file.swp") is True
        assert self.handler._is_temporary_file("/path/.file.swx") is True


class TestIsFileStable:
    def setup_method(self):
        self.handler = watch.FileHandler()
        # Speed up stability checks for tests
        self.handler._stability_wait_seconds = 0.01
        self.handler._stability_checks = 2

    def test_stable_file(self, tmp_path):
        f = tmp_path / "stable.mp4"
        f.write_bytes(b"content")
        assert self.handler._is_file_stable(str(f)) is True

    def test_nonexistent_file(self):
        assert self.handler._is_file_stable("/nonexistent/file.mp4") is False

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")
        assert self.handler._is_file_stable(str(f)) is False

    def test_file_deleted_during_check(self, tmp_path, monkeypatch):
        f = tmp_path / "deleting.mp4"
        f.write_bytes(b"content")
        path = str(f)

        original_exists = os.path.exists
        call_count = [0]

        def mock_exists(p):
            call_count[0] += 1
            if call_count[0] > 2:
                return False
            return original_exists(p)

        monkeypatch.setattr(os.path, "exists", mock_exists)
        assert self.handler._is_file_stable(path) is False


class TestFileHandlerHandle:
    def setup_method(self):
        self.handler = watch.FileHandler()
        self.handler._debounce_seconds = 0.01

    def test_ignores_unsupported_extension(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_bytes(b"x")
        self.handler._handle(str(f))
        assert str(f) not in self.handler._timers

    def test_ignores_temp_file(self, tmp_path):
        f = tmp_path / "file.tmp"
        f.write_bytes(b"x")
        self.handler._handle(str(f))
        assert str(f) not in self.handler._timers

    def test_queues_supported_file(self, tmp_path, monkeypatch):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"x")
        monkeypatch.setattr(watch, "enqueue_job", MagicMock(return_value=(1, True)))
        self.handler._handle(str(f))
        assert str(f) in self.handler._timers

    def test_debounce_replaces_timer(self, tmp_path, monkeypatch):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"x")
        monkeypatch.setattr(watch, "enqueue_job", MagicMock(return_value=(1, True)))
        self.handler._handle(str(f))
        first_timer = self.handler._timers[str(f)]
        self.handler._handle(str(f))
        assert self.handler._timers[str(f)] is not first_timer


class TestProcessAfterDebounce:
    def setup_method(self):
        self.handler = watch.FileHandler()
        self.handler._debounce_seconds = 0.01
        self.handler._stability_wait_seconds = 0.01
        self.handler._stability_checks = 1

    def test_enqueues_stable_file(self, tmp_path, monkeypatch):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"content")
        mock_enqueue = MagicMock(return_value=(5, True))
        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        monkeypatch.setattr(watch, "initialize_db", lambda: None)
        self.handler._process_after_debounce(str(f))
        mock_enqueue.assert_called_once()
        args = mock_enqueue.call_args
        assert args[0][0] == str(f) or args.kwargs.get("path") == str(f)

    def test_skips_nonexistent_file(self, monkeypatch):
        mock_enqueue = MagicMock()
        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        monkeypatch.setattr(watch, "initialize_db", lambda: None)
        self.handler._process_after_debounce("/nonexistent.mp4")
        mock_enqueue.assert_not_called()

    def test_skips_unstable_file(self, tmp_path, monkeypatch):
        f = tmp_path / "unstable.mp4"
        f.write_bytes(b"content")
        monkeypatch.setattr(self.handler, "_is_file_stable", lambda p: False)
        mock_enqueue = MagicMock()
        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        monkeypatch.setattr(watch, "initialize_db", lambda: None)
        self.handler._process_after_debounce(str(f))
        mock_enqueue.assert_not_called()

    def test_source_type_mapping(self, tmp_path, monkeypatch):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"content")
        mock_enqueue = MagicMock(return_value=(1, True))
        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        monkeypatch.setattr(watch, "initialize_db", lambda: None)
        self.handler._process_after_debounce(str(f))
        args = mock_enqueue.call_args
        assert "pdf" in str(args)

    def test_note_source_type(self, tmp_path, monkeypatch):
        f = tmp_path / "note.md"
        f.write_bytes(b"content")
        mock_enqueue = MagicMock(return_value=(1, True))
        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        monkeypatch.setattr(watch, "initialize_db", lambda: None)
        self.handler._process_after_debounce(str(f))
        args = mock_enqueue.call_args
        assert "note" in str(args)


class TestInitialScan:
    def test_scans_supported_files(self, tmp_path, monkeypatch):
        (tmp_path / "a.mp4").write_bytes(b"a")
        (tmp_path / "b.pdf").write_bytes(b"b")
        (tmp_path / "c.md").write_bytes(b"c")
        (tmp_path / "d.txt").write_bytes(b"d")
        (tmp_path / "e.xyz").write_bytes(b"e")  # unsupported

        queued = []

        def mock_enqueue(path, source_type=None):
            queued.append((path, source_type))
            return (len(queued), True)

        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        watch.initial_scan(str(tmp_path))
        assert len(queued) == 4  # a, b, c, d (not e)

    def test_handles_subdirectories(self, tmp_path, monkeypatch):
        subdir = tmp_path / "subfolder"
        subdir.mkdir()
        (subdir / "deep.mp4").write_bytes(b"x")

        queued = []
        monkeypatch.setattr(
            watch,
            "enqueue_job",
            lambda p, source_type=None: (queued.append(p), (1, True))[1],
        )
        watch.initial_scan(str(tmp_path))
        assert len(queued) == 1

    def test_skips_nonexistent_files(self, tmp_path, monkeypatch):
        # This is an edge case: file exists during os.walk but deleted before enqueue
        (tmp_path / "ghost.mp4").write_bytes(b"x")
        monkeypatch.setattr(
            watch, "enqueue_job", MagicMock(side_effect=Exception("file gone"))
        )
        # Should not raise
        watch.initial_scan(str(tmp_path))

    def test_empty_folder(self, tmp_path, monkeypatch):
        mock_enqueue = MagicMock()
        monkeypatch.setattr(watch, "enqueue_job", mock_enqueue)
        watch.initial_scan(str(tmp_path))
        mock_enqueue.assert_not_called()
