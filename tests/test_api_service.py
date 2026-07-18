"""Unit tests for api_service.py — service layer functions.

All sql_database functions are monkeypatched. No real DB needed.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.search_and_index import api_service


@pytest.fixture
def mock_db(monkeypatch):
    """Stub all sql_database functions."""
    from backend.search_and_index import sql_database
    mocks = {}
    funcs = [
        "initialize_db", "get_jobs", "list_jobs", "get_job", "retry_job", "cancel_job",
        "search_to_json", "enqueue_job", "get_db_stats", "integrity_check",
        "create_backup", "get_media_detail", "get_media_segments",
        "cancel_jobs_for_path", "delete_file_records", "get_setting", "set_setting",
    ]
    for fn_name in funcs:
        mock = MagicMock(return_value={})
        mocks[fn_name] = mock
        monkeypatch.setattr(sql_database, fn_name, mock, raising=False)
    return mocks


class TestHealthStatus:
    def test_healthy(self, mock_db):
        mock_db["initialize_db"].return_value = None
        result = api_service.health_status()
        assert result["status"] == "up"
        assert result["database"] == "ok"

    def test_db_error(self, mock_db):
        mock_db["initialize_db"].side_effect = Exception("DB error")
        result = api_service.health_status()
        assert result["status"] == "error"
        assert result["database"] == "fail"


class TestGetJobs:
    def test_calls_sql_database(self, mock_db):
        mock_db["list_jobs"].return_value = [{"id": 1}]
        result = api_service.get_jobs(status="queued", limit=50)
        assert len(result) == 1
        mock_db["list_jobs"].assert_called_once_with(status="queued", limit=50)

    def test_default_params(self, mock_db):
        mock_db["list_jobs"].return_value = []
        api_service.get_jobs()
        mock_db["list_jobs"].assert_called_once_with(status=None, limit=100)


class TestGetJobOrNone:
    def test_returns_job(self, mock_db):
        mock_db["get_job"].return_value = {"id": 5}
        result = api_service.get_job_or_none(5)
        assert result["id"] == 5

    def test_returns_none(self, mock_db):
        mock_db["get_job"].return_value = None
        result = api_service.get_job_or_none(999)
        assert result is None


class TestRetryJobById:
    def test_success(self, mock_db):
        mock_db["retry_job"].return_value = True
        assert api_service.retry_job_by_id(5) is True

    def test_failure(self, mock_db):
        mock_db["retry_job"].return_value = False
        assert api_service.retry_job_by_id(999) is False


class TestCancelJobById:
    def test_success(self, mock_db):
        mock_db["cancel_job"].return_value = True
        assert api_service.cancel_job_by_id(5) is True

    def test_failure(self, mock_db):
        mock_db["cancel_job"].return_value = False
        assert api_service.cancel_job_by_id(999) is False


class TestSearchKeyword:
    def test_returns_normalized_results(self, mock_db):
        mock_db["search_to_json"].return_value = [
            {"file_name": "v.mp4", "file_path": "/v.mp4", "start": 0, "end": 1, "text": "test", "score": 0.5}
        ]
        result = api_service.search_keyword("test")
        assert len(result) == 1
        assert "matched_by" in result[0]

    def test_empty_results(self, mock_db):
        mock_db["search_to_json"].return_value = []
        result = api_service.search_keyword("nothing")
        assert result == []


class TestIngestFile:
    def test_calls_enqueue(self, mock_db):
        mock_db["enqueue_job"].return_value = (1, True)
        result = api_service.ingest_file("/v.mp4", "video", 3)
        assert result["job_id"] == 1
        assert result["created"] is True


class TestIngestFolder:
    def test_queues_files(self, mock_db, tmp_path):
        f1 = tmp_path / "a.mp4"
        f1.write_bytes(b"a")
        f2 = tmp_path / "b.pdf"
        f2.write_bytes(b"b")
        (tmp_path / "c.xyz").write_bytes(b"c")  # unsupported

        mock_db["enqueue_job"].return_value = (1, True)
        result = api_service.ingest_folder(str(tmp_path), recursive=True)
        assert result["queued"] == 2

    def test_skips_duplicates(self, mock_db, tmp_path):
        f1 = tmp_path / "a.mp4"
        f1.write_bytes(b"a")
        mock_db["enqueue_job"].return_value = (1, False)
        result = api_service.ingest_folder(str(tmp_path))
        assert result["skipped_duplicates"] == 1

    def test_non_recursive(self, mock_db, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "deep.mp4").write_bytes(b"x")
        (tmp_path / "top.mp4").write_bytes(b"y")
        mock_db["enqueue_job"].return_value = (1, True)
        result = api_service.ingest_folder(str(tmp_path), recursive=False)
        assert result["queued"] == 1  # only top.mp4


class TestReindexFile:
    def test_cancels_deletes_reingests(self, mock_db):
        mock_db["enqueue_job"].return_value = (2, True)
        result = api_service.reindex_file("/v.mp4")
        mock_db["cancel_jobs_for_path"].assert_called_once_with("/v.mp4")
        mock_db["delete_file_records"].assert_called_once_with("/v.mp4")
        assert result["job_id"] == 2


class TestDeleteFile:
    def test_cancels_and_deletes(self, mock_db):
        api_service.delete_file("/v.mp4")
        mock_db["cancel_jobs_for_path"].assert_called_once_with("/v.mp4")
        mock_db["delete_file_records"].assert_called_once_with("/v.mp4")


class TestSystemStatus:
    def test_returns_health_and_stats(self, mock_db):
        mock_db["get_db_stats"].return_value = {"media_files": 5, "jobs_total": 10}
        result = api_service.system_status()
        assert "health" in result
        assert "db_stats" in result
        assert result["db_stats"]["media_files"] == 5


class TestRunIntegrityCheck:
    def test_delegates_to_db(self, mock_db):
        mock_db["integrity_check"].return_value = {"sqlite_integrity": "ok"}
        result = api_service.run_integrity_check()
        assert result["sqlite_integrity"] == "ok"


class TestCreateBackup:
    def test_delegates_to_db(self, mock_db):
        mock_db["create_backup"].return_value = {"backup_path": "/backup"}
        result = api_service.create_backup(label="test")
        assert result["backup_path"] == "/backup"


class TestOnboarding:
    def test_get_onboarding_status_false(self, mock_db):
        mock_db["get_setting"].return_value = "false"
        assert api_service.get_onboarding_status() is False

    def test_get_onboarding_status_true(self, mock_db):
        mock_db["get_setting"].return_value = "true"
        assert api_service.get_onboarding_status() is True

    def test_set_onboarding_completed(self, mock_db):
        api_service.set_onboarding_completed(True)
        mock_db["set_setting"].assert_called_once_with("onboarding_completed", "true")


class TestNormalizeResultItem:
    def test_normalizes_fields(self):
        item = {
            "file_name": "v.mp4", "file_path": "/v.mp4",
            "start": 0, "end": 1, "text": "hello",
            "score": 0.9, "matched_by": ["semantic"],
            "source_type": "video", "added_at": "2026-01-01",
        }
        result = api_service.normalize_result_item(item)
        assert result["file_name"] == "v.mp4"
        assert result["score"] == 0.9
        assert result["matched_by"] == ["semantic"]
        assert result["semantic_rank"] is None
        assert result["keyword_rank"] is None

    def test_handles_matched_by_alias(self):
        item = {"file_path": "/v.mp4", "score": 0.5, "matched-by": ["keyword"]}
        result = api_service.normalize_result_item(item)
        assert result["matched_by"] == ["keyword"]

    def test_defaults_missing_fields(self):
        item = {"file_path": "/v.mp4"}
        result = api_service.normalize_result_item(item)
        assert result["score"] == 0.0
        assert result["matched_by"] == []
        assert result["file_name"] is None
