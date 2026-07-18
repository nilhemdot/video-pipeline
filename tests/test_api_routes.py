"""Unit tests for FastAPI API routes using TestClient.

All api_service functions are monkeypatched so no DB or AI models are needed.
Tests verify HTTP status codes, response envelope structure, and routing.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Create a TestClient with all backend services stubbed."""
    monkeypatch.setenv("TOBU_WATCH_FOLDER", str(tmp_path / "watch"))

    # Patch DB + worker before importing app (app starts worker in lifespan)
    from backend.search_and_index import sql_database

    monkeypatch.setattr(sql_database, "DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(sql_database, "VECTOR_DB_PATH", str(tmp_path / "vectors"))
    monkeypatch.setattr(sql_database, "THUMBNAIL_PATH", str(tmp_path / "thumbs"))
    monkeypatch.setattr(sql_database, "initialize_db", lambda: None)

    # Patch worker loop to do nothing
    from backend.search_and_index import runtime_service

    monkeypatch.setattr(runtime_service, "worker_loop", lambda **kw: None)

    # Patch the watch module's initial_scan + FileHandler so lifespan doesn't fail
    from backend.search_and_index import watch

    monkeypatch.setattr(watch, "initial_scan", lambda folder: None)
    monkeypatch.setattr(watch, "FileHandler", MagicMock)

    # Patch watchdog Observer in the api_app module's import scope
    mock_observer = MagicMock()
    mock_observer_class = MagicMock(return_value=mock_observer)
    # The api_app imports Observer inside lifespan, so patch at module level
    monkeypatch.setattr("watchdog.observers.Observer", mock_observer_class)

    from backend.search_and_index.api_app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_api_service(monkeypatch):
    """Stub all api_service functions with controllable mocks."""
    from backend.search_and_index import api_service

    mocks = {}
    funcs = [
        "health_status",
        "system_status",
        "run_integrity_check",
        "create_backup",
        "get_jobs",
        "get_job_or_none",
        "retry_job_by_id",
        "cancel_job_by_id",
        "search_hybrid",
        "search_semantic",
        "search_keyword",
        "ingest_file",
        "ingest_folder",
        "delete_file",
        "reindex_file",
        "get_media_by_id",
        "get_media_segments",
        "get_media_list",
    ]
    for fn_name in funcs:
        mock = MagicMock(return_value={})
        mocks[fn_name] = mock
        # raising=False: some functions may not exist on api_service yet
        monkeypatch.setattr(api_service, fn_name, mock, raising=False)

    return mocks


# ---------------------------------------------------------------------------
# Health + System routes
# ---------------------------------------------------------------------------


class TestSystemRoutes:
    def test_health_ok(self, client, mock_api_service):
        mock_api_service["health_status"].return_value = {
            "status": "up",
            "database": "ok",
        }
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["data"]["status"] == "up"

    def test_health_db_error(self, client, mock_api_service):
        mock_api_service["health_status"].return_value = {
            "status": "error",
            "database": "fail",
        }
        resp = client.get("/api/v1/health")
        assert resp.status_code == 503

    def test_system_status(self, client, mock_api_service):
        mock_api_service["system_status"].return_value = {"disk": "1GB", "jobs": 5}
        resp = client.get("/api/v1/system/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["jobs"] == 5

    def test_system_integrity(self, client, mock_api_service):
        mock_api_service["run_integrity_check"].return_value = {
            "ok": True,
            "errors": [],
        }
        resp = client.get("/api/v1/system/integrity")
        assert resp.status_code == 200
        assert resp.json()["data"]["ok"] is True

    def test_system_backup(self, client, mock_api_service):
        mock_api_service["create_backup"].return_value = {"path": "/backup/test.db"}
        resp = client.post("/api/v1/system/backup")
        assert resp.status_code == 200
        assert "path" in resp.json()["data"]


# ---------------------------------------------------------------------------
# Jobs routes
# ---------------------------------------------------------------------------


class TestJobsRoutes:
    def test_list_jobs_empty(self, client, mock_api_service):
        mock_api_service["get_jobs"].return_value = []
        resp = client.get("/api/v1/jobs/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["data"]["count"] == 0

    def test_list_jobs_with_items(self, client, mock_api_service):
        mock_api_service["get_jobs"].return_value = [
            {"id": 1, "file_path": "/v.mp4", "status": "done"}
        ]
        resp = client.get("/api/v1/jobs/")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1

    def test_list_jobs_with_status_filter(self, client, mock_api_service):
        mock_api_service["get_jobs"].return_value = []
        resp = client.get("/api/v1/jobs/?status=queued")
        assert resp.status_code == 200
        mock_api_service["get_jobs"].assert_called_with(status="queued", limit=100)

    def test_list_jobs_invalid_status(self, client, mock_api_service):
        resp = client.get("/api/v1/jobs/?status=invalid")
        assert resp.status_code == 422  # Pydantic validation error

    def test_get_job_found(self, client, mock_api_service):
        mock_api_service["get_job_or_none"].return_value = {
            "id": 5,
            "file_path": "/v.mp4",
            "source_type": "video",
            "status": "done",
            "stage": "finished",
            "progress": 1.0,
            "retries": 0,
            "max_retries": 3,
            "error_message": None,
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
        }
        resp = client.get("/api/v1/jobs/5")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == 5

    def test_get_job_not_found(self, client, mock_api_service):
        mock_api_service["get_job_or_none"].return_value = None
        resp = client.get("/api/v1/jobs/999")
        assert resp.status_code == 404

    def test_retry_job_success(self, client, mock_api_service):
        mock_api_service["retry_job_by_id"].return_value = True
        resp = client.post("/api/v1/jobs/5/retry")
        assert resp.status_code == 200
        assert resp.json()["data"]["retried"] is True

    def test_retry_job_not_found(self, client, mock_api_service):
        mock_api_service["retry_job_by_id"].return_value = False
        resp = client.post("/api/v1/jobs/999/retry")
        assert resp.status_code == 404

    def test_cancel_job_success(self, client, mock_api_service):
        mock_api_service["cancel_job_by_id"].return_value = True
        resp = client.post("/api/v1/jobs/5/cancel")
        assert resp.status_code == 200
        assert resp.json()["data"]["cancelled"] is True

    def test_cancel_job_not_found(self, client, mock_api_service):
        mock_api_service["cancel_job_by_id"].return_value = False
        resp = client.post("/api/v1/jobs/999/cancel")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Search routes
# ---------------------------------------------------------------------------


class TestSearchRoutes:
    def test_hybrid_search(self, client, mock_api_service):
        mock_api_service["search_hybrid"].return_value = [
            {"file_path": "/v.mp4", "score": 0.9, "matched_by": ["semantic"]}
        ]
        resp = client.post("/api/v1/search/hybrid", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["count"] == 1
        assert data["data"]["items"][0]["score"] == 0.9

    def test_hybrid_search_empty_query(self, client, mock_api_service):
        resp = client.post("/api/v1/search/hybrid", json={"query": ""})
        assert resp.status_code == 422  # Pydantic min_length=1

    def test_semantic_search(self, client, mock_api_service):
        mock_api_service["search_semantic"].return_value = [
            {"file_path": "/v.mp4", "score": 0.8}
        ]
        resp = client.post("/api/v1/search/semantic?query=test&limit=10")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1

    def test_semantic_search_missing_query(self, client, mock_api_service):
        resp = client.post("/api/v1/search/semantic")
        assert resp.status_code == 422

    def test_keyword_search(self, client, mock_api_service):
        mock_api_service["search_keyword"].return_value = [
            {"file_path": "/v.mp4", "score": 0.5}
        ]
        resp = client.post("/api/v1/search/keyword?query=test")
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 1


# ---------------------------------------------------------------------------
# Ingest routes
# ---------------------------------------------------------------------------


class TestIngestRoutes:
    def test_ingest_file(self, client, mock_api_service):
        mock_api_service["ingest_file"].return_value = {"job_id": 1, "created": True}
        resp = client.post("/api/v1/ingest/file", json={"file_path": "/v.mp4"})
        assert resp.status_code == 200
        assert resp.json()["data"]["job_id"] == 1

    def test_ingest_file_with_source_type(self, client, mock_api_service):
        mock_api_service["ingest_file"].return_value = {"job_id": 2, "created": True}
        resp = client.post(
            "/api/v1/ingest/file", json={"file_path": "/d.pdf", "source_type": "pdf"}
        )
        assert resp.status_code == 200
        mock_api_service["ingest_file"].assert_called_with("/d.pdf", "pdf", 3)

    def test_ingest_file_missing_path(self, client, mock_api_service):
        resp = client.post("/api/v1/ingest/file", json={})
        assert resp.status_code == 422

    def test_delete_file(self, client, mock_api_service):
        resp = client.delete("/api/v1/ingest/file?file_path=/v.mp4")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

    def test_ingest_folder(self, client, mock_api_service):
        mock_api_service["ingest_folder"].return_value = {"queued": 5, "skipped": 2}
        resp = client.post(
            "/api/v1/ingest/folder", json={"folder_path": "/media", "recursive": True}
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["queued"] == 5

    def test_reindex_file(self, client, mock_api_service):
        mock_api_service["reindex_file"].return_value = {"job_id": 3, "created": True}
        resp = client.post("/api/v1/ingest/reindex", json={"file_path": "/v.mp4"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Response envelope structure
# ---------------------------------------------------------------------------


class TestResponseEnvelope:
    def test_success_envelope_structure(self, client, mock_api_service):
        mock_api_service["get_jobs"].return_value = []
        resp = client.get("/api/v1/jobs/")
        data = resp.json()
        assert "ok" in data
        assert "data" in data
        assert data["ok"] is True

    def test_error_envelope_structure(self, client, mock_api_service):
        mock_api_service["get_job_or_none"].return_value = None
        resp = client.get("/api/v1/jobs/999")
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data
