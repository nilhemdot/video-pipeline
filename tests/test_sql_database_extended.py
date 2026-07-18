"""Additional unit tests for sql_database.py — list_jobs, retry, cancel, media detail, stats, backup.

Uses the same temp_db fixture as test_sql_database.py.
"""

import os
import sqlite3
import pytest

from backend.search_and_index import sql_database as db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect all DB paths to temp dir."""
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    monkeypatch.setattr(db, "DATABASE_PATH", str(db_dir / "test.db"))
    monkeypatch.setattr(db, "VECTOR_DB_PATH", str(db_dir / "vectors"))
    monkeypatch.setattr(db, "THUMBNAIL_PATH", str(tmp_path / "thumbs"))
    monkeypatch.setattr(db, "PROJECT_ROOT", str(tmp_path))
    db.initialize_db()
    yield db


@pytest.fixture
def temp_file(tmp_path):
    f = tmp_path / "sample.mp4"
    f.write_bytes(b"fake video content")
    return str(f)


class TestListJobs:
    def test_empty_jobs(self, temp_db):
        jobs = temp_db.list_jobs()
        assert jobs == []

    def test_lists_queued_job(self, temp_db, temp_file):
        temp_db.enqueue_job(temp_file)
        jobs = temp_db.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "queued"
        assert jobs[0]["file_path"] == os.path.abspath(temp_file)

    def test_filter_by_status(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.fetch_next_job()  # transitions to running
        jobs = temp_db.list_jobs(status="running")
        assert len(jobs) == 1
        jobs_queued = temp_db.list_jobs(status="queued")
        assert len(jobs_queued) == 0

    def test_limit_applied(self, temp_db, tmp_path):
        for i in range(5):
            f = tmp_path / f"v{i}.mp4"
            f.write_bytes(b"x")
            temp_db.enqueue_job(str(f))
        jobs = temp_db.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_ordered_by_created_at_desc(self, temp_db, tmp_path):
        f1 = tmp_path / "a.mp4"
        f1.write_bytes(b"unique content A")
        f2 = tmp_path / "b.mp4"
        f2.write_bytes(b"unique content B")
        jid1, _ = temp_db.enqueue_job(str(f1))
        jid2, _ = temp_db.enqueue_job(str(f2))
        assert jid1 > 0 and jid2 > 0, f"Expected both jobs to enqueue: {jid1}, {jid2}"
        jobs = temp_db.list_jobs()
        assert len(jobs) == 2
        job_ids = [j["id"] for j in jobs]
        assert jid1 in job_ids
        assert jid2 in job_ids


class TestRetryJob:
    def test_retry_failed_job(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        # Mark as failed
        temp_db.update_job_status(jid, "failed", stage="failed", error_message="test error")
        result = temp_db.retry_job(jid)
        assert result is True
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute("SELECT status, stage FROM indexing_jobs WHERE id=?", (jid,)).fetchone()
        assert row[0] == "queued"
        assert row[1] == "pending"

    def test_retry_cancelled_job(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.cancel_job(jid)
        result = temp_db.retry_job(jid)
        assert result is True

    def test_retry_queued_job_fails(self, temp_db, temp_file):
        """Can't retry a job that's already queued."""
        jid, _ = temp_db.enqueue_job(temp_file)
        result = temp_db.retry_job(jid)
        assert result is False  # already queued, not in failed/cancelled

    def test_retry_nonexistent_job(self, temp_db):
        result = temp_db.retry_job(99999)
        assert result is False


class TestCancelJob:
    def test_cancel_queued_job(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        result = temp_db.cancel_job(jid)
        assert result is True
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute("SELECT status FROM indexing_jobs WHERE id=?", (jid,)).fetchone()
        assert row[0] == "cancelled"

    def test_cancel_running_job(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.fetch_next_job()  # running
        result = temp_db.cancel_job(jid)
        assert result is True

    def test_cancel_done_job_fails(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.fetch_next_job()
        temp_db.update_job_status(jid, "done", stage="finished", progress=1.0)
        result = temp_db.cancel_job(jid)
        assert result is False  # can't cancel completed job

    def test_cancel_nonexistent_job(self, temp_db):
        result = temp_db.cancel_job(99999)
        assert result is False


class TestGetJob:
    def test_get_existing_job(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        job = temp_db.get_job(jid)
        assert job is not None
        assert job["id"] == jid
        assert job["file_path"] == os.path.abspath(temp_file)

    def test_get_nonexistent_job(self, temp_db):
        job = temp_db.get_job(99999)
        assert job is None


class TestGetMediaDetail:
    def test_get_existing_media(self, temp_db, temp_file):
        media_id = temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "hello"}],
        )
        detail = temp_db.get_media_detail(media_id)
        assert detail is not None
        assert detail["file_name"] == "sample.mp4"
        assert detail["duration_seconds"] == 60.0

    def test_get_nonexistent_media(self, temp_db):
        detail = temp_db.get_media_detail(99999)
        assert detail is None


class TestGetMediaSegments:
    def test_returns_segments(self, temp_db, temp_file):
        media_id = temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [
                {"start": 0, "end": 5, "text": "first segment"},
                {"start": 5, "end": 10, "text": "second segment"},
            ],
        )
        segments = temp_db.get_media_segments(media_id)
        assert len(segments) == 2
        assert segments[0]["text"] == "first segment"
        assert segments[1]["text"] == "second segment"

    def test_limit_applied(self, temp_db, temp_file):
        transcript = [{"start": i, "end": i+1, "text": f"seg{i}"} for i in range(10)]
        media_id = temp_db.save_to_db(temp_file, "sample.mp4", 60.0, transcript)
        segments = temp_db.get_media_segments(media_id, limit=5)
        assert len(segments) == 5


class TestGetDbStats:
    def test_empty_stats(self, temp_db):
        stats = temp_db.get_db_stats()
        assert stats["media_files"] == 0
        assert stats["jobs_total"] == 0
        assert stats["jobs_queued"] == 0

    def test_with_data(self, temp_db, tmp_path):
        f1 = tmp_path / "a.mp4"
        f1.write_bytes(b"a")
        f2 = tmp_path / "b.mp4"
        f2.write_bytes(b"b")
        temp_db.save_to_db(str(f1), "a.mp4", 10.0, [{"start": 0, "end": 1, "text": "a"}])
        temp_db.enqueue_job(str(f2))
        temp_db.enqueue_job(str(f1))  # re-enqueue (hash unchanged, returns -1)
        stats = temp_db.get_db_stats()
        assert stats["media_files"] == 1
        assert stats["jobs_total"] >= 1


class TestIntegrityCheck:
    def test_returns_dict(self, temp_db):
        result = temp_db.integrity_check()
        assert "sqlite_integrity" in result
        assert "vector_store" in result
        assert "database_path" in result

    def test_sqlite_ok(self, temp_db):
        result = temp_db.integrity_check()
        assert result["sqlite_integrity"] == "ok"


class TestCreateBackup:
    def test_creates_backup_dir(self, temp_db, tmp_path):
        result = temp_db.create_backup(label="test")
        assert "backup_path" in result
        assert os.path.exists(result["backup_path"])

    def test_backup_without_label(self, temp_db):
        result = temp_db.create_backup()
        assert "backup_path" in result
        assert "backup_" in result["backup_path"]

    def test_backup_copies_db(self, temp_db, temp_file):
        temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "hello"}],
        )
        result = temp_db.create_backup(label="test")
        backup_db = os.path.join(result["backup_path"], "brain.db")
        assert os.path.exists(backup_db)


class TestSettings:
    def test_get_default_setting(self, temp_db):
        val = temp_db.get_setting("nonexistent_key", "default_val")
        assert val == "default_val"

    def test_set_and_get_setting(self, temp_db):
        temp_db.set_setting("test_key", "test_value")
        val = temp_db.get_setting("test_key", "")
        assert val == "test_value"

    def test_update_existing_setting(self, temp_db):
        temp_db.set_setting("key", "first")
        temp_db.set_setting("key", "second")
        assert temp_db.get_setting("key", "") == "second"
