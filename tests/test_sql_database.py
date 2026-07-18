"""Unit tests for sql_database.py — SQLite job queue, media records, search, hash dedup.

Tests use an in-memory SQLite DB ( monkeypatched DATABASE_PATH ) so no
real data is touched.  No AI models required.
"""

import os
import sqlite3
import tempfile
import hashlib

import pytest

from backend.search_and_index import sql_database as db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Redirect all DB + vector paths to a temp dir."""
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    monkeypatch.setattr(db, "DATABASE_PATH", str(db_dir / "test.db"))
    monkeypatch.setattr(db, "VECTOR_DB_PATH", str(db_dir / "vectors"))
    monkeypatch.setattr(db, "THUMBNAIL_PATH", str(tmp_path / "thumbs"))
    db.initialize_db()
    yield db


@pytest.fixture
def temp_file(tmp_path):
    f = tmp_path / "sample.mp4"
    f.write_bytes(b"fake video content")
    return str(f)


# ---------------------------------------------------------------------------
# initialize_db
# ---------------------------------------------------------------------------

class TestInitializeDB:
    def test_creates_all_tables(self, temp_db):
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "media_files" in tables
        assert "transcripts_fts" in tables
        assert "indexing_jobs" in tables
        assert "app_settings" in tables

    def test_creates_onboarding_setting(self, temp_db):
        """Onboarding default should be false."""
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key='onboarding_completed'"
            ).fetchone()
        assert row is not None
        assert row[0] == "false"


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_returns_sha256_hex(self, temp_file):
        h = db.compute_file_hash(temp_file)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_consistent_for_same_content(self, temp_file):
        h1 = db.compute_file_hash(temp_file)
        h2 = db.compute_file_hash(temp_file)
        assert h1 == h2

    def test_changes_when_content_changes(self, temp_file):
        h1 = db.compute_file_hash(temp_file)
        with open(temp_file, "wb") as f:
            f.write(b"different content")
        h2 = db.compute_file_hash(temp_file)
        assert h1 != h2


# ---------------------------------------------------------------------------
# should_process
# ---------------------------------------------------------------------------

class TestShouldProcess:
    def test_new_file_should_process(self, temp_db, temp_file):
        should, h = temp_db.should_process(temp_file)
        assert should is True
        assert h is not None

    def test_nonexistent_file_should_not_process(self, temp_db):
        should, h = temp_db.should_process("/nonexistent/file.mp4")
        assert should is False
        assert h is None

    def test_unchanged_file_should_not_process(self, temp_db, temp_file):
        temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "hello"}],
            current_hash=temp_db.compute_file_hash(temp_file),
        )
        should, h = temp_db.should_process(temp_file)
        assert should is False

    def test_changed_file_should_process(self, temp_db, temp_file):
        temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "hello"}],
            current_hash="old_hash",
        )
        should, h = temp_db.should_process(temp_file)
        assert should is True


# ---------------------------------------------------------------------------
# save_to_db
# ---------------------------------------------------------------------------

class TestSaveToDB:
    def test_inserts_new_media(self, temp_db, temp_file):
        media_id = temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "hello world"}],
        )
        assert media_id is not None
        assert isinstance(media_id, int)

    def test_updates_existing_media(self, temp_db, temp_file):
        mid1 = temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "hello"}],
        )
        mid2 = temp_db.save_to_db(
            temp_file, "sample.mp4", 62.0,
            [{"start": 0, "end": 5, "text": "updated transcript"}],
        )
        assert mid1 == mid2

    def test_transcript_fts_populated(self, temp_db, temp_file):
        temp_db.save_to_db(
            temp_file, "sample.mp4", 60.0,
            [{"start": 0, "end": 5, "text": "unique keyword applesauce"}],
        )
        results = temp_db.search_to_json("applesauce")
        assert len(results) >= 1
        assert "applesauce" in results[0]["text"]


# ---------------------------------------------------------------------------
# search_to_json (keyword search)
# ---------------------------------------------------------------------------

class TestSearchToJSON:
    def test_finds_matching_text(self, temp_db, tmp_path):
        f1 = tmp_path / "a.mp4"
        f1.write_bytes(b"a")
        f2 = tmp_path / "b.mp4"
        f2.write_bytes(b"b")
        temp_db.save_to_db(
            str(f1), "a.mp4", 10.0,
            [{"start": 0, "end": 5, "text": "machine learning is fun"}],
        )
        temp_db.save_to_db(
            str(f2), "b.mp4", 10.0,
            [{"start": 0, "end": 5, "text": "cooking pasta recipe"}],
        )
        results = temp_db.search_to_json("machine")
        assert len(results) >= 1
        assert "machine" in results[0]["text"].lower()

    def test_returns_empty_for_no_match(self, temp_db):
        results = temp_db.search_to_json("nonexistentword12345")
        assert results == []

    def test_results_have_expected_fields(self, temp_db, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        temp_db.save_to_db(
            str(f), "v.mp4", 10.0,
            [{"start": 1, "end": 2, "text": "searchable content here"}],
        )
        results = temp_db.search_to_json("searchable")
        assert len(results) >= 1
        r = results[0]
        assert "file_name" in r
        assert "file_path" in r
        assert "start" in r
        assert "end" in r
        assert "text" in r
        assert "score" in r


# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------

class TestJobQueue:
    def test_enqueue_creates_job(self, temp_db, temp_file):
        job_id, created = temp_db.enqueue_job(temp_file)
        assert created is True
        assert job_id > 0

    def test_enqueue_duplicate_returns_existing(self, temp_db, temp_file):
        jid1, c1 = temp_db.enqueue_job(temp_file)
        jid2, c2 = temp_db.enqueue_job(temp_file)
        assert c1 is True
        assert c2 is False
        assert jid1 == jid2

    def test_fetch_next_job(self, temp_db, temp_file):
        temp_db.enqueue_job(temp_file)
        job = temp_db.fetch_next_job()
        assert job is not None
        assert job["file_path"] == os.path.abspath(temp_file)
        # fetch_next_job transitions queued→running but returns the original row dict
        # (status field may not be in returned dict; verify via DB)
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute(
                "SELECT status FROM indexing_jobs WHERE id=?", (job["id"],)
            ).fetchone()
        assert row[0] == "running"

    def test_fetch_next_job_empty(self, temp_db):
        job = temp_db.fetch_next_job()
        assert job is None

    def test_update_job_status(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.fetch_next_job()  # mark running
        temp_db.update_job_status(jid, "done", stage="finished", progress=1.0)
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute(
                "SELECT status, stage, progress FROM indexing_jobs WHERE id=?", (jid,)
            ).fetchone()
        assert row[0] == "done"
        assert row[1] == "finished"
        assert row[2] == 1.0

    def test_increment_retry(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.increment_retry(jid)
        retries, max_retries = temp_db.get_job_retries(jid)
        assert retries == 1
        assert max_retries == 3

    def test_requeue_job(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.fetch_next_job()  # running
        temp_db.requeue_job(jid)
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute(
                "SELECT status, stage FROM indexing_jobs WHERE id=?", (jid,)
            ).fetchone()
        assert row[0] == "queued"
        assert row[1] == "pending"

    def test_reset_stale_running_jobs(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file)
        temp_db.fetch_next_job()  # running
        # simulate crash: job stuck in running
        temp_db.reset_stale_running_jobs()
        with sqlite3.connect(temp_db.DATABASE_PATH) as conn:
            row = conn.execute(
                "SELECT status FROM indexing_jobs WHERE id=?", (jid,)
            ).fetchone()
        assert row[0] == "queued"

    def test_max_retries_respected(self, temp_db, temp_file):
        jid, _ = temp_db.enqueue_job(temp_file, max_retries=2)
        temp_db.increment_retry(jid)
        temp_db.increment_retry(jid)
        retries, max_r = temp_db.get_job_retries(jid)
        assert retries == 2
        assert max_r == 2
