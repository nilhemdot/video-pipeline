# API Reference

Base URL: `http://127.0.0.1:8000/api/v1`

## System

### GET /health
Health check. Returns database status.

**Response:** `200 OK` or `503 Service Unavailable`
```json
{"ok": true, "data": {"status": "up", "database": "ok"}}
```

### GET /system/status
System status with DB stats.

### GET /system/integrity
Database integrity check (SQLite + LanceDB).

### POST /system/backup
Create a backup of the database + vectors + thumbnails.

## Jobs

### GET /jobs/
List indexing jobs.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | null | Filter: queued, running, failed, done, cancelled |
| `limit` | int | 100 | Max results (1-500) |

### GET /jobs/{job_id}
Get job details by ID. Returns `404` if not found.

### POST /jobs/{job_id}/retry
Retry a failed or cancelled job. Returns `404` if not retryable.

### POST /jobs/{job_id}/cancel
Cancel a queued or running job. Returns `404` if already finished.

## Search

### POST /search/hybrid
Hybrid semantic + keyword search with Reciprocal Rank Fusion (RRF).

**Request body:**
```json
{
  "query": "machine learning",
  "limit": 20,
  "semantic_limit": 40,
  "keyword_limit": 40,
  "k": 60,
  "source_types": ["video", "pdf"],
  "folders": ["/media/videos"],
  "date_from": "2026-01-01",
  "date_to": "2026-12-31",
  "min_score": 0.0
}
```

**Response:**
```json
{
  "ok": true,
  "data": {
    "query": "machine learning",
    "count": 5,
    "items": [
      {
        "file_name": "lecture.mp4",
        "file_path": "/media/lecture.mp4",
        "start": 12.5,
        "end": 18.3,
        "text": "Machine learning is a subset of AI",
        "score": 0.0328,
        "matched_by": ["semantic", "keyword"],
        "semantic_rank": 1,
        "keyword_rank": 3,
        "source_type": "video",
        "added_at": "2026-07-18 12:00:00"
      }
    ]
  }
}
```

### POST /search/semantic
Semantic-only search (vector similarity).

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query (min 1 char) |
| `limit` | int | No | Max results (default 20, max 200) |

### POST /search/keyword
Keyword-only search (SQLite FTS5).

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query |

## Ingest

### POST /ingest/file
Queue a single file for indexing.

**Request body:**
```json
{
  "file_path": "/path/to/video.mp4",
  "source_type": "video",
  "max_retries": 3
}
```

### DELETE /ingest/file
Delete a file from the index.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Yes | File path to remove |

### POST /ingest/folder
Queue all supported files in a folder.

**Request body:**
```json
{
  "folder_path": "/media/videos",
  "recursive": true
}
```

### POST /ingest/reindex
Cancel existing job + delete records + re-queue a file.

**Request body:**
```json
{
  "file_path": "/path/to/video.mp4"
}
```

## Response Envelope

All responses follow a standard envelope:

**Success:**
```json
{"ok": true, "data": {...}}
```

**Error:**
```json
{"ok": false, "error": {"code": "not_found", "message": "Job not found"}}
```
