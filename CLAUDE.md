# TOBU — Project Rules

## Identity
TOBU: fully offline desktop app for local video/PDF/note search. 100% local inference, no cloud, no telemetry.
Quality bar: production-grade for personal use. Every model runs on-device.

## Tech Stack
- Backend: Python 3.10+, FastAPI 0.115+, PyInstaller (frozen packaging)
- Frontend: React + Vite (JSX, not TS)
- Desktop: Electron 30
- Storage: SQLite (media_files, transcripts_fts, indexing_jobs, app_settings) + LanceDB (vector embeddings)
- AI Models (all local): Whisper distil-large-v3 (speech), CLIP ViT-B/32 (vision), MiniLM-L6-v2 (text embeddings), DistilBART-cnn-6-6 (summaries)
- Media: ffmpeg (audio extraction + frame sampling)

## Commands
- Backend dev: `cd backend/search_and_index && python main.py --mode worker`
- API server: `cd backend/search_and_index && python -m uvicorn api_app:app --port 8000`
- Frontend dev: `cd client && npm run dev`
- Electron dev: `npm start` (root)
- Smoke tests: `python test_api_smoke.py` (requires running server on :8000)
- Unit tests: `python -m pytest tests/ -v`
- Build backend: `python scripts/build_backend.py`
- Build desktop: `cd desktop && npm run dist`
- Model download: `python backend/search_and_index/model_downloader.py`

## Code Conventions
- Lazy model loading: singleton pattern (`_MODEL = None`, `get_model()` function). Never load at import time.
- Dual import paths: `if __package__:` for package mode, `else:` for script mode. Both must work.
- Frozen path detection: `getattr(sys, 'frozen', False)` → route data to `~/.tobu`. Dev → project root.
- SQLite: parameterized queries only (`?` placeholders). Never f-string SQL with user input.
- LanceDB: `table.delete(f"media_id = {int(media_id)}")` — cast to int before interpolation (current pattern, safe but smell).
- FastAPI routes: thin handlers, delegate logic to `api_service.py`. Pydantic models in `api_models.py`.
- Engine modules: one responsibility each (aural, visual, document, semantic, summarizer). No cross-engine imports.
- `runtime_service.py`: orchestrates engines + job processing + hybrid search. This is the only module that imports all engines.

## Boundaries
- Never add cloud dependencies. All models local, all data on-disk.
- Never commit model weights (gitignored `models/`).
- Never commit runtime data (gitignored `data/`).
- Never hardcode file paths — use `PROJECT_ROOT` pattern.
- Never load AI models at import time — use lazy `get_model()` pattern.
- Never use f-string SQL with untrusted input — parameterized queries only.
- Test before commit: `python -m pytest tests/ -v`.

## Patterns
- Job queue: SQLite-backed. Status enum: queued → running → done | failed | cancelled. Retry logic with max_retries=3.
- Watch folder: watchdog observer + 2s debounce + 1.2s stability wait. Handles create/modify/delete.
- Search: hybrid RRF (Reciprocal Rank Fusion). `runtime_service.hybrid_search_rrf()` merges semantic + keyword results.
- Result deduplication: `_result_key()` tuple of (file_path, start, end, text). Same segment from both arms = merged.
- Filters: source_type, folder prefix, date range, min_score. Applied post-RRF.

## Context Block — LAZY-LOAD ON DEMAND
- `backend/search_and_index/`: core engine + API. Load when working on backend.
- `client/src/`: React frontend. Load when working on UI.
- `desktop/`: Electron shell. Load when working on desktop packaging.
- `README.md`: architecture diagram + API docs. Load when onboarding or checking API contract.
- `GPU_TRANSCRIBTION_TEMPORARY_FIX.md`: CUDA DLL workaround. Load only when debugging GPU transcription.

## Cadences
- Every merge: run `python -m pytest tests/ -v`. Numbers in PR.
- Every new engine function: add unit test in `tests/`.
- Every schema change (SQLite tables, API models): update `tests/test_sql_database.py` + `tests/test_api_models.py`.
- Every search change: verify RRF results don't regress (golden queries in `tests/golden_queries.json`).

## Known Issues
- `sql_database.py` is 739 lines — god module. Split planned into media_repo, job_repo, search_repo, settings_repo.
- `GPU_TRANSCRIBTION_TEMPORARY_FIX.md` — CUDA DLL workaround not resolved. End users hit this on GPU.
- No API input validation on ingest paths (can ingest outside watch folders).
- No rate limiting on `ingest_folder()` — 10K files = 10K jobs queued instantly.
