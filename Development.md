# Development

## Project Rules (CLAUDE.md)

The project uses a `CLAUDE.md` rules file for AI-assisted development. Key rules:

### Stack
- Backend: Python 3.10+, FastAPI, PyInstaller
- Frontend: React + Vite (JSX)
- Desktop: Electron 30
- Storage: SQLite + LanceDB
- Models: Whisper, CLIP, MiniLM, DistilBART (all local)

### Conventions
- **Lazy model loading**: singleton pattern (`_MODEL = None`, `get_model()`)
- **Dual import paths**: `if __package__:` for package mode, `else:` for script mode
- **Frozen path detection**: `getattr(sys, 'frozen', False)` → `~/.tobu`
- **Parameterized SQL**: `?` placeholders only, never f-string SQL
- **FastAPI routes**: thin handlers, delegate to `api_service.py`
- **Engine modules**: one responsibility each, no cross-imports

### Boundaries
- Never add cloud dependencies
- Never commit model weights (`models/` is gitignored)
- Never commit runtime data (`data/` is gitignored)
- Never hardcode file paths — use `PROJECT_ROOT`
- Test before commit: `python -m pytest tests/ -v`

## Commands

```bash
# Backend dev
cd backend/search_and_index && python main.py --mode worker

# API server
cd backend/search_and_index && python -m uvicorn api_app:app --port 8000

# Frontend dev
cd client && npm run dev

# Electron dev
npm start  # from root

# Tests
python -m pytest tests/ -v

# Build backend (PyInstaller)
python scripts/build_backend.py

# Build desktop (Electron)
cd desktop && npm run dist
```

## Known Issues

1. `sql_database.py` is 739 lines — god module. Split planned into media_repo, job_repo, search_repo, settings_repo.
2. `GPU_TRANSCRIBTION_TEMPORARY_FIX.md` — CUDA DLL workaround not resolved.
3. No API input validation on ingest paths (can ingest outside watch folders).
4. No rate limiting on `ingest_folder()`.

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Commit with conventional messages: `feat:`, `fix:`, `test:`, `docs:`
6. Open a Pull Request

## Commit Message Convention

| Prefix | Use |
|--------|-----|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `test:` | Test additions/changes |
| `docs:` | Documentation |
| `ci:` | CI/CD changes |
| `chore:` | Maintenance |
| `refactor:` | Code restructuring (no behavior change) |
