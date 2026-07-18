# Testing

## Overview

- **240 unit tests** across 13 test files
- **60% code coverage** (branch + line)
- **No GPU or AI models required** — all heavy deps mocked via `conftest.py`
- **Runtime**: ~3 seconds for full suite

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
python -m pytest tests/ -v --tb=short --cov=backend/search_and_index --cov-report=term-missing --cov-branch

# Run a specific test file
python -m pytest tests/test_sql_database.py -v

# Run a specific test class
python -m pytest tests/test_runtime_service.py::TestHybridSearchRRF -v
```

## Test Files

| File | Module | Tests | Coverage |
|------|--------|-------|----------|
| `test_api_models.py` | api_models.py | 15 | 96% |
| `test_api_routes.py` | api_routes_*.py | 28 | 39-100% |
| `test_api_service.py` | api_service.py | 27 | 76% |
| `test_aural_engine.py` | aural_engine.py | 6 | 75% |
| `test_document_engine.py` | document_engine.py | 11 | 93% |
| `test_model_downloader.py` | model_downloader.py | 11 | 85% |
| `test_runtime_service.py` | runtime_service.py | 22 | 54% |
| `test_semantic_engine.py` | semantic_engine.py | 18 | 91% |
| `test_sql_database.py` | sql_database.py | 29 | 48% |
| `test_sql_database_extended.py` | sql_database.py | 29 | 67% |
| `test_summarizer.py` | summarizer.py | 8 | 95% |
| `test_visual_engine.py` | visual_engine.py | 12 | 70% |
| `test_watch.py` | watch.py | 20 | 65% |

## CI Pipeline

7 jobs run on every push/PR to `main` or `dev`:

1. **lint** — ruff (Python linting)
2. **type-check** — mypy on api_models.py
3. **unit-tests** — pytest with coverage + Codecov upload
4. **frontend-lint** — ESLint on client/src/
5. **frontend-build** — Vite production build
6. **smoke-tests** — API smoke test file validation
7. **validate-config** — JSON + Python syntax validation

## Adding New Tests

1. Create `tests/test_<module_name>.py`
2. Use the `temp_db` fixture pattern for SQLite tests (see `test_sql_database.py`)
3. Mock AI models using `conftest.py` stubs (already set up for 11 heavy deps)
4. Run `python -m pytest tests/ -v` to verify
5. Commit with message `test: add <module> tests`

## Test Conventions

- Test classes: `Test<ModuleName>`
- Test functions: `test_<behavior>`
- Use `pytest.raises()` for expected errors
- Use `monkeypatch` for mocking (not `unittest.mock.patch` where possible)
- Use `tmp_path` fixture for temp files (auto-cleaned)
- No network calls — all external deps mocked
