# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Electron Shell (desktop/)                  │
│  main.js → spawns Python backend → loads React UI           │
└─────────────────┬───────────────────────────────────────────┘
                  │ http://127.0.0.1:8000
┌─────────────────▼───────────────────────────────────────────┐
│              FastAPI Backend (backend/search_and_index/)      │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ aural_engine│  │visual_engine │  │  document_engine  │   │
│  │  (Whisper)  │  │   (CLIP)     │  │ (PyMuPDF/frontmatter)│ │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬─────────┘   │
│         │                │                     │             │
│  ┌──────▼────────────────▼─────────────────────▼───────────┐ │
│  │            semantic_engine (MiniLM + LanceDB)           │ │
│  │          Hybrid RRF Search (semantic + keyword)         │ │
│  └────────────────────────┬────────────────────────────────┘ │
│                           │                                   │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │         sql_database.py  (SQLite — jobs + media)        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────┐   ┌───────────────────────────────┐│
│  │  watchdog FileWatcher│   │  Background Worker Thread     ││
│  │  (auto-enqueue files)│   │  (job queue consumer)         ││
│  └──────────────────────┘   └───────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│             React Frontend (client/ — Vite + JSX)           │
│   Search · Jobs · Ingest · System panels                    │
└─────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| Aural Engine | `aural_engine.py` | Audio extraction (ffmpeg), Whisper transcription |
| Visual Engine | `visual_engine.py` | Frame extraction (cv2), CLIP embeddings, thumbnails |
| Document Engine | `document_engine.py` | PDF parsing (PyMuPDF), Markdown/TXT chunking |
| Semantic Engine | `semantic_engine.py` | MiniLM embeddings, LanceDB vector storage, semantic search |
| SQL Database | `sql_database.py` | SQLite: media records, FTS5 transcripts, job queue, settings |
| Runtime Service | `runtime_service.py` | Job processing, RRF hybrid search, filtering |
| Summarizer | `summarizer.py` | DistilBART auto-summaries |
| Watch | `watch.py` | Filesystem watcher, debounced enqueue, initial scan |
| API App | `api_app.py` | FastAPI app, lifespan, worker thread, watcher setup |

## Data Flow

1. **File dropped** in watch folder → watchdog detects → debounced (2s) → stability check → `enqueue_job()`
2. **Worker picks up** job → `process_media()` dispatches by extension
3. **Video**: extract audio → transcribe (Whisper) → summarize (DistilBART) → save to SQLite + LanceDB → index frames (CLIP)
4. **PDF**: extract pages (PyMuPDF) → chunk → summarize → save to SQLite + LanceDB
5. **Markdown/TXT**: read → split on `\n\n` → summarize → save to SQLite + LanceDB
6. **Search**: query → semantic search (LanceDB vector) + keyword search (SQLite FTS5) → RRF fusion → filter → return

## AI Models

| Model | Purpose | Size | Device |
|-------|---------|------|--------|
| distil-large-v3 | Speech-to-text | ~750 MB | GPU (CUDA) or CPU |
| clip-ViT-B-32 | Visual frame understanding | ~600 MB | GPU (CUDA) or CPU |
| all-MiniLM-L6-v2 | Semantic text embeddings | ~90 MB | CPU |
| distilbart-cnn-6-6 | Summarization | ~230 MB | CPU |

## Storage

- **SQLite** (`data/database/brain.db`): media metadata, transcripts (FTS5), job queue, app settings
- **LanceDB** (`data/database/vector_data/`): semantic segments, summary segments, visual moments
- **Thumbnails** (`data/thumbnails/`): extracted frame thumbnails (320x320, JPEG quality 80)
