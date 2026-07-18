# Installation

## Prerequisites

- **Python** 3.10 or later
- **Node.js** 18 or later
- **FFmpeg** on your system PATH ([download](https://ffmpeg.org/download.html))
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: Optional — NVIDIA with CUDA 12 for faster transcription (see [[GPU-Setup]])
- **Disk**: ~2 GB for models + app + data

## Steps

### 1. Clone the Repository

```bash
git clone https://github.com/nilhemdot/video-pipeline.git
cd video-pipeline
```

### 2. Set Up Python Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux / WSL
source .venv/bin/activate

pip install -r backend/search_and_index/requirements.txt
```

> **GPU Acceleration (Optional):** If you have an NVIDIA GPU with CUDA 12, uncomment the CUDA lines in `requirements.txt` before installing.

### 3. Install Node Dependencies

```bash
# Root (Electron shell)
npm install

# React frontend
cd client && npm install && cd ..
```

### 4. Download AI Models

On first launch, the onboarding wizard handles model downloads automatically (~1.5 GB).

To pre-download manually:

```bash
python backend/search_and_index/model_downloader.py
```

Models are saved to `models/` directory.

### 5. Run the Application

```bash
npm start
```

This launches:
- Python FastAPI backend on `http://127.0.0.1:8000`
- Electron desktop shell with React UI
- Background worker thread for indexing
- File watcher on the `watch/` folder

### 6. Index Your First Video

1. Drop a video file (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`) into the `watch/` folder
2. Watch the Jobs panel for indexing progress
3. Once indexed, search via the UI or API:

```bash
# Semantic search
curl -X POST "http://127.0.0.1:8000/api/v1/search/semantic?query=machine+learning&limit=10"

# Hybrid search (semantic + keyword)
curl -X POST "http://127.0.0.1:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "gradient descent", "limit": 20}'
```

## Supported File Types

| Type | Extensions |
|------|------------|
| Video | `.mp4` `.mkv` `.avi` `.mov` `.webm` |
| Documents | `.pdf` |
| Notes | `.md` `.txt` |
