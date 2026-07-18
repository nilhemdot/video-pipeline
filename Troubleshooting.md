# Troubleshooting

## Common Issues

### 1. `RuntimeError: Library cublas64_12.dll is not found`

**Cause:** CUDA DLLs not on PATH when using GPU mode on Windows.

**Fix:**
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Then copy DLLs from:
```
.venv\Lib\site-packages\nvidia\cublas\bin
.venv\Lib\site-packages\nvidia\cudnn\bin
```
To:
```
.venv\Lib\site-packages\ctranslate2
```

See `backend/search_and_index/GPU_TRANSCRIBTION_TEMPORARY_FIX.md`.

### 2. `Semantic model not found at .../all-MiniLM-L6-v2`

**Cause:** Models not downloaded.

**Fix:**
```bash
python backend/search_and_index/model_downloader.py
```

Or launch the app and complete the onboarding wizard.

### 3. Whisper transcription is very slow

**Cause:** Running on CPU instead of GPU.

**Fix:** Verify CUDA is available:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If `False`, install CUDA dependencies (see [[GPU-Setup]]).

### 4. Search returns no results

**Cause:** Videos not indexed yet, or index is empty.

**Fix:**
1. Check jobs: `curl http://127.0.0.1:8000/api/v1/jobs/`
2. Check DB stats: `curl http://127.0.0.1:8000/api/v1/system/status`
3. Re-index: `curl -X POST "http://127.0.0.1:8000/api/v1/ingest/file" -H "Content-Type: application/json" -d '{"file_path": "/path/to/video.mp4"}'`

### 5. `ffmpeg` not found

**Cause:** FFmpeg not installed or not on PATH.

**Fix:**
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH
- **macOS:** `brew install ffmpeg`
- **Linux/WSL:** `sudo apt install ffmpeg`

### 6. Port 8000 already in use

**Cause:** Another process using port 8000.

**Fix:** Change the port in `backend/search_and_index/api_app.py` (line 140) or kill the process:
```bash
# Linux/macOS
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### 7. Electron app won't start

**Cause:** Missing Node.js dependencies or backend build.

**Fix:**
```bash
npm install
cd client && npm install && cd ..
npm start
```

### 8. Database locked error

**Cause:** Multiple processes accessing SQLite simultaneously.

**Fix:** Ensure only one instance of the app is running. If the worker thread crashed:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/system/integrity"
```

The app automatically resets stale running jobs on startup.
