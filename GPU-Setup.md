# GPU Setup

## VRAM Requirements

TOBU runs all AI models locally. Two modes are supported:

| Mode | VRAM | RAM | Speed | Notes |
|------|------|-----|-------|-------|
| **CPU-only** | 0 GB | 8 GB min (16 GB rec.) | Slow (3-10x realtime) | Works on any machine |
| **GPU (CUDA)** | 4 GB min (2 GB usable) | 8 GB | Fast (near-realtime) | NVIDIA GPU with CUDA 12 |

## Model VRAM Breakdown (GPU Mode)

| Model | Purpose | VRAM (GPU) | Device |
|-------|---------|-----------|--------|
| Whisper distil-large-v3 (int8) | Speech-to-text | ~750 MB | GPU (CUDA) |
| CLIP ViT-B/32 (fp32) | Visual frame embeddings | ~600 MB | GPU (CUDA) |
| all-MiniLM-L6-v2 | Semantic text embeddings | 0 MB | CPU (default) |
| distilbart-cnn-6-6 | Summarization | 0 MB | CPU (default) |
| **Total resident** | | **~1.4 GB** + ~500 MB working | **~2 GB min, 4 GB recommended** |

Any GPU with 4 GB+ VRAM works (RTX 3050, GTX 1660, RTX 4050, etc.).

## Enabling GPU on Windows

### 1. Install CUDA Dependencies

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

### 2. Fix DLL Path (if needed)

If you encounter `RuntimeError: Library cublas64_12.dll is not found`:

1. Locate the DLLs:
   ```
   .venv\Lib\site-packages\nvidia\cublas\bin
   .venv\Lib\site-packages\nvidia\cudnn\bin
   ```

2. Copy them to:
   ```
   .venv\Lib\site-packages\ctranslate2
   ```

See `backend/search_and_index/GPU_TRANSCRIBTION_TEMPORARY_FIX.md` for details.

### 3. Verify GPU is Active

```python
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If `True`, Whisper and CLIP will use GPU automatically.

## Enabling GPU on WSL

```bash
# Install CUDA toolkit in WSL
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## CPU-Only Mode

If you don't have a GPU, TOBU works fine on CPU. Transcription is 3-10x slower but fully functional. No special configuration needed — the models detect CPU automatically and fall back gracefully.
