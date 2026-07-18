# Video Pipeline Wiki

Welcome to the **video-pipeline** wiki — your local-first, offline video search engine.

## What is video-pipeline?

video-pipeline is a fully offline desktop application that turns your personal archive of videos, PDFs, and plain-text notes into a searchable, queryable knowledge base. No cloud, no subscriptions, no data leaving your machine.

## Key Features

- **Speech-to-text transcription** via Whisper distil-large-v3
- **Visual frame search** via CLIP ViT-B/32
- **Semantic text search** via MiniLM-L6-v2 with LanceDB vectors
- **Hybrid search** (semantic + keyword) with Reciprocal Rank Fusion (RRF)
- **Auto-summarization** via DistilBART
- **Watch folder** auto-indexing with debounced file detection
- **100% local** — all models run on-device, no cloud calls

## Quick Links

- [[Architecture]] — system design, module layout, data flow
- [[Installation]] — setup guide for Windows, macOS, and Linux
- [[GPU-Setup]] — VRAM requirements and CUDA configuration
- [[Testing]] — how to run tests, coverage, and CI
- [[API-Reference]] — REST API endpoints and examples
- [[Development]] — contributing guide, conventions, CLAUDE.md rules
- [[Troubleshooting]] — common issues and fixes
