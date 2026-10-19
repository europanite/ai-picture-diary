# [ai-picture-diary](https://github.com/europanite/ai-picture-diary "ai-picture-diary")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11|%203.12|%203.13-blue)](https://www.python.org/)

[![Feed](https://github.com/europanite/ai-picture-diary/actions/workflows/feed.yml/badge.svg)](https://github.com/europanite/ai-picture-diary/actions/workflows/feed.yml)
[![Pages](https://github.com/europanite/ai-picture-diary/actions/workflows/pages.yml/badge.svg)](https://github.com/europanite/ai-picture-diary/actions/workflows/pages.yml)
[![News Agent Tests](https://github.com/europanite/ai-picture-diary/actions/workflows/news-agent-tests.yml/badge.svg)](https://github.com/europanite/ai-picture-diary/actions/workflows/news-agent-tests.yml)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React Native](https://img.shields.io/badge/react_native-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Jest](https://img.shields.io/badge/-jest-%23C21325?style=for-the-badge&logo=jest&logoColor=white)
![Expo](https://img.shields.io/badge/expo-1C1E24?style=for-the-badge&logo=expo&logoColor=#D04A37)


An automated **English fictional picture diary** driven by RSS news headlines, local language models, and image generation.

Each entry combines three keywords extracted from unrelated recent English-language news articles. A pretrained MiniLM embedding model scores combinations by semantic distance; the local LLM writes an original first-person diary, and the illustration worker draws it. Headlines are creative inspiration, **not factual claims**. The diary-writing prompt receives only the dynamically selected three terms,
not the underlying headlines or source articles. The feed retains provenance
and distance scores for reproducibility.

**News agent setup and architecture:** [docs/NEWS_AGENT.md](docs/NEWS_AGENT.md).

**Automated publishing:** GitHub Actions `Feed` runs every day at **12:17 JST**
(03:17 UTC), collecting live English news, generating a fictional diary using
Ollama, generating a 512px SD-Turbo illustration on the hosted CPU runner,
committing the feed, and deploying GitHub Pages. Image generation is enabled by
default; set `NEWS_IMAGE_ENABLED=false` as a repository variable to disable it. See
[Scheduled GitHub Actions](docs/NEWS_AGENT.md#scheduled-github-actions).

To generate and publish a news-driven English diary on Linux with a running
local Ollama instance:

```sh
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
docker compose -f docker-compose.news.yml run --rm news-agent
docker compose -f docker-compose.news.yml run --rm diary-writer
docker compose -f docker-compose.news.yml run --rm diary-publish
```

Locally, the image stage can be run with `diary-illustrate`, followed by
`diary-index`. The web UI hides source links; source records are still retained
in the feed JSON for audit and reproducibility.

---
The project combines:

- a FastAPI backend for authentication, RAG, and text generation APIs
- PostgreSQL for application data
- ChromaDB-backed retrieval for local JSON documents
- Ollama or OpenAI-compatible models for generation and embedding
- an Expo / React Native frontend that can be exported for the web
- an illustration worker that generates images for feed entries
- GitHub Actions workflows for feed generation and GitHub Pages deployment

---

## Main services

| Service | Purpose | Default ports |
| --- | --- | --- |
| `db` | PostgreSQL 16 database | `5432` |
| `ollama` | Local LLM / embedding model server | `11434` |
| `backend` | FastAPI API server | `8000` |
| `frontend` | Expo development server | `19000`, `19001`, `19002`, `8081` |
| `illustrate` | One-shot image generation worker | none |
| `llm_finetune` | Optional LoRA fine-tuning service | none |
| `ollama_import` | Optional Ollama adapter import service | none |

## Requirements

- Docker
- Docker Compose v2
- Git
- Ollama models
- ChromaDB
- Optional: an OpenAI API key when using the OpenAI provider
- Optional: Hugging Face access/cache settings when using model or LoRA assets that require them

## Local development

Start the main stack:

```bash
docker compose --env-file .env up --build db ollama backend frontend
```

Check backend health:

```bash
curl http://localhost:8000/health
```

Open the FastAPI docs:

```text
http://localhost:8000/docs
```

The frontend container runs Expo. Use the URL printed in the `frontend` logs. Metro and Expo development ports are exposed by Compose.

```bash
docker compose logs -f frontend
```

Stop the stack:

```bash
docker compose down
```

Remove local database and model volumes only when you intentionally want a clean reset:

```bash
docker compose down -v
```

## Backend API

The FastAPI app mounts three router groups.

### Health

```bash
curl http://localhost:8000/health
```

### Authentication

```text
POST /auth/signup
POST /auth/signin
GET  /auth/me
```

### RAG

```text
GET  /rag/status
POST /rag/reindex
POST /rag/ingest
POST /rag/query
```

Example:

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What should be introduced today?",
    "top_k": 5,
    "include_debug": true
  }'
```

### Sentence generation

```text
GET  /diary/health
POST /diary/generate
```

Example:

```bash
curl -X POST http://localhost:8000/diary/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "diary",
    "max_retries": 3,
    "temperature": 0.9
  }'
```

## RAG data

The backend reads JSON documents from `DOCS_DIR`, which is mounted from `./data/json` in Docker Compose.

To rebuild the ChromaDB index from local JSON files:

```bash
curl -X POST http://localhost:8000/rag/reindex
```

To check the current index status:

```bash
curl http://localhost:8000/rag/status
```

The persisted ChromaDB directory is mounted through `./chroma_db` and `CHROMA_DB_DIR`.

## Feed generation

The feed assets are stored under `frontend/app/public`.

Important paths:

```text
frontend/app/public/latest.json
frontend/app/public/feed/
frontend/app/public/snapshot/
frontend/app/public/posts/
frontend/app/public/image/
```

Generate a sentence with the local script through the backend environment:

```bash
docker compose run --rm backend python /scripts/generate_sentence.py
```

## Illustration generation

The `illustrate` service reads the latest feed entry and writes/patches image references.

Run it after `frontend/app/public/latest.json` exists:

```bash
docker compose run --rm illustrate
```

Rebuild the feed index and post pages after modifying public feed assets:

```bash
docker compose run --rm illustrate \
  python scripts/build_feed_pages.py \
  --public-dir frontend/app/public \
  --base-url .
```

## GitHub Actions

The repository includes workflows for automated feed generation and deployment.

| Workflow | Purpose |
| --- | --- |
| `feed.yml` | Scheduled/manual entry point for feed generation |
| `feed_slot_event.yml` | Event feed wrapper |
| `event_core.yml` | Generates an event payload, fixes it into public feed state, illustrates it, and triggers Pages |
| `fixed_entry.yml` | Writes a fixed entry JSON into public feed assets |
| `feed_core.yml` | RAG-driven feed generation pipeline |
| `ingest.yml` | Restores/builds ChromaDB data from Google Drive, S3, or local source |
| `pages.yml` | Exports the Expo web app and deploys to GitHub Pages |
