---
title: Fast API Ingestion Cloud Service
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# fast-api-ingestion-cloud-service

This service exposes a FastAPI endpoint to ingest documents and index them into a ChromaDB collection.

## Endpoints

- `GET /doc` – simple health/info endpoint.
- `POST /ingest` – upload a file and index it into a Chroma collection.

## Environment variables

The app reads Chroma Cloud configuration from environment variables (optionally via `.env`):

- `CHROMA_API_KEY`
- `CHROMA_TENET`
- `CHROMA_DB`

You can place these in a `.env` file for local development; in Docker, pass them via `-e` or an env file.

## Running locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run the FastAPI app
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

## Containerising with Docker

### 1. Build the image

From the project root (where the `Dockerfile` lives):

```bash
docker build -t fast-api-ingestion-service .
```

### 2. Run the container

Make sure you provide the Chroma Cloud environment variables so the `db/chroma.py` client can connect.

```bash
docker run \
  -p 8000:8000 \
  -e CHROMA_API_KEY=your_key \
  -e CHROMA_TENET=your_tenant \
  -e CHROMA_DB=your_db \
  fast-api-ingestion-service
```

Alternatively, if you have a `.env` file in the project root:

```bash
docker run --env-file .env -p 8000:8000 fast-api-ingestion-service
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/doc

# Ingest a file (example with curl)
curl -X POST "http://localhost:8000/ingest?collection_name=my_collection" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" 
```

The container will write temporary uploads into the `temp/` directory inside the container and clean them up after indexing.
