# Corporate Sales Strategist — User Guide

This guide explains how to use the **Corporate Sales Strategist** system as an end user (client). It covers how to access the dashboard, run analyses, and generate the Analysis Overview report.

## 1) What this system does

Corporate Sales Strategist is an AI-driven decision-support platform for enterprise sales teams. It provides:

- Company/product understanding (Enterprise Analyst)
- Market segmentation and customer clustering (Market Decipherer)
- Sales strategy recommendations (Strategy Engine)
- Performance monitoring and forecasting (Performance Governor)
- Business/resource optimization (Business Optimizer)
- Explainability tooling (when available)
- An **Analysis Overview** post-analysis report that compiles outputs into an executive brief

## 2) Accessing the system

You interact with the system through a web dashboard hosted by the API server.

### Dashboard URLs (local)

- Dashboard home: `http://127.0.0.1:5000/dashboard/`
- Analysis Overview page: `http://127.0.0.1:5000/dashboard/analysis-overview`

If you see a JSON response at `http://127.0.0.1:5000/`, that’s normal — it’s the API root endpoint.

## 3) Running the system (for the person hosting it)

If your team is running the system on your machine or a server, there are two common ways:

### Option A — Docker (recommended for databases + worker)

1. Ensure Docker Desktop is running.
2. From the project folder, start services:

   - `docker compose up -d --build`

3. Confirm containers are healthy:

   - `docker compose ps`

This typically starts Postgres, MongoDB, Redis, and the Celery worker used for asynchronous jobs.

### Option B — Local API + Docker databases

1. Start the databases + Redis via Docker:

   - `docker compose up -d postgres mongodb redis worker`

2. Start the Flask API locally:

   - `python -m flask --app src.api.app run --host=127.0.0.1 --port=5000`

## 4) Using the dashboard

Open `http://127.0.0.1:5000/dashboard/`.

### Key pages

- **Market Analysis**: Generates/visualizes market segmentation and clustering results.
- **Strategy**: Produces recommended sales strategies from available inputs.
- **Performance**: Displays forecasts, alerts, and monitoring outputs.
- **Explainability**: Shows model transparency artifacts (when available).
- **Business Optimization**: Suggests resource allocation and optimization outputs.
- **Analysis Overview**: Creates a unified executive brief report from stored module outputs.

## 5) Analysis Overview (Executive Brief)

The Analysis Overview feature compiles the latest stored module outputs into a single report.

Notes:
- If any required module was not executed for the run, the report will explicitly mark it as "Not executed in this run" (e.g., "Market Decipherer: Not executed in this run").
- The dashboard focuses on the latest run only; it does not require storing or browsing older runs.

### Generate a new Analysis Overview run

1. Go to: `http://127.0.0.1:5000/dashboard/analysis-overview`
2. Click **Generate Analysis Overview**.
4. The run status updates as it processes (queued → collecting → generating_report → ready).

### Generate a Podcast Summary (after the report)

Once the report is fully generated, the dashboard shows **Generate Podcast Summary**.

- The system first generates a short **podcast narration script** from the final report text (LLM when configured; safe fallback otherwise).
- Then it converts that script to an MP3 via text-to-speech.
- It becomes downloadable immediately once ready.
- It does not rerun any modules.

### View and download artifacts

Once a run is **ready**, you can:

- Read the report in the dashboard
- Download `report.md` (Markdown)
- Download `report.pdf` (best-effort; appears when PDF export is available)
- Download `podcast_script.txt` (the narration script used for audio)
- Download `audio.mp3` after generating the podcast summary and dependencies are present

### Where runs are stored

Each run writes to a folder like:

- `logs/post_analysis/<run_id>/`
  - `snapshot.json`
  - `report.md`, `report.txt`, `report.pdf`
  - `audio.mp3` (if enabled)
  - `status.json`

## 6) APIs (optional)

If you use the system programmatically, the API is available under the prefix `/api/v1`.

### API documentation

- Swagger UI: `http://127.0.0.1:5000/api/v1/docs`

### Analysis Overview API

- Trigger generation:
  - `POST /api/v1/post_analysis/generate`
  - Body example:

    ```json
    {
      "task_ids": {
        "enterprise": "<task_id>",
        "market": "<task_id>",
        "strategy": "<task_id>",
        "performance": "<task_id>",
        "optimization": "<task_id>"
      }
    }
    ```

- Poll status:
  - `GET /api/v1/post_analysis/runs/<run_id>/status`

- Generate podcast summary:
  - `POST /api/v1/post_analysis/runs/<run_id>/podcast`

- Download artifacts:
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/report.md?download=1`
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/report.pdf?download=1`
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/podcast_script.txt?download=1`
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/audio.mp3?download=1`

## 7) Troubleshooting

### “Page shows JSON”

- Use the dashboard routes under `/dashboard/...`.
- `/` is expected to return JSON.

### Analysis Overview stuck on “queued”

Most commonly:

- Redis is not running
- The Celery worker is not running

Fix:

- `docker compose up -d redis worker`
- Check worker logs: `docker compose logs --tail 200 worker`

### Audio (MP3) not generated

Audio generation requires optional dependencies (Coqui TTS + MP3 encoder + numpy). If missing, audio can fail even if the report succeeds.

If you do not need audio, leave the toggle off.

### Database connection errors

If you’re using Docker databases:

- Ensure Postgres/Mongo are healthy: `docker compose ps`
- Ensure your `.env` matches your exposed host ports and credentials.

---

If you want a shorter “1-page quickstart” version of this guide, tell me your target audience (sales user vs technical admin) and I’ll tailor it.
