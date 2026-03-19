# Corporate Sales Strategist

AI-driven, modular decision-support system for enterprise sales teams. The platform combines NLP, representation learning, forecasting, optimization, and explainability into one workflow and a unified dashboard.

## What This Project Solves

Sales organizations often face avoidable losses due to:

- Weak product-market alignment
- Slow reaction to demand changes
- Inconsistent pricing and campaign strategy
- Poor visibility into forecast risk
- Low explainability of model-driven recommendations

Corporate Sales Strategist addresses these gaps by coordinating six domain modules and a post-analysis layer that produces executive-ready outputs.

## Core Modules

### 1. Enterprise Analyst

Purpose: company and product understanding from unstructured business text.

- Extracts semantic business signals
- Produces structured context for downstream modules
- Supports company/product text flows in dashboard and API

### 2. Market Decipherer

Purpose: customer and market segmentation using representation learning and clustering.

- Learns latent patterns from high-dimensional market data
- Produces segment profiles and cluster-level insights
- Feeds strategy and post-analysis sections

### 3. Strategy Engine

Purpose: strategy recommendation and pricing/action guidance.

- Generates sales actions from market state
- Includes fallback heuristics when model checkpoints are unavailable
- Returns actionable insights suitable for dashboard use

### 4. Performance Governor

Purpose: forecast and risk monitoring.

- Monitors trends and expected trajectory
- Surfaces alerts and risk signals
- Supports readiness and operational decisioning

### 5. Business Manager

Purpose: resource and production optimization.

- Optimizes portfolio-level allocations
- Normalizes legacy payload variants
- Returns optimization metrics and recommendations used by dashboard and post-analysis

### 6. Model Transparency (SHAP)

Purpose: explainability and confidence context.

- Supports local/global explanation workflows
- Provides feature contribution signals
- Integrated into Analysis Overview confidence section


## Analysis Overview (Post-Analysis)

A newly added Post-Analysis capability now aggregates module outputs into a single executive report and optional audio summary.

### What it does

- Collects latest outputs from:
  - Enterprise Analyst
  - Market Decipherer
  - Strategy Engine
  - Performance Governor
  - Business Manager
  - Model Transparency (SHAP)
- Builds a unified Analysis Overview report in Markdown and text
- Exports PDF when PDF dependencies are available
- Optionally generates a podcast-style MP3 summary
- Stores artifacts under `logs/post_analysis/<run_id>/`

### Dashboard experience

- Analysis Overview page: `http://127.0.0.1:5000/dashboard/analysis-overview`
- Floating action button appears when all required module outputs are available in the current app session
- If a module did not run, the report clearly marks it as not executed in this run

### API endpoints

- Trigger report generation: `POST /api/v1/post_analysis/generate`
- Run status: `GET /api/v1/post_analysis/runs/<run_id>/status`
- Trigger podcast summary: `POST /api/v1/post_analysis/runs/<run_id>/podcast`
- Download artifacts:
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/report.md?download=1`
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/report.pdf?download=1`
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/podcast_script.txt?download=1`
  - `GET /api/v1/post_analysis/runs/<run_id>/artifact/audio.mp3?download=1`

## Architecture Overview

The system is organized into:

- API layer: Flask routes and health/readiness endpoints
- Dashboard layer: Jinja templates + static JS/CSS
- Async layer: Celery task processing with Redis
- Data layer: PostgreSQL + MongoDB + Redis
- Post-analysis layer: snapshot collection, report generation, artifact storage, audio generation

## Repository Layout

Key areas:

- `src/api/`: API app, routes, async routes, health, database integration
- `src/dashboard/`: dashboard pages, frontend assets, templates
- `src/services/`: business logic per module
- `src/post_analysis/`: analysis overview orchestration, reporting, storage, audio
- `tests/`: unit and integration tests
- `scripts/`: local run and utility scripts
- `deployment/`, `k8s/`: deployment assets

## Quick Start

### Prerequisites

- Python 3.10+
- Docker Desktop (recommended for local infra)
- Optional for audio summary: Coqui TTS and MP3 encoder dependencies

### 1) Clone and install

```bash
git clone <your-repo-url>
cd NNUFinalProject
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment

Create an `.env` file (you can start from `.env.example`) and set:

- DB connection variables
- Redis variables
- Optional LLM keys for report/script generation
- Optional post-analysis tuning flags

### 3) Choose a run mode

### Mode A: Local API with Docker infra (recommended)

```powershell
./scripts/run_local_with_docker_infra.ps1 -BindAddress 0.0.0.0 -Port 5000
```

This brings up required containers and runs Flask locally.

### Mode B: No-DB local mode (fast dev/testing)

```powershell
python -m src.api.run_nodb --host 0.0.0.0 --port 5000
```

Notes:

- Disables DB initialization
- Uses sync post-analysis behavior by default
- Opens dashboard automatically unless `--no-open-browser` is used

### Mode C: Docker Compose full stack

```bash
docker compose up -d --build
```

Optional MLflow profile:

```bash
docker compose --profile mlflow up -d
```

## Access URLs

- API root: `http://127.0.0.1:5000/`
- Dashboard home: `http://127.0.0.1:5000/dashboard/`
- Enterprise page: `http://127.0.0.1:5000/dashboard/enterprise`
- Analysis Overview page: `http://127.0.0.1:5000/dashboard/analysis-overview`
- Swagger (if enabled): `http://127.0.0.1:5000/api/v1/docs`

## Environment Flags (Important)

Common runtime toggles used by this project:

- `API_ENABLE_CORE_ROUTES`: enable/disable core ML-heavy routes
- `API_ENABLE_ASYNC_ROUTES`: enable/disable async route set
- `API_ENABLE_SWAGGER`: enable/disable docs endpoints
- `CELERY_INCLUDE_CORE_TASKS`: include/exclude heavy core Celery task modules
- `DB_INIT_DISABLE`: skip DB initialization entirely
- `DB_INIT_STRICT`: fail hard on DB init errors when set
- `POST_ANALYSIS_USE_CELERY`: force async/sync post-analysis execution mode

## Post-Analysis Artifacts

Generated artifacts are written to:

- `logs/post_analysis/<run_id>/snapshot.json`
- `logs/post_analysis/<run_id>/report.md`
- `logs/post_analysis/<run_id>/report.txt`
- `logs/post_analysis/<run_id>/report.pdf` (best effort)
- `logs/post_analysis/<run_id>/podcast_script.txt` (when podcast generated)
- `logs/post_analysis/<run_id>/audio.mp3` (when podcast generated)
- `logs/post_analysis/<run_id>/status.json`

The latest run pointer is stored in:

- `logs/post_analysis/latest.json`

## Testing

Run the full test suite:

```bash
pytest -q
```

Run only post-analysis tests:

```bash
pytest -q tests/test_post_analysis_*.py
```

## Deployment

Deployment assets are available for:

- Docker-based environments (`Dockerfile*`, `docker-compose.yml`)
- Kubernetes (`k8s/` manifests)
- Terraform-assisted AWS deployment (`deployment/terraform-aws.tf`)

See deployment notes in:

- `deployment/README.md`

## Documentation

- User guide: `docs/USER_GUIDE.md`

## Tech Stack

- Python, Flask, Celery
- PostgreSQL, MongoDB, Redis
- PyTorch, NumPy, Pandas, scikit-learn
- SHAP-based explainability patterns
- Markdown + PDF artifact generation for reporting

## Notes

- Some advanced model paths/checkpoints are optional in development.
- Dashboard and post-analysis components are designed to degrade gracefully when optional infra is unavailable.
- For minimal Docker images, use the provided split requirement files (`requirements.docker.txt`, `requirements.worker.txt`).

## License

See `LICENSE`.
