# Multi-stage Dockerfile for Flask API Service
# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim as production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from base stage
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs models/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src.api.app
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Run Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
