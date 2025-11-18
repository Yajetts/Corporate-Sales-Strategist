# PowerShell deployment script for Docker Compose

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Sales Strategist - Docker Deployment" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if .env file exists
if (-not (Test-Path .env)) {
    Write-Host "Error: .env file not found!" -ForegroundColor Red
    Write-Host "Please copy .env.example to .env and configure it." -ForegroundColor Yellow
    exit 1
}

# Build Docker images
Write-Host ""
Write-Host "Building Docker images..." -ForegroundColor Green
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker build failed!" -ForegroundColor Red
    exit 1
}

# Start services
Write-Host ""
Write-Host "Starting services..." -ForegroundColor Green
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start services!" -ForegroundColor Red
    exit 1
}

# Wait for services to be healthy
Write-Host ""
Write-Host "Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service health
Write-Host ""
Write-Host "Checking service health..." -ForegroundColor Green
docker-compose ps

# Run database migrations
Write-Host ""
Write-Host "Running database migrations..." -ForegroundColor Green
docker-compose exec -T api python -m alembic upgrade head

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  - API: http://localhost:5000"
Write-Host "  - Dashboard: http://localhost:5002"
Write-Host "  - MLflow: http://localhost:5001"
Write-Host ""
Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Cyan
Write-Host "To stop services: docker-compose down" -ForegroundColor Cyan
Write-Host ""
