#!/bin/bash
# Deployment script for Docker Compose

set -e

echo "========================================="
echo "Sales Strategist - Docker Deployment"
echo "========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# Build Docker images
echo ""
echo "Building Docker images..."
docker-compose build

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
docker-compose ps

# Run database migrations
echo ""
echo "Running database migrations..."
docker-compose exec -T api python -m alembic upgrade head

echo ""
echo "========================================="
echo "Deployment complete!"
echo "========================================="
echo ""
echo "Services:"
echo "  - API: http://localhost:5000"
echo "  - Dashboard: http://localhost:5002"
echo "  - MLflow: http://localhost:5001"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"
echo ""
