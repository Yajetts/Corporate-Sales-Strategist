#!/bin/bash
# Deployment script for Kubernetes

set -e

echo "========================================="
echo "Sales Strategist - Kubernetes Deployment"
echo "========================================="

# Configuration
NAMESPACE="sales-strategist"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed!"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster!"
    exit 1
fi

# Build and push Docker images
echo ""
echo "Building Docker images..."
docker build -t ${DOCKER_REGISTRY}/sales-strategist-api:${IMAGE_TAG} -f Dockerfile .
docker build -t ${DOCKER_REGISTRY}/sales-strategist-worker:${IMAGE_TAG} -f Dockerfile.worker .

echo ""
echo "Pushing Docker images to registry..."
docker push ${DOCKER_REGISTRY}/sales-strategist-api:${IMAGE_TAG}
docker push ${DOCKER_REGISTRY}/sales-strategist-worker:${IMAGE_TAG}

# Create namespace
echo ""
echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Apply ConfigMaps and Secrets
echo ""
echo "Applying ConfigMaps and Secrets..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy databases
echo ""
echo "Deploying databases..."
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/mongodb-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml

# Wait for databases to be ready
echo ""
echo "Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=mongodb -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=300s

# Deploy application services
echo ""
echo "Deploying application services..."
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml

# Wait for application services to be ready
echo ""
echo "Waiting for application services to be ready..."
kubectl wait --for=condition=ready pod -l app=api -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=worker -n ${NAMESPACE} --timeout=300s

# Apply Ingress
echo ""
echo "Applying Ingress configuration..."
kubectl apply -f k8s/ingress.yaml

# Apply HPA
echo ""
echo "Applying HorizontalPodAutoscaler..."
kubectl apply -f k8s/hpa.yaml

# Run database migrations
echo ""
echo "Running database migrations..."
API_POD=$(kubectl get pod -l app=api -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${API_POD} -- python -m alembic upgrade head

echo ""
echo "========================================="
echo "Deployment complete!"
echo "========================================="
echo ""
echo "Checking deployment status..."
kubectl get all -n ${NAMESPACE}
echo ""
echo "To view logs:"
echo "  kubectl logs -f deployment/api -n ${NAMESPACE}"
echo "  kubectl logs -f deployment/worker -n ${NAMESPACE}"
echo ""
echo "To access the API:"
echo "  kubectl port-forward -n ${NAMESPACE} service/api-service 5000:5000"
echo ""
