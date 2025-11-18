# Deployment Scripts and Infrastructure

This directory contains deployment scripts and infrastructure-as-code templates for the Sales Strategist system.

## Contents

- **deploy-docker.sh**: Bash script for Docker Compose deployment (Linux/Mac)
- **deploy-docker.ps1**: PowerShell script for Docker Compose deployment (Windows)
- **deploy-k8s.sh**: Bash script for Kubernetes deployment
- **terraform-aws.tf**: Terraform configuration for AWS infrastructure

## Quick Start

### Docker Compose Deployment

**Linux/Mac:**
```bash
chmod +x deploy-docker.sh
./deploy-docker.sh
```

**Windows:**
```powershell
.\deploy-docker.ps1
```

### Kubernetes Deployment

```bash
# Configure environment variables
export DOCKER_REGISTRY=your-registry.com
export IMAGE_TAG=v1.0.0

# Run deployment script
chmod +x deploy-k8s.sh
./deploy-k8s.sh
```

### AWS Deployment with Terraform

```bash
# Initialize Terraform
terraform init

# Create terraform.tfvars with your configuration
cat > terraform.tfvars <<EOF
aws_region   = "us-east-1"
environment  = "production"
cluster_name = "sales-strategist-prod"
db_password  = "your-secure-password"
EOF

# Plan and apply
terraform plan
terraform apply
```

## Prerequisites

### For Docker Compose
- Docker 20.10+
- Docker Compose 2.0+
- `.env` file configured (copy from `.env.example`)

### For Kubernetes
- kubectl 1.25+
- Access to Kubernetes cluster
- Container registry credentials
- Kubernetes manifests in `../k8s/` directory

### For Terraform (AWS)
- Terraform 1.0+
- AWS CLI configured
- AWS credentials with appropriate permissions
- S3 bucket for Terraform state (update backend configuration)

## Configuration

### Environment Variables

All deployment scripts use environment variables for configuration:

**Docker Compose:**
- Configured via `.env` file in project root
- See `.env.example` for all available options

**Kubernetes:**
- Configured via ConfigMaps and Secrets in `k8s/` directory
- Update `k8s/configmap.yaml` and `k8s/secrets.yaml`

**Terraform:**
- Configured via `terraform.tfvars` file
- See `terraform-aws.tf` for available variables

### Customization

#### Docker Compose

Edit `docker-compose.yml` to:
- Change service configurations
- Add new services
- Modify resource limits
- Update volume mounts

#### Kubernetes

Edit manifests in `k8s/` directory to:
- Change replica counts
- Update resource requests/limits
- Modify HPA settings
- Configure Ingress rules

#### Terraform

Edit `terraform-aws.tf` to:
- Change instance types
- Modify cluster size
- Add additional AWS resources
- Update networking configuration

## Deployment Workflow

### 1. Pre-Deployment

```bash
# Verify prerequisites
docker --version
docker-compose --version
kubectl version --client

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Build and test locally
docker-compose build
docker-compose up -d
docker-compose ps
```

### 2. Deployment

Choose your deployment method:

**Local Development:**
```bash
./deploy-docker.sh
```

**Kubernetes (Staging/Production):**
```bash
./deploy-k8s.sh
```

**AWS (Production):**
```bash
terraform apply
./deploy-k8s.sh  # After infrastructure is ready
```

### 3. Post-Deployment

```bash
# Verify deployment
curl http://localhost:5000/api/v1/health

# Run database migrations
docker-compose exec api python -m alembic upgrade head
# OR for Kubernetes:
kubectl exec -n sales-strategist deployment/api -- python -m alembic upgrade head

# Check logs
docker-compose logs -f
# OR for Kubernetes:
kubectl logs -f deployment/api -n sales-strategist
```

### 4. Monitoring

```bash
# Check service status
docker-compose ps
# OR for Kubernetes:
kubectl get all -n sales-strategist

# View metrics
# Access Grafana at http://localhost:3000
# Access MLflow at http://localhost:5001
```

## Troubleshooting

### Docker Compose Issues

**Services not starting:**
```bash
# Check logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]

# Rebuild and restart
docker-compose up -d --build [service-name]
```

**Database connection errors:**
```bash
# Check database is running
docker-compose ps postgres

# Test connection
docker-compose exec api python scripts/test_database_connection.py
```

### Kubernetes Issues

**Pods not starting:**
```bash
# Check pod status
kubectl get pods -n sales-strategist

# Describe pod for events
kubectl describe pod [pod-name] -n sales-strategist

# Check logs
kubectl logs [pod-name] -n sales-strategist
```

**Image pull errors:**
```bash
# Check image exists in registry
docker pull ${DOCKER_REGISTRY}/sales-strategist-api:${IMAGE_TAG}

# Create image pull secret if needed
kubectl create secret docker-registry regcred \
  --docker-server=${DOCKER_REGISTRY} \
  --docker-username=${DOCKER_USER} \
  --docker-password=${DOCKER_PASSWORD} \
  -n sales-strategist
```

### Terraform Issues

**State lock errors:**
```bash
# Force unlock (use with caution)
terraform force-unlock [LOCK_ID]
```

**Resource already exists:**
```bash
# Import existing resource
terraform import [resource_type].[resource_name] [resource_id]
```

## Rollback Procedures

### Docker Compose

```bash
# Stop services
docker-compose down

# Restore from backup
docker-compose exec postgres psql -U postgres sales_strategist < backup.sql

# Start with previous image version
docker-compose up -d
```

### Kubernetes

```bash
# Rollback deployment
kubectl rollout undo deployment/api -n sales-strategist

# Rollback to specific revision
kubectl rollout undo deployment/api --to-revision=2 -n sales-strategist

# Check rollout status
kubectl rollout status deployment/api -n sales-strategist
```

### Terraform

```bash
# Revert to previous state
terraform state pull > current-state.json
# Restore from backup
terraform state push previous-state.json

# Or destroy and recreate
terraform destroy
terraform apply
```

## Security Considerations

1. **Never commit secrets** to version control
2. **Use strong passwords** for all services
3. **Enable TLS/SSL** for production deployments
4. **Rotate credentials** regularly
5. **Use secrets management** (Vault, AWS Secrets Manager, etc.)
6. **Scan images** for vulnerabilities before deployment
7. **Enable network policies** in Kubernetes
8. **Use RBAC** for access control
9. **Enable audit logging**
10. **Keep dependencies updated**

## Maintenance

### Regular Tasks

**Daily:**
- Monitor service health
- Check error logs
- Review alerts

**Weekly:**
- Review resource usage
- Check for security updates
- Backup databases

**Monthly:**
- Update dependencies
- Review and optimize costs
- Test disaster recovery procedures
- Rotate credentials

### Updates

**Application Updates:**
```bash
# Docker Compose
docker-compose pull
docker-compose up -d

# Kubernetes
kubectl set image deployment/api api=${DOCKER_REGISTRY}/sales-strategist-api:${NEW_TAG} -n sales-strategist
```

**Infrastructure Updates:**
```bash
# Terraform
terraform plan
terraform apply
```

## Support

For issues or questions:
1. Check the [Deployment Guide](../docs/deployment_guide.md)
2. Review [Monitoring and Logging Guide](../docs/monitoring_logging_guide.md)
3. Check application logs for error messages
4. Consult API documentation at `/api/v1/docs`

## Additional Resources

- [Main Documentation](../docs/)
- [API Gateway Guide](../docs/api_gateway_guide.md)
- [Database Setup Guide](../docs/database_setup_guide.md)
- [Kubernetes Manifests](../k8s/)
- [Docker Compose Configuration](../docker-compose.yml)
