# Docker Guide

This document provides comprehensive information about using Docker with the Quant Trading System.

## Table of Contents

- [Overview](#overview)
- [Docker Images](#docker-images)
- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Production Deployment](#production-deployment)
- [Service Configurations](#service-configurations)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)

## Overview

The Quant Trading System uses a multi-stage Docker architecture that provides:

- **Production-ready images** with minimal attack surface
- **Development environment** with full tooling
- **Testing environment** for CI/CD pipelines
- **Jupyter environment** for data analysis
- **API service** for web interfaces
- **Full stack deployment** with databases and monitoring

## Docker Images

### Available Targets

The Dockerfile provides multiple build targets:

| Target | Purpose | Size | Use Case |
|--------|---------|------|----------|
| `production` | Production deployment | ~200MB | Production servers |
| `development` | Development work | ~400MB | Local development |
| `testing` | Running tests | ~400MB | CI/CD pipelines |
| `jupyter` | Data analysis | ~450MB | Research and analysis |
| `api` | Web API service | ~220MB | API endpoints |

### Image Tags

Images are tagged following semantic versioning:

```bash
# Latest stable release
quant-system:latest

# Specific version
quant-system:v1.2.3

# Development builds
quant-system:dev
quant-system:main-<commit-sha>

# Environment-specific
quant-system:test
quant-system:jupyter
```

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ available RAM
- 10GB+ available disk space

### Basic Setup

1. **Clone and Build**

```bash
git clone <repository-url>
cd quant-system

# Build all images
./scripts/run-docker.sh build
```

2. **Start Development Environment**

```bash
# Start development container
./scripts/run-docker.sh dev

# Access development shell
./scripts/run-docker.sh shell
```

3. **Run Basic Commands**

```bash
# Check system status
docker run --rm quant-system:latest python -m src.cli.unified_cli cache stats

# Download sample data
docker run --rm -v $(pwd)/cache:/app/cache quant-system:latest \
    python -m src.cli.unified_cli data download --symbols AAPL --start-date 2023-01-01 --end-date 2023-01-31
```

## Development Environment

### Starting Development Container

```bash
# Method 1: Using helper script
./scripts/run-docker.sh dev

# Method 2: Using docker-compose directly
docker-compose --profile dev up -d quant-dev
```

### Development Workflow

1. **Access Development Shell**

```bash
# Interactive shell
./scripts/run-docker.sh shell

# Or manually
docker-compose --profile dev exec quant-dev bash
```

2. **Run Development Commands**

```bash
# Inside container
poetry install
poetry run pytest tests/
poetry run python -m src.cli.unified_cli --help
```

3. **File Synchronization**

The development container mounts the entire project directory:

```yaml
volumes:
  - .:/app  # Live code updates
```

Changes on your host machine are immediately reflected in the container.

### Development Features

- **Hot Reload**: Code changes are immediately available
- **Full Tooling**: All development dependencies included
- **Interactive Debugging**: VS Code and debugger support
- **Test Environment**: Complete test suite available

## Production Deployment

### Single Container Deployment

```bash
# Build production image
docker build -t quant-system:prod --target production .

# Run production container
docker run -d \
  --name quant-system \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/exports:/app/exports \
  -v $(pwd)/config:/app/config:ro \
  -e ENVIRONMENT=production \
  quant-system:prod
```

### Multi-Service Deployment

```bash
# Start API with database
./scripts/run-docker.sh full

# Start with monitoring
docker-compose --profile api --profile database --profile monitoring up -d
```

### Production Configuration

```yaml
# docker-compose.override.yml for production
services:
  quant-system:
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
      - CACHE_SIZE_GB=50
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Security Considerations

1. **Non-root User**: Containers run as `quantuser`
2. **Read-only Config**: Configuration mounted read-only
3. **Minimal Image**: Production images contain only necessary components
4. **Secret Management**: Use Docker secrets or environment variables

```bash
# Using Docker secrets
echo "your-api-key" | docker secret create bybit_api_key -
```

## Service Configurations

### API Service

```bash
# Start API server
./scripts/run-docker.sh api

# Access API
curl http://localhost:8000/health
open http://localhost:8000/docs
```

**Configuration:**
- Port: 8000
- Health check: `/health`
- Documentation: `/docs`
- Metrics: `/metrics`

### Jupyter Service

```bash
# Start Jupyter Lab
./scripts/run-docker.sh jupyter

# Access Jupyter
open http://localhost:8888
```

**Features:**
- Pre-installed analysis libraries
- Access to quant system modules
- Persistent notebook storage
- Sample notebooks included

### Database Services

```bash
# Start PostgreSQL
docker-compose --profile database up -d postgres

# Start Redis
docker-compose --profile cache up -d redis
```

**PostgreSQL:**
- Port: 5432
- Database: `quant_system`
- User: `quantuser`
- Password: `quantpass`

**Redis:**
- Port: 6379
- Persistence: Enabled
- Configuration: `/data`

### Full Stack Deployment

```bash
# Start everything
docker-compose --profile api --profile database --profile cache --profile monitoring up -d
```

This starts:
- API service (port 8000)
- PostgreSQL database (port 5432)
- Redis cache (port 6379)
- Prometheus monitoring (port 9090)
- Grafana dashboards (port 3000)

## Monitoring and Logging

### Prometheus Monitoring

```bash
# Start monitoring stack
./scripts/run-docker.sh monitoring

# Access Prometheus
open http://localhost:9090

# Access Grafana
open http://localhost:3000  # admin/admin
```

### Available Metrics

- API response times
- Database connections
- Cache hit rates
- System resources
- Application errors

### Log Management

```bash
# View logs
./scripts/run-docker.sh logs

# Follow logs
./scripts/run-docker.sh logs --follow

# Specific service logs
docker-compose logs -f api
```

### Log Configuration

```yaml
# docker-compose.yml
services:
  quant-system:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Health Checks

All services include health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -m src.cli.unified_cli cache stats || exit 1
```

Check health status:

```bash
docker ps  # Shows health status
docker-compose ps  # Shows service status
```

## Advanced Configuration

### Environment Variables

```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO
CACHE_DIR=/app/cache
MAX_WORKERS=4

# API keys
BYBIT_API_KEY=your_key
BYBIT_SECRET_KEY=your_secret
ALPHA_VANTAGE_API_KEY=your_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379
```

### Volume Mounts

```yaml
volumes:
  # Persistent data
  - ./cache:/app/cache
  - ./exports:/app/exports
  - ./logs:/app/logs
  
  # Configuration (read-only)
  - ./config:/app/config:ro
  
  # Development (read-write)
  - .:/app
```

### Network Configuration

```yaml
networks:
  quant-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## Performance Optimization

### Image Optimization

1. **Multi-stage builds** reduce final image size
2. **Layer caching** speeds up builds
3. **Dependency optimization** minimizes packages

```dockerfile
# Use .dockerignore to exclude unnecessary files
# Combine RUN commands to reduce layers
# Use specific package versions
```

### Runtime Optimization

```bash
# Use production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.api.main:app"]

# Enable container restart policies
restart: unless-stopped

# Set appropriate resource limits
deploy:
  resources:
    limits:
      memory: 2G
```

### Build Optimization

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build .

# Use build cache
docker build --cache-from quant-system:latest .

# Multi-platform builds
docker buildx build --platform linux/amd64,linux/arm64 .
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ci.yml
- name: Build Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: ./Dockerfile
    push: true
    tags: |
      ghcr.io/${{ github.repository }}:latest
      ghcr.io/${{ github.repository }}:${{ github.sha }}
```

### Registry Deployment

```bash
# Build and tag for registry
docker build -t ghcr.io/your-org/quant-system:latest .

# Push to registry
docker push ghcr.io/your-org/quant-system:latest

# Pull and run from registry
docker run ghcr.io/your-org/quant-system:latest
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**

```bash
# Check logs
docker logs <container-name>

# Check health status
docker inspect <container-name> | grep Health

# Verify environment
docker exec <container-name> env
```

2. **Port Conflicts**

```bash
# Check port usage
netstat -tulpn | grep :8000

# Use different ports
docker run -p 8001:8000 quant-system:latest
```

3. **Volume Mount Issues**

```bash
# Check permissions
ls -la /path/to/volume

# Fix permissions
sudo chown -R 1000:1000 ./cache
```

4. **Memory Issues**

```bash
# Check container memory usage
docker stats

# Increase memory limits
docker run --memory=4g quant-system:latest
```

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG quant-system:latest

# Interactive debugging
docker run -it --entrypoint=/bin/bash quant-system:latest
```

### Performance Monitoring

```bash
# Monitor container resources
docker stats --all

# Check container processes
docker exec <container> ps aux

# Monitor disk usage
docker system df
```

### Cleanup

```bash
# Clean up containers
./scripts/run-docker.sh clean

# Remove unused images
docker image prune -a

# Clean everything
docker system prune -a --volumes
```

## Best Practices

### Development

1. Use development target for active coding
2. Mount source code as volume for hot reload
3. Keep development and production environments similar
4. Use consistent naming conventions

### Production

1. Use minimal production images
2. Implement proper health checks
3. Set resource limits
4. Use secrets management
5. Enable logging and monitoring
6. Plan for scaling

### Security

1. Run as non-root user
2. Use read-only filesystems where possible
3. Scan images for vulnerabilities
4. Keep base images updated
5. Use specific image tags, not `latest`

This Docker guide provides everything needed to successfully deploy and manage the Quant Trading System in any environment.
