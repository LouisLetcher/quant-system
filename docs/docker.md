# Docker Guide

Guide for running the Quant Trading System with Docker.

## Quick Start

### Run the System
```bash
# Clone and setup
git clone https://github.com/LouisLetcher/quant-system.git
cd quant-system

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Run the system
docker-compose up quant-system
```

### Access Services
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Jupyter Lab**: http://localhost:8888 (password: `quant`)

## Docker Compose Services

### Core Services
```bash
# Main application
docker-compose up quant-system

# Development environment
docker-compose up dev

# Testing environment
docker-compose up test
```

### Extended Services
```bash
# With database
docker-compose --profile database up

# With API server
docker-compose --profile api up

# With monitoring
docker-compose --profile monitoring up

# Full stack
docker-compose --profile database --profile api --profile monitoring up
```

## Available Profiles

### `dev` - Development
- Hot-reload enabled
- Debug logging
- Development dependencies
- Volume mounts for code

### `test` - Testing
- Test environment
- Isolated test database
- Coverage reporting

### `api` - Web API
- FastAPI server
- OpenAPI documentation
- REST endpoints

### `database` - PostgreSQL
- Persistent data storage
- Automated backups
- Connection pooling

### `cache` - Redis
- Data caching
- Session storage
- Performance optimization

### `monitoring` - Observability
- Prometheus metrics
- Grafana dashboards
- Health checks

### `jupyter` - Analysis
- Jupyter Lab
- Data science tools
- Interactive analysis

## Environment Configuration

### Required Variables
```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key

# System Settings
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

### Database Configuration
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:pass@postgres:5432/quant_db
POSTGRES_USER=quant_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=quant_db
```

### Monitoring Configuration
```bash
# Prometheus
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin
```

## Volume Mounts

The Docker setup includes several volume mounts:

```yaml
volumes:
  - ./config:/app/config           # Configuration files
  - ./cache:/app/cache             # Data cache
  - ./reports_output:/app/reports_output  # Generated reports
  - ./exports:/app/exports         # Data exports
  - postgres_data:/var/lib/postgresql/data  # Database
  - redis_data:/data               # Cache storage
```

## Production Deployment

### Build Production Image
```bash
# Build optimized image
docker build -t quant-system:prod .

# Or use multi-stage build
docker build --target production -t quant-system:prod .
```

### Deploy with Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml quant-stack
```

### Deploy with Kubernetes
```bash
# Convert docker-compose to k8s
kompose convert

# Apply manifests
kubectl apply -f .
```

## Health Checks

Health checks are configured for all services:

```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs quant-system

# Execute health check manually
docker-compose exec quant-system python -c "
import requests
response = requests.get('http://localhost:8000/health')
print(response.status_code, response.json())
"
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :8000

   # Use different ports
   export API_PORT=8001
   docker-compose up
   ```

2. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER ./cache ./reports_output

   # Or use Docker user
   export UID=$(id -u)
   export GID=$(id -g)
   docker-compose up
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory

   # Or limit container memory
   docker-compose up --memory=2g
   ```

4. **Database Connection**
   ```bash
   # Check database status
   docker-compose logs postgres

   # Connect to database
   docker-compose exec postgres psql -U quant_user -d quant_db
   ```

### Debug Commands
```bash
# Enter container shell
docker-compose exec quant-system bash

# View container logs
docker-compose logs -f quant-system

# Check resource usage
docker stats

# Inspect container
docker-compose exec quant-system python --version
docker-compose exec quant-system pip list
```

## Performance Optimization

### Resource Limits
```yaml
services:
  quant-system:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Caching Strategy
- Use Redis for session/API caching
- Mount cache directory as volume
- Implement cache warming

### Database Optimization
- Use connection pooling
- Optimize queries
- Regular maintenance

## Monitoring and Logging

### Prometheus Metrics
Access metrics at http://localhost:9090

Key metrics:
- API request duration
- Cache hit rate
- Database connections
- Memory usage

### Grafana Dashboards
Access dashboards at http://localhost:3000

Default dashboards:
- System Overview
- API Performance
- Database Metrics
- Cache Performance

### Log Aggregation
```bash
# View all logs
docker-compose logs

# Follow specific service
docker-compose logs -f quant-system

# Export logs
docker-compose logs > system.log
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U quant_user quant_db > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U quant_user quant_db < backup.sql
```

### Data Backup
```bash
# Backup cache and reports
tar -czf backup.tar.gz cache/ reports_output/ config/

# Restore data
tar -xzf backup.tar.gz
```

## Security

### Network Security
- Use internal networks
- Expose only necessary ports
- Implement SSL/TLS

### Secrets Management
```bash
# Use Docker secrets
echo "api_key_value" | docker secret create alpha_vantage_key -

# Or use external secret management
# - HashiCorp Vault
# - AWS Secrets Manager
# - Azure Key Vault
```

### Image Security
```bash
# Scan images for vulnerabilities
docker scan quant-system:latest

# Use distroless base images
# Use multi-stage builds
# Regular security updates
```
