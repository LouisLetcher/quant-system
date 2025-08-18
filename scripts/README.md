# Scripts Directory

Utility scripts for development, debugging, and maintenance.

## Directory Structure

### `/debug/`
Development and debugging scripts used during development. These are temporary scripts used to investigate specific issues and should be reviewed/cleaned up periodically.

- `debug_*.py` - Scripts for investigating specific issues during development

### `/fixes/`
One-time database fix scripts used to address data inconsistencies. These scripts should be reviewed and potentially removed after successful execution.

- `fix_*.py` - Scripts for correcting database issues or data migration

### Root Scripts
Production and CI/CD utility scripts that are part of the regular development workflow.

- `init-db.sql` - Database initialization schema
- `run-tests.sh` - Test execution script with linting, type checking, and security
- `check_precommit.sh` - Pre-commit hook validation
- `ci_validation.py` - CI/CD pipeline validation
- `generate_changelog.py` - Automated changelog generation
- `test_cicd_pipeline.py` - CI/CD testing script

## Usage

```bash
# Run production scripts
./scripts/run-tests.sh

# Execute debug scripts (development only)
docker-compose run --rm -v $(pwd)/scripts/debug/debug_metrics.py:/app/debug_metrics.py quant python debug_metrics.py

# Execute fix scripts (one-time use)
docker-compose run --rm -v $(pwd)/scripts/fixes/fix_metrics.py:/app/fix_metrics.py quant python fix_metrics.py
```

## Cleanup Guidelines

- **Debug scripts**: Should be reviewed monthly and removed if no longer needed
- **Fix scripts**: Should be removed after successful execution and verification
- **Production scripts**: Should be maintained and updated as needed
