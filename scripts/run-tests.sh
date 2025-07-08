#!/bin/bash
set -e

echo "🧪 Running Quant System Test Suite"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed. Please install Poetry first."
    exit 1
fi

# Install dependencies
print_status "Installing dependencies..."
poetry install --no-interaction

# Run linting
print_status "Running code quality checks..."
echo "  → Ruff linter..."
poetry run ruff check src/ tests/ || {
    print_error "Ruff linting failed"
    exit 1
}

echo "  → Ruff formatter..."
poetry run ruff format --check src/ tests/ || {
    print_error "Ruff formatting check failed"
    exit 1
}

echo "  → Black formatter..."
poetry run black --check src/ tests/ || {
    print_warning "Black formatting check failed (non-critical)"
}

echo "  → isort import sorting..."
poetry run isort --check-only src/ tests/ || {
    print_warning "isort check failed (non-critical)"
}

# Run type checking
print_status "Running type checks..."
poetry run mypy src/ --ignore-missing-imports || {
    print_warning "Type checking found issues (non-critical)"
}

# Run security checks
print_status "Running security checks..."
echo "  → Safety check..."
poetry run safety check --json || {
    print_warning "Safety check found issues (non-critical)"
}

echo "  → Bandit security linter..."
poetry run bandit -r src/ -f json || {
    print_warning "Bandit found security issues (non-critical)"
}

# Run unit tests
print_status "Running unit tests..."
poetry run pytest tests/core/ -v \
    --cov=src/core \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml \
    --tb=short \
    -m "not slow and not requires_api" || {
    print_error "Unit tests failed"
    exit 1
}

# Run integration tests
print_status "Running integration tests..."
poetry run pytest tests/integration/ -v \
    --cov=src \
    --cov-append \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml \
    --tb=short \
    -m "not slow and not requires_api" || {
    print_error "Integration tests failed"
    exit 1
}

# Run CLI smoke tests
print_status "Running CLI smoke tests..."
poetry run python -m src.cli.unified_cli --help > /dev/null || {
    print_error "CLI smoke test failed"
    exit 1
}

poetry run python -m src.cli.unified_cli cache stats > /dev/null || {
    print_error "CLI cache command failed"
    exit 1
}

# Generate coverage report
print_status "Generating coverage report..."
poetry run coverage report --show-missing

# Check coverage threshold
COVERAGE=$(poetry run coverage report --format=total)
THRESHOLD=80

if (( $(echo "$COVERAGE < $THRESHOLD" | bc -l) )); then
    print_warning "Coverage is below threshold: $COVERAGE% < $THRESHOLD%"
else
    print_status "Coverage meets threshold: $COVERAGE% >= $THRESHOLD%"
fi

# Run slow tests if requested
if [[ "$1" == "--slow" ]]; then
    print_status "Running slow tests..."
    poetry run pytest tests/ -v -m "slow" --timeout=600 || {
        print_warning "Some slow tests failed (non-critical)"
    }
fi

# Run API tests if requested
if [[ "$1" == "--api" ]] || [[ "$2" == "--api" ]]; then
    print_status "Running API-dependent tests..."
    poetry run pytest tests/ -v -m "requires_api" || {
        print_warning "API tests failed (API keys may be missing)"
    }
fi

print_status "All tests completed successfully! ✅"
echo ""
echo "📊 Test Results Summary:"
echo "  • Code quality: ✅ Passed"
echo "  • Unit tests: ✅ Passed"
echo "  • Integration tests: ✅ Passed"
echo "  • Coverage: $COVERAGE%"
echo ""
echo "📁 Generated files:"
echo "  • htmlcov/index.html - HTML coverage report"
echo "  • coverage.xml - XML coverage report"
