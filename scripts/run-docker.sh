#!/bin/bash
set -e

echo "🐳 Quant System Docker Management"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build       Build Docker images"
    echo "  test        Run tests in Docker"
    echo "  dev         Start development environment"
    echo "  prod        Start production environment"
    echo "  jupyter     Start Jupyter notebook server"
    echo "  api         Start API server"
    echo "  full        Start full stack (API + DB + Redis)"
    echo "  monitoring  Start monitoring stack"
    echo "  stop        Stop all services"
    echo "  clean       Clean up containers and images"
    echo "  logs        Show logs for services"
    echo "  shell       Open shell in development container"
    echo ""
    echo "Options:"
    echo "  --rebuild   Force rebuild of images"
    echo "  --no-cache  Build without cache"
    echo "  --follow    Follow logs"
}

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Parse command line arguments
COMMAND=${1:-help}
REBUILD=false
NO_CACHE=false
FOLLOW=false

for arg in "$@"; do
    case $arg in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --follow)
            FOLLOW=true
            shift
            ;;
    esac
done

# Build options
BUILD_OPTS=""
if [ "$REBUILD" = true ]; then
    BUILD_OPTS="$BUILD_OPTS --force-recreate"
fi
if [ "$NO_CACHE" = true ]; then
    BUILD_OPTS="$BUILD_OPTS --no-cache"
fi

case $COMMAND in
    build)
        print_header "Building Docker images..."
        docker-compose build $BUILD_OPTS
        print_status "Build completed successfully!"
        ;;

    test)
        print_header "Running tests in Docker..."
        docker-compose --profile test build quant-test $BUILD_OPTS
        docker-compose --profile test run --rm quant-test
        print_status "Tests completed!"
        ;;

    dev)
        print_header "Starting development environment..."
        docker-compose --profile dev up -d $BUILD_OPTS
        print_status "Development environment started!"
        echo ""
        echo "📝 To access the development container:"
        echo "   docker-compose exec quant-dev bash"
        ;;

    prod)
        print_header "Starting production environment..."
        docker-compose up -d quant-system $BUILD_OPTS
        print_status "Production environment started!"
        ;;

    jupyter)
        print_header "Starting Jupyter notebook server..."
        docker-compose --profile jupyter up -d jupyter $BUILD_OPTS
        print_status "Jupyter server started!"
        echo ""
        echo "📊 Access Jupyter at: http://localhost:8888"
        ;;

    api)
        print_header "Starting API server..."
        docker-compose --profile api up -d api $BUILD_OPTS
        print_status "API server started!"
        echo ""
        echo "🌐 API available at: http://localhost:8000"
        echo "📋 API docs at: http://localhost:8000/docs"
        ;;

    full)
        print_header "Starting full stack (API + Database + Redis)..."
        docker-compose --profile api --profile database --profile cache up -d $BUILD_OPTS
        print_status "Full stack started!"
        echo ""
        echo "🌐 API: http://localhost:8000"
        echo "🗄️  PostgreSQL: localhost:5432"
        echo "📦 Redis: localhost:6379"
        ;;

    monitoring)
        print_header "Starting monitoring stack..."
        docker-compose --profile monitoring up -d $BUILD_OPTS
        print_status "Monitoring stack started!"
        echo ""
        echo "📊 Prometheus: http://localhost:9090"
        echo "📈 Grafana: http://localhost:3000 (admin/admin)"
        ;;

    stop)
        print_header "Stopping all services..."
        docker-compose down
        print_status "All services stopped!"
        ;;

    clean)
        print_header "Cleaning up containers and images..."
        docker-compose down --volumes --remove-orphans
        docker system prune -f
        print_status "Cleanup completed!"
        ;;

    logs)
        print_header "Showing service logs..."
        if [ "$FOLLOW" = true ]; then
            docker-compose logs -f
        else
            docker-compose logs --tail=100
        fi
        ;;

    shell)
        print_header "Opening shell in development container..."
        docker-compose --profile dev exec quant-dev bash || {
            print_warning "Development container not running. Starting it first..."
            docker-compose --profile dev up -d quant-dev
            sleep 2
            docker-compose --profile dev exec quant-dev bash
        }
        ;;

    status)
        print_header "Service status..."
        docker-compose ps
        ;;

    help|--help|-h)
        show_usage
        ;;

    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
