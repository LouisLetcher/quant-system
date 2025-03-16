#!/bin/bash
# Run all unit tests for the quant trading system

# Set up colored output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running unit tests for Quant Trading System...${NC}"

# Create test directory if it doesn't exist
mkdir -p tests/reports

# Run the test suite
echo -e "${YELLOW}Running main test suite...${NC}"
result=$(poetry run python -m tests.test_suite)

# Check if tests passed
if [[ $result == *"FAILED"* ]]; then
    echo -e "${RED}Tests failed!${NC}"
    echo "$result"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
fi

# Run coverage report
echo -e "${YELLOW}Generating coverage report...${NC}"
poetry run coverage run -m tests.test_suite
poetry run coverage report
poetry run coverage html -d tests/reports/coverage

echo -e "${GREEN}Test run complete. Coverage report available in tests/reports/coverage/index.html${NC}"
