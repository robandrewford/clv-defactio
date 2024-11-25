.PHONY: install test test-convergence test-performance test-integration test-clv clean setup

# Initial setup
install:
	poetry install

# Activate virtual environment
shell:
	poetry shell

# Run all tests
test:
	poetry run pytest

# Run specific test suites
test-convergence:
	poetry run pytest tests/test_clv/test_model_diagnostics.py

test-performance:
	poetry run pytest tests/test_clv/test_performance.py

test-integration:
	poetry run pytest tests/test_clv/test_integration.py

# Run CLV tests specifically
test-clv:
	poetry run pytest tests/test_clv -v

# Run with coverage
test-coverage:
	poetry run pytest --cov=src --cov-report=html

# Run with specific markers
test-statistical:
	poetry run pytest -v -m "statistical"

# Clean up
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.pyc" -exec rm -rf {} +

# Setup development environment
setup: clean install