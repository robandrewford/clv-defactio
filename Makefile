.PHONY: test test-convergence test-performance test-integration

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