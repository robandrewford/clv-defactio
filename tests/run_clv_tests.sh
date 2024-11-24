# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_clv/test_model.py

# Run with coverage report
poetry run pytest --cov=src --cov-report=html

# Run tests matching a pattern
poetry run pytest -k "model"

# Open htmlcov/index.html in browser
open htmlcov/index.html