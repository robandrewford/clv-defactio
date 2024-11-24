# Run all diagnostic tests
poetry run pytest -v tests/test_clv/test_model_diagnostics.py

# Run specific diagnostic
poetry run pytest tests/test_clv/test_model_diagnostics.py::TestModelDiagnostics::test_model_convergence_quality

# Run with detailed output
poetry run pytest -v -s tests/test_clv/test_model_diagnostics.py