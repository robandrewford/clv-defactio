# Run only integration tests
poetry run pytest -v tests/test_clv/test_integration.py

# Run with increased verbosity
poetry run pytest -v -s tests/test_clv/test_integration.py

# Run specific test
poetry run pytest tests/test_clv/test_integration.py::TestCLVPipelineIntegration::test_end_to_end_pipeline