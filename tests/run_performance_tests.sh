# Run all performance tests
poetry run pytest -v tests/test_clv/test_performance.py

# Run specific performance test
poetry run pytest tests/test_clv/test_performance.py::TestCLVPipelinePerformance::test_preprocessing_performance

# Run with memory profiling
poetry run pytest --profile tests/test_clv/test_performance.py

# Run specific data size
poetry run pytest tests/test_clv/test_performance.py -k "n_records_1000"