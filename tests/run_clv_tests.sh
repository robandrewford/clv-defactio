#!/bin/bash

# Run all CLV tests
pytest tests/test_clv/ -v

# Run specific test suites
pytest tests/test_clv/test_model_diagnostics.py -v
pytest tests/test_clv/test_integration.py -v
pytest tests/test_clv/test_performance.py -v