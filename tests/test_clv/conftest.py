import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import yaml

# Common test constants
TEST_FEATURES = ['recency', 'frequency', 'monetary']
TEST_DATES = pd.date_range('2023-01-01', periods=10, freq='D')

# Import all fixtures from parent conftest
from ..conftest import (
    config_loader,
    sample_model_data,
    sample_transaction_data,
    mock_gcs_bucket,
    mock_bigquery_client,
    sample_customer_features
)

@pytest.fixture
def base_test_data():
    """Fixture providing basic test DataFrame with required features"""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4],
        'recency': [10, 20, 30, 40],
        'frequency': [2, 3, 4, 5],
        'monetary': [100, 200, 300, 400],
        'transaction_date': TEST_DATES[:4]
    })

@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing"""
    mock = MagicMock()
    mock.predict.return_value = pd.Series([1, 2, 3, 4])
    return mock

@pytest.fixture
def seaborn_style():
    """Fixture to handle seaborn style context"""
    with sns.set_style('darkgrid'):
        yield

@pytest.fixture
def test_config_dir(tmp_path):
    """Create a temporary config directory with test config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create test configs
    configs = {
        "model_config.yaml": {
            "model_type": "hierarchical_clv",
            "parameters": {
                "chains": 4,
                "draws": 2000,
                "tune": 1000,
                "target_accept": 0.8,
                "random_seed": 42
            }
        },
        "deployment_config.yaml": {
            "environment": "test",
            "model_serving": {
                "endpoint": "localhost:8080",
                "timeout": 30,
                "max_retries": 3
            }
        },
        "segment_config.yaml": {
            "segmentation": {
                "method": "kmeans",
                "n_clusters": 5,
                "features": ["recency", "frequency", "monetary"],
                "scaling": "standard"
            }
        },
        "pipeline_config.yaml": {
            "pipeline": {
                "input": {
                    "table": "test_table",
                    "dataset": "test_dataset",
                    "project": "test-project"
                },
                "output": {
                    "bucket": "gs://test-bucket",
                    "prefix": "test_predictions"
                },
                "processing": {
                    "batch_size": 100,
                    "workers": 2
                }
            }
        }
    }
    
    # Write config files
    for filename, content in configs.items():
        with open(config_dir / filename, 'w') as f:
            yaml.dump(content, f)
    
    return config_dir

@pytest.fixture
def mock_config_loader(test_config_dir):
    """Return a config loader that uses the test config directory."""
    from src.pipeline.clv.config import CLVConfigLoader
    return CLVConfigLoader(config_dir=test_config_dir)

# Import fixtures from parent conftest
pytest.fixture(autouse=True)(lambda: None)  # Ensures parent fixtures are loaded