import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
from unittest.mock import MagicMock

# Define test data constants
CATEGORIES = ['Electronics', 'Clothing', 'Food', 'Home']
BRANDS = ['BrandA', 'BrandB', 'BrandC', 'BrandD']

@pytest.fixture
def config_loader():
    """Create a mock config loader for testing"""
    class MockConfigLoader:
        def __init__(self):
            self.pipeline_config = {
                'data_processing': {
                    'preprocessing': {
                        'feature_columns': ['frequency', 'recency', 'monetary'],
                        'target_column': 'customer_value'
                    },
                    'feature_engineering': {
                        'transforms': ['log', 'standardize'],
                        'interaction_terms': True,
                        'polynomial_degree': 2
                    }
                },
                'storage': {
                    'model_path': 'models/',
                    'data_path': 'data/',
                    'gcs': {
                        'bucket_name': 'test-bucket',
                        'project_id': 'test-project'
                    }
                },
                'visualization': {
                    'plot_style': 'seaborn',
                    'figure_size': (10, 6)
                },
                'model': {
                    'hyperparameters': {
                        'prior_settings': {
                            'alpha_shape': 1.0,
                            'beta_shape': 1.0
                        }
                    }
                },
                'segment_rules': {
                    'rfm': {
                        'recency_bins': [0, 30, 60, 90],
                        'frequency_bins': [1, 2, 3, 4],
                        'monetary_bins': [0, 100, 500, 1000]
                    }
                },
                'segment_config': {
                    'n_segments': 3,
                    'features': ['recency', 'frequency', 'monetary'],
                    'method': 'kmeans'
                }
            }
            self.config = self.pipeline_config  # For backward compatibility
            
        def get(self, key, default=None):
            """Get configuration value by key"""
            # Handle nested keys with dot notation
            if '.' in key:
                keys = key.split('.')
                value = self.pipeline_config
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k, default)
                    else:
                        return default
                return value
            
            # Handle top level keys
            return self.pipeline_config.get(key, default)
            
        def get_config(self, key, default=None):
            """Alias for get() method for backward compatibility"""
            return self.get(key, default)
            
    return MockConfigLoader()

@pytest.fixture
def sample_model_data():
    """Fixture to provide sample data for model testing"""
    n_customers = 100
    n_transactions = 500
    
    data = pd.DataFrame({
        'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
        'transaction_date': pd.date_range(
            start='2020-01-01', 
            periods=n_transactions, 
            freq='D'
        ),
        'amount': np.random.lognormal(3, 1, n_transactions)
    })
    
    return data

@pytest.fixture
def sample_transaction_data():
    """Create a sample transaction DataFrame for testing"""
    return pd.DataFrame({
        'customer_id': [1, 1, 2, 2],
        'transaction_date': ['2023-01-01', '2023-02-01', '2023-01-15', '2023-03-01'],
        'amount': [100, 200, 150, 300]
    })

@pytest.fixture
def mock_gcs_bucket():
    """Mock Google Cloud Storage bucket for testing."""
    mock_bucket = MagicMock()
    
    # Mock common bucket operations
    mock_bucket.blob.return_value.exists.return_value = True
    mock_bucket.blob.return_value.download_as_string.return_value = b"mock_data"
    mock_bucket.blob.return_value.upload_from_string.return_value = None
    
    return mock_bucket

@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client for testing."""
    mock_client = MagicMock()
    
    # Mock common BigQuery operations
    mock_client.query.return_value.result.return_value = []
    mock_client.get_table.return_value = MagicMock()
    
    return mock_client 

@pytest.fixture
def sample_customer_features():
    """Generate sample customer features for testing"""
    n_samples = 100
    return pd.DataFrame({
        'customer_id': range(n_samples),
        'frequency': np.random.randint(1, 10, n_samples),
        'recency': np.random.randint(1, 365, n_samples),
        'monetary': np.random.uniform(10, 1000, n_samples),
        'transaction_amount': np.random.uniform(10, 1000, n_samples),
        'customer_age_days': np.random.randint(1, 1000, n_samples),
        'sms_active': np.random.choice([0, 1], n_samples),
        'email_active': np.random.choice([0, 1], n_samples),
        'transaction_date': pd.date_range(start='2023-01-01', periods=n_samples),
        'is_loyalty_member': np.random.choice([0, 1], n_samples)
    }) 