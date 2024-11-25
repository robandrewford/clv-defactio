import os
import sys
from pathlib import Path
import warnings

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

# Add at the top of conftest.py, before other fixtures
def pytest_configure(config):
    """Configure pytest to ignore specific warnings"""
    warnings.filterwarnings('ignore', category=DeprecationWarning, 
                          module='seaborn._oldcore')
    warnings.filterwarnings('ignore', category=FutureWarning, 
                          module='seaborn.categorical')
    warnings.filterwarnings('ignore', category=UserWarning, 
                          module='arviz.data.base')

@pytest.fixture
def config_loader():
    """Mock config loader with required configuration"""
    class MockConfigLoader:
        def __init__(self):
            # Add pipeline_config attribute
            self.pipeline_config = {
                'storage': {
                    'type': 'gcs',  # Specify storage type
                    'path': '/tmp/models',
                    'model_prefix': 'models/clv',
                    # Add GCS config at root level of storage
                    'gcs': {
                        'bucket_name': 'mock-bucket',
                        'project_id': 'mock-project',
                        'model_prefix': 'models/clv',
                        'path': 'models'
                    },
                    'model_storage': {
                        'type': 'gcs',
                        'path': '/tmp/models'
                    },
                    'model_registry': {
                        'bucket': 'mock-bucket',
                        'path': 'models',
                        'metadata_file': 'model_metadata.json'
                    }
                },
                'data_processing': {
                    'features': ['recency', 'frequency', 'monetary'],
                    'remove_outliers': True,
                    'outlier_threshold': 3
                },
                'segmentation': {
                    'n_clusters': 2,
                    'features': ['recency', 'frequency', 'monetary']
                },
                'model_parameters': {
                    'chains': 4,
                    'draws': 2000,
                    'tune': 1000,
                    'target_accept': 0.8,
                    'random_seed': 42
                }
            }

            self.model_config = {
                'model_type': 'hierarchical_clv',
                'parameters': {
                    'chains': 4,
                    'draws': 2000,
                    'tune': 1000,
                    'target_accept': 0.8,
                    'random_seed': 42
                }
            }

        def get(self, *args, **kwargs):
            if args[0] == 'model':
                return self.model_config
            elif args[0] == 'pipeline':
                return self.pipeline_config
            return {}
            
        def get_config(self, config_type):
            """Mock get_config method for model configuration"""
            if config_type == 'model':
                return self.model_config
            elif config_type == 'pipeline':
                return self.pipeline_config
            return {}
            
        def __getitem__(self, key):
            return self.get(key)
            
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
    """Create sample transaction data with required features"""
    data = pd.DataFrame({
        'customer_id': [1, 1, 2, 2],
        'transaction_date': ['2023-01-01', '2023-02-01', '2023-01-15', '2023-03-01'],
        'transaction_amount': [100, 200, 150, 300],
        'recency': [30, 30, 45, 45],
        'frequency': [2, 2, 2, 2],
        'monetary': [150, 150, 225, 225],
        'T': [90, 90, 90, 90]
    })
    # Ensure segment_ids match customer_ids
    data['segment'] = data['customer_id'] - 1  # 0-based indexing for segments
    return data

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