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
    """Fixture to provide model and pipeline configuration"""
    class SimpleConfigLoader:
        def __init__(self):
            self.model_config = {
                'segment_config': {
                    'use_engagement': True,
                    'use_covariates': False
                }
            }
            self.pipeline_config = {
                'data_processing': {
                    'validation': {
                        'required_columns': ['customer_id', 'transaction_date', 'transaction_amount'],
                        'min_rows': 10,
                        'max_missing_pct': 0.1
                    },
                    'cleaning': {
                        'outlier_method': 'iqr',
                        'outlier_threshold': 1.5
                    }
                },
                'feature_engineering': {
                    'time_features': {'enable': True, 'features': []},
                    'customer_features': {'enable': True, 'features': []},
                    'product_features': {'enable': True, 'features': []}
                }
            }
            
        def get_storage_config(self):
            return {
                'model_storage': {
                    'type': 'local',
                    'path': '/tmp/models'
                },
                'registry': {
                    'type': 'simple',
                    'path': '/tmp/registry'
                },
                'gcs': {
                    'bucket_name': 'test-bucket',
                    'project_id': 'test-project',
                    'model_prefix': 'models/clv'
                }
            }
    
    return SimpleConfigLoader()

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
    """Generate sample transaction data"""
    n_transactions = 1000
    data = pd.DataFrame({
        'customer_id': np.random.randint(1, 101, n_transactions),
        'transaction_id': range(n_transactions),
        'transaction_date': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_transactions)
        ],
        'transaction_amount': np.random.lognormal(3, 1, n_transactions),
        'category': np.random.choice(CATEGORIES, n_transactions),
        'brand': np.random.choice(BRANDS, n_transactions),
        'channel': np.random.choice(['online', 'store'], n_transactions),
        'sms_active': np.random.randint(0, 2, n_transactions),
        'email_active': np.random.randint(0, 2, n_transactions),
        'is_loyalty_member': np.random.randint(0, 2, n_transactions),
        'loyalty_points': np.random.randint(0, 1000, n_transactions),
        'transaction_amount_std': np.random.rand(n_transactions),
        'price_sensitivity': np.random.rand(n_transactions),
    })
    
    # Add monetary column (same as transaction_amount)
    data['monetary'] = data['transaction_amount']
    
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