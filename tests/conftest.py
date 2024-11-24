import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.pipeline.clv import CLVConfigLoader

@pytest.fixture
def config_loader():
    """Provide test configuration"""
    return CLVConfigLoader("tests/test_config")

@pytest.fixture
def sample_model_data():
    """Generate sample model data"""
    n_samples = 100
    return {
        'frequency': np.random.poisson(5, n_samples),
        'recency': np.random.randint(0, 365, n_samples),
        'monetary_value': np.random.lognormal(3, 1, n_samples),
        'T': np.random.randint(100, 1000, n_samples),
        'segment_ids': np.zeros(n_samples)
    }

@pytest.fixture
def sample_transaction_data():
    """Generate sample transaction data"""
    n_transactions = 1000
    return pd.DataFrame({
        'customer_id': np.random.randint(1, 101, n_transactions),
        'transaction_date': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_transactions)
        ],
        'transaction_amount': np.random.lognormal(3, 1, n_transactions)
    }) 