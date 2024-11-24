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

@pytest.fixture
def config_loader():
    """Fixture to provide model configuration"""
    class SimpleConfigLoader:
        def __init__(self, config_path):
            self.config_path = config_path
            with open(config_path) as f:
                self.model_config = yaml.safe_load(f)
    
    return SimpleConfigLoader('src/config/model_config.yaml')

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
    return pd.DataFrame({
        'customer_id': np.random.randint(1, 101, n_transactions),
        'transaction_date': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_transactions)
        ],
        'transaction_amount': np.random.lognormal(3, 1, n_transactions)
    }) 