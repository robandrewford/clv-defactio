import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.pipeline.clv.config import CLVConfigLoader

@pytest.fixture
def sample_transaction_data():
    """Generate sample transaction data for testing"""
    np.random.seed(42)
    n_customers = 100
    n_transactions = 1000
    
    # Generate customer IDs
    customer_ids = np.random.randint(1000, 9999, n_customers)
    
    # Generate transactions
    data = {
        'customer_id': np.random.choice(customer_ids, n_transactions),
        'transaction_date': [
            datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            for _ in range(n_transactions)
        ],
        'transaction_amount': np.random.lognormal(3, 1, n_transactions),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], n_transactions),
        'brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C'], n_transactions),
        'channel': np.random.choice(['online', 'store'], n_transactions)
    }
    
    df = pd.DataFrame(data)
    
    # Sort by customer and date
    df = df.sort_values(['customer_id', 'transaction_date'])
    
    return df

@pytest.fixture
def sample_customer_features(sample_transaction_data):
    """Generate sample customer features from transactions"""
    df = sample_transaction_data.copy()
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'transaction_date': ['min', 'max', 'count'],
        'transaction_amount': ['sum', 'mean', 'std'],
        'category': 'nunique',
        'brand': 'nunique'
    })
    
    # Flatten column names
    customer_metrics.columns = [
        'first_purchase', 'last_purchase', 'frequency',
        'monetary', 'avg_transaction_value', 'transaction_std',
        'distinct_categories', 'distinct_brands'
    ]
    
    # Calculate recency and customer age
    now = df['transaction_date'].max() + timedelta(days=1)
    customer_metrics['recency'] = (now - customer_metrics['last_purchase']).dt.days
    customer_metrics['customer_age_days'] = (
        customer_metrics['last_purchase'] - customer_metrics['first_purchase']
    ).dt.days
    
    # Add engagement metrics
    customer_metrics['sms_active'] = np.random.choice([0, 1], len(customer_metrics))
    customer_metrics['email_active'] = np.random.choice([0, 1], len(customer_metrics))
    customer_metrics['is_loyalty_member'] = np.random.choice([0, 1], len(customer_metrics))
    customer_metrics['loyalty_points'] = np.random.poisson(100, len(customer_metrics))
    
    # Add channel preferences
    channel_counts = df.groupby('customer_id')['channel'].value_counts().unstack(fill_value=0)
    customer_metrics['has_online_purchases'] = (channel_counts['online'] > 0).astype(int)
    customer_metrics['has_store_purchases'] = (channel_counts['store'] > 0).astype(int)
    
    # Reset index to make customer_id a column
    customer_metrics = customer_metrics.reset_index()
    
    return customer_metrics

@pytest.fixture
def sample_model_data(sample_customer_features):
    """Generate sample data formatted for model training"""
    return {
        'customer_id': sample_customer_features['customer_id'].values,
        'frequency': sample_customer_features['frequency'].values,
        'recency': sample_customer_features['recency'].values,
        'monetary_value': sample_customer_features['monetary'].values,
        'T': sample_customer_features['customer_age_days'].values,
        'segment_ids': np.zeros(len(sample_customer_features)),  # Default segment
        'customer_features': sample_customer_features[[
            'avg_transaction_value',
            'distinct_categories',
            'distinct_brands',
            'loyalty_points'
        ]].values
    }

@pytest.fixture
def config_loader():
    """Load test configuration"""
    return CLVConfigLoader("tests/test_config")

@pytest.fixture
def mock_gcs_bucket(monkeypatch):
    """Mock GCS bucket for testing"""
    class MockBlob:
        def __init__(self, name):
            self.name = name
            self._data = None
        
        def upload_from_string(self, data):
            self._data = data
        
        def download_as_string(self):
            return self._data
    
    class MockBucket:
        def __init__(self):
            self.blobs = {}
        
        def blob(self, name):
            if name not in self.blobs:
                self.blobs[name] = MockBlob(name)
            return self.blobs[name]
        
        def list_blobs(self, prefix=None):
            return [blob for name, blob in self.blobs.items() 
                   if not prefix or name.startswith(prefix)]
    
    mock_bucket = MockBucket()
    monkeypatch.setattr("google.cloud.storage.Bucket", lambda _: mock_bucket)
    return mock_bucket

@pytest.fixture
def mock_bigquery_client(monkeypatch):
    """Mock BigQuery client for testing"""
    class MockBigQueryClient:
        def __init__(self):
            self.tables = {}
        
        def get_table(self, table_ref):
            return self.tables.get(table_ref)
        
        def create_table(self, table):
            self.tables[table.table_id] = table
    
    mock_client = MockBigQueryClient()
    monkeypatch.setattr("google.cloud.bigquery.Client", lambda: mock_client)
    return mock_client 