import pytest
import pandas as pd
import numpy as np
from src.pipeline.clv.preprocessing import CLVDataPreprocessor

def test_preprocessor_initialization(config_loader):
    """Test preprocessor initialization"""
    preprocessor = CLVDataPreprocessor(config_loader)
    assert preprocessor is not None
    assert hasattr(preprocessor, 'process_data')

def test_data_cleaning(config_loader, sample_transaction_data):
    """Test data cleaning functionality"""
    preprocessor = CLVDataPreprocessor(config_loader)
    processed_data = preprocessor.process_data(sample_transaction_data)
    
    assert not processed_data.empty
    assert processed_data['monetary'].isnull().sum() == 0
    
    assert len(processed_data.columns) > 0
    assert 'customer_id' in processed_data.columns
    assert 'monetary' in processed_data.columns
    
def test_feature_engineering(sample_transaction_data, config_loader):
    """Test feature engineering with transaction data"""
    preprocessor = CLVDataPreprocessor(config_loader)
    processed_data = preprocessor.process_data(sample_transaction_data)
    
    # Check engineered features exist
    assert 'frequency' in processed_data.columns
    assert 'recency' in processed_data.columns
    assert 'monetary' in processed_data.columns
    
    # Verify calculations
    customer = processed_data.iloc[0]
    assert customer['frequency'] >= 1
    assert customer['recency'] >= 0
    assert customer['monetary'] > 0 