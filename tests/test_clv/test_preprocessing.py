import pytest
from src.pipeline.clv.preprocessing import CLVDataPreprocessor

def test_preprocessor_initialization(config_loader):
    """Test preprocessor initialization"""
    preprocessor = CLVDataPreprocessor(config_loader)
    assert preprocessor is not None
    assert preprocessor.pipeline_config is not None
    assert preprocessor.feature_config is not None

def test_data_cleaning(config_loader, sample_transaction_data):
    """Test data cleaning functionality"""
    preprocessor = CLVDataPreprocessor(config_loader)
    
    # Add some duplicates and missing values
    dirty_data = sample_transaction_data.copy()
    dirty_data = pd.concat([dirty_data, dirty_data.head(10)])
    dirty_data.loc[0:5, 'transaction_amount'] = np.nan
    
    # Process data
    clean_data = preprocessor.process_data(dirty_data)
    
    assert len(clean_data) < len(dirty_data)  # Duplicates removed
    assert clean_data['transaction_amount'].isna().sum() == 0  # Missing values handled

def test_feature_engineering(sample_transaction_data, config_loader):
    """Test feature engineering with transaction data"""
    preprocessor = CLVDataPreprocessor(config_loader)
    processed_data = preprocessor.process_data(sample_transaction_data)
    
    # Check engineered features
    assert 'frequency' in processed_data.columns
    assert 'recency' in processed_data.columns
    assert 'monetary' in processed_data.columns
    assert 'avg_transaction_value' in processed_data.columns
    
    # Verify calculations
    customer = processed_data.iloc[0]
    assert customer['frequency'] >= 1
    assert customer['recency'] >= 0
    assert customer['monetary'] > 0 