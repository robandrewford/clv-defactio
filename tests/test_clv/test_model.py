import pytest
import numpy as np
from src.pipeline.clv.model import HierarchicalCLVModel

def test_model_initialization(config_loader):
    """Test model initialization"""
    model = HierarchicalCLVModel(config_loader)
    assert model is not None
    assert model.model is None
    assert model.trace is None

def test_model_building(config_loader, sample_customer_features):
    """Test model building"""
    model = HierarchicalCLVModel(config_loader)
    
    # Prepare data for model
    data = {
        'frequency': sample_customer_features['frequency'].values,
        'recency': sample_customer_features['recency'].values,
        'monetary_value': sample_customer_features['monetary'].values,
        'T': sample_customer_features['customer_age_days'].values,
        'segment_ids': np.zeros(len(sample_customer_features))
    }
    
    # Build model
    built_model = model.build_model(data)
    assert built_model is not None
    assert model.model is not None

def test_model_sampling(config_loader, sample_customer_features):
    """Test model sampling"""
    model = HierarchicalCLVModel(config_loader)
    
    # Prepare data
    data = {
        'frequency': sample_customer_features['frequency'].values,
        'recency': sample_customer_features['recency'].values,
        'monetary_value': sample_customer_features['monetary'].values,
        'T': sample_customer_features['customer_age_days'].values,
        'segment_ids': np.zeros(len(sample_customer_features))
    }
    
    # Build and sample
    model.build_model(data)
    trace = model.sample(draws=100, tune=50, chains=2)
    
    assert trace is not None
    assert model.trace is not None 

def test_model_training(sample_model_data, config_loader):
    """Test model training with prepared data"""
    model = HierarchicalCLVModel(config_loader)
    model.build_model(sample_model_data)
    
    # Train with small number of samples for testing
    trace = model.sample(draws=50, tune=25, chains=2)
    assert trace is not None
    
    # Check predictions
    predictions = model.predict(sample_model_data, prediction_period=30)
    assert len(predictions) == len(sample_model_data['customer_id']) 