import pytest
import numpy as np
import pymc as pm
from src.pipeline.clv import HierarchicalCLVModel
from src.pipeline.clv.base import BaseModel

class TestHierarchicalCLVModel:
    """Test suite for HierarchicalCLVModel"""

    def test_model_initialization(self, config_loader):
        """Test model initialization"""
        model = HierarchicalCLVModel(config_loader)
        assert isinstance(model, BaseModel)
        assert model.model is None
        assert model.trace is None

    def test_model_building(self, config_loader, sample_customer_features):
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
        
        # Verify model structure
        assert isinstance(built_model, pm.Model)
        assert 'alpha' in built_model.named_vars
        assert 'beta' in built_model.named_vars
        assert 'lambda' in built_model.named_vars
        assert 'freq_obs' in built_model.named_vars
        assert 'mu_m' in built_model.named_vars
        assert 'sigma_m' in built_model.named_vars
        assert 'monetary' in built_model.named_vars

    def test_model_interface(self, config_loader):
        """Test model interface compliance"""
        model = HierarchicalCLVModel(config_loader)
        
        # Verify required methods exist
        assert hasattr(model, 'build_model')
        assert hasattr(model, 'train_model')
        assert hasattr(model, 'evaluate_model')
        
        # Verify method signatures
        from inspect import signature
        assert 'data' in signature(model.build_model).parameters