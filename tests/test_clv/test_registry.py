import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.pipeline.clv import CLVModelRegistry, HierarchicalCLVModel
from src.pipeline.clv.base import BaseModel

class TestCLVModelRegistry:
    """Test suite for CLVModelRegistry"""

    def test_registry_initialization(self, config_loader):
        """Test registry initialization"""
        registry = CLVModelRegistry(config_loader)
        assert registry.storage_config is not None
        assert registry.bucket_name is not None
        assert registry.model_prefix is not None

    def test_model_saving(self, config_loader, mock_gcs_bucket):
        """Test model saving functionality"""
        registry = CLVModelRegistry(config_loader)
        model = HierarchicalCLVModel(config_loader)
        
        # Verify model is BaseModel instance
        assert isinstance(model, BaseModel)
        
        metrics = {
            'rmse': 0.5,
            'mae': 0.3,
            'r2': 0.8
        }
        
        version = registry.save_model(model, metrics)
        assert version is not None

    def test_model_loading(self, config_loader, mock_gcs_bucket):
        """Test model loading functionality"""
        registry = CLVModelRegistry(config_loader)
        
        # Save a model first
        original_model = HierarchicalCLVModel(config_loader)
        metrics = {'rmse': 0.5}
        version = registry.save_model(original_model, metrics)
        
        # Load the model
        loaded_model, loaded_metrics = registry.load_model(version)
        
        # Verify loaded model is correct type
        assert isinstance(loaded_model, BaseModel)
        assert isinstance(loaded_model, HierarchicalCLVModel)
        assert loaded_metrics == metrics

    def test_invalid_model_type(self, config_loader, mock_gcs_bucket):
        """Test handling of invalid model types"""
        registry = CLVModelRegistry(config_loader)
        
        class InvalidModel:
            pass
            
        invalid_model = InvalidModel()
        metrics = {'rmse': 0.5}
        
        with pytest.raises(TypeError):
            registry.save_model(invalid_model, metrics)

    def test_version_management(self, config_loader, mock_gcs_bucket):
        """Test version management functionality"""
        registry = CLVModelRegistry(config_loader)
        model = HierarchicalCLVModel(config_loader)
        
        # Save multiple versions
        versions = []
        for i in range(3):
            metrics = {'rmse': 0.5 - i * 0.1}
            version = registry.save_model(model, metrics)
            versions.append(version)
            
        # Verify we can load each version
        for version in versions:
            loaded_model, _ = registry.load_model(version)
            assert isinstance(loaded_model, BaseModel)