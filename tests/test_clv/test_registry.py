import pytest
import os
import tempfile
from src.pipeline.clv.registry import CLVModelRegistry

@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_registry_initialization(config_loader):
    """Test registry initialization"""
    registry = CLVModelRegistry(config_loader)
    assert registry is not None
    assert registry.storage_config is not None

def test_model_saving(config_loader, temp_model_dir, sample_customer_features):
    """Test model saving functionality"""
    registry = CLVModelRegistry(config_loader)
    
    # Create dummy model and metrics
    dummy_model = {"weights": [1, 2, 3]}
    metrics = {
        "accuracy": 0.95,
        "r_squared": 0.85
    }
    
    # Save model
    version = registry.save_model(dummy_model, metrics)
    assert version is not None

def test_model_loading(config_loader, temp_model_dir):
    """Test model loading functionality"""
    registry = CLVModelRegistry(config_loader)
    
    # Save and then load model
    dummy_model = {"weights": [1, 2, 3]}
    metrics = {"accuracy": 0.95}
    
    version = registry.save_model(dummy_model, metrics)
    loaded_model, loaded_metrics = registry.load_model(version)
    
    assert loaded_model is not None
    assert loaded_metrics == metrics 