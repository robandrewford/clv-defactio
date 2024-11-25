# Standard library imports
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Third-party imports
import pytest
import pandas as pd
import numpy as np

# Local imports
from src.pipeline.clv.base import BaseProcessor, BaseModel

# Test utilities
class MockConfigLoader:
    """Mock config loader for testing"""
    def __init__(self):
        self.config = {}
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class TestBaseProcessor(BaseProcessor):
    """Test implementation of BaseProcessor"""
    def process_data(self, data):
        if not isinstance(self.config, (dict, MockConfigLoader)):
            raise ValueError("Invalid config type")
        
        required_features = ['recency', 'frequency', 'monetary']
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return data

class TestBaseModel(BaseModel):
    """Test implementation of BaseModel"""
    def build_model(self, data):
        self._model = {"data": data}
        return self._model
        
    def predict(self, data):
        return data

def test_base_processor_initialization():
    """Test BaseProcessor initialization"""
    config = {"test": "config"}
    processor = TestBaseProcessor(config)
    assert processor.config == config
    
    with pytest.raises(ValueError):
        TestBaseProcessor(None)

def test_processor_interface(config_loader):
    """Test processor interface consistency"""
    processor = TestBaseProcessor(config_loader)
    
    # Test with missing features
    invalid_data = pd.DataFrame({
        'customer_id': [1, 2],
        'recency': [30, 45],
        'frequency': [2, 2]
        # monetary is missing
    })
    
    with pytest.raises(ValueError, match="Missing required features: \\['monetary'\\]"):
        processor.process_data(invalid_data)
    
    # Test with all required features
    valid_data = pd.DataFrame({
        'customer_id': [1, 2],
        'recency': [30, 45],
        'frequency': [2, 2],
        'monetary': [150, 225]
    })
    
    result = processor.process_data(valid_data)
    assert isinstance(result, pd.DataFrame)

def test_processor_config_validation():
    """Test processor configuration validation"""
    class InvalidConfig:
        def get(self, key):
            return None
    
    with pytest.raises(ValueError, match="Invalid config type"):
        TestBaseProcessor(InvalidConfig())
    
    with pytest.raises(ValueError, match="Configuration cannot be None"):
        TestBaseProcessor(None)

def test_base_model_initialization():
    """Test BaseModel initialization"""
    config = {"test": "config"}
    model = TestBaseModel(config)
    assert model.config == config
    
    with pytest.raises(ValueError):
        TestBaseModel(None)

def test_base_model_abstract_methods():
    """Test BaseModel abstract method enforcement"""
    class InvalidModel(BaseModel):
        pass
        
    config = {"test": "config"}
    with pytest.raises(TypeError):
        InvalidModel(config)

def test_base_model_properties():
    """Test BaseModel property access"""
    config = {"test": "config"}
    model = TestBaseModel(config)
    assert model.model is None
    
    test_data = {"input": "data"}
    built_model = model.build_model(test_data)
    assert model.model == built_model
