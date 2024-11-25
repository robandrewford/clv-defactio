import pytest
from src.pipeline.clv.base import BaseProcessor, BaseModel

class TestBaseProcessor(BaseProcessor):
    """Test implementation of BaseProcessor"""
    def process_data(self, data):
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

def test_base_processor_abstract_methods():
    """Test BaseProcessor abstract method enforcement"""
    class InvalidProcessor(BaseProcessor):
        pass
        
    config = {"test": "config"}
    with pytest.raises(TypeError):
        InvalidProcessor(config)

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
