import pytest
import pandas as pd
from src.pipeline.clv.base import BaseProcessor, BaseModel
from src.pipeline.clv.preprocessing import CLVDataPreprocessor
from src.pipeline.clv.model import HierarchicalCLVModel
from src.pipeline.clv.segmentation import CustomerSegmentation

def test_processor_inheritance(config_loader):
    """Test processor class inheritance relationships"""
    preprocessor = CLVDataPreprocessor(config_loader)
    segmenter = CustomerSegmentation(config_loader)
    
    # Test inheritance
    assert isinstance(preprocessor, BaseProcessor)
    assert isinstance(segmenter, BaseProcessor)
    
    # Test abstract methods implementation
    assert hasattr(preprocessor, 'process_data')
    assert hasattr(segmenter, 'process_data')

def test_model_hierarchy(config_loader):
    """Test model class hierarchy"""
    model = HierarchicalCLVModel(config_loader)
    
    # Test inheritance
    assert isinstance(model, BaseModel)
    
    # Test abstract methods
    assert hasattr(model, 'build_model')
    assert hasattr(model, 'train_model')
    assert hasattr(model, 'evaluate_model')

def test_processor_interface(config_loader, sample_transaction_data):
    """Test processor interface consistency"""
    # Add required features to sample data
    sample_data = sample_transaction_data.copy()
    sample_data['recency'] = 1
    sample_data['frequency'] = 1
    
    processors = [
        CLVDataPreprocessor(config_loader),
        CustomerSegmentation(config_loader)
    ]
    
    for processor in processors:
        # All processors should accept DataFrame input
        result = processor.process_data(sample_data)
        
        # Result should be DataFrame or tuple containing DataFrame
        if isinstance(result, tuple):
            assert isinstance(result[0], pd.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)

def test_processor_config_validation(config_loader):
    """Test processor configuration validation"""
    # Test with valid config
    segmenter = CustomerSegmentation(config_loader)
    assert hasattr(segmenter, 'config')
    assert segmenter.config is not None
    
    # Test basic config access
    assert hasattr(segmenter, 'get_config')
    
    # Test with invalid config
    class InvalidConfig:
        pass
    
    with pytest.raises(ValueError):
        BaseProcessor(InvalidConfig())

@pytest.mark.parametrize('config_fixture', [None])
def test_processor_abstract_methods(config_loader):
    """Test that abstract methods raise NotImplementedError when not implemented"""
    class InvalidProcessor(BaseProcessor):
        def __init__(self, config):
            super().__init__(config)

    with pytest.raises(NotImplementedError):
        processor = InvalidProcessor(config_loader)
        processor.process_data(pd.DataFrame())
