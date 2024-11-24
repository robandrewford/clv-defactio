import pytest
import pandas as pd
from src.pipeline.clv.base import BaseProcessor, BaseModel
from src.pipeline.clv.preprocessing import CLVDataPreprocessor
from src.pipeline.clv.model import HierarchicalCLVModel
from src.pipeline.clv.segmentation import CustomerSegmentation

def test_processor_hierarchy(config_loader):
    """Test processor class hierarchy"""
    preprocessor = CLVDataPreprocessor(config_loader)
    segmenter = CustomerSegmentation(config_loader)
    
    # Test inheritance
    assert isinstance(preprocessor, BaseProcessor)
    assert isinstance(segmenter, BaseProcessor)
    
    # Test abstract methods
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
    processors = [
        CLVDataPreprocessor(config_loader),
        CustomerSegmentation(config_loader)
    ]
    
    for processor in processors:
        # All processors should accept DataFrame input
        result = processor.process_data(sample_transaction_data)
        
        # Result should be DataFrame or tuple containing DataFrame
        if isinstance(result, tuple):
            assert isinstance(result[0], pd.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame) 