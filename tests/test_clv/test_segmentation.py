import pytest
import numpy as np
import pandas as pd
from src.pipeline.clv import CustomerSegmentation
from src.pipeline.clv.base import BaseProcessor
from typing import Tuple, Dict, Any

class TestCustomerSegmentation:
    """Test suite for CustomerSegmentation"""

    def test_segmentation_initialization(self, config_loader):
        """Test segmentation initialization"""
        segmenter = CustomerSegmentation(config_loader)
        assert isinstance(segmenter, BaseProcessor)
        assert segmenter.segment_config is not None

    def test_process_data_interface(self, config_loader):
        """Test process_data method implementation"""
        # Create segmenter with config
        segmenter = CustomerSegmentation(config_loader)
        
        # Create test data
        test_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'purchase_amount': [100, 200, 300],
            'frequency': [5, 3, 7],
            'recency': [10, 20, 30],
            'monetary': [150, 250, 350],
            'transaction_date': pd.date_range('2023-01-01', periods=3),
            'sms_active': [1, 0, 1],
            'email_active': [1, 1, 0]
        })
        
        # Process the data
        result = segmenter.process_data(test_data)
        
        # Verify return type and structure
        assert isinstance(result, tuple), "process_data should return a tuple"
        assert len(result) == 2, "process_data should return a tuple of (DataFrame, Dict)"
        
        processed_data, metadata = result
        assert isinstance(processed_data, pd.DataFrame), "First element should be a DataFrame"
        assert isinstance(metadata, dict), "Second element should be a dictionary"
        
        # Verify required columns are present
        required_columns = [
            'customer_id',
            'frequency',
            'recency',
            'monetary'
        ]
        for col in required_columns:
            assert col in processed_data.columns, f"Required column {col} missing from processed data"
        
        # Verify metadata structure
        assert 'n_customers' in metadata, "Metadata should contain n_customers"
        assert 'processing_timestamp' in metadata, "Metadata should contain processing_timestamp"

    def test_rfm_segmentation(self, config_loader, sample_customer_features):
        """Test RFM segmentation"""
        segmenter = CustomerSegmentation(config_loader)
        df, model_data = segmenter.create_segments(sample_customer_features)
        
        # Check RFM scores exist
        assert 'R_score' in df.columns
        assert 'F_score' in df.columns
        assert 'M_score' in df.columns
        assert 'RFM_score' in df.columns
        
        # Check model data structure
        assert 'segment_ids' in model_data
        assert 'customer_features' in model_data

    def test_engagement_segmentation(self, config_loader, sample_customer_features):
        """Test engagement segmentation"""
        # Add engagement metrics
        data = sample_customer_features.copy()
        data['sms_active'] = np.random.randint(0, 2, len(data))
        data['email_active'] = np.random.randint(0, 2, len(data))
        
        segmenter = CustomerSegmentation(config_loader)
        df, _ = segmenter.create_segments(data)
        
        assert 'engagement_score' in df.columns
        assert 'engagement_level' in df.columns

    def test_segment_decoding(self, config_loader, sample_customer_features):
        """Test segment decoding functionality"""
        segmenter = CustomerSegmentation(config_loader)
        _, model_data = segmenter.create_segments(sample_customer_features)
        
        decoded = segmenter.decode_segments(model_data['segment_ids'])
        assert not decoded.empty

    def test_processor_interface_compliance(self, config_loader):
        """Test compliance with BaseProcessor interface"""
        segmenter = CustomerSegmentation(config_loader)
        
        # Check required methods
        assert hasattr(segmenter, 'process_data')
        
        # Check method signature
        from inspect import signature
        sig = signature(segmenter.process_data)
        
        # Check parameter exists and has correct type annotation
        assert 'data' in sig.parameters, "process_data should have 'data' parameter"
        assert sig.parameters['data'].annotation == pd.DataFrame, "data parameter should be annotated as pd.DataFrame"
        
        # Check return type annotation
        expected_return = Tuple[pd.DataFrame, Dict[str, Any]]
        assert sig.return_annotation == expected_return, "Return type should be Tuple[pd.DataFrame, Dict[str, Any]]"