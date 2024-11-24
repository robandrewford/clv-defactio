import pytest
import numpy as np
from src.pipeline.clv.segmentation import CustomerSegmentation

def test_segmentation_initialization(config_loader):
    """Test segmentation initialization"""
    segmenter = CustomerSegmentation(config_loader)
    assert segmenter is not None
    assert segmenter.segment_config is not None

def test_rfm_segmentation(config_loader, sample_customer_features):
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

def test_engagement_segmentation(config_loader, sample_customer_features):
    """Test engagement segmentation"""
    # Add engagement metrics
    data = sample_customer_features.copy()
    data['sms_active'] = np.random.randint(0, 2, len(data))
    data['email_active'] = np.random.randint(0, 2, len(data))
    
    segmenter = CustomerSegmentation(config_loader)
    df, _ = segmenter.create_segments(data)
    
    assert 'engagement_score' in df.columns
    assert 'engagement_level' in df.columns

def test_segment_decoding(config_loader, sample_customer_features):
    """Test segment decoding functionality"""
    segmenter = CustomerSegmentation(config_loader)
    _, model_data = segmenter.create_segments(sample_customer_features)
    
    decoded = segmenter.decode_segments(model_data['segment_ids'])
    assert not decoded.empty 