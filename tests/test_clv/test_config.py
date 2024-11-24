import pytest
from src.pipeline.clv.config import CLVConfigLoader

def test_config_loading():
    """Test configuration loading"""
    config_loader = CLVConfigLoader()
    
    assert config_loader.model_config is not None
    assert config_loader.segment_config is not None
    assert config_loader.pipeline_config is not None

def test_vertex_config(config_loader):
    """Test Vertex AI configuration"""
    vertex_config = config_loader.get_vertex_config()
    
    assert 'project_id' in vertex_config
    assert 'location' in vertex_config
    assert 'pipeline_root' in vertex_config

def test_storage_config(config_loader):
    """Test storage configuration"""
    storage_config = config_loader.get_storage_config()
    
    assert 'gcs' in storage_config
    assert 'bucket_name' in storage_config['gcs']
    assert 'bigquery' in storage_config

def test_monitoring_config(config_loader):
    """Test monitoring configuration"""
    monitoring_config = config_loader.get_monitoring_config()
    
    assert 'enable_monitoring' in monitoring_config
    assert 'metrics' in monitoring_config

"""Test configuration constants"""

TEST_CONFIG = {
    'model': {
        'mcmc_samples': 500,
        'mcmc_tune': 200,
        'chains': 2,
        'target_accept': 0.8
    },
    'data': {
        'min_transactions': 5,
        'max_customers': 1000
    }
} 