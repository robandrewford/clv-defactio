import pytest
from pathlib import Path
from src.pipeline.clv.config import CLVConfigLoader

def test_load_config_success(mock_config_loader):
    """Test that valid config files can be loaded."""
    model_config = mock_config_loader.get_config('model')
    assert isinstance(model_config, dict)
    assert model_config['model_type'] == 'hierarchical_clv'
    
def test_load_config_missing_file():
    """Test that appropriate error is raised for missing config."""
    with pytest.raises(FileNotFoundError):
        CLVConfigLoader(config_dir="nonexistent_directory")
        
def test_verify_configs_complete(mock_config_loader):
    """Test that all required configs are present and valid."""
    configs = {
        'model': mock_config_loader.get_config('model'),
        'deployment': mock_config_loader.get_config('deployment'),
        'segment': mock_config_loader.get_config('segment')
    }
    required_configs = {'model', 'deployment', 'segment'}
    assert set(configs.keys()) == required_configs
    
def test_config_schema(mock_config_loader):
    """Test that configs have required fields."""
    model_config = mock_config_loader.get_config('model')
    required_fields = ['model_type', 'parameters']
    assert all(field in model_config for field in required_fields) 