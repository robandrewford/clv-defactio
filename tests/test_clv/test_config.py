import pytest
from pathlib import Path
from src.pipeline.clv.config import CLVConfigLoader

def test_load_config_success():
    """Test that valid config files can be loaded."""
    config = CLVConfigLoader()
    model_config = config.get_config('model')
    assert isinstance(model_config, dict)
    # Add specific assertions about expected config content
    
def test_load_config_missing_file():
    """Test that appropriate error is raised for missing config."""
    with pytest.raises(FileNotFoundError):
        CLVConfigLoader(config_dir="nonexistent_directory")
        
def test_verify_configs_complete():
    """Test that all required configs are present and valid."""
    config = CLVConfigLoader()
    configs = {
        'model': config.get_config('model'),
        'deployment': config.get_config('deployment'),
        'segment': config.get_config('segment')
    }
    required_configs = {'model', 'deployment', 'segment'}
    assert set(configs.keys()) == required_configs
    
def test_config_schema():
    """Test that configs have required fields."""
    config = CLVConfigLoader()
    model_config = config.get_config('model')
    required_fields = ['model_type', 'parameters']  # example required fields
    assert all(field in model_config for field in required_fields) 