import pytest
from src.config.config_loader import load_config, verify_configs
from pathlib import Path

def test_load_config_success():
    """Test that valid config files can be loaded."""
    model_config = load_config('model')
    assert isinstance(model_config, dict)
    # Add specific assertions about expected config content
    
def test_load_config_missing_file():
    """Test that appropriate error is raised for missing config."""
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent')
        
def test_verify_configs_complete():
    """Test that all required configs are present and valid."""
    configs = verify_configs()
    required_configs = {'model', 'data_processing', 'deployment'}
    assert set(configs.keys()) == required_configs
    
def test_config_schema():
    """Test that configs have required fields."""
    model_config = load_config('model')
    required_fields = ['model_type', 'parameters']  # example required fields
    assert all(field in model_config for field in required_fields) 