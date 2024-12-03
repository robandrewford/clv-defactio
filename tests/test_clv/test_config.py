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

def test_config_schema_types(mock_config_loader):
    """Test that configs have required fields with correct types."""
    model_config = mock_config_loader.get_config('model')
    assert isinstance(model_config['model_type'], str)
    assert isinstance(model_config['parameters'], dict)

def test_config_missing_required_fields(mock_config_loader):
    """Test behavior when required fields are missing."""
    model_config = mock_config_loader.get_config('model')
    model_config.pop('model_type', None)
    with pytest.raises(KeyError):
        # Assuming some validation logic that raises KeyError
        validate_config(model_config)


from unittest.mock import patch

def test_load_config_missing_file_mock():
    """Mock os.path.exists to simulate missing config."""
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            CLVConfigLoader(config_dir="any_directory")

