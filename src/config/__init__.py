from pathlib import Path
import yaml
import os

def load_config(config_name: str):
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_data_processing_config():
    """Load and return the data processing configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'data_processing_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at '{config_path}'")

def get_deployment_config():
    """Get deployment configuration"""
    return load_config("deployment_config")

def get_development_config():
    """Get development configuration"""
    return load_config("development_config")

def get_model_config():
    """Get model configuration"""
    return load_config("model_config")

def get_pipeline_config():
    """Get pipeline configuration"""
    return load_config("pipeline_config")

def get_system_config():
    """Get system configuration"""
    return load_config("system_config")

__all__ = [
    "load_config",
    "get_data_processing_config",
    "get_deployment_config",
    "get_development_config",
    "get_model_config",
    "get_pipeline_config",
    "get_system_config",
]
