from pathlib import Path

import yaml


def load_config(config_name: str):
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    "get_model_config",
    "get_pipeline_config",
    "get_system_config",
]
