import yaml
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ConfigLoader:
    """Picklable configuration loader"""
    def __init__(self, config_path="config"):
        self.config_path = config_path
        self.model_config = self._load_config("model_config.yaml")
        self.pipeline_config = self._load_config("pipeline_config.yaml")
        self.bucket_name = self.model_config.get('storage', {}).get('bucket_name', 'default-bucket')
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = os.path.join(self.config_path, filename)
        if not os.path.exists(config_file):
            return {}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def __getstate__(self):
        """Make class picklable"""
        return {
            'config_path': self.config_path,
            'model_config': self.model_config,
            'pipeline_config': self.pipeline_config,
            'bucket_name': self.bucket_name
        }
    
    def __setstate__(self, state):
        """Restore pickled state"""
        self.__dict__.update(state)

def load_config(config_type: str) -> Dict[str, Any]:
    """
    Load configuration file by type
    
    Args:
        config_type (str): Type of configuration to load (model, data_processing, deployment)
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_path = os.path.join('config', f'{config_type}_config.yaml')
    if not os.path.exists(config_path):
        return {}
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def verify_configs() -> Dict[str, Dict[str, Any]]:
    """Load and verify all configuration files."""
    configs = {}
    
    try:
        configs['model'] = load_config('model')
        configs['data_processing'] = load_config('data_processing')
        configs['deployment'] = load_config('deployment')
        
        print("✅ All configuration files loaded successfully!")
        return configs
        
    except Exception as e:
        print(f"❌ Error loading configurations: {str(e)}")
        raise