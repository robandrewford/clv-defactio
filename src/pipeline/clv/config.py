import yaml
from pathlib import Path
from typing import Dict, Any
import os

class CLVConfigLoader:
    """Handles loading and validation of CLV pipeline configs"""
    
    def __init__(self, config_dir: str = "src/config"):
        self.config_dir = Path(config_dir)
        self._configs = {}  # Store configs in instance variable
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configuration files"""
        config_files = {
            "deployment": "deployment_config.yaml",
            "model": "model_config.yaml",
            "segment": "segment_config.yaml",
            "pipeline": "pipeline_config.yaml"
        }
        
        for config_type, filename in config_files.items():
            config_path = self.config_dir / filename
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path) as f:
                self._configs[config_type] = yaml.safe_load(f)

    def get_config(self, config_type: str) -> Dict[str, Any]:
        """Get configuration by type"""
        return self._configs.get(config_type, {})

    def __getstate__(self):
        """Support for pickling"""
        return {
            'config_dir': self.config_dir,
            '_configs': self._configs
        }

    def __setstate__(self, state):
        """Support for unpickling"""
        self.config_dir = state['config_dir']
        self._configs = state['_configs']

    def get_vertex_config(self) -> Dict[str, Any]:
        """Get Vertex AI specific configuration"""
        return self.get_config("deployment").get("vertex_ai", {})

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.get_config("deployment").get("storage", {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get_config("deployment").get("monitoring", {}) 