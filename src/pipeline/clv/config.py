import yaml
from pathlib import Path
from typing import Dict, Any

class CLVConfigLoader:
    """Handles loading and validation of CLV pipeline configs"""
    
    def __init__(self, config_dir: str = "src/config"):
        self.config_dir = Path(config_dir)
        self.deployment_config = self._load_config("deployment_config.yaml")
        self.model_config = self._load_config("model_config.yaml")
        self.segment_config = self._load_config("segment_config.yaml")
        self.pipeline_config = self._load_config("pipeline_config.yaml")

    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a config file"""
        config_path = self.config_dir / filename
        with open(config_path) as f:
            return yaml.safe_load(f)

    def get_vertex_config(self) -> Dict[str, Any]:
        """Get Vertex AI specific configuration"""
        return self.deployment_config.get("vertex_ai", {})

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.deployment_config.get("storage", {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.deployment_config.get("monitoring", {}) 