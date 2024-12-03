import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ConfigLoader1126:
    """Dynamic configuration loader that is picklable"""
    config_path: str = "config"
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Dynamically load all configurations upon initialization
        self.configs = self._load_all_configs()

    def _load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all YAML configuration files in the directory dynamically."""
        configs = {}
        for file in os.listdir(self.config_path):
            if file.endswith(".yaml"):
                config_name = file.replace(".yaml", "")
                configs[config_name] = self._load_config(file)
        return configs

    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a specific YAML file."""
        config_file = os.path.join(self.config_path, filename)
        if not os.path.exists(config_file):
            # Use logging to warn about missing files
            print(f"Warning: Configuration file {filename} not found.")
            return {}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get(self, config_name: str, default=None) -> Dict[str, Any]:
        """Retrieve a specific configuration by name."""
        return self.configs.get(config_name, default)

    def __getstate__(self):
        """Ensure the class is picklable."""
        # Return only the necessary state for pickling
        return {
            'config_path': self.config_path,
            'configs': self.configs
        }

    def __setstate__(self, state):
        """Restore the object's state from a pickled state."""
        self.config_path = state['config_path']
        self.configs = state['configs']

    def validate_config(self, config: dict, required_fields: list):
        """Validate that the config contains all required fields."""
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

    def test_config_schema_validation(self, mock_config_loader):
        """Test that configs match the required schema."""
        model_config = mock_config_loader.get_config('model')
        
        schema = {
            'model_type': str,
            'parameters': dict
        }
        
        # Test with a valid config
        self.validate_config(model_config, schema)
        
        # Test with a missing field
        model_config.pop('model_type', None)
        with pytest.raises(KeyError, match="Missing required field: model_type"):
            self.validate_config(model_config, schema)
        
        # Test with incorrect field type
        model_config['model_type'] = 123  # Invalid type
        with pytest.raises(TypeError, match="Field 'model_type' should be of type str, got int"):
            self.validate_config(model_config, schema)