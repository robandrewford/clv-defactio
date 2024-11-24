import yaml
from pathlib import Path

def load_config(config_name: str) -> dict:
    """
    Load a configuration file from the config directory.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        
    Returns:
        dict: Configuration data
    """
    config_path = Path(__file__).parent / f"{config_name}_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def verify_configs():
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