from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple

class BaseProcessor:
    def __init__(self, config):
        """Initialize base processor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate configuration"""
        if not hasattr(self, 'config'):
            raise ValueError("Configuration not properly initialized")

    def process_data(self, data):
        """Process data method to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement process_data method")

    def get_config(self, key, default=None):
        """Safely get configuration value
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key not found
        """
        return self.config.get(key, default)

class BaseModel:
    """Base class for model components"""
    
    def __init__(self, config):
        """Initialize base model
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.hyper_priors = {}
        self.group_params = {}
        self.coef_priors = {}
        self.convergence_history = []
        
    @abstractmethod
    def build_model(self, data: Dict[str, Any]) -> Any:
        """Build the model"""
        pass
        
    @abstractmethod
    def train_model(self, data: Dict[str, Any]) -> None:
        """Train the model"""
        pass
        
    @abstractmethod
    def evaluate_model(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model"""
        pass

class BaseSegmentation(BaseProcessor):
    def __init__(self, config):
        """Initialize base segmentation
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
    @abstractmethod
    def create_segments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create segments and return both DataFrame and metadata"""
        pass

    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process data and return segmented data with metadata"""
        return self.create_segments(df)