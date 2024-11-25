from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """
    Base class for all data processing components in the CLV pipeline.
    """
    def __init__(self, config):
        """
        Initialize base processor.
        
        Args:
            config: Configuration object or dict containing processor settings
        """
        self._config = config
        self._validate_config()
        logger.debug(f"Initialized {self.__class__.__name__} with config")

    def _validate_config(self):
        """Validate the configuration"""
        if self._config is None:
            raise ValueError("Configuration cannot be None")

    @property
    def config(self):
        """Get configuration"""
        return self._config

    @abstractmethod
    def process_data(self, data):
        """
        Process input data. Must be implemented by child classes.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
            
        Raises:
            NotImplementedError: If child class doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement process_data method")

    def validate_input(self, data):
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if data is None:
            raise ValueError("Input data cannot be None")

class BaseModel(ABC):
    """
    Base class for all models in the CLV pipeline.
    """
    def __init__(self, config):
        """
        Initialize base model.
        
        Args:
            config: Configuration object or dict containing model settings
        """
        self._config = config
        self._model = None
        self._validate_config()
        logger.debug(f"Initialized {self.__class__.__name__} with config")

    def _validate_config(self):
        """Validate the configuration"""
        if self._config is None:
            raise ValueError("Configuration cannot be None")

    @property
    def config(self):
        """Get configuration"""
        return self._config

    @property
    def model(self):
        """Get the underlying model"""
        return self._model

    @model.setter
    def model(self, value):
        """Set the underlying model"""
        self._model = value

    @abstractmethod
    def build_model(self, data):
        """
        Build the model. Must be implemented by child classes.
        
        Args:
            data: Input data for model building
            
        Returns:
            Built model
            
        Raises:
            NotImplementedError: If child class doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement build_model method")

    @abstractmethod
    def predict(self, data):
        """
        Make predictions. Must be implemented by child classes.
        
        Args:
            data: Input data for predictions
            
        Returns:
            Model predictions
            
        Raises:
            NotImplementedError: If child class doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement predict method")

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