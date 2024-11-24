from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple

class BaseProcessor(ABC):
    """Base class for data processing components"""
    
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Process data and return DataFrame or DataFrame with metadata"""
        pass

class BaseModel(ABC):
    """Base class for model components"""
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build the model"""
        pass
        
    @abstractmethod
    def train_model(self) -> Any:
        """Train the model"""
        pass
        
    @abstractmethod
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the model"""
        pass

class BaseSegmentation(BaseProcessor):
    @abstractmethod
    def create_segments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create segments and return both DataFrame and metadata"""
        pass

    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process data and return segmented data with metadata"""
        return self.create_segments(df)