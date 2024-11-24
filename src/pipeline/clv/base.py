from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class BaseProcessor(ABC):
    """Base class for data processing components"""
    
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process input data"""
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