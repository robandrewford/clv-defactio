from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class CLVProcessor(ABC):
    """Abstract base class for CLV processing"""
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def build_model(self) -> Any:
        pass
        
    @abstractmethod
    def train_model(self) -> Any:
        pass
        
    @abstractmethod
    def evaluate_model(self) -> Dict[str, Any]:
        pass 