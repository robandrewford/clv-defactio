from typing import Dict, Type

from .base import BayesianModel
from .clv_model import CLVModel


class ModelRegistry:
    """Registry for managing different model versions."""
    
    def __init__(self):
        self._models: Dict[str, Type[BayesianModel]] = {}
        
        # Register default models
        self.register_model("clv_hierarchical", CLVModel)
    
    def register_model(self, name: str, model_class: Type[BayesianModel]) -> None:
        """Register a new model class."""
        if not issubclass(model_class, BayesianModel):
            raise ValueError(f"{model_class.__name__} must inherit from BayesianModel")
        self._models[name] = model_class
    
    def get_model(self, name: str, config: Dict) -> BayesianModel:
        """Get a model instance by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name](config)
    
    @property
    def available_models(self) -> list:
        """List available model types."""
        return list(self._models.keys()) 