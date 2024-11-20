from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pymc as pm
import arviz as az

class BayesianModel(ABC):
    """Abstract base class for Bayesian models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.trace = None
        
    @abstractmethod
    def build_model(self, data: Dict[str, np.ndarray]) -> pm.Model:
        """Build the PyMC model."""
        pass
    
    @abstractmethod
    def sample(self, 
               draws: int = 2000, 
               tune: int = 1000,
               chains: int = 4,
               target_accept: float = 0.8,
               return_inferencedata: bool = True,
               **kwargs) -> Any:
        """Sample from the posterior distribution."""
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics (R-hat, ESS, etc.)."""
        if self.trace is None:
            raise ValueError("Model hasn't been sampled yet.")
        
        return {
            "r_hat": pm.stats.rhat(self.trace),
            "ess": pm.stats.ess(self.trace)
        }
    