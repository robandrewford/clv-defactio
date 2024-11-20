from typing import Any, Dict, Optional

import numpy as np
import pymc as pm

from .base import BayesianModel


class CLVModel(BayesianModel):
    """Hierarchical Bayesian model for Customer Lifetime Value prediction."""
    
    def build_model(self, data: Dict[str, np.ndarray]) -> pm.Model:
        """
        Build hierarchical CLV model.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary containing:
            - customer_value: Historical customer value
            - customer_features: Customer-level features
            - segment_ids: Customer segment identifiers
        """
        with pm.Model() as model:
            # Hyperpriors for customer segments
            segment_mu = pm.Normal("segment_mu", mu=0, sigma=10)
            segment_sigma = pm.HalfNormal("segment_sigma", sigma=5)
            
            # Segment-level parameters
            n_segments = len(np.unique(data["segment_ids"]))
            segment_effects = pm.Normal("segment_effects",
                                      mu=segment_mu,
                                      sigma=segment_sigma,
                                      shape=n_segments)
            
            # Customer-level parameters
            beta = pm.Normal("beta", mu=0, sigma=2, 
                           shape=data["customer_features"].shape[1])
            
            # Expected value
            mu = (segment_effects[data["segment_ids"]] + 
                 pm.dot(data["customer_features"], beta))
            
            # Likelihood
            sigma = pm.HalfNormal("sigma", sigma=5)
            y = pm.Normal("y",
                         mu=mu,
                         sigma=sigma,
                         observed=data["customer_value"])
            
            self.model = model
            
        return model
    
    def sample(self,
               draws: int = 2000,
               tune: int = 1000,
               chains: int = 4,
               target_accept: float = 0.8,
               return_inferencedata: bool = True,
               **kwargs) -> Any:
        """
        Sample from the posterior distribution.
        
        Returns
        -------
        InferenceData or MultiTrace
            Sampling results
        """
        if self.model is None:
            raise ValueError("Model hasn't been built yet.")
            
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=return_inferencedata,
                **kwargs
            )
            
        return self.trace
    
    def predict(self, 
                new_data: Dict[str, np.ndarray],
                samples: int = 1000) -> np.ndarray:
        """
        Generate predictions for new customers.
        
        Parameters
        ----------
        new_data : Dict[str, np.ndarray]
            New customer data
        samples : int
            Number of posterior samples to use
            
        Returns
        -------
        np.ndarray
            Predicted customer lifetime values
        """
        if self.trace is None:
            raise ValueError("Model hasn't been sampled yet.")
            
        # Extract posterior samples
        segment_effects = self.trace.posterior["segment_effects"].mean(dim=["chain", "draw"]).values
        beta = self.trace.posterior["beta"].mean(dim=["chain", "draw"]).values
        
        # Generate predictions
        predictions = (segment_effects[new_data["segment_ids"]] + 
                      np.dot(new_data["customer_features"], beta))
        
        return predictions 