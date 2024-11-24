from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from .base import BayesianModel
from src.utils.gpu import GPUManager
from src.utils.monitoring import ResourceMonitor


class ModelError(Exception):
    """Base exception for model-related errors"""
    pass


class HierarchicalCLVModel(BayesianModel):
    """
    Enhanced Hierarchical Bayesian model for Customer Lifetime Value prediction.
    Combines BG/NBD purchase model with monetary value prediction.
    """

    def __init__(self, config: Dict[str, Any], segment_config: Optional[Dict] = None):
        super().__init__(config)
        self.segment_config = segment_config or config.get("segment_config", {})
        self.gpu_manager = GPUManager(config)
        self.resource_monitor = ResourceMonitor(config)
        self.initialize_components()

    def initialize_components(self):
        """Initialize model components"""
        self.hyper_priors = {}
        self.group_params = {}
        self.coef_priors = {}
        self.convergence_history = []

    def build_model(self, data: Dict[str, np.ndarray]) -> pm.Model:
        """
        Build hierarchical CLV model combining purchase frequency and monetary value.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary containing:
            - frequency: Purchase frequency
            - recency: Time since last purchase
            - T: Customer age
            - monetary_value: Average transaction value
            - customer_features: Customer-level features
            - segment_ids: Customer segment identifiers
        """
        try:
            with pm.Model() as model:
                # Add hierarchical priors
                self._add_hierarchical_priors(data)
                
                # Add covariate effects if specified
                if self.segment_config.get("use_covariates", False):
                    self._add_covariate_effects(data)
                
                # Add likelihood components
                self._add_likelihood(data)
                
                self.model = model
                return model
                
        except Exception as e:
            raise ModelError(f"Failed to build model: {str(e)}")

    def _add_hierarchical_priors(self, data: Dict[str, np.ndarray]) -> None:
        """Add hierarchical priors to the model"""
        n_segments = len(np.unique(data["segment_ids"]))

        # Global hyperpriors
        self.hyper_priors.update({
            'r_mu': pm.Gamma('r_mu', alpha=1.0, beta=1.0),
            'r_sigma': pm.HalfNormal('r_sigma', sigma=1.0),
            'alpha_mu': pm.Gamma('alpha_mu', alpha=1.0, beta=1.0),
            'alpha_sigma': pm.HalfNormal('alpha_sigma', sigma=1.0),
            'beta_mu': pm.Gamma('beta_mu', alpha=1.0, beta=1.0),
            'beta_sigma': pm.HalfNormal('beta_sigma', sigma=1.0)
        })

        # Segment-level parameters
        shape = n_segments if n_segments > 1 else None
        self.group_params.update({
            'r': pm.Gamma('r', 
                         alpha=self.hyper_priors['r_mu'],
                         beta=self.hyper_priors['r_sigma'],
                         shape=shape),
            'alpha': pm.Gamma('alpha',
                            alpha=self.hyper_priors['alpha_mu'],
                            beta=self.hyper_priors['alpha_sigma'],
                            shape=shape),
            'beta': pm.Gamma('beta',
                           alpha=self.hyper_priors['beta_mu'],
                           beta=self.hyper_priors['beta_sigma'],
                           shape=shape)
        })

    def _add_covariate_effects(self, data: Dict[str, np.ndarray]) -> None:
        """Add covariate effects to the model"""
        covariates = self.segment_config.get("covariates", [])
        if not covariates:
            return

        X = data["customer_features"]
        n_covariates = X.shape[1]

        # Covariate coefficients
        self.coef_priors['gamma'] = pm.Normal(
            'gamma',
            mu=0,
            sigma=1,
            shape=(n_covariates, 3)  # One for each parameter (r, alpha, beta)
        )

        # Calculate and apply covariate effects
        covariate_effects = pm.math.dot(X, self.coef_priors['gamma'])
        
        for idx, param in enumerate(['r', 'alpha', 'beta']):
            self.group_params[param] = pm.math.exp(
                pm.math.log(self.group_params[param]) + covariate_effects[:, idx]
            )

    def _add_likelihood(self, data: Dict[str, np.ndarray]) -> None:
        """Add likelihood components to the model"""
        # Purchase frequency likelihood
        pm.NegativeBinomial(
            'frequency_obs',
            mu=self.group_params['r'] * self.group_params['beta'] * data['T'],
            alpha=self.group_params['r'],
            observed=data['frequency']
        )

        # Monetary value likelihood
        pm.Gamma(
            'monetary_obs',
            alpha=self.group_params['alpha'],
            beta=self.group_params['beta'],
            observed=data['monetary_value']
        )

    def sample(self,
               draws: int = 2000,
               tune: int = 1000,
               chains: int = 4,
               target_accept: float = 0.8,
               **kwargs: Any) -> az.InferenceData:
        """
        Sample from the posterior distribution with GPU optimization if available.
        """
        if self.model is None:
            raise ModelError("Model hasn't been built yet.")

        if self.gpu_manager.gpu_enabled:
            self.gpu_manager.optimize_memory()

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                **kwargs
            )

        self._check_convergence()
        return self.trace

    def _check_convergence(self) -> None:
        """Check MCMC convergence using Gelman-Rubin statistic"""
        try:
            gr_stats = az.rhat(self.trace)
            
            self.convergence_history.append({
                'timestamp': pd.Timestamp.now(),
                'gelman_rubin': gr_stats,
                'n_samples': self.trace.posterior.sizes['draw'],
                'n_chains': self.trace.posterior.sizes['chain']
            })

            # Check for non-convergence
            non_converged = {k: v for k, v in gr_stats.items() if v > 1.1}
            if non_converged:
                print("Warning: The following parameters haven't converged:")
                for param, value in non_converged.items():
                    print(f"{param}: {value:.3f}")
                    
        except Exception as e:
            print(f"Error checking convergence: {str(e)}")

    def predict(self, 
                new_data: Dict[str, np.ndarray],
                prediction_period: int,
                samples: int = 1000) -> pd.DataFrame:
        """
        Generate predictions for new customers.
        
        Parameters
        ----------
        new_data : Dict[str, np.ndarray]
            New customer data
        prediction_period : int
            Future time period for predictions
        samples : int
            Number of posterior samples to use
            
        Returns
        -------
        pd.DataFrame
            Predicted purchases and values with uncertainty intervals
        """
        if self.trace is None:
            raise ModelError("Model hasn't been sampled yet.")

        # Extract posterior samples
        trace_data = az.extract(self.trace)
        
        # Generate predictions for each customer
        predictions = []
        for i in range(len(new_data['customer_id'])):
            # Calculate expected purchases
            r = trace_data['r'].values
            beta = trace_data['beta'].values
            alpha = trace_data['alpha'].values
            
            expected_purchases = r * beta * prediction_period
            expected_value = (alpha / beta) * expected_purchases
            
            predictions.append({
                'customer_id': new_data['customer_id'][i],
                'predicted_purchases': expected_purchases.mean(),
                'predicted_value': expected_value.mean(),
                'purchases_lower': np.percentile(expected_purchases, 2.5),
                'purchases_upper': np.percentile(expected_purchases, 97.5),
                'value_lower': np.percentile(expected_value, 2.5),
                'value_upper': np.percentile(expected_value, 97.5)
            })
            
        return pd.DataFrame(predictions)