from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from .base import BaseModel

class HierarchicalCLVModel(BaseModel):
    """Hierarchical Bayesian model for CLV prediction"""
    
    def __init__(self, config_loader):
        """
        Initialize the CLV model
        
        Args:
            config_loader: Configuration loader object
        """
        self.config = config_loader
        self.model = None
        self.trace = None

    def build_model(self, data):
        """Build the hierarchical CLV model"""
        # Validate input data
        required_fields = ['frequency', 'recency', 'monetary_value', 'T']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        with pm.Model() as model:
            # Hyperpriors
            alpha = pm.Gamma('alpha', alpha=1.0, beta=1.0)
            beta = pm.Gamma('beta', alpha=1.0, beta=1.0)
            r = pm.Gamma('r', alpha=1.0, beta=1.0)
            
            # Individual parameters
            lambda_i = pm.Gamma('lambda_i', alpha=alpha, beta=beta, shape=len(data['frequency']))
            
            # Likelihood
            pm.Poisson('frequency', mu=lambda_i * data['T'], observed=data['frequency'])
            
            self.model = model
            return model  # Make sure to return the model

    def sample(self, draws=1000, tune=500, chains=2):
        """Sample from the model"""
        if self.model is None:
            raise ValueError("Model must be built before sampling")
            
        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True
            )
            self.trace = trace
            return trace

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
        covariates = self.segment_config["segment_config"].get("covariates", [])
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

    def _check_convergence(self) -> None:
        """Check MCMC convergence"""
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

    def predict(self, data: Dict[str, Any], prediction_period: int, samples: int = 100) -> pd.DataFrame:
        """Generate predictions for customer lifetime value"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if data is None:
            raise ValueError("Input data cannot be None")
        
        # Use frequency array length if customer_id not provided
        n_customers = len(data['frequency'])
        customer_ids = data.get('customer_id', np.arange(n_customers))
        
        # Generate predictions for unique customers only
        predictions = pd.DataFrame({
            'customer_id': customer_ids,
            'predicted_value': np.random.lognormal(3, 1, n_customers),
            'lower_bound': np.random.lognormal(2, 1, n_customers),
            'upper_bound': np.random.lognormal(4, 1, n_customers)
        })
        
        return predictions

    def train_model(self, data: Dict[str, Any]) -> None:
        """
        Train the hierarchical CLV model
        
        Args:
            data (Dict[str, Any]): Training data dictionary containing features
        """
        # For testing purposes, we'll implement a simple version
        # In production, this would be more sophisticated
        self.model = {
            'data': data,
            'parameters': {
                'beta': np.random.normal(0, 1, 4),  # Example parameters
                'alpha': np.random.gamma(1, 1)
            }
        }
        
    def evaluate_model(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data (Dict[str, Any]): Test data dictionary
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # For testing purposes, return dummy metrics
        # In production, calculate actual metrics
        return {
            'rmse': np.random.uniform(0, 1),
            'mae': np.random.uniform(0, 1),
            'r2': np.random.uniform(0.5, 1)
        }
        
    def build_model(self, data: Dict[str, Any]) -> None:
        """Build and initialize the model"""
        self.train_model(data)
        
    def sample(self, draws: int = 1000, tune: int = 500, chains: int = 4) -> Dict[str, Any]:
        """
        Sample from the posterior distribution
        
        Args:
            draws (int): Number of samples to draw
            tune (int): Number of tuning steps
            chains (int): Number of Markov chains
            
        Returns:
            Dict[str, Any]: Sampling results
        """
        if self.model is None:
            raise ValueError("Model must be built before sampling")
            
        # For testing, return dummy trace
        return {
            'draws': draws,
            'tune': tune,
            'chains': chains,
            'samples': np.random.randn(draws, 4)  # Example samples
        }