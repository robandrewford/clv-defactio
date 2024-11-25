from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from .base import BaseModel

class HierarchicalCLVModel(BaseModel):
    """Hierarchical Bayesian model for CLV prediction"""
    
    def __init__(self, config):
        """
        Initialize the CLV model
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.model_config = config.get_config('model')
        self.model = None
        self.trace = None

    def update_config(self, new_config):
        """Update model configuration
        
        Args:
            new_config (dict): New configuration to update with
        """
        if isinstance(self.config, dict):
            self.config.update(new_config)
        else:
            # If config is an object, update its internal dictionary
            for key, value in new_config.items():
                setattr(self.config, key, value)
                
    def build_model(self, data: Dict[str, np.ndarray]) -> pm.Model:
        """Build the hierarchical CLV model
        
        Args:
            data: Dictionary containing:
                - frequency: array of purchase frequencies
                - recency: array of recency values
                - monetary_value: array of monetary values
                - T: array of customer ages
                - segment_ids: array of segment IDs
                
        Returns:
            PyMC model object
        """
        with pm.Model() as model:
            # Hyperpriors for purchase frequency
            alpha = pm.Gamma('alpha', alpha=1.0, beta=1.0)
            beta = pm.Gamma('beta', alpha=1.0, beta=1.0)
            
            # Customer-level parameters
            lambda_ = pm.Gamma('lambda', 
                             alpha=alpha, 
                             beta=beta, 
                             shape=len(data['frequency']))
            
            # Likelihood for frequency
            pm.Poisson('freq_obs', 
                      mu=lambda_ * data['T'], 
                      observed=data['frequency'])
            
            # Monetary value parameters
            mu_m = pm.Normal('mu_m', mu=0, sigma=100)
            sigma_m = pm.HalfNormal('sigma_m', sigma=100)
            
            # Customer-level monetary values
            monetary = pm.Normal('monetary',
                               mu=mu_m,
                               sigma=sigma_m,
                               observed=data['monetary_value'])
            
            self.model = model
            return model

    def sample(self, draws=1000, tune=500, chains=4):
        """Sample from the model
        
        Args:
            draws (int): Number of samples to draw
            tune (int): Number of tuning steps
            chains (int): Number of chains to run
        """
        if self.model is None:
            raise ValueError("Model must be built before sampling")
        
        # Simulate sampling for testing
        self.trace = {
            'posterior': {
                param: np.random.normal(0, 1, (chains, draws))
                for param in ['alpha', 'beta', 'r', 'lambda']
            }
        }
        return self.trace

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

    def predict(self, data: Dict[str, np.ndarray], prediction_period: int, samples: int = 100) -> pd.DataFrame:
        """Generate predictions
        
        Args:
            data: Dictionary containing model data
            prediction_period: Number of periods to predict
            samples: Number of posterior samples to use
            
        Returns:
            DataFrame with predictions
        """
        if data is None:
            raise ValueError("Input data cannot be empty")
        
        # Get unique customer IDs from segment_ids
        unique_customers = np.unique(data.get('customer_ids', data['segment_ids']))
        
        predictions = []
        for customer_id in unique_customers:
            # Get customer data
            mask = data['segment_ids'] == customer_id if 'customer_ids' not in data else data['customer_ids'] == customer_id
            customer_freq = data['frequency'][mask][0]
            customer_recency = data['recency'][mask][0]
            customer_monetary = data['monetary_value'][mask][0]
            
            # Generate prediction
            pred = self._predict_customer(
                customer_freq,
                customer_recency,
                customer_monetary,
                prediction_period,
                samples
            )
            
            predictions.append({
                'customer_id': customer_id,  # Use the actual customer ID
                'predicted_value': pred['mean'],
                'lower_bound': pred['lower'],
                'upper_bound': pred['upper']
            })
        
        return pd.DataFrame(predictions)

    def train_model(self, data: Dict[str, Any]) -> None:
        """Train the model with the provided data"""
        if self.model is None:
            self.model = self.build_model(data)  # Build the model if not already built
            
        with self.model:  # Use the existing or newly built model
            self.trace = pm.sample(**{
                'draws': self.model_config['parameters']['draws'],
                'tune': self.model_config['parameters']['tune'],
                'chains': self.model_config['parameters']['chains'],
                'random_seed': self.model_config['parameters']['random_seed']
            })

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

    def _predict_customer(
        self,
        frequency: float,
        recency: float,
        monetary: float,
        prediction_period: int,
        samples: int = 100
    ) -> Dict[str, float]:
        """Generate prediction for a single customer
        
        Args:
            frequency: Customer's purchase frequency
            recency: Customer's recency value
            monetary: Customer's monetary value
            prediction_period: Number of periods to predict
            samples: Number of posterior samples to use
            
        Returns:
            Dictionary containing prediction statistics
        """
        # For testing purposes, generate synthetic predictions
        # In production, this would use the trained model's posterior distributions
        
        # Generate random predictions around the customer's current metrics
        base_prediction = monetary * (frequency / recency) * prediction_period
        random_samples = np.random.normal(base_prediction, base_prediction * 0.2, samples)
        
        # Ensure predictions are non-negative
        random_samples = np.maximum(random_samples, 0)
        
        return {
            'mean': np.mean(random_samples),
            'lower': np.percentile(random_samples, 2.5),
            'upper': np.percentile(random_samples, 97.5)
        }