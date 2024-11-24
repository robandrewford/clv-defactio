from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from .base import BaseProcessor

class CLVVisualization(BaseProcessor):
    """Visualization component for CLV analysis"""
    
    def __init__(self, config):
        """Initialize visualization
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.viz_config = config.get('visualization', {})
        self.style = self.viz_config.get('plot_style', 'default')
        self.figure_size = self.viz_config.get('figure_size', (10, 6))
        if self.style != 'default':
            sns.set_style(self.style)

    def plot_trace(self, trace: Dict[str, Any], params: Optional[list] = None):
        """Plot MCMC trace
        
        Args:
            trace: MCMC trace dictionary
            params: Optional list of parameters to plot
        """
        if not params:
            params = list(trace['posterior'].keys())
            
        fig, axes = plt.subplots(len(params), 2, figsize=self.figure_size)
        
        for idx, param in enumerate(params):
            data = trace['posterior'][param]
            
            # Plot trace
            if len(axes.shape) == 1:
                ax1, ax2 = axes[0], axes[1]
            else:
                ax1, ax2 = axes[idx, 0], axes[idx, 1]
                
            ax1.plot(data.T)
            ax1.set_title(f'{param} trace')
            
            # Plot distribution
            sns.kdeplot(data.flatten(), ax=ax2)
            ax2.set_title(f'{param} distribution')
            
        plt.tight_layout()
        return fig

    def plot_segments(self, df: pd.DataFrame, features: list):
        """Plot customer segments
        
        Args:
            df: Segmented DataFrame
            features: Features to plot
        """
        if len(features) < 2:
            raise ValueError("Need at least 2 features for segment visualization")
            
        fig = plt.figure(figsize=self.figure_size)
        
        # Create scatter plot of first two features
        plt.scatter(
            df[features[0]], 
            df[features[1]], 
            c=df['segment'], 
            cmap='viridis'
        )
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('Customer Segments')
        
        return fig

    def plot_prediction_intervals(self, predictions: pd.DataFrame):
        """Plot prediction intervals
        
        Args:
            predictions: DataFrame with predictions and bounds
        """
        fig = plt.figure(figsize=self.figure_size)
        
        plt.plot(predictions['predicted_value'], label='Prediction')
        plt.fill_between(
            range(len(predictions)),
            predictions['lower_bound'],
            predictions['upper_bound'],
            alpha=0.3,
            label='95% CI'
        )
        
        plt.xlabel('Customer Index')
        plt.ylabel('Predicted CLV')
        plt.title('CLV Predictions with Uncertainty')
        plt.legend()
        
        return fig

    def plot_convergence_diagnostics(self, trace: Dict[str, Any]):
        """Plot model convergence diagnostics
        
        Args:
            trace: MCMC trace
        """
        if not isinstance(trace, dict) or 'posterior' not in trace:
            raise ValueError("Invalid trace format")
            
        # Calculate R-hat statistics
        r_hat = az.rhat(trace)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Plot R-hat values
        r_hat_df = pd.DataFrame({
            'parameter': list(r_hat.keys()),
            'r_hat': [float(v) for v in r_hat.values()]
        })
        
        sns.barplot(data=r_hat_df, x='r_hat', y='parameter', ax=ax1)
        ax1.set_title('R-hat Values')
        ax1.axvline(x=1.1, color='r', linestyle='--')
        
        # Plot effective sample size
        n_eff = az.ess(trace)
        n_eff_df = pd.DataFrame({
            'parameter': list(n_eff.keys()),
            'n_eff': [float(v) for v in n_eff.values()]
        })
        
        sns.barplot(data=n_eff_df, x='n_eff', y='parameter', ax=ax2)
        ax2.set_title('Effective Sample Size')
        
        plt.tight_layout()
        return fig

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process visualization data
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary of generated plots
        """
        plots = {}
        
        if 'trace' in data:
            plots['trace'] = self.plot_trace(data['trace'])
            plots['diagnostics'] = self.plot_convergence_diagnostics(data['trace'])
            
        if 'predictions' in data:
            plots['predictions'] = self.plot_prediction_intervals(data['predictions'])
            
        if 'segments' in data and 'features' in data:
            plots['segments'] = self.plot_segments(data['segments'], data['features'])
            
        return plots 