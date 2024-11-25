from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from .base import BaseProcessor
import numpy as np

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

    def _create_barplot(self, data: pd.DataFrame, x: str, y: str, ax: plt.Axes, title: str):
        """Create a barplot using matplotlib to avoid seaborn deprecation warning
        
        Args:
            data: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            ax: Matplotlib axes object
            title: Plot title
        """
        # Convert to categorical if not already
        if not isinstance(data[y].dtype, pd.CategoricalDtype):
            data = data.copy()
            data[y] = pd.Categorical(data[y])
            
        # Create barplot manually
        bars = ax.barh(range(len(data)), data[x], align='center')
        
        # Set y-tick labels
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data[y])
        
        # Set labels and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        
        return bars

    def plot_trace(self, trace: Dict[str, Any], params: Optional[list] = None):
        """Plot MCMC trace
        
        Args:
            trace: MCMC trace dictionary
            params: Optional list of parameters to plot
        """
        # Convert the trace format to match what's expected
        posterior = {
            'alpha': trace['samples'][:, 0],
            'beta': trace['samples'][:, 1],
            'lambda': trace['samples'][:, 2],
            'r': trace['samples'][:, 3]
        }
        trace['posterior'] = posterior
        
        if not params:
            params = list(trace['posterior'].keys())
            
        fig, axes = plt.subplots(len(params), 2, figsize=(12, 4*len(params)))
        
        for i, param in enumerate(params):
            # Plot trace
            axes[i, 0].plot(trace['posterior'][param])
            axes[i, 0].set_title(f'{param} trace')
            
            # Plot histogram
            axes[i, 1].hist(trace['posterior'][param], bins=30)
            axes[i, 1].set_title(f'{param} histogram')
            
        plt.tight_layout()
        return fig

    def plot_segments(self, data, features: List[str]):
        """Plot customer segments
        
        Args:
            data: DataFrame with customer features
            features: List of features to use for visualization
        """
        if len(features) < 2:
            raise ValueError("Need at least 2 features for visualization")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            data[features[0]], 
            data[features[1]],
            c=data.get('segment', np.zeros(len(data))),
            cmap='viridis'
        )
        
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        plt.colorbar(scatter, label='Segment')
        
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
            trace: Either MCMC trace or diagnostics dictionary
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Check if we're dealing with raw diagnostics or a trace
        if all(key in trace for key in ['convergence', 'effective_sample_size', 'r_hat']):
            # Use provided diagnostics directly
            r_hat_values = trace['r_hat']
            n_eff_values = trace['effective_sample_size']
            parameters = [f'param_{i}' for i in range(len(r_hat_values))]
        elif 'posterior' in trace:
            # Calculate diagnostics from trace
            r_hat = az.rhat(trace)
            n_eff = az.ess(trace)
            
            r_hat_values = [float(v) for v in r_hat.values()]
            n_eff_values = [float(v) for v in n_eff.values()]
            parameters = list(r_hat.keys())
        else:
            raise ValueError("Invalid trace format. Must contain either 'posterior' or diagnostic metrics.")
        
        # Create DataFrames for plotting
        r_hat_df = pd.DataFrame({
            'parameter': parameters,
            'r_hat': r_hat_values
        })
        
        n_eff_df = pd.DataFrame({
            'parameter': parameters,
            'n_eff': n_eff_values
        })
        
        # Create plots using the new method
        self._create_barplot(
            data=r_hat_df,
            x='r_hat',
            y='parameter',
            ax=ax1,
            title='R-hat Values'
        )
        ax1.axvline(x=1.1, color='r', linestyle='--')
        
        self._create_barplot(
            data=n_eff_df,
            x='n_eff',
            y='parameter',
            ax=ax2,
            title='Effective Sample Size'
        )
        
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