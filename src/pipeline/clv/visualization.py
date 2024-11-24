from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .base import BaseProcessor

class CLVVisualization(BaseProcessor):
    """Component for CLV-related visualizations"""
    
    def __init__(self, config_loader):
        self.config = config_loader
        self.viz_config = config_loader.pipeline_config.get('visualization', {})
        plt.style.use('default')
        if plt.get_backend() == 'agg':
            sns.set_style('whitegrid')
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for visualization"""
        return df
        
    def plot_trace(self, trace, params: Optional[list] = None) -> plt.Figure:
        """Plot MCMC trace with configurable parameters"""
        config = self.viz_config.get('trace_plots', {})
        fig, axes = plt.subplots(
            3, 2,
            figsize=config.get('figsize', (12, 8)),
            dpi=config.get('dpi', 100)
        )
        
        params = params or ['r', 'alpha', 'beta']
        
        # Handle both ArviZ InferenceData and dictionary traces
        if hasattr(trace, 'posterior'):
            # ArviZ InferenceData
            n_chains = trace.posterior.chain.size
            for i, param in enumerate(params):
                if param in trace.posterior:
                    param_data = trace.posterior[param].values
                    self._plot_param_trace(axes[i, 0], axes[i, 1], param, param_data, n_chains)
        else:
            # Dictionary trace
            n_chains = trace.get('chains', 1)
            samples = trace.get('samples', {})
            for i, param in enumerate(params):
                if param in samples:
                    param_data = samples[param]
                    self._plot_param_trace(axes[i, 0], axes[i, 1], param, param_data, n_chains)
        
        plt.tight_layout()
        return fig
        
    def _plot_param_trace(self, ax_trace, ax_hist, param, data, n_chains):
        """Helper method to plot trace and histogram for a parameter"""
        if len(data.shape) == 1:
            # Reshape 1D array to 2D (chains x samples)
            samples_per_chain = len(data) // n_chains
            data = data.reshape(n_chains, samples_per_chain)
        
        for chain in range(n_chains):
            ax_trace.plot(data[chain], alpha=0.5)
        ax_trace.set_title(f'{param} trace')
        
        ax_hist.hist(data.flatten(), bins=30)
        ax_hist.set_title(f'{param} histogram')
        
    def plot_segments(self, df: pd.DataFrame) -> plt.Figure:
        """Plot customer segments with configurable styling"""
        config = self.viz_config['segment_plots']
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=config.get('figsize', (15, 6)),
            dpi=self.viz_config.get('dpi', 100)
        )
        
        palette = config.get('palette', 'deep')
        alpha = config.get('bar_alpha', 0.8)
        
        # RFM Score Distribution
        if 'RFM_score' in df.columns:
            sns.histplot(
                data=df,
                x='RFM_score',
                ax=ax1,
                palette=palette,
                alpha=alpha
            )
            ax1.set_title('RFM Score Distribution')
            
        # Segment Sizes
        if 'segment_ids' in df.columns:
            segment_sizes = df['segment_ids'].value_counts()
            segment_sizes.plot(
                kind='bar',
                ax=ax2,
                alpha=alpha,
                color=sns.color_palette(palette)
            )
            ax2.set_title('Segment Sizes')
            
        plt.tight_layout()
        return fig
        
    def save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        directory: Optional[str] = None
    ) -> None:
        """Save plot in configured formats"""
        formats = self.viz_config.get('formats', ['png'])
        directory = directory or 'plots'
        
        import os
        os.makedirs(directory, exist_ok=True)
        
        for fmt in formats:
            path = os.path.join(directory, f"{filename}.{fmt}")
            fig.savefig(
                path,
                dpi=self.viz_config.get('dpi', 100),
                bbox_inches='tight'
            )
        
    def plot_prediction_intervals(self, predictions: pd.DataFrame) -> plt.Figure:
        """Plot prediction intervals"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by predicted value for better visualization
        sorted_preds = predictions.sort_values('predicted_value')
        
        # Plot predictions with confidence intervals
        ax.plot(sorted_preds.index, sorted_preds['predicted_value'], 'b-', label='Prediction')
        ax.fill_between(
            sorted_preds.index,
            sorted_preds['lower_bound'],
            sorted_preds['upper_bound'],
            alpha=0.3,
            label='95% CI'
        )
        
        ax.set_title('CLV Predictions with Confidence Intervals')
        ax.set_xlabel('Customer Index')
        ax.set_ylabel('Predicted CLV')
        ax.legend()
        
        return fig
        
    def plot_diagnostics(self, diagnostics: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot model diagnostics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Convergence plot
        if 'convergence' in diagnostics:
            axes[0, 0].plot(diagnostics['convergence'])
            axes[0, 0].set_title('Convergence')
            
        # Effective sample size
        if 'effective_sample_size' in diagnostics:
            sns.histplot(diagnostics['effective_sample_size'], ax=axes[0, 1])
            axes[0, 1].set_title('Effective Sample Size')
            
        # R-hat values
        if 'r_hat' in diagnostics:
            sns.histplot(diagnostics['r_hat'], ax=axes[1, 0])
            axes[1, 0].set_title('R-hat Values')
            axes[1, 0].axvline(x=1.1, color='r', linestyle='--')
            
        plt.tight_layout()
        return fig 

    def _plot_rhat(self, r_hat_values: pd.Series) -> plt.Figure:
        """Plot R-hat convergence statistics"""
        # Convert xarray DataArray to pandas DataFrame
        if hasattr(r_hat_values, 'to_dataframe'):
            r_hat_df = r_hat_values.to_dataframe('r_hat')
        else:
            r_hat_df = pd.DataFrame({'r_hat': r_hat_values})
        
        r_hat_df = r_hat_df.reset_index()
        r_hat_df.columns = ['parameter', 'r_hat']
        
        # Convert to numeric, dropping any non-numeric values
        r_hat_df['r_hat'] = pd.to_numeric(r_hat_df['r_hat'], errors='coerce')
        r_hat_df = r_hat_df.dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=r_hat_df, x='r_hat', y='parameter', ax=ax)
        ax.axvline(x=1.1, color='r', linestyle='--', alpha=0.5)
        ax.set_title('R-hat Values by Parameter')
        
        return fig 