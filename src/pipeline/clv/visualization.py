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
        self.viz_config = config_loader.pipeline_config['visualization']
        plt.style.use(self.viz_config.get('style', 'seaborn'))
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for visualization"""
        return df
        
    def plot_trace(self, trace, params: Optional[list] = None) -> plt.Figure:
        """Plot MCMC trace with configurable parameters"""
        config = self.viz_config['trace_plots']
        fig, axes = plt.subplots(
            3, 2, 
            figsize=config.get('figsize', (12, 8)),
            dpi=self.viz_config.get('dpi', 100)
        )
        
        params = params or ['r', 'alpha', 'beta']
        n_chains = min(
            config.get('n_chains_display', 4),
            trace.posterior.chain.size
        )
        
        for i, param in enumerate(params):
            # Trace plot
            for chain in range(n_chains):
                axes[i, 0].plot(
                    trace.posterior[param].isel(chain=chain),
                    alpha=0.7,
                    label=f'Chain {chain+1}'
                )
            axes[i, 0].set_title(f'{param} Trace')
            axes[i, 0].legend()
            
            # Histogram
            sns.histplot(
                trace.posterior[param].values.flatten(),
                ax=axes[i, 1],
                bins=config.get('hist_bins', 30)
            )
            axes[i, 1].set_title(f'{param} Distribution')
            
        plt.tight_layout()
        return fig
        
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
            sorted_preds['value_lower'],
            sorted_preds['value_upper'],
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