import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, Any
from .test_model_diagnostics import ModelStatisticalAnalyzer

class StatisticalVisualizer:
    """Visualize statistical analysis results"""
    
    def __init__(self, analyzer: ModelStatisticalAnalyzer):
        self.analyzer = analyzer
        self.results = analyzer.run_statistical_tests()
        plt.style.use('seaborn')
        
    def create_visualizations(self, save_dir: Path):
        """Create all statistical visualizations"""
        save_dir = save_dir / 'statistical_analysis'
        save_dir.mkdir(exist_ok=True)
        
        # Create individual plots
        self.plot_convergence_tests(save_dir)
        self.plot_performance_distributions(save_dir)
        self.plot_correlation_matrix(save_dir)
        self.plot_effect_sizes(save_dir)
        self.plot_stationarity_analysis(save_dir)
        
        # Create summary dashboard
        self.create_dashboard(save_dir)
    
    def plot_convergence_tests(self, save_dir: Path):
        """Visualize convergence test results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R-hat trend analysis
        r_hats = self.analyzer.runs_data['convergence_metrics'].apply(
            lambda x: x.get('max_r_hat', np.nan)
        )
        
        sns.regplot(
            x=np.arange(len(r_hats)),
            y=r_hats,
            ax=ax1,
            scatter_kws={'alpha': 0.5}
        )
        ax1.set_title('R-hat Trend Analysis')
        ax1.set_xlabel('Run Index')
        ax1.set_ylabel('Maximum R-hat')
        
        # Add trend test results
        trend_test = self.results['convergence_tests']['r_hat']
        ax1.text(
            0.05, 0.95,
            f"Trend p-value: {trend_test['trend_pvalue']:.3f}",
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # ESS autocorrelation
        ess_values = self.analyzer.runs_data['convergence_metrics'].apply(
            lambda x: x.get('mean_ess', np.nan)
        )
        
        pd.plotting.autocorrelation_plot(ess_values.dropna(), ax=ax2)
        ax2.set_title('ESS Autocorrelation')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'convergence_tests.png')
        plt.close()
    
    def plot_performance_distributions(self, save_dir: Path):
        """Visualize performance metric distributions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Duration distribution
        durations = self.analyzer.runs_data['duration']
        sns.histplot(durations, kde=True, ax=ax1)
        ax1.set_title('Run Duration Distribution')
        
        # Add normality test results
        norm_test = self.results['distribution_tests']['duration_normality']
        ax1.text(
            0.05, 0.95,
            f"Normality p-value: {norm_test['pvalue']:.3f}",
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Q-Q plot for duration
        stats.probplot(durations, dist="norm", plot=ax2)
        ax2.set_title('Duration Q-Q Plot')
        
        # Memory usage distribution
        memory_usage = self.analyzer.runs_data['performance_metrics'].apply(
            lambda x: float(x.get('memory_peak', '0').rstrip('GB'))
        )
        sns.histplot(memory_usage, kde=True, ax=ax3)
        ax3.set_title('Memory Usage Distribution')
        
        # Performance by configuration
        sns.boxplot(
            data=self.analyzer.runs_data,
            x='config_hash',
            y='duration',
            ax=ax4
        )
        ax4.set_title('Duration by Configuration')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_distributions.png')
        plt.close()
    
    def plot_correlation_matrix(self, save_dir: Path):
        """Visualize correlation matrix with significance"""
        metrics = pd.DataFrame({
            'duration': self.analyzer.runs_data['duration'],
            'r_hat': self.analyzer.runs_data['convergence_metrics'].apply(
                lambda x: x.get('max_r_hat', np.nan)
            ),
            'ess': self.analyzer.runs_data['convergence_metrics'].apply(
                lambda x: x.get('mean_ess', np.nan)
            )
        })
        
        # Create correlation matrix
        corr_matrix = metrics.corr()
        
        # Plot
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0
        )
        
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(save_dir / 'correlation_matrix.png')
        plt.close()
    
    def plot_effect_sizes(self, save_dir: Path):
        """Visualize effect sizes across configurations"""
        effect_sizes = []
        
        for config, stats in self.results['performance_tests'].items():
            effect_sizes.append({
                'config': config,
                'effect_size': stats['effect_size'],
                'significant': stats['duration_pvalue'] < 0.05
            })
        
        df = pd.DataFrame(effect_sizes)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            df['config'],
            df['effect_size'],
            color=[
                'green' if sig else 'gray'
                for sig in df['significant']
            ]
        )
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title("Configuration Effect Sizes")
        plt.xlabel("Configuration")
        plt.ylabel("Cohen's d")
        plt.xticks(rotation=45)
        
        # Add significance markers
        for i, significant in enumerate(df['significant']):
            if significant:
                plt.text(
                    i,
                    bars[i].get_height(),
                    '*',
                    ha='center',
                    va='bottom'
                )
        
        plt.tight_layout()
        plt.savefig(save_dir / 'effect_sizes.png')
        plt.close()
    
    def plot_stationarity_analysis(self, save_dir: Path):
        """Visualize stationarity analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Duration time series
        durations = self.analyzer.runs_data['duration']
        ax1.plot(durations, marker='o')
        ax1.set_title('Duration Time Series')
        
        # Add stationarity test results
        adf_test = self.results['stationarity_tests']['duration']
        ax1.text(
            0.05, 0.95,
            f"ADF p-value: {adf_test['pvalue']:.3f}",
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Rolling statistics
        window = 5
        rolling_mean = durations.rolling(window=window).mean()
        rolling_std = durations.rolling(window=window).std()
        
        ax2.plot(rolling_mean, label='Rolling Mean')
        ax2.plot(rolling_std, label='Rolling Std')
        ax2.set_title(f'Rolling Statistics (window={window})')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'stationarity_analysis.png')
        plt.close()
    
    def create_dashboard(self, save_dir: Path):
        """Create summary dashboard of all statistical results"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # Convergence summary
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_convergence_summary(ax1)
        
        # Performance summary
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_performance_summary(ax2)
        
        # Distribution summary
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_distribution_summary(ax3)
        
        # Correlation summary
        ax4 = fig.add_subplot(gs[1:, 2])
        self._plot_correlation_summary(ax4)
        
        # Effect size summary
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_effect_size_summary(ax5)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'statistical_dashboard.png')
        plt.close() 