import pytest
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from src.pipeline.clv import HierarchicalCLVModel, CLVConfigLoader
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from tests.test_clv.test_metadata_analysis import ModelMetadataAnalyzer
from tests.test_clv.test_statistical_analysis import ModelStatisticalAnalyzer
from tests.test_clv.test_statistical_visualization import StatisticalVisualizer
from tests.test_clv.test_report_generation import ModelAnalysisReport

@dataclass
class RunMetadata:
    """Metadata for a model diagnostic run"""
    timestamp: str
    config_hash: str
    git_commit: Optional[str]
    model_params: Dict[str, Any]
    convergence_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    hardware_info: Dict[str, str]
    duration: float
    status: str
    notes: Optional[str] = None

class TestModelDiagnostics:
    """Tests for model configuration and convergence diagnostics with visualizations"""
    
    def setup_method(self):
        """Setup for tests with metadata tracking"""
        # Create directory structure (previous code)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_diagnostic_dir = Path("diagnostics")
        self.run_dir = self.base_diagnostic_dir / f"model_run_{timestamp}"
        
        # Create directories
        for dir_name in ['convergence', 'efficiency', 'sensitivity', 'metadata']:
            (self.run_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.run_metadata = RunMetadata(
            timestamp=timestamp,
            config_hash=self._hash_config(),
            git_commit=self._get_git_commit(),
            model_params={},
            convergence_metrics={},
            performance_metrics={},
            hardware_info=self._get_hardware_info(),
            duration=0,
            status='started'
        )
        
        # Set style for plots
        plt.style.use('seaborn')
        
    def _hash_config(self) -> str:
        """Create hash of current configuration"""
        import hashlib
        config_str = str(self.config_loader.model_config)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha[:8]
        except:
            return None
    
    def _get_hardware_info(self) -> Dict[str, str]:
        """Get hardware information"""
        import platform
        import psutil
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
            'cpu_count': str(psutil.cpu_count())
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu'] = torch.cuda.get_device_name(0)
        except:
            info['gpu'] = 'none'
            
        return info
    
    def save_metadata(self):
        """Save run metadata to file"""
        metadata_file = self.run_dir / 'metadata' / 'run_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(asdict(self.run_metadata), f, indent=2)
            
        # Also save configuration
        config_file = self.run_dir / 'metadata' / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config_loader.model_config, f)
    
    def update_metadata(self, **kwargs):
        """Update metadata with new information"""
        for key, value in kwargs.items():
            if hasattr(self.run_metadata, key):
                setattr(self.run_metadata, key, value)
    
    def save_plot(self, name: str, category: str):
        """Save plot to appropriate subdirectory"""
        category_dir = getattr(self, f"{category}_dir")
        plt.savefig(category_dir / f"{name}.png")
        plt.close()
    
    def test_model_convergence_quality(self, sample_model_data, config_loader):
        """Test and visualize model convergence with PDF report"""
        start_time = time.time()
        
        try:
            # Previous convergence test code...
            model = HierarchicalCLVModel(config_loader)
            model.build_model(sample_model_data)
            trace = model.sample(draws=500, tune=200, chains=2)
            
            # Calculate metrics
            r_hat = az.rhat(trace)
            ess = az.ess(trace)
            
            # Update metadata with convergence metrics
            self.update_metadata(
                convergence_metrics={
                    'max_r_hat': float(max(r_hat.values())),
                    'min_ess': float(min(ess.values())),
                    'mean_ess': float(np.mean(ess.values()))
                },
                model_params=model.config,
                duration=time.time() - start_time,
                status='completed'
            )
            
            # Save plots...
            
        except Exception as e:
            self.update_metadata(
                status='failed',
                notes=str(e)
            )
            raise
        finally:
            self.save_metadata()
        
        # After running tests, analyze metadata
        analyzer = ModelMetadataAnalyzer()
        analysis = analyzer.analyze_runs()
        
        # Print analysis results
        print("\nRun Analysis:")
        print(f"Total Runs: {analysis.run_summary['total_runs'][0]}")
        print(f"Success Rate: {analysis.run_summary['successful_runs'][0] / analysis.run_summary['total_runs'][0]:.2%}")
        print("\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"- {rec}")
        
        # Create analysis plots
        analyzer.plot_analysis()
        
        # Run statistical analysis
        analyzer = ModelStatisticalAnalyzer()
        
        # Generate and save statistical report
        report = analyzer.generate_statistical_report()
        report_path = self.run_dir / 'metadata' / 'statistical_analysis.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Run tests and assert on critical metrics
        results = analyzer.run_statistical_tests()
        
        # Check convergence
        assert results['convergence_tests']['r_hat']['trend_pvalue'] > 0.05, \
            "Significant trend in R-hat values indicates convergence issues"
        
        # Check performance stability
        assert results['stationarity_tests']['duration']['pvalue'] > 0.05, \
            "Non-stationary performance metrics detected"
        
        # Create statistical visualizations
        stat_analyzer = ModelStatisticalAnalyzer()
        visualizer = StatisticalVisualizer(stat_analyzer)
        visualizer.create_visualizations(self.run_dir)
        
        # Generate PDF report
        report_generator = ModelAnalysisReport(stat_analyzer, analyzer)
        
        pdf_path = report_generator.generate_report(self.run_dir)
        print(f"\nAnalysis report generated: {pdf_path}")
    
    @classmethod
    def compare_runs(cls, n_runs: int = 5):
        """Compare metadata from recent runs"""
        diagnostic_dir = Path("diagnostics")
        runs = sorted(diagnostic_dir.glob("model_run_*"))[-n_runs:]
        
        comparisons = []
        for run_dir in runs:
            metadata_file = run_dir / 'metadata' / 'run_info.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    comparisons.append(json.load(f))
        
        # Create comparison plots
        cls._plot_run_comparisons(comparisons)
    
    @staticmethod
    def _plot_run_comparisons(comparisons):
        """Create comparison plots between runs"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot R-hat trends
        r_hats = [c['convergence_metrics']['max_r_hat'] for c in comparisons]
        axes[0, 0].plot(r_hats, marker='o')
        axes[0, 0].set_title('Max R-hat Across Runs')
        axes[0, 0].axhline(y=1.1, color='r', linestyle='--')
        
        # Plot ESS trends
        ess = [c['convergence_metrics']['mean_ess'] for c in comparisons]
        axes[0, 1].plot(ess, marker='o')
        axes[0, 1].set_title('Mean ESS Across Runs')
        
        # Plot duration
        durations = [c['duration'] for c in comparisons]
        axes[1, 0].plot(durations, marker='o')
        axes[1, 0].set_title('Run Duration (s)')
        
        # Plot status
        statuses = [c['status'] for c in comparisons]
        status_counts = pd.Series(statuses).value_counts()
        status_counts.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Run Status Distribution')
        
        plt.tight_layout()
        plt.savefig('diagnostics/run_comparisons.png')
        plt.close()
    
    def _plot_rhat(self, r_hat):
        """Create R-hat diagnostic plot"""
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame({'parameter': r_hat.index, 'r_hat': r_hat.values})
        
        sns.barplot(data=data, x='r_hat', y='parameter')
        plt.axvline(x=1.1, color='r', linestyle='--', label='Threshold (1.1)')
        plt.title('R-hat Values by Parameter')
        plt.xlabel('R-hat')
        plt.tight_layout()
        
    def _plot_ess(self, ess):
        """Create ESS diagnostic plot"""
        plt.figure(figsize=(10, 6))
        data = pd.DataFrame({'parameter': ess.index, 'ess': ess.values})
        
        sns.barplot(data=data, x='ess', y='parameter')
        plt.axvline(x=400, color='r', linestyle='--', label='Minimum ESS (400)')
        plt.title('Effective Sample Size by Parameter')
        plt.xlabel('ESS')
        plt.tight_layout()
    
    def test_sampling_efficiency(self, sample_model_data, config_loader):
        """Test and visualize sampling efficiency"""
        results = []
        configs = [
            {'mcmc_samples': 1000, 'chains': 2},
            {'mcmc_samples': 2000, 'chains': 4},
            {'mcmc_samples': 500, 'chains': 2}
        ]
        
        for config in configs:
            model = HierarchicalCLVModel(config_loader)
            model.build_model(sample_model_data)
            
            start_time = time.time()
            trace = model.sample(
                draws=config['mcmc_samples'],
                tune=config['mcmc_samples'] // 2,
                chains=config['chains']
            )
            sampling_time = time.time() - start_time
            
            results.append({
                'config': f"{config['mcmc_samples']}s_{config['chains']}c",
                'time_per_sample': sampling_time / (config['mcmc_samples'] * config['chains']),
                'ess_per_second': float(np.mean(az.ess(trace).values())) / sampling_time,
                'max_r_hat': float(max(az.rhat(trace).values()))
            })
        
        # Plot efficiency metrics
        self._plot_efficiency_comparison(results)
        self.save_plot("sampling_efficiency", "efficiency")
        
    def _plot_efficiency_comparison(self, results):
        """Create efficiency comparison plots"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Time per sample
        sns.barplot(data=df, x='config', y='time_per_sample', ax=axes[0])
        axes[0].set_title('Time per Sample')
        axes[0].set_ylabel('Seconds')
        
        # ESS per second
        sns.barplot(data=df, x='config', y='ess_per_second', ax=axes[1])
        axes[1].set_title('ESS per Second')
        
        # Max R-hat
        sns.barplot(data=df, x='config', y='max_r_hat', ax=axes[2])
        axes[2].set_title('Maximum R-hat')
        axes[2].axhline(y=1.1, color='r', linestyle='--')
        
        plt.tight_layout()
    
    def test_prior_sensitivity(self, sample_model_data, config_loader):
        """Test and visualize prior sensitivity"""
        results = []
        prior_configs = [
            {'alpha_shape': 1.0, 'beta_shape': 1.0},
            {'alpha_shape': 0.5, 'beta_shape': 0.5},
            {'alpha_shape': 2.0, 'beta_shape': 2.0}
        ]
        
        for prior in prior_configs:
            model = HierarchicalCLVModel(config_loader)
            model.config['hyperparameters']['prior_settings'].update(prior)
            model.build_model(sample_model_data)
            trace = model.sample(draws=500, tune=200, chains=2)
            
            # Store posterior summaries
            summary = az.summary(trace)
            results.append({
                'prior': f"α={prior['alpha_shape']},β={prior['beta_shape']}",
                'summary': summary
            })
        
        # Plot prior sensitivity comparison
        self._plot_prior_sensitivity(results)
        self.save_plot("prior_sensitivity", "sensitivity")
    
    def _plot_prior_sensitivity(self, results):
        """Create prior sensitivity comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prior Sensitivity Analysis')
        
        params = ['r', 'alpha', 'beta', 'gamma']
        
        for i, param in enumerate(params):
            ax = axes[i // 2, i % 2]
            data = []
            
            for result in results:
                if param in result['summary'].index:
                    data.append({
                        'prior': result['prior'],
                        'mean': result['summary'].loc[param, 'mean'],
                        'sd': result['summary'].loc[param, 'sd']
                    })
            
            if data:
                df = pd.DataFrame(data)
                sns.barplot(data=df, x='prior', y='mean', yerr=df['sd'], ax=ax)
                ax.set_title(f'Parameter: {param}')
                ax.set_ylabel('Posterior Mean')
                
        plt.tight_layout()
    
    def test_model_stability(self, sample_model_data, config_loader):
        """Test model stability across multiple runs"""
        n_runs = 3
        results = []
        
        for i in range(n_runs):
            model = HierarchicalCLVModel(config_loader)
            model.build_model(sample_model_data)
            trace = model.sample(draws=500, tune=200, chains=2)
            
            # Get posterior means
            posterior_means = {
                var: float(trace.posterior[var].mean())
                for var in trace.posterior.variables
                if var not in ['likelihood']
            }
            
            results.append(posterior_means)
            
        # Calculate variation in estimates
        variations = {}
        for param in results[0].keys():
            values = [r[param] for r in results]
            variations[param] = np.std(values) / np.mean(values)  # CV
            
        print("\nParameter Stability:")
        for param, cv in variations.items():
            print(f"{param} CV: {cv:.3f}")
            assert cv < 0.1, f"High variability in {param}" 