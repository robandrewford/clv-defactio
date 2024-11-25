import pytest
import numpy as np
import pandas as pd
import arviz as az
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from src.pipeline.clv.model import HierarchicalCLVModel
import json
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import torch
import xarray as xr

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

@pytest.fixture
def sample_model_data():
    """Fixture to generate sample model data"""
    n_customers = 100
    return {
        'frequency': np.random.poisson(5, n_customers),
        'recency': np.random.randint(0, 365, n_customers),
        'monetary_value': np.random.lognormal(3, 1, n_customers),
        'T': np.random.randint(100, 1000, n_customers),
        'segment_ids': np.zeros(n_customers)
    }

class TestModelDiagnostics:
    """Tests for model configuration and convergence diagnostics with visualizations"""
    
    def setup_method(self):
        """Setup for tests with metadata tracking"""
        # Create directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_diagnostic_dir = Path("diagnostics")
        self.run_dir = self.base_diagnostic_dir / f"model_run_{timestamp}"
        
        # Create directories
        for dir_name in ['convergence', 'efficiency', 'sensitivity', 'metadata']:
            (self.run_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.run_metadata = RunMetadata(
            timestamp=timestamp,
            config_hash="",  # Will be set when config_loader is available
            git_commit=self._get_git_commit(),
            model_params={},
            convergence_metrics={},
            performance_metrics={},
            hardware_info=self._get_hardware_info(),
            duration=0,
            status='started'
        )
        
        # Set style for plots
        try:
            sns.set_style("whitegrid")
        except ImportError:
            plt.style.use('default')
        
        # Add config_loader attribute initialization
        self.config_loader = None
    
    def _hash_config(self, config_loader) -> str:
        """Create hash of current configuration"""
        import hashlib
        config_str = str(config_loader.model_config)
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
    
    @pytest.mark.tryfirst
    def test_model_convergence_quality(self, sample_model_data, config_loader):
        """Test and visualize model convergence with PDF report"""
        try:
            # Build and sample model
            model = HierarchicalCLVModel(config_loader)
            model.build_model(sample_model_data)
            
            # Get parameters from config
            model_params = config_loader.get('model', {}).get('parameters', {})
            chains = model_params.get('chains', 4)
            draws = model_params.get('draws', 2000)
            tune = model_params.get('tune', 1000)
            
            trace_dict = model.sample(
                draws=draws,
                tune=tune,
                chains=chains
            )
            
            # Extract samples from trace_dict
            samples = trace_dict.get('samples', [])
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples)
            
            # Handle nested dictionary structure
            if samples.dtype == object:
                numeric_samples = []
                for sample in samples.flat:
                    if isinstance(sample, dict) and 'parameter' in sample:
                        param_array = np.array(sample['parameter'])
                        numeric_samples.append(param_array)
                samples = np.stack(numeric_samples)
            
            # Ensure we have numeric data
            samples = samples.astype(np.float64)
            
            # Calculate dimensions based on actual data
            total_samples = len(samples)
            n_params = samples.shape[-1] if samples.ndim > 1 else 1
            samples_per_chain = total_samples // chains  # This should match draws
            
            # Reshape samples correctly using calculated dimensions
            samples = samples.reshape((chains, samples_per_chain, n_params))
            
            # Create proper structure for arviz
            posterior_data = xr.DataArray(
                samples,
                dims=['chain', 'draw', 'param'],
                coords={
                    'chain': np.arange(chains),
                    'draw': np.arange(samples_per_chain),
                    'param': [f'param_{i}' for i in range(n_params)]
                }
            )
            
            # Convert to InferenceData
            trace = az.InferenceData(
                posterior=xr.Dataset({'theta': posterior_data})
            )
            
            # Calculate convergence metrics
            r_hat_values = az.rhat(trace.posterior.theta)
            
            # Convert r_hat values to numpy array and handle NaN values
            if isinstance(r_hat_values, xr.Dataset):
                r_hat_array = r_hat_values.to_array().values
            elif isinstance(r_hat_values, xr.DataArray):
                r_hat_array = r_hat_values.values
            else:
                r_hat_array = np.asarray(r_hat_values)
            
            # Ensure r_hat_array is 1D
            r_hat_array = np.ravel(r_hat_array)
            
            # Convert to proper format for plotting
            valid_values = []
            for i, r_hat in enumerate(r_hat_array):
                try:
                    r_hat_float = float(r_hat)
                    if not pd.isna(r_hat_float):
                        valid_values.append((f'param_{i}', r_hat_float))
                except (TypeError, ValueError):
                    continue
            
            r_hat_df = pd.DataFrame(valid_values, columns=['parameter', 'r_hat'])
            
            # Fix categorical dtype warning by using explicit CategoricalDtype
            r_hat_df['parameter'] = pd.Series(
                r_hat_df['parameter'],
                dtype=pd.CategoricalDtype(
                    categories=r_hat_df['parameter'].unique(),
                    ordered=True
                )
            )
            
            # Use newer seaborn API with explicit categorical ordering
            sns.barplot(
                data=r_hat_df,
                x='r_hat',
                y='parameter',
                errorbar=None,
                order=r_hat_df['parameter'].unique()  # Explicit ordering
            )
            plt.title('R-hat Values by Parameter')
            plt.tight_layout()
            
            # Assertions
            assert len(r_hat_df) > 0, "No valid R-hat values found"
            assert all(r_hat_df['r_hat'] < 1.1), "Some parameters show poor convergence"
            
        except Exception as e:
            pytest.fail(f"Model convergence test failed: {str(e)}")
    
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
        try:
            results = []
            model_params = config_loader.get('model', {}).get('parameters', {})
            base_chains = model_params.get('chains', 4)
            base_draws = model_params.get('draws', 2000)
            base_tune = model_params.get('tune', 1000)
            
            # Modify the configs to have more draws than chains
            configs = [
                {
                    'draws': 1000,
                    'chains': 2,
                    'tune': 500
                },
                {
                    'draws': 2000,
                    'chains': 2,
                    'tune': 1000
                },
                {
                    'draws': 4000,
                    'chains': 2,
                    'tune': 2000
                }
            ]
            
            for config in configs:
                model = HierarchicalCLVModel(config_loader)
                model.build_model(sample_model_data)
                
                start_time = time.time()
                trace = model.sample(
                    chains=config['chains'],
                    draws=config['draws'],
                    tune=config['tune'],

                )
                sampling_time = time.time() - start_time
                
                total_samples = config['draws'] * config['chains']  # Calculate total samples
                
                # Ensure proper reshaping for ArviZ
                if isinstance(trace, dict):
                    # Reshape the samples to have more draws than chains
                    for param_name, param_samples in trace.items():
                        if isinstance(param_samples, np.ndarray):
                            n_samples = param_samples.shape[0]
                            # Reshape to have more draws than chains
                            n_chains = 2
                            n_draws = n_samples // n_chains
                            trace[param_name] = param_samples.reshape(n_chains, n_draws, -1)
                
                # Calculate ESS more carefully
                try:
                    ess_values = az.ess(trace)
                    if isinstance(ess_values, (xr.Dataset, xr.DataArray)):
                        # Extract numeric values from xarray object
                        ess_nums = []
                        for var_name, var in ess_values.items():
                            if isinstance(var, xr.DataArray):
                                # Convert DataArray to numpy array and flatten
                                var_values = var.values
                                if var_values.size > 0:  # Check if array is not empty
                                    ess_nums.extend(var_values.flatten())
                            elif isinstance(var, np.ndarray):
                                if var.size > 0:  # Check if array is not empty
                                    ess_nums.extend(var.flatten())
                            else:
                                try:
                                    val = float(var)
                                    if not np.isnan(val):
                                        ess_nums.append(val)
                                except (TypeError, ValueError):
                                    continue
                        
                        # Calculate mean ESS if we have valid numbers
                        if ess_nums:
                            mean_ess = float(np.nanmean(ess_nums))
                        else:
                            mean_ess = np.nan
                    else:
                        # Handle case where ess_values is a dict
                        ess_nums = []
                        for val in ess_values.values():
                            if isinstance(val, np.ndarray):
                                if val.size > 0:  # Check if array is not empty
                                    ess_nums.extend(val.flatten())
                            else:
                                try:
                                    val_float = float(val)
                                    if not np.isnan(val_float):
                                        ess_nums.append(val_float)
                                except (TypeError, ValueError):
                                    continue
                        
                        # Calculate mean ESS if we have valid numbers
                        if ess_nums:
                            mean_ess = float(np.nanmean(ess_nums))
                        else:
                            mean_ess = np.nan

                except Exception as e:
                    print(f"Warning: Error calculating ESS: {str(e)}")
                    mean_ess = np.nan

                # Calculate R-hat values carefully
                try:
                    r_hat_values = az.rhat(trace)
                    if isinstance(r_hat_values, (xr.Dataset, xr.DataArray)):
                        # Convert xarray values to numpy array and handle NaN values
                        if isinstance(r_hat_values, xr.Dataset):
                            r_hat_array = np.array([v.values for v in r_hat_values.values()])
                        else:
                            r_hat_array = r_hat_values.values
                        
                        # Flatten array and remove NaN values
                        r_hat_array = r_hat_array.flatten()
                        r_hat_array = r_hat_array[~np.isnan(r_hat_array)]
                        
                        # Calculate max R-hat if we have valid values
                        max_r_hat = float(np.max(r_hat_array)) if r_hat_array.size > 0 else np.nan
                    else:
                        # Handle dictionary case
                        r_hat_array = np.array([
                            v.flatten() if isinstance(v, np.ndarray) else v 
                            for v in r_hat_values.values()
                        ])
                        r_hat_array = r_hat_array[~np.isnan(r_hat_array)]
                        max_r_hat = float(np.max(r_hat_array)) if r_hat_array.size > 0 else np.nan
                except Exception as e:
                    print(f"Warning: Error calculating R-hat: {str(e)}")
                    max_r_hat = np.nan

                results.append({
                    'config': f"{config['draws']}d_{config['chains']}c",
                    'time_per_sample': sampling_time / total_samples,
                    'ess_per_second': mean_ess / sampling_time if not np.isnan(mean_ess) else np.nan,
                    'max_r_hat': max_r_hat
                })
            
            # Plot efficiency metrics
            self._plot_efficiency_comparison(results)
            self.save_plot("sampling_efficiency", "efficiency")
            
        except Exception as e:
            self.update_metadata(
                status='failed',
                notes=f"Sampling efficiency test failed: {str(e)}"
            )
            raise
    
    def _plot_efficiency_comparison(self, results):
        """Create efficiency comparison plots"""
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Fix categorical dtype warning
        df['config'] = pd.Series(
            df['config'],
            dtype=pd.CategoricalDtype(
                categories=df['config'].unique(),
                ordered=True
            )
        )
        
        # Use explicit groupby with observed=True
        grouped_df = df.groupby('config', observed=True).agg({
            'time_per_sample': 'mean',
            'ess_per_second': 'mean',
            'max_r_hat': 'mean'
        }).reset_index()
        
        # Update barplots with explicit ordering
        for ax, column, title in zip(
            axes,
            ['time_per_sample', 'ess_per_second', 'max_r_hat'],
            ['Time per Sample', 'ESS per Second', 'Maximum R-hat']
        ):
            sns.barplot(
                data=grouped_df,
                x='config',
                y=column,
                ax=ax,
                errorbar=None,
                order=df['config'].unique()  # Explicit ordering
            )
            ax.set_title(title)
            if column == 'max_r_hat':
                ax.axhline(y=1.1, color='r', linestyle='--')
        
        plt.tight_layout()
    
    def test_prior_sensitivity(self, sample_model_data, config_loader):
        """Test and visualize prior sensitivity"""
        model_params = config_loader.get('model', {}).get('parameters', {})
        chains = model_params.get('chains', 4)
        draws = model_params.get('draws', 2000)
        tune = model_params.get('tune', 1000)
        
        results = []
        prior_configs = [
            {'alpha_shape': 1.0, 'beta_shape': 1.0},
            {'alpha_shape': 0.5, 'beta_shape': 0.5},
            {'alpha_shape': 2.0, 'beta_shape': 2.0}
        ]
        
        try:
            for prior_config in prior_configs:
                model_config = config_loader.get('model', {})
                if isinstance(model_config, dict):
                    model_config = model_config.copy()
                else:
                    model_config = {}
                    
                model_config['hyperparameters'] = {
                    'prior_settings': prior_config
                }
                
                # Build and sample model with new priors
                model = HierarchicalCLVModel(config_loader)
                model.update_config(model_config)
                model.build_model(sample_model_data)
                trace_dict = model.sample(draws=draws, tune=tune, chains=chains)
                
                # Extract samples - assuming beta is the second parameter
                samples = trace_dict.get('samples', [])
                if not isinstance(samples, np.ndarray):
                    samples = np.array(samples)
                
                # Take mean of the second parameter (beta)
                beta_values = samples[..., 1] if samples.ndim > 1 else samples
                posterior_mean = float(np.mean(beta_values))
                
                results.append({
                    'prior_config': prior_config,
                    'posterior_mean': posterior_mean
                })
                
            assert len(results) == len(prior_configs)
            
        except Exception as e:
            pytest.fail(f"Prior sensitivity test failed: {str(e)}")
    
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
        model_params = config_loader.get('model', {}).get('parameters', {})
        chains = model_params.get('chains', 4)
        draws = model_params.get('draws', 2000)
        tune = model_params.get('tune', 1000)
        
        n_runs = 3
        posterior_means = []
        
        try:
            for _ in range(n_runs):
                model = HierarchicalCLVModel(config_loader)
                model.build_model(sample_model_data)
                trace_dict = model.sample(draws=draws, tune=tune, chains=chains)
                
                # Extract posterior means for key parameters
                run_means = {}
                for param in ['alpha', 'beta', 'r', 'lambda']:
                    param_key = next((k for k in trace_dict.keys() if param in k.lower()), None)
                    if param_key:
                        run_means[param] = float(np.mean(trace_dict[param_key]))
                        
                posterior_means.append(run_means)
            
            # Convert to DataFrame for analysis
            results_df = pd.DataFrame(posterior_means)
            
            # Calculate coefficient of variation (CV) for each parameter
            cv_threshold = 0.1  # 10% variation threshold
            for param in results_df.columns:
                cv = results_df[param].std() / results_df[param].mean()
                assert cv < cv_threshold, f"Parameter {param} shows high variation (CV={cv:.3f})"
                
        except Exception as e:
            pytest.fail(f"Model stability test failed: {str(e)}")
    
    @property
    def convergence_dir(self):
        return self.run_dir / 'convergence'

    @property
    def efficiency_dir(self):
        return self.run_dir / 'efficiency'

    @property
    def sensitivity_dir(self):
        return self.run_dir / 'sensitivity' 