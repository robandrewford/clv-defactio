import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ModelStatisticalAnalyzer:
    """Statistical analysis of model performance and convergence"""
    
    def __init__(self, diagnostic_dir: str = "diagnostics"):
        self.diagnostic_dir = Path(diagnostic_dir)
        self.runs_data = self._load_runs_data()
        self.significance_level = 0.05
        
    def run_statistical_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive statistical analysis"""
        results = {
            'convergence_tests': self._test_convergence(),
            'stationarity_tests': self._test_stationarity(),
            'performance_tests': self._test_performance(),
            'distribution_tests': self._test_distributions(),
            'correlation_tests': self._test_correlations()
        }
        
        return results
    
    def _test_convergence(self) -> Dict[str, Dict[str, float]]:
        """Statistical tests for convergence metrics"""
        results = {}
        
        # Extract convergence metrics
        r_hats = self.runs_data['convergence_metrics'].apply(
            lambda x: x.get('max_r_hat', np.nan)
        )
        ess_values = self.runs_data['convergence_metrics'].apply(
            lambda x: x.get('mean_ess', np.nan)
        )
        
        # Test for trend in R-hat values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mann_kendall_rhat = stats.kendalltau(
                np.arange(len(r_hats)),
                r_hats.dropna()
            )
        
        # Test for autocorrelation in ESS
        ljung_box_ess = acorr_ljungbox(
            ess_values.dropna(),
            lags=[10],
            return_df=True
        )
        
        results['r_hat'] = {
            'trend_statistic': mann_kendall_rhat.statistic,
            'trend_pvalue': mann_kendall_rhat.pvalue,
            'mean': float(r_hats.mean()),
            'std': float(r_hats.std())
        }
        
        results['ess'] = {
            'autocorr_statistic': float(ljung_box_ess['lb_stat'].iloc[0]),
            'autocorr_pvalue': float(ljung_box_ess['lb_pvalue'].iloc[0]),
            'mean': float(ess_values.mean()),
            'std': float(ess_values.std())
        }
        
        return results
    
    def _test_stationarity(self) -> Dict[str, Dict[str, float]]:
        """Test for stationarity in performance metrics"""
        results = {}
        
        # Test duration stationarity
        duration_adf = adfuller(
            self.runs_data['duration'].dropna(),
            regression='ct'
        )
        
        # Test memory usage stationarity
        memory_usage = self.runs_data['performance_metrics'].apply(
            lambda x: float(x.get('memory_peak', '0').rstrip('GB'))
        )
        memory_adf = adfuller(
            memory_usage.dropna(),
            regression='ct'
        )
        
        results['duration'] = {
            'adf_statistic': duration_adf[0],
            'pvalue': duration_adf[1],
            'critical_values': duration_adf[4]
        }
        
        results['memory'] = {
            'adf_statistic': memory_adf[0],
            'pvalue': memory_adf[1],
            'critical_values': memory_adf[4]
        }
        
        return results
    
    def _test_performance(self) -> Dict[str, Dict[str, float]]:
        """Statistical tests for performance metrics"""
        results = {}
        
        # Group runs by configuration
        for config_hash in self.runs_data['config_hash'].unique():
            config_runs = self.runs_data[
                self.runs_data['config_hash'] == config_hash
            ]
            
            # Compare performance metrics
            other_runs = self.runs_data[
                self.runs_data['config_hash'] != config_hash
            ]
            
            # T-test for duration
            t_stat, p_val = stats.ttest_ind(
                config_runs['duration'],
                other_runs['duration']
            )
            
            results[config_hash] = {
                'duration_tstat': t_stat,
                'duration_pvalue': p_val,
                'effect_size': self._compute_cohens_d(
                    config_runs['duration'],
                    other_runs['duration']
                )
            }
            
        return results
    
    def _test_distributions(self) -> Dict[str, Dict[str, float]]:
        """Test distribution properties of metrics"""
        results = {}
        
        # Test normality of convergence metrics
        r_hats = self.runs_data['convergence_metrics'].apply(
            lambda x: x.get('max_r_hat', np.nan)
        )
        shapiro_rhat = stats.shapiro(r_hats.dropna())
        
        # Test normality of performance metrics
        shapiro_duration = stats.shapiro(self.runs_data['duration'].dropna())
        
        results['r_hat_normality'] = {
            'statistic': shapiro_rhat.statistic,
            'pvalue': shapiro_rhat.pvalue
        }
        
        results['duration_normality'] = {
            'statistic': shapiro_duration.statistic,
            'pvalue': shapiro_duration.pvalue
        }
        
        return results
    
    def _test_correlations(self) -> Dict[str, Dict[str, float]]:
        """Test correlations between metrics"""
        results = {}
        
        # Extract metrics
        metrics = pd.DataFrame({
            'duration': self.runs_data['duration'],
            'r_hat': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('max_r_hat', np.nan)
            ),
            'ess': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('mean_ess', np.nan)
            )
        })
        
        # Compute correlation matrix
        corr_matrix = metrics.corr()
        
        # Test significance of correlations
        for col1 in metrics.columns:
            for col2 in metrics.columns:
                if col1 < col2:  # Avoid duplicate tests
                    correlation = stats.pearsonr(
                        metrics[col1].dropna(),
                        metrics[col2].dropna()
                    )
                    results[f"{col1}_vs_{col2}"] = {
                        'correlation': correlation.statistic,
                        'pvalue': correlation.pvalue
                    }
        
        return results
    
    def _compute_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        
        # Pooled standard deviation
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (group1.mean() - group2.mean()) / pooled_se
    
    def generate_statistical_report(self) -> str:
        """Generate detailed statistical analysis report"""
        results = self.run_statistical_tests()
        
        report = ["Statistical Analysis Report", "=" * 25, ""]
        
        # Convergence Analysis
        report.extend([
            "Convergence Analysis",
            "-" * 20,
            f"R-hat Trend Test (Mann-Kendall):",
            f"  Statistic: {results['convergence_tests']['r_hat']['trend_statistic']:.3f}",
            f"  p-value: {results['convergence_tests']['r_hat']['trend_pvalue']:.3f}",
            "",
            f"ESS Autocorrelation Test (Ljung-Box):",
            f"  Statistic: {results['convergence_tests']['ess']['autocorr_statistic']:.3f}",
            f"  p-value: {results['convergence_tests']['ess']['autocorr_pvalue']:.3f}",
            ""
        ])
        
        # Performance Analysis
        report.extend([
            "Performance Analysis",
            "-" * 20
        ])
        
        for config, stats in results['performance_tests'].items():
            report.extend([
                f"Configuration {config}:",
                f"  Duration t-statistic: {stats['duration_tstat']:.3f}",
                f"  p-value: {stats['duration_pvalue']:.3f}",
                f"  Effect size: {stats['effect_size']:.3f}",
                ""
            ])
        
        # Add recommendations
        report.extend([
            "Recommendations",
            "-" * 20
        ])
        
        # Add recommendations based on statistical tests
        if results['convergence_tests']['r_hat']['trend_pvalue'] < self.significance_level:
            report.append("- Significant trend in R-hat values detected. Review convergence settings.")
            
        if results['stationarity_tests']['duration']['pvalue'] < self.significance_level:
            report.append("- Performance metrics show non-stationarity. Consider stabilizing factors.")
        
        return "\n".join(report) 