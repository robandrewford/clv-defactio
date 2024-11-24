import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox 
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import json
from typing import Dict, Any

class ModelStatisticalAnalyzer:
    """Analyzes statistical properties of model runs"""

    def __init__(self):
        self.diagnostic_dir = Path("diagnostics")
        self.runs_data = self._load_runs_data()

    def _load_runs_data(self) -> list:
        """Load data from all model runs"""
        runs = []
        for run_dir in self.diagnostic_dir.glob("model_run_*"):
            metadata_file = run_dir / 'metadata' / 'run_info.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    runs.append(json.load(f))
        return runs

    def run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical tests on model metrics"""
        if not self.runs_data:
            return {}

        # Extract time series data
        r_hats = [run['convergence_metrics']['max_r_hat'] for run in self.runs_data]
        durations = [run['duration'] for run in self.runs_data]

        results = {
            'convergence_tests': {
                'r_hat': self._test_stationarity(r_hats),
            },
            'stationarity_tests': {
                'duration': self._test_stationarity(durations),
            }
        }

        return results

    def _test_stationarity(self, series: list) -> Dict[str, float]:
        """Run stationarity tests on a time series"""
        # Convert to numpy array
        series = np.array(series)
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series)
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(series, lags=[10], return_df=True)
        
        return {
            'adf_statistic': float(adf_result[0]),
            'adf_pvalue': float(adf_result[1]),
            'lb_statistic': float(lb_result['lb_stat'].iloc[0]),
            'lb_pvalue': float(lb_result['lb_pvalue'].iloc[0]),
            'trend_pvalue': float(adf_result[1])  # Using ADF p-value as trend indicator
        }

    def generate_statistical_report(self) -> str:
        """Generate a text report of statistical analysis"""
        if not self.runs_data:
            return "No model runs data available for analysis."

        results = self.run_statistical_tests()
        
        report = ["Statistical Analysis Report", "=" * 30, ""]
        
        # Convergence Analysis
        report.append("Convergence Analysis:")
        report.append("-" * 20)
        r_hat_tests = results['convergence_tests']['r_hat']
        report.append(f"R-hat Trend Test p-value: {r_hat_tests['trend_pvalue']:.4f}")
        report.append(f"R-hat Stationarity Test p-value: {r_hat_tests['adf_pvalue']:.4f}")
        report.append("")
        
        # Performance Analysis
        report.append("Performance Analysis:")
        report.append("-" * 20)
        duration_tests = results['stationarity_tests']['duration']
        report.append(f"Duration Stationarity Test p-value: {duration_tests['adf_pvalue']:.4f}")
        report.append(f"Duration Autocorrelation Test p-value: {duration_tests['lb_pvalue']:.4f}")
        
        return "\n".join(report)

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for key metrics"""
        if not self.runs_data:
            return {}

        metrics = {
            'r_hat': [run['convergence_metrics']['max_r_hat'] for run in self.runs_data],
            'duration': [run['duration'] for run in self.runs_data]
        }

        summary = {}
        for metric, values in metrics.items():
            values = np.array(values)
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'cv': float(np.std(values) / np.mean(values))  # coefficient of variation
            }

        return summary 