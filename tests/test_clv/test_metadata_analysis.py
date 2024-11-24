import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MetadataAnalysis:
    """Analysis results for model run metadata"""
    run_summary: pd.DataFrame
    convergence_trends: Dict[str, List[float]]
    performance_trends: Dict[str, List[float]]
    config_impact: Dict[str, float]
    hardware_impact: Dict[str, float]
    recommendations: List[str]

class ModelMetadataAnalyzer:
    """Analyzer for model run metadata"""
    
    def __init__(self, diagnostic_dir: str = "diagnostics"):
        self.diagnostic_dir = Path(diagnostic_dir)
        self.runs_data = self._load_runs_data()
        
    def _load_runs_data(self) -> pd.DataFrame:
        """Load metadata from all runs into DataFrame"""
        runs = []
        for run_dir in self.diagnostic_dir.glob("model_run_*"):
            metadata_file = run_dir / 'metadata' / 'run_info.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    run_data = json.load(f)
                    runs.append(run_data)
        
        return pd.DataFrame(runs)
    
    def analyze_runs(self) -> MetadataAnalysis:
        """Perform comprehensive analysis of run metadata"""
        # Basic summary
        run_summary = self._create_run_summary()
        
        # Analyze trends
        convergence_trends = self._analyze_convergence_trends()
        performance_trends = self._analyze_performance_trends()
        
        # Analyze impacts
        config_impact = self._analyze_config_impact()
        hardware_impact = self._analyze_hardware_impact()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            convergence_trends,
            performance_trends,
            config_impact
        )
        
        return MetadataAnalysis(
            run_summary=run_summary,
            convergence_trends=convergence_trends,
            performance_trends=performance_trends,
            config_impact=config_impact,
            hardware_impact=hardware_impact,
            recommendations=recommendations
        )
    
    def _create_run_summary(self) -> pd.DataFrame:
        """Create summary statistics for runs"""
        summary = pd.DataFrame({
            'total_runs': len(self.runs_data),
            'successful_runs': sum(self.runs_data['status'] == 'completed'),
            'failed_runs': sum(self.runs_data['status'] == 'failed'),
            'avg_duration': self.runs_data['duration'].mean(),
            'avg_r_hat': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('max_r_hat', np.nan)
            ).mean(),
            'avg_ess': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('mean_ess', np.nan)
            ).mean()
        }, index=[0])
        
        return summary
    
    def _analyze_convergence_trends(self) -> Dict[str, List[float]]:
        """Analyze convergence metric trends over time"""
        trends = {
            'r_hat_trend': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('max_r_hat', np.nan)
            ).tolist(),
            'ess_trend': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('mean_ess', np.nan)
            ).tolist()
        }
        
        return trends
    
    def _analyze_performance_trends(self) -> Dict[str, List[float]]:
        """Analyze performance metric trends"""
        return {
            'duration_trend': self.runs_data['duration'].tolist(),
            'memory_trend': self.runs_data['performance_metrics'].apply(
                lambda x: float(x.get('memory_peak', '0').rstrip('GB'))
            ).tolist()
        }
    
    def _analyze_config_impact(self) -> Dict[str, float]:
        """Analyze impact of different configurations on performance"""
        impacts = {}
        
        # Group by config hash and analyze performance
        for config_hash in self.runs_data['config_hash'].unique():
            config_runs = self.runs_data[self.runs_data['config_hash'] == config_hash]
            impacts[config_hash] = {
                'avg_duration': config_runs['duration'].mean(),
                'avg_r_hat': config_runs['convergence_metrics'].apply(
                    lambda x: x.get('max_r_hat', np.nan)
                ).mean()
            }
            
        return impacts
    
    def _analyze_hardware_impact(self) -> Dict[str, float]:
        """Analyze impact of hardware on performance"""
        impacts = {}
        
        # Group by GPU type and analyze performance
        for gpu in self.runs_data['hardware_info'].apply(lambda x: x.get('gpu')).unique():
            gpu_runs = self.runs_data[
                self.runs_data['hardware_info'].apply(lambda x: x.get('gpu')) == gpu
            ]
            impacts[str(gpu)] = {
                'avg_duration': gpu_runs['duration'].mean(),
                'success_rate': (
                    sum(gpu_runs['status'] == 'completed') / len(gpu_runs)
                )
            }
            
        return impacts
    
    def _generate_recommendations(
        self,
        convergence_trends: Dict[str, List[float]],
        performance_trends: Dict[str, List[float]],
        config_impact: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check convergence trends
        recent_r_hats = convergence_trends['r_hat_trend'][-5:]
        if any(r > 1.1 for r in recent_r_hats):
            recommendations.append(
                "Consider increasing number of samples or tuning steps "
                "to improve convergence"
            )
        
        # Check performance trends
        recent_durations = performance_trends['duration_trend'][-5:]
        if np.mean(recent_durations) > np.mean(performance_trends['duration_trend']):
            recommendations.append(
                "Recent runs showing increased duration. "
                "Consider optimizing configuration or hardware resources"
            )
        
        # Find best configuration
        best_config = min(
            config_impact.items(),
            key=lambda x: x[1]['avg_r_hat']
        )[0]
        recommendations.append(f"Best performing configuration: {best_config}")
        
        return recommendations
    
    def plot_analysis(self, save_dir: Optional[Path] = None):
        """Create visualization of analysis results"""
        save_dir = save_dir or self.diagnostic_dir / 'analysis'
        save_dir.mkdir(exist_ok=True)
        
        # Plot convergence trends
        self._plot_convergence_trends(save_dir)
        
        # Plot performance trends
        self._plot_performance_trends(save_dir)
        
        # Plot configuration comparison
        self._plot_config_comparison(save_dir)
        
        # Plot hardware comparison
        self._plot_hardware_comparison(save_dir)
    
    def _plot_convergence_trends(self, save_dir: Path):
        """Plot convergence metric trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # R-hat trend
        r_hats = self.runs_data['convergence_metrics'].apply(
            lambda x: x.get('max_r_hat', np.nan)
        )
        ax1.plot(r_hats, marker='o')
        ax1.axhline(y=1.1, color='r', linestyle='--', label='Threshold')
        ax1.set_title('Maximum R-hat Over Time')
        ax1.set_ylabel('R-hat')
        
        # ESS trend
        ess = self.runs_data['convergence_metrics'].apply(
            lambda x: x.get('mean_ess', np.nan)
        )
        ax2.plot(ess, marker='o')
        ax2.set_title('Mean ESS Over Time')
        ax2.set_ylabel('ESS')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'convergence_trends.png')
        plt.close()
    
    def _plot_performance_trends(self, save_dir: Path):
        """Plot performance metric trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Duration trend
        ax1.plot(self.runs_data['duration'], marker='o')
        ax1.set_title('Run Duration Over Time')
        ax1.set_ylabel('Seconds')
        
        # Memory trend
        memory = self.runs_data['performance_metrics'].apply(
            lambda x: float(x.get('memory_peak', '0').rstrip('GB'))
        )
        ax2.plot(memory, marker='o')
        ax2.set_title('Memory Usage Over Time')
        ax2.set_ylabel('GB')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_trends.png')
        plt.close() 