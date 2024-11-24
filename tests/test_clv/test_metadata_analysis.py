import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import json

@dataclass
class MetadataAnalysis:
    """Analysis results for model runs"""
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
        """Load metadata from all runs"""
        runs = []
        for run_dir in self.diagnostic_dir.glob("model_run_*"):
            metadata_file = run_dir / 'metadata' / 'run_info.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    runs.append(json.load(f))
        return pd.DataFrame(runs)
    
    def analyze_runs(self) -> MetadataAnalysis:
        """Analyze all runs"""
        run_summary = pd.DataFrame({
            'total_runs': len(self.runs_data),
            'successful_runs': sum(self.runs_data['status'] == 'completed'),
            'failed_runs': sum(self.runs_data['status'] == 'failed'),
            'avg_duration': self.runs_data['duration'].mean()
        }, index=[0])
        
        convergence_trends = {
            'r_hat': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('max_r_hat', np.nan)
            ).tolist(),
            'ess': self.runs_data['convergence_metrics'].apply(
                lambda x: x.get('mean_ess', np.nan)
            ).tolist()
        }
        
        performance_trends = {
            'duration': self.runs_data['duration'].tolist()
        }
        
        config_impact = self._analyze_config_impact()
        hardware_impact = self._analyze_hardware_impact()
        recommendations = self._generate_recommendations()
        
        return MetadataAnalysis(
            run_summary=run_summary,
            convergence_trends=convergence_trends,
            performance_trends=performance_trends,
            config_impact=config_impact,
            hardware_impact=hardware_impact,
            recommendations=recommendations
        )
    
    def _analyze_config_impact(self) -> Dict[str, float]:
        """Analyze impact of different configurations"""
        impacts = {}
        for config_hash in self.runs_data['config_hash'].unique():
            config_runs = self.runs_data[self.runs_data['config_hash'] == config_hash]
            impacts[config_hash] = config_runs['duration'].mean()
        return impacts
    
    def _analyze_hardware_impact(self) -> Dict[str, float]:
        """Analyze impact of hardware configurations"""
        impacts = {}
        for gpu in self.runs_data['hardware_info'].apply(lambda x: x.get('gpu')).unique():
            gpu_runs = self.runs_data[
                self.runs_data['hardware_info'].apply(lambda x: x.get('gpu')) == gpu
            ]
            impacts[str(gpu)] = gpu_runs['duration'].mean()
        return impacts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check convergence trends
        recent_r_hats = self.runs_data['convergence_metrics'].apply(
            lambda x: x.get('max_r_hat', np.nan)
        ).tail(5)
        
        if (recent_r_hats > 1.1).any():
            recommendations.append(
                "Consider increasing number of samples or tuning steps"
            )
            
        # Check performance trends
        recent_durations = self.runs_data['duration'].tail(5)
        if recent_durations.mean() > self.runs_data['duration'].mean():
            recommendations.append(
                "Recent runs showing increased duration. Consider optimization."
            )
            
        return recommendations