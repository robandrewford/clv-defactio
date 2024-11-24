import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.pipeline.clv.visualization import CLVVisualization
from src.pipeline.clv.base import BaseProcessor
from src.pipeline.clv import HierarchicalCLVModel

class TestCLVVisualization:
    """Test suite for CLV visualization components"""

    def test_visualization_initialization(self, config_loader):
        """Test visualization component initialization"""
        viz = CLVVisualization(config_loader)
        assert isinstance(viz, BaseProcessor)

    def test_trace_plot(self, config_loader, sample_customer_features):
        """Test MCMC trace visualization"""
        # Setup model and generate trace
        model = HierarchicalCLVModel(config_loader)
        data = {
            'frequency': sample_customer_features['frequency'].values,
            'recency': sample_customer_features['recency'].values,
            'monetary_value': sample_customer_features['monetary'].values,
            'T': sample_customer_features['customer_age_days'].values,
            'segment_ids': np.zeros(len(sample_customer_features))
        }
        
        model.build_model(data)
        trace = model.sample(draws=50, tune=25, chains=2)
        
        # Generate visualization
        viz = CLVVisualization(config_loader)
        fig = viz.plot_trace(trace)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_segment_visualization(self, config_loader, sample_customer_features):
        """Test segment visualization"""
        viz = CLVVisualization(config_loader)
        fig = viz.plot_segments(sample_customer_features)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_prediction_intervals(self, config_loader, sample_customer_features):
        """Test prediction interval visualization"""
        # Generate predictions
        model = HierarchicalCLVModel(config_loader)
        data = {
            'frequency': sample_customer_features['frequency'].values,
            'recency': sample_customer_features['recency'].values,
            'monetary_value': sample_customer_features['monetary'].values,
            'T': sample_customer_features['customer_age_days'].values,
            'segment_ids': np.zeros(len(sample_customer_features))
        }
        
        model.build_model(data)
        model.sample(draws=50, tune=25, chains=2)
        predictions = model.predict(data, prediction_period=30, samples=50)
        
        # Generate visualization
        viz = CLVVisualization(config_loader)
        fig = viz.plot_prediction_intervals(predictions)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_diagnostic_plots(self, config_loader, sample_customer_features):
        """Test model diagnostic visualizations"""
        viz = CLVVisualization(config_loader)
        
        # Generate diagnostic data
        diagnostics = {
            'convergence': np.random.rand(10),
            'effective_sample_size': np.random.rand(10),
            'r_hat': np.random.rand(10)
        }
        
        fig = viz.plot_diagnostics(diagnostics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig) 