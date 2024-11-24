import pytest
import time
import psutil
import numpy as np
import pandas as pd
import git
import memory_profiler
from memory_profiler import profile
from datetime import datetime, timedelta
from src.pipeline.clv import (
    CLVDataPreprocessor,
    CustomerSegmentation,
    HierarchicalCLVModel,
    CLVModelRegistry
)

class TestCLVPipelinePerformance:
    """Performance tests for CLV pipeline components"""
    
    @pytest.mark.performance
    @pytest.mark.parametrize('n_records', [1000, 10000, 100000])
    def test_preprocessing_performance(self, n_records, config_loader):
        """Test preprocessing performance with different data sizes"""
        # Generate test data
        data = self._generate_test_data(n_records)
        preprocessor = CLVDataPreprocessor(config_loader)
        
        # Measure execution time
        start_time = time.time()
        processed_data = preprocessor.process_data(data)
        execution_time = time.time() - start_time
        
        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Log metrics
        print(f"\nPreprocessing Performance (n={n_records}):")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Memory Usage: {memory_usage:.2f} MB")
        print(f"Records/Second: {n_records/execution_time:.2f}")
        
        assert execution_time < self._get_time_threshold(n_records)
        
    @pytest.mark.performance
    @pytest.mark.parametrize('n_customers', [100, 1000, 5000])
    def test_model_training_performance(self, n_customers, config_loader):
        """Test model training performance with different customer counts"""
        # Generate customer data
        data = self._generate_customer_data(n_customers)
        model = HierarchicalCLVModel(config_loader)
        
        # Measure training time
        start_time = time.time()
        model.build_model(data)
        model.sample(draws=100, tune=50, chains=2)  # Reduced for testing
        training_time = time.time() - start_time
        
        # Get GPU memory if available
        gpu_memory = self._get_gpu_memory_usage() if hasattr(model, 'gpu_enabled') else 0
        
        print(f"\nModel Training Performance (n={n_customers}):")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"GPU Memory: {gpu_memory:.2f} MB")
        
        assert training_time < self._get_training_threshold(n_customers)
        
    @pytest.mark.performance
    def test_memory_leaks(self, config_loader):
        """Test for memory leaks during repeated operations"""
        initial_memory = psutil.Process().memory_info().rss
        
        for _ in range(5):  # Repeat operations
            # Generate new data each time
            data = self._generate_test_data(1000)
            
            # Run pipeline components
            preprocessor = CLVDataPreprocessor(config_loader)
            processed_data = preprocessor.process_data(data)
            
            segmenter = CustomerSegmentation(config_loader)
            segmented_data, model_data = segmenter.create_segments(processed_data)
            
            model = HierarchicalCLVModel(config_loader)
            model.build_model(model_data)
            model.sample(draws=50, tune=25, chains=2)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"\nMemory Growth: {memory_growth:.2f} MB")
        assert memory_growth < 100  # Allow for some growth, but not excessive
        
    @pytest.mark.performance
    def test_concurrent_operations(self, config_loader):
        """Test performance with concurrent operations"""
        import concurrent.futures
        
        def run_pipeline(size):
            data = self._generate_test_data(size)
            preprocessor = CLVDataPreprocessor(config_loader)
            return preprocessor.process_data(data)
        
        sizes = [1000, 2000, 3000]
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_pipeline, size) for size in sizes]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        total_time = time.time() - start_time
        
        print(f"\nConcurrent Processing Time: {total_time:.2f} seconds")
        assert len(results) == len(sizes)
        
    def _generate_test_data(self, n_records):
        """Generate test transaction data"""
        return pd.DataFrame({
            'customer_id': np.random.randint(1, 1000, n_records),
            'transaction_date': [
                datetime.now() - timedelta(days=np.random.randint(0, 365))
                for _ in range(n_records)
            ],
            'transaction_amount': np.random.lognormal(3, 1, n_records)
        })
        
    def _generate_customer_data(self, n_customers):
        """Generate test customer data for modeling"""
        return {
            'frequency': np.random.poisson(5, n_customers),
            'recency': np.random.randint(0, 365, n_customers),
            'monetary_value': np.random.lognormal(3, 1, n_customers),
            'T': np.random.randint(100, 1000, n_customers),
            'segment_ids': np.zeros(n_customers)
        }
        
    def _get_time_threshold(self, n_records):
        """Get execution time threshold based on data size"""
        if n_records <= 1000:
            return 5  # seconds
        elif n_records <= 10000:
            return 30
        return 180
        
    def _get_training_threshold(self, n_customers):
        """Get training time threshold based on customer count"""
        if n_customers <= 100:
            return 60  # seconds
        elif n_customers <= 1000:
            return 300
        return 900
        
    def _get_gpu_memory_usage(self):
        """Get GPU memory usage if available"""
        try:
            import torch
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        except:
            return 0 