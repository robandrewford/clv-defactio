import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.pipeline.clv import (
    CLVDataPreprocessor,
    CustomerSegmentation,
    HierarchicalCLVModel,
    CLVModelRegistry
)
from src.pipeline.clv.base import BaseProcessor, BaseModel
from src.pipeline.clv.config import CLVConfigLoader

@pytest.mark.integration
class TestCLVPipelineIntegration:
    """Integration tests for the full CLV pipeline"""
    
    def test_end_to_end_pipeline(
        self,
        sample_transaction_data,
        config_loader,
        mock_gcs_bucket,
        mock_bigquery_client
    ):
        """Test full pipeline from data preprocessing to model deployment"""
        try:
            # Use the provided config_loader instead of creating new one
            preprocessor = CLVDataPreprocessor(config_loader)
            
            # 1. Preprocess Data
            assert isinstance(preprocessor, BaseProcessor)
            processed_data = preprocessor.process_data(sample_transaction_data)
            
            # 2. Create Segments
            segmenter = CustomerSegmentation(config_loader)
            assert isinstance(segmenter, BaseProcessor)
            segmented_data, model_data = segmenter.create_segments(processed_data)
            
            # 3. Train Model
            model = HierarchicalCLVModel(config_loader)
            assert isinstance(model, BaseModel)
            model.build_model(model_data)
            
            # Train with small sample size for testing
            trace = model.sample(draws=50, tune=25, chains=2)
            assert trace is not None
            
            # 4. Generate Predictions
            predictions = model.predict(
                model_data,
                prediction_period=30,
                samples=50
            )
            
            assert len(predictions) == len(processed_data['customer_id'].unique())
            assert 'predicted_value' in predictions.columns
            
            # 5. Save Model
            registry = CLVModelRegistry(config_loader)
            metrics = {
                'rmse': np.random.rand(),
                'mae': np.random.rand(),
                'r2': np.random.rand()
            }
            
            version = registry.save_model(model, metrics)
            assert version is not None
            
            # 6. Load Model
            loaded_model, loaded_metrics = registry.load_model(version)
            assert loaded_model is not None
            assert loaded_metrics == metrics
            
        except Exception as e:
            pytest.fail(f"Pipeline integration test failed: {str(e)}")
            
    def test_incremental_update(
        self,
        sample_transaction_data,
        config_loader,
        mock_gcs_bucket
    ):
        """Test incremental model update with new data"""
        # Ensure we have data on both sides of the date split with enough variation
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', periods=len(sample_transaction_data))
        values = np.array([
            [100, 1, 100],  # Different RFM values for each transaction
            [200, 2, 150],
            [300, 3, 200],
            [400, 4, 250]
        ])
        
        sample_transaction_data['transaction_date'] = dates
        sample_transaction_data['frequency'] = values[:, 1]
        sample_transaction_data['monetary'] = values[:, 2]
        sample_transaction_data['recency'] = np.linspace(10, 100, len(sample_transaction_data))
        
        # 1. Train initial model
        preprocessor = CLVDataPreprocessor(config_loader)
        segmenter = CustomerSegmentation(config_loader)
        
        initial_data = sample_transaction_data[
            sample_transaction_data['transaction_date'] < '2023-06-01'
        ]
        
        processed_data = preprocessor.process_data(initial_data)
        segmented_data, model_data = segmenter.create_segments(processed_data)
        
        model = HierarchicalCLVModel(config_loader)
        model.build_model(model_data)
        model.sample(draws=50, tune=25, chains=2)
        
        # 2. Process new data
        new_data = sample_transaction_data[
            sample_transaction_data['transaction_date'] >= '2023-06-01'
        ]
        
        new_processed = preprocessor.process_data(new_data)
        new_segmented, new_model_data = segmenter.create_segments(new_processed)
        
        # 3. Update predictions
        updated_predictions = model.predict(
            new_model_data,
            prediction_period=30,
            samples=50
        )
        
        assert len(updated_predictions) == len(new_processed['customer_id'].unique())
        
    def test_error_handling(
        self,
        sample_transaction_data,
        config_loader
    ):
        """Test error handling across pipeline components"""
        # 1. Test missing data handling
        bad_data = sample_transaction_data.copy()
        # Only set some values to NaN, not all
        bad_data.loc[0:1, 'transaction_amount'] = np.nan
        
        preprocessor = CLVDataPreprocessor(config_loader)
        cleaned_data = preprocessor.process_data(bad_data)
        
        # Check that the data was processed but NaN values were handled
        assert not cleaned_data['transaction_amount'].isna().any()
        
        # Also verify that we have reasonable values
        assert (cleaned_data['transaction_amount'] > 0).all()
        
        # 2. Test invalid segment configuration
        segmenter = CustomerSegmentation(config_loader)
        # Fix: Need to process the data first
        processed_data = preprocessor.process_data(sample_transaction_data)
        processed_data['frequency'] = -1  # Invalid values
        
        with pytest.raises(ValueError):
            segmenter.create_segments(processed_data)
            
        # 3. Test model validation
        model = HierarchicalCLVModel(config_loader)
        
        with pytest.raises(ValueError):
            model.predict(None, prediction_period=30)  # Invalid input
            
    @pytest.mark.parametrize('n_customers', [10, 50, 100])
    def test_scalability(
        self,
        n_customers,
        config_loader,
        mock_gcs_bucket,
        mock_bigquery_client
    ):
        """Test pipeline with different data sizes"""
        # Generate scaled test data with all required columns
        categories = ['Electronics', 'Clothing', 'Food', 'Home']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD']
        
        test_data = pd.DataFrame({
            'customer_id': range(n_customers),
            'transaction_date': [
                datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
                for _ in range(n_customers)
            ],
            'transaction_amount': np.random.lognormal(3, 1, n_customers),
            'monetary': np.random.lognormal(3, 1, n_customers),
            'recency': np.random.randint(1, 365, n_customers),  # Added recency
            'frequency': np.random.randint(1, 10, n_customers), # Added frequency
            'transaction_id': range(n_customers),
            'category': np.random.choice(categories, n_customers),
            'brand': np.random.choice(brands, n_customers),
            'channel': np.random.choice(['online', 'store'], n_customers),
            'sms_active': np.random.randint(0, 2, n_customers),
            'email_active': np.random.randint(0, 2, n_customers),
            'is_loyalty_member': np.random.randint(0, 2, n_customers),
            'loyalty_points': np.random.randint(0, 1000, n_customers),
            'transaction_amount_std': np.random.rand(n_customers)
        })
        
        # Run pipeline with test mode
        preprocessor = CLVDataPreprocessor(config_loader, test_mode=True)
        processed_data = preprocessor.process_data(test_data)
        
        segmenter = CustomerSegmentation(config_loader)
        segmented_data, model_data = segmenter.create_segments(processed_data)
        
        model = HierarchicalCLVModel(config_loader)
        model.build_model(model_data)
        
        assert len(segmented_data) == n_customers 