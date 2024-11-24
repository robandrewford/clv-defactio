import pytest
import pandas as pd
import numpy as np
from src.pipeline.clv import (
    CLVConfigLoader,
    CLVDataPreprocessor,
    CustomerSegmentation,
    HierarchicalCLVModel,
    CLVModelRegistry
)

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
            # 1. Preprocess Data
            preprocessor = CLVDataPreprocessor(config_loader)
            processed_data = preprocessor.process_data(sample_transaction_data)
            
            assert not processed_data.empty
            assert 'frequency' in processed_data.columns
            assert 'recency' in processed_data.columns
            
            # 2. Create Segments
            segmenter = CustomerSegmentation(config_loader)
            segmented_data, model_data = segmenter.create_segments(processed_data)
            
            assert 'segment_ids' in model_data
            assert 'RFM_score' in segmented_data.columns
            
            # 3. Train Model
            model = HierarchicalCLVModel(config_loader)
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
        bad_data.loc[0:10, 'transaction_amount'] = np.nan
        
        preprocessor = CLVDataPreprocessor(config_loader)
        processed_data = preprocessor.process_data(bad_data)
        
        assert processed_data['transaction_amount'].isna().sum() == 0
        
        # 2. Test invalid segment configuration
        segmenter = CustomerSegmentation(config_loader)
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
        mock_gcs_bucket
    ):
        """Test pipeline with different data sizes"""
        # Generate scaled test data
        test_data = pd.DataFrame({
            'customer_id': range(n_customers),
            'transaction_date': [
                datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
                for _ in range(n_customers)
            ],
            'transaction_amount': np.random.lognormal(3, 1, n_customers)
        })
        
        # Run pipeline
        preprocessor = CLVDataPreprocessor(config_loader)
        processed_data = preprocessor.process_data(test_data)
        
        segmenter = CustomerSegmentation(config_loader)
        segmented_data, model_data = segmenter.create_segments(processed_data)
        
        model = HierarchicalCLVModel(config_loader)
        model.build_model(model_data)
        
        assert len(segmented_data) == n_customers 