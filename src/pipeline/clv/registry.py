from typing import Dict, Any, Tuple, Optional
from google.cloud import storage, bigquery
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from .base import BaseModel
import os
import pickle

logger = logging.getLogger(__name__)

class CLVModelRegistry:
    """Handles model versioning, storage, and metadata management"""
    
    def __init__(self, config_loader):
        self.config = config_loader
        self.storage_config = config_loader.pipeline_config.get('storage', {})
        
        # Set default values
        self.storage_type = self.storage_config.get('type', 'local')
        self.storage_path = self.storage_config.get('path', '/tmp/models')
        self.model_prefix = self.storage_config.get('model_prefix', 'models/clv')
        
        # Use local storage for testing
        if self.storage_config.get('model_storage', {}).get('type') == 'local':
            self.storage_type = 'local'
            self.storage_path = self.storage_config['model_storage']['path']
        else:
            self.storage_type = 'gcs'
            self.bucket_name = self.storage_config['gcs']['bucket_name']
            self.model_prefix = self.storage_config['gcs']['model_prefix']
        
    def save_model(
        self,
        model: BaseModel,
        metrics: Dict[str, float],
        version: Optional[str] = None
    ) -> str:
        """Save model and its metadata to registry"""
        try:
            if self.storage_type == 'local':
                # Save locally for testing
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(self.storage_path, f"model_{version}")
                os.makedirs(model_path, exist_ok=True)
                
                # Save model and metrics
                with open(os.path.join(model_path, "model.pkl"), "wb") as f:
                    pickle.dump(model, f)
                with open(os.path.join(model_path, "metrics.json"), "w") as f:
                    json.dump(metrics, f)
                    
                return version
            else:
                # Generate version if not provided
                version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create model path
                model_path = f"{self.model_prefix}/clv_model_{version}"
                
                # Save model file
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.bucket_name)
                
                # Save model
                model_blob = bucket.blob(f"{model_path}/model.joblib")
                with model_blob.open('wb') as f:
                    joblib.dump(model, f)
                    
                # Save metrics
                metrics_blob = bucket.blob(f"{model_path}/metrics.json")
                with metrics_blob.open('w') as f:
                    json.dump(metrics, f)
                    
                # Save to BigQuery for tracking
                self._save_metadata_to_bq(version, metrics)
                
                logger.info(f"Model saved successfully: {model_path}")
                return version
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
            
    def load_model(
        self,
        version: Optional[str] = None
    ) -> tuple[BaseModel, Dict[str, float]]:
        """Load model and its metadata from registry"""
        try:
            if self.storage_type == 'local':
                # Load from local storage
                model_path = os.path.join(self.storage_path, f"model_{version}")
                
                # Load model
                with open(os.path.join(model_path, "model.pkl"), "rb") as f:
                    model = pickle.load(f)
                    
                # Load metrics
                with open(os.path.join(model_path, "metrics.json"), "r") as f:
                    metrics = json.load(f)
                    
                return model, metrics
            else:
                # Get latest version if not specified
                if not version:
                    version = self._get_latest_version()
                    
                model_path = f"{self.model_prefix}/clv_model_{version}"
                
                # Load from GCS
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.bucket_name)
                
                # Load model
                model_blob = bucket.blob(f"{model_path}/model.joblib")
                with model_blob.open('rb') as f:
                    model = joblib.load(f)
                    
                # Load metrics
                metrics_blob = bucket.blob(f"{model_path}/metrics.json")
                with metrics_blob.open('r') as f:
                    metrics = json.load(f)
                    
                return model, metrics
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def _get_latest_version(self) -> str:
        """Get the latest model version"""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.model_prefix)
        versions = []
        
        for blob in blobs:
            if blob.name.endswith('model.joblib'):
                version = blob.name.split('_')[-2]
                versions.append(version)
                
        if not versions:
            raise ValueError("No models found in registry")
            
        return sorted(versions)[-1]
        
    def _save_metadata_to_bq(
        self,
        version: str,
        metrics: Dict[str, float]
    ) -> None:
        """Save model metadata to BigQuery"""
        client = bigquery.Client()
        dataset_id = self.storage_config['bigquery']['dataset_id']
        table_id = self.storage_config['bigquery']['metrics_table']
        
        # Prepare row
        row = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': f"{self.model_prefix}/clv_model_{version}"
        }
        
        # Insert into BigQuery
        table_ref = f"{client.project}.{dataset_id}.{table_id}"
        errors = client.insert_rows_json(table_ref, [row])
        
        if errors:
            logger.error(f"Failed to insert metadata: {errors}")
            raise ValueError(f"Failed to insert metadata: {errors}") 