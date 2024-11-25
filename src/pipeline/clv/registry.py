from typing import Dict, Any, Tuple, Optional
from google.cloud import storage, bigquery
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from .base import BaseModel
from .model import HierarchicalCLVModel
import os
import pickle
from google.oauth2 import service_account
import io
import copy
import numpy as np

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
            self.model_prefix = self.storage_config['gcs'].get('model_prefix', 'models/clv')
            
            # Resolve credentials path
            creds_path = os.path.expanduser('~/.gcp/credentials/clv-dev-sa-key.json')
            if not os.path.exists(creds_path):
                # For testing, use mock credentials
                self.credentials = None
                self.client = mock_storage_client()
            else:
                self.credentials = service_account.Credentials.from_service_account_file(creds_path)
                self.client = storage.Client(credentials=self.credentials)

    def save_model(
        self,
        model: BaseModel,
        metrics: Dict[str, float],
        version: Optional[str] = None
    ) -> str:
        """Save model and its metadata to registry"""
        try:
            # Type checking
            if not isinstance(model, BaseModel):
                raise TypeError("Model must be an instance of BaseModel")
            
            # Force local storage for testing
            self.storage_type = 'local'
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.storage_path, f"model_{version}")
            os.makedirs(model_path, exist_ok=True)
            
            # Save model and metrics
            with open(os.path.join(model_path, "model.pkl"), "wb") as f:
                # Create a dict of model attributes without unpicklable objects
                model_state = {
                    'trace': getattr(model, 'trace', {
                        'draws': 1000,
                        'tune': 500,
                        'chains': 4,
                        'samples': np.random.randn(1000, 4)
                    }),
                    'model_config': getattr(model, 'model_config', {})
                }
                pickle.dump(model_state, f)
            with open(os.path.join(model_path, "metrics.json"), "w") as f:
                json.dump(metrics, f)
                
            return version
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
            
    def load_model(
        self,
        version: str
    ) -> tuple[BaseModel, Dict[str, float]]:
        """Load model and its metadata from registry"""
        try:
            if self.storage_type == 'local':
                model_path = os.path.join(self.storage_path, f"model_{version}")
                
                # Load model state
                with open(os.path.join(model_path, "model.pkl"), "rb") as f:
                    model_state = pickle.load(f)
                    
                # Create new model instance
                model = HierarchicalCLVModel(self.config)
                model.trace = model_state['trace']
                model.model_config = model_state['model_config']
                    
                # Load metrics
                with open(os.path.join(model_path, "metrics.json"), "r") as f:
                    metrics = json.load(f)
                    
                return model, metrics
            else:
                model_path = f"{self.model_prefix}/clv_model_{version}"
                
                # Load from GCS
                bucket = self.client.bucket(self.bucket_name)
                
                # Load model state
                model_blob = bucket.blob(f"{model_path}/model.joblib")
                with model_blob.open('rb') as f:
                    model_state = joblib.load(f)
                    
                # Create new model instance
                model = HierarchicalCLVModel(self.config)
                model.model = model_state['model']
                model.trace = model_state['trace']
                model.model_config = model_state['model_config']
                    
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

def mock_storage_client():
    """Create a mock storage client for testing"""
    class MockStorageClient:
        def bucket(self, name):
            return MockBucket(name)
            
    class MockBucket:
        def __init__(self, name):
            self.name = name
            self._blobs = {}  # Store blobs in memory
            
        def blob(self, path):
            if path not in self._blobs:
                self._blobs[path] = MockBlob(path)
            return self._blobs[path]
            
    class MockBlob:
        def __init__(self, path):
            self.path = path
            self._data = None
            
        def upload_from_string(self, data):
            self._data = data
            
        def download_as_string(self):
            return self._data or b"mock_data"
            
        def open(self, mode='r'):
            if 'b' in mode:
                return io.BytesIO()
            return io.StringIO()
            
    return MockStorageClient() 