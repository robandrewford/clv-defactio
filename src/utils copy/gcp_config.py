# Development GCP configuration for CLV 360
# src/utils/gcp_config.py
import os

from dotenv import load_dotenv
from google.cloud import aiplatform, bigquery, storage


class GCPDevConfig:
    """Manages GCP development configuration"""

    def __init__(self):
        load_dotenv()  # Load .env file
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.region = os.getenv("GCP_REGION")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def initialize_clients(self):
        """Initialize GCP clients for development"""
        try:
            # Initialize clients
            self.storage_client = storage.Client()
            self.bq_client = bigquery.Client()
            self.ai_client = aiplatform.gapic.PipelineServiceClient()

            # Test connections
            self._test_connections()

            print("GCP clients initialized successfully")

        except Exception as e:
            print(f"Failed to initialize GCP clients: {str(e)}")
            raise

    def _test_connections(self):
        """Test GCP connections"""
        try:
            # Test Storage
            buckets = list(self.storage_client.list_buckets())
            print(f"Connected to Storage, found {len(buckets)} buckets")

            # Test BigQuery
            datasets = list(self.bq_client.list_datasets())
            print(f"Connected to BigQuery, found {len(datasets)} datasets")

        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            raise
