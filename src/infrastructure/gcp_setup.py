from google.cloud import storage, bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def check_gcp_credentials() -> bool:
    """Verify GCP credentials are properly configured"""
    try:
        credentials, project = default()
        return credentials.valid
    except DefaultCredentialsError as e:
        logger.error(f"GCP credentials not found: {str(e)}")
        return False

def setup_gcp_resources(
    project_id: str,
    bucket_name: str,
    dataset_id: str,
    location: str = "US"
) -> Tuple[bool, Optional[str]]:
    """Setup required GCP resources for CLV pipeline"""
    try:
        # Check credentials first
        if not check_gcp_credentials():
            raise ValueError("Invalid GCP credentials")

        # Setup Storage
        storage_client = storage.Client()
        try:
            bucket = storage_client.get_bucket(bucket_name)
            logger.info(f"Using existing bucket: {bucket_name}")
        except Exception:
            bucket = storage_client.create_bucket(bucket_name, location=location)
            logger.info(f"Created new bucket: {bucket_name}")

        # Setup BigQuery
        bq_client = bigquery.Client()
        try:
            dataset = bq_client.get_dataset(f"{project_id}.{dataset_id}")
            logger.info(f"Using existing dataset: {dataset_id}")
        except Exception:
            dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
            dataset.location = location
            dataset = bq_client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Created new dataset: {dataset_id}")

        return True, None

    except Exception as e:
        error_msg = f"Failed to setup GCP resources: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def cleanup_gcp_resources(
    project_id: str,
    bucket_name: str,
    dataset_id: str
) -> Tuple[bool, Optional[str]]:
    """Clean up GCP resources (for testing/development)"""
    try:
        if not check_gcp_credentials():
            raise ValueError("Invalid GCP credentials")

        # Clean up Storage
        storage_client = storage.Client()
        try:
            bucket = storage_client.get_bucket(bucket_name)
            bucket.delete(force=True)
            logger.info(f"Deleted bucket: {bucket_name}")
        except Exception as e:
            logger.warning(f"Could not delete bucket: {str(e)}")

        # Clean up BigQuery
        bq_client = bigquery.Client()
        try:
            dataset_ref = f"{project_id}.{dataset_id}"
            bq_client.delete_dataset(
                dataset_ref,
                delete_contents=True,
                not_found_ok=True
            )
            logger.info(f"Deleted dataset: {dataset_id}")
        except Exception as e:
            logger.warning(f"Could not delete dataset: {str(e)}")

        return True, None

    except Exception as e:
        error_msg = f"Failed to cleanup GCP resources: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
