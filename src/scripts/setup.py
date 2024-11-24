import click
import logging
from pathlib import Path
from src.infrastructure.gcp_setup import setup_gcp_resources, check_gcp_credentials
from src.pipeline.clv.config import CLVConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--project-id', required=True, help='GCP Project ID')
@click.option('--bucket-name', required=True, help='GCS Bucket Name')
@click.option('--dataset-id', required=True, help='BigQuery Dataset ID')
@click.option('--location', default='US', help='GCP Resource Location')
def setup_environment(project_id: str, bucket_name: str, dataset_id: str, location: str):
    """Setup development environment and GCP resources"""
    try:
        # Verify configs exist
        config_loader = CLVConfigLoader()
        
        # Check GCP credentials
        if not check_gcp_credentials():
            raise ValueError("GCP credentials not configured")
            
        # Setup GCP resources
        success, error = setup_gcp_resources(
            project_id=project_id,
            bucket_name=bucket_name,
            dataset_id=dataset_id,
            location=location
        )
        
        if not success:
            raise ValueError(f"Failed to setup GCP resources: {error}")
            
        logger.info("Environment setup completed successfully")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    setup_environment() 