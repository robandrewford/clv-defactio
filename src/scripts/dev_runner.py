# scripts/dev_runner.py
import click
import logging
from pathlib import Path
from src.pipeline.clv import HierarchicalCLVRunner
from src.infrastructure.gcp_setup import check_gcp_credentials

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_development_environment():
    """Setup development environment and verify credentials"""
    try:
        # Check GCP credentials
        if not check_gcp_credentials():
            raise ValueError("GCP credentials not properly configured")
        
        # Verify config files exist
        config_dir = Path("src/config")
        required_configs = [
            "deployment_config.yaml",
            "model_config.yaml",
            "segment_config.yaml",
            "pipeline_config.yaml"
        ]
        
        for config in required_configs:
            if not (config_dir / config).exists():
                raise FileNotFoundError(f"Missing config file: {config}")
                
        logger.info("Development environment setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Development setup failed: {str(e)}")
        return False

@click.group()
def cli():
    """Development CLI for CLV Pipeline"""
    pass

@cli.command()
@click.option('--input-table', required=True, help='BigQuery input table')
@click.option('--output-bucket', required=True, help='GCS output bucket')
@click.option('--local-test', is_flag=True, help='Run in local test mode')
def run_pipeline(input_table, output_bucket, local_test):
    """Run CLV pipeline in development"""
    try:
        # Setup environment
        if not setup_development_environment():
            raise ValueError("Development environment setup failed")
            
        # Initialize runner
        runner = HierarchicalCLVRunner()
        
        # Run pipeline
        logger.info(f"Starting pipeline with input: {input_table}")
        job = runner.run_pipeline(
            input_table=input_table,
            output_bucket=output_bucket
        )
        
        logger.info(f"Pipeline job started: {job.job_id}")
        
        # Monitor job status
        status = runner.get_pipeline_status(job.job_id)
        logger.info(f"Initial job status: {status['state']}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    cli()
